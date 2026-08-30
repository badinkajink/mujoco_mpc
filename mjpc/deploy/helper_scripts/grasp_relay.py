#!/usr/bin/env python3
"""grasp_relay.py -- STRAT 27 close-gate relay: h12_control_node <-> magpie gripper.

SCOPE (one job): carry the lean task's grasp close command to the magpie
gripper and carry the aperture verdict back. The GATE DECISION (when to close,
retry budget, ladder regress) lives node-side in lean.cc's grasp-gate lambda
(the release-gate precedent: gates in one process, one clock); this relay only
ACTUATES and JUDGES the close. Design: docs/strat27_retrieval_design_2026-08-24.md.

CHAIN:
  DDS rt/grasp_gate  (std_msgs String, "CLOSE" @10Hz while the gate holds)
    -> ROS2 /right/gripper/close (std_srvs Trigger, magpie_hand_bridge surface)
    -> wait settle, read /right/gripper/state (magpie_msgs GripperState,
       .position = aperture mm)
    -> aperture >= --object-min-mm (default 25; a held 50 mm block reads ~50,
       an empty close reads ~0)  =>  DDS rt/grasp_ack "closed"
       else "empty" + REOPEN (/right/gripper/open) so the node-side retry
       re-runs the approach with clear jaws.

THERMAL RULE (08-24 research, ticket 10): retries only make sense when the
session entered with plan >= 30/s. The relay enforces the operator side of
that: pass --no-retry to answer EMPTY exactly once and then answer every
further CLOSE with "closed" (advance = recover empty-handed) instead of
feeding the node's retry loop on a soaked session.

Run (needs ROS2 overlay + magpie_msgs sourced, same shell as the bridges):
  python3 grasp_relay.py                # real: right gripper
  python3 grasp_relay.py --selftest     # DDS/ROS-free logic check
"""
import argparse
import os
import sys
import time


def judge(aperture_mm, object_min_mm, open_min_mm=80.0):
    """closed-on-object iff the jaws stopped >= object_min_mm apart AND
    actually moved (< open_min_mm). Real 29_55 (2026-08-30): the magpie sat
    in OVERLOAD after closing on nothing, ignored the close, aperture stayed
    96 mm (= fully open) and the old judge called that 'object held'."""
    return object_min_mm <= aperture_mm < open_min_mm


def _selftest():
    ok = True
    for ap, want in [(50.0, True), (48.0, True), (25.0, True),
                     (24.9, False), (2.0, False), (0.0, False),
                     (96.0, False), (80.0, False), (79.9, True), (-1.0, False)]:
        got = judge(ap, 25.0)
        ok &= got == want
        print(f"[selftest] aperture {ap:5.1f}mm -> "
              f"{'closed' if got else 'empty'} ({'ok' if got == want else 'FAIL'})")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="right")
    ap.add_argument("--gate-topic", default="rt/grasp_gate")
    ap.add_argument("--ack-topic", default="rt/grasp_ack")
    ap.add_argument("--object-min-mm", type=float, default=25.0,
                    help="aperture at/above this after close = object held "
                         "(5x5cm block reads ~50; empty reads ~0)")
    ap.add_argument("--open-min-mm", type=float, default=80.0,
                    help="aperture at/above this after a close = the jaws did "
                         "NOT move (open reads ~96): reset_overload + re-close "
                         "once, and never report 'held'")
    ap.add_argument("--settle-sec", type=float, default=1.2,
                    help="wait after the close call before judging aperture")
    ap.add_argument("--no-retry", action="store_true",
                    help="thermal rule: soaked session (entry plan <30/s) -- "
                         "answer EMPTY once, then 'closed' so the ladder "
                         "recovers empty-handed instead of retrying")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(_selftest())

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from base_estimator_node import _pick_iface
    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                             ChannelSubscriber, ChannelPublisher)
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_

    iface, why = _pick_iface(a.iface)
    print(f"[relay] DDS interface = {iface or 'autodetermine'} ({why})")
    ChannelFactoryInitialize(a.domain, iface) if iface else ChannelFactoryInitialize(a.domain)

    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger
    from magpie_msgs.msg import GripperState

    state = {"cmd": None, "aperture": None, "busy": False, "empties": 0}
    gate_sub = ChannelSubscriber(a.gate_topic, String_)
    gate_sub.Init(lambda m: state.__setitem__("cmd", m.data), 10)
    ack_pub = ChannelPublisher(a.ack_topic, String_)
    ack_pub.Init()
    ack_msg = std_msgs_msg_dds__String_()

    class Relay(Node):
        def __init__(self):
            super().__init__("grasp_relay")
            ns = f"/{a.side}/gripper"
            self.cli_close = self.create_client(Trigger, f"{ns}/close")
            self.cli_open = self.create_client(Trigger, f"{ns}/open")
            # 2026-08-30: closing on nothing / on an edge trips the magpie
            # OVERLOAD latch; while latched it ignores open AND close (real
            # 29_54 -> 29_55). reset_overload clears it.
            self.cli_reset = self.create_client(Trigger, f"{ns}/reset_overload")
            self.create_subscription(GripperState, f"{ns}/state",
                                     lambda m: state.__setitem__("aperture",
                                                                 m.position), 10)
            # 2026-08-29: tick() is driven from the MAIN loop (see below), not a
            # ROS timer -- call() spins the executor to wait for the service
            # future, and spinning from inside a timer callback wedges the
            # single-threaded executor: the relay answered ONE command and then
            # went deaf for the rest of the session (real 29_33->34, 29_35->41).
            self.get_logger().info(
                f"grasp_relay: {a.gate_topic} -> {ns}/close -> aperture judge "
                f"(>= {a.object_min_mm}mm = object) -> {a.ack_topic}"
                + (" [NO-RETRY thermal mode]" if a.no_retry else ""))

        def call(self, cli, name):
            if not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().error(f"{name} service unavailable")
                return False
            fut = cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.5)
            return fut.done() and fut.result() is not None and fut.result().success

        def tick(self):
            # 2026-08-29 OPEN-on-stand: the node publishes "OPEN" for ~1 s on
            # every stand_up entry; open the jaws once per burst.
            if state["cmd"] == "OPEN" and not state["busy"]:
                state["cmd"] = None
                if time.time() - state.get("last_open", 0.0) < 3.0:
                    return                      # same 1 s burst: already opened
                state["busy"] = True
                try:
                    self.get_logger().info("OPEN received -> reset_overload + opening gripper")
                    self.call(self.cli_reset, "reset_overload")
                    time.sleep(0.3)
                    self.call(self.cli_open, "open")
                    state["last_open"] = time.time()
                finally:
                    state["busy"] = False
                return
            if state["cmd"] != "CLOSE" or state["busy"]:
                return
            state["busy"] = True
            try:
                # 2026-08-30 (29_66/67): EVERY first close after an open was
                # ignored (aperture stayed 96-97 mm) and only the post-reset
                # re-close moved the jaws -- the overload latch is set by the
                # open itself. Clear it before every close, not just after.
                # 29_68: reset+close 0.3 s apart was STILL ignored on the first
                # try every time and only the second reset+close moved the jaws
                # -> the reset needs longer to take. 1.2 s.
                self.get_logger().info("CLOSE received -> reset_overload + closing gripper")
                self.call(self.cli_reset, "reset_overload")
                time.sleep(1.2)
                ok = self.call(self.cli_close, "close")
                # 2026-08-29: the magpie close service blocks for the whole
                # motion and often outlives the future wait (real 29_33/29_42:
                # 'FAILED' while the jaws had closed on the block, aperture
                # 51 mm). Do NOT answer blind on a timed-out call -- fall
                # through and let the APERTURE decide, same as a clean call.
                if not ok:
                    self.get_logger().warn("close call did not return in time "
                                           "-- judging by aperture anyway")
                time.sleep(a.settle_sec)
                aper = state["aperture"]
                if aper is not None and aper >= a.open_min_mm:
                    # jaws never moved: overload latch -> clear it, close again
                    self.get_logger().warn(
                        f"jaws did not move (aperture {aper:.0f}mm) -> "
                        "reset_overload + re-close")
                    self.call(self.cli_reset, "reset_overload")
                    time.sleep(0.3)
                    self.call(self.cli_close, "close")
                    time.sleep(a.settle_sec)
                    aper = state["aperture"]
                if aper is None:
                    self.answer("closed", "no aperture telemetry -- advancing")
                    return
                if judge(aper, a.object_min_mm, a.open_min_mm):
                    self.answer("closed", f"object held (aperture {aper:.0f}mm)")
                elif a.no_retry and state["empties"] >= 1:
                    self.answer("closed",
                                f"EMPTY (aperture {aper:.0f}mm) but NO-RETRY "
                                "thermal mode -- advancing empty")
                else:
                    state["empties"] += 1
                    self.call(self.cli_reset, "reset_overload")  # empty close latches overload
                    time.sleep(0.3)
                    self.call(self.cli_open, "open")   # clear jaws for retry
                    self.answer("empty", f"EMPTY (aperture {aper:.0f}mm) -> "
                                         "reopened for retry")
            finally:
                state["busy"] = False

        def answer(self, verdict, why):
            self.get_logger().info(f"ack '{verdict}': {why}")
            ack_msg.data = verdict
            ack_pub.Write(ack_msg)
            state["cmd"] = None   # consume; node re-raises CLOSE on next fire

    rclpy.init()
    node = Relay()
    # 2026-08-29: start from a known state -- jaws OPEN.
    node.get_logger().info("startup -> reset_overload + opening gripper")
    node.call(node.cli_reset, "reset_overload")
    time.sleep(0.3)
    node.call(node.cli_open, "open")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)   # subscriptions/timers
            node.tick()                               # main thread: safe to spin inside call()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
