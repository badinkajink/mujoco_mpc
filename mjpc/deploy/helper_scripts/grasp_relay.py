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


def judge(aperture_mm, object_min_mm):
    """closed-on-object iff the jaws stopped >= object_min_mm apart."""
    return aperture_mm >= object_min_mm


def _selftest():
    ok = True
    for ap, want in [(50.0, True), (48.0, True), (25.0, True),
                     (24.9, False), (2.0, False), (0.0, False)]:
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
            self.create_subscription(GripperState, f"{ns}/state",
                                     lambda m: state.__setitem__("aperture",
                                                                 m.position), 10)
            self.create_timer(0.1, self.tick)
            self.get_logger().info(
                f"grasp_relay: {a.gate_topic} -> {ns}/close -> aperture judge "
                f"(>= {a.object_min_mm}mm = object) -> {a.ack_topic}"
                + (" [NO-RETRY thermal mode]" if a.no_retry else ""))

        def call(self, cli, name):
            if not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().error(f"{name} service unavailable")
                return False
            fut = cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=4.0)
            return fut.done() and fut.result() is not None and fut.result().success

        def tick(self):
            if state["cmd"] != "CLOSE" or state["busy"]:
                return
            state["busy"] = True
            try:
                self.get_logger().info("CLOSE received -> closing gripper")
                if not self.call(self.cli_close, "close"):
                    # actuation failed: answer closed so the ladder advances to
                    # recovery instead of hanging on the ack (fail-safe forward)
                    self.answer("closed", "close call FAILED -- advancing")
                    return
                time.sleep(a.settle_sec)
                aper = state["aperture"]
                if aper is None:
                    self.answer("closed", "no aperture telemetry -- advancing")
                    return
                if judge(aper, a.object_min_mm):
                    self.answer("closed", f"object held (aperture {aper:.0f}mm)")
                elif a.no_retry and state["empties"] >= 1:
                    self.answer("closed",
                                f"EMPTY (aperture {aper:.0f}mm) but NO-RETRY "
                                "thermal mode -- advancing empty")
                else:
                    state["empties"] += 1
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
