#!/usr/bin/env python3
"""sim_tag_anchor.py -- TWIN-BENCH stand-in for tag_bridge_node.py.

Simulates the v4+april anchor at the INFORMATION level: subscribes to the
twin's ground-truth stream (rt/sportmodestate, published by h12_mujoco.py)
and republishes its xy as a mode-2 POSITION-ONLY aux sample on rt/aux_odom
with the real bridge's measured characteristics:

    rate     ~15 Hz   (real: 10-20 Hz depending on visible tags)
    noise    3 mm 1-sigma per axis (real still capture: 2-5 mm/120 s windows)
    latency  60 ms    (capture + detect + solve, single frame)

No camera rendering / PnP theater -- the real anchor's entire contract with
v4 is "xy in the world frame, mm-level, position only" (tag_bridge_node.py
mode=2), and that is exactly what is reproduced here. Frame note: on the
real robot the anchor xy lives in the table/IMU-world frame; in the twin the
truth topic is twin-world. v4's offset-latch (aux_offset_xy) absorbs any
constant frame offset in both cases, so the fusion regime is identical.

Usage (bench wiring, see realchain_bench.sh EST_MODE=v4a):
  .venv/bin/python sim_tag_anchor.py --truth-topic rt/sportmodestate
"""
import argparse
import collections
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth-topic", default="rt/sportmodestate",
                    help="twin ground-truth SportModeState_ stream")
    ap.add_argument("--aux-topic", default="rt/aux_odom")
    ap.add_argument("--rate", type=float, default=15.0, help="anchor publish Hz")
    ap.add_argument("--noise-mm", type=float, default=3.0, help="1-sigma per axis")
    ap.add_argument("--latency-ms", type=float, default=60.0,
                    help="age of the sample when published")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None,
                    help="DDS interface, e.g. lo for a same-host twin. Must "
                         "match the twin and the estimator: unset means the "
                         "SDK autodetermines, and an anchor that autodetermines "
                         "while everyone else is pinned to lo publishes "
                         "perfectly good samples onto a bus nobody reads "
                         "(v4 just keeps saying aux=IDLE).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-offset", type=float, nargs=2, default=[0.0, 0.0],
                    help="2026-08-13: constant xy added to every sample -- "
                         "emulates a mis-latched/mis-calibrated anchor frame "
                         "(0 0 = absolute truth, the tag_bridge --abs-world "
                         "contract)")
    a = ap.parse_args()

    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                             ChannelSubscriber, ChannelPublisher)
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

    if a.iface:
        ChannelFactoryInitialize(a.domain, a.iface)
    else:
        ChannelFactoryInitialize(a.domain)
    rng = np.random.default_rng(a.seed)
    # (t_mono, xy) ring buffer so the published sample is latency-ms OLD
    buf = collections.deque(maxlen=256)

    def on_truth(m):
        buf.append((time.monotonic(), float(m.position[0]), float(m.position[1])))

    sub = ChannelSubscriber(a.truth_topic, SportModeState_)
    sub.Init(on_truth, 20)
    pub = ChannelPublisher(a.aux_topic, SportModeState_)
    pub.Init()
    out = unitree_go_msg_dds__SportModeState_()

    lat = a.latency_ms / 1000.0
    n, t0, last_note = 0, time.monotonic(), 0.0
    print(f"[simtag] {a.truth_topic} -> {a.aux_topic} mode=2 "
          f"@{a.rate:.0f}Hz noise {a.noise_mm}mm latency {a.latency_ms:.0f}ms",
          flush=True)
    while True:
        time.sleep(1.0 / a.rate)
        now = time.monotonic()
        # newest sample at least `lat` old
        pick = None
        for t, x, y in reversed(buf):
            if now - t >= lat:
                pick = (x, y)
                break
        if pick is None:
            continue
        nx, ny = rng.normal(0.0, a.noise_mm / 1000.0, 2)
        out.position[0] = pick[0] + nx + a.frame_offset[0]
        out.position[1] = pick[1] + ny + a.frame_offset[1]
        out.position[2] = 0.0
        for k in range(3):
            out.velocity[k] = 0.0
        out.mode = 2                    # POSITION-ONLY (v4 contract)
        pub.Write(out)
        n += 1
        if now - last_note > 10.0:
            last_note = now
            print(f"[simtag] {n} anchors ({n / max(now - t0, 1e-6):.1f}Hz)", flush=True)


if __name__ == "__main__":
    main()
