#!/usr/bin/env python3
"""aux_bus_probe.py -- who is publishing on rt/aux_odom right now?

2026-10-15. Several sources share the v4 aux bus (SportModeState_.mode):
  2 = tag_bridge (AprilTag position-only; robot-side since 09-12 via lean_bringup.sh)
  1 = lio_bridge --publish twist  (legacy: xy + the never-filled FAST-LIO twist)
  3 = lio_bridge --publish position (LiDAR position-only)
The 08-30 v4l stand mixed modes 1 and 2 without anyone noticing. Listen for a
few seconds and print per-mode rate, so a bringup can refuse/skip accordingly.

  python3 aux_bus_probe.py              # human-readable
  python3 aux_bus_probe.py --modes-only # prints e.g. "2,3" (empty = silent bus)
"""
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aux-topic", default="rt/aux_odom")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--modes-only", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from base_estimator_node import _pick_iface
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    iface, why = _pick_iface(a.iface)
    if iface:
        ChannelFactoryInitialize(a.domain, iface)
    else:
        ChannelFactoryInitialize(a.domain)

    counts, last = {}, {}

    def _on(m):
        k = int(m.mode)
        counts[k] = counts.get(k, 0) + 1
        last[k] = (float(m.position[0]), float(m.position[1]), float(m.position[2]))

    sub = ChannelSubscriber(a.aux_topic, SportModeState_)
    sub.Init(_on, 10)
    time.sleep(a.seconds)

    modes = sorted(counts)
    if a.modes_only:
        print(",".join(str(k) for k in modes))
        return
    print(f"[aux-probe] {a.aux_topic} over {a.seconds:.1f}s (iface {iface or 'auto'}: {why})")
    if not modes:
        print("[aux-probe]   silent -- no aux source publishing")
    names = {1: "lio_bridge twist (legacy)", 2: "tag_bridge", 3: "lio_bridge position"}
    for k in modes:
        x, y, z = last[k]
        print(f"[aux-probe]   mode {k} ({names.get(k, '?')}): {counts[k] / a.seconds:5.1f} Hz  "
              f"last xy=({x:+.3f},{y:+.3f}) p2={z:+.3f}")


if __name__ == "__main__":
    main()
