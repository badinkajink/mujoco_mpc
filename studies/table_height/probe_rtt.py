#!/usr/bin/env python3
"""Where the reaching jaw tip can actually get, per slab height.

Rung 2's advance target is built from three constants in the strategy JSON,
`reach_target_table` = [0.55, 0.04, 0.15]: x is measured in from the slab's near
edge, y out from the table centre line, z up from the face. They were fitted at
the compiled 0.985 m face and they do not move with the slab.

This replays every logged run and reports, for each height, the SAME three
numbers computed from where the jaw tip actually got -- the reachable waypoint
nearest the demanded one. The difference between the two is the gate error.

usage: probe_rtt.py --runs LABEL=dir [...] [--json out.json]
"""
import argparse, csv, json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height

TIP = np.array([0.2254, -0.0118, -0.1062])   # kGripperTipLocal, lean.cc
RTT = np.array([0.55, 0.04, 0.15])


def slab_rtt(m, d, tgc, grip, qcsv, scsv):
    """closest approach in phase 2, returned as the achieved rtt triple."""
    ctr, half = d.geom_xpos[tgc].copy(), m.geom_size[tgc].copy()
    tgt = np.array([ctr[0] - half[0] + RTT[0], ctr[1] - RTT[1],
                    ctr[2] + half[2] + RTT[2]])
    ph = {round(float(r["t"]), 3): int(float(r["phase"]))
          for r in csv.DictReader(open(scsv))}
    best, tip_best, t_best = 1e9, None, None
    for r in csv.DictReader(open(qcsv)):
        t = round(float(r["t"]), 3)
        if ph.get(t) != 2:
            continue
        d.qpos[:] = [float(r["q%d" % i]) for i in range(m.nq)]
        mujoco.mj_kinematics(m, d)
        tip = d.xpos[grip] + d.xmat[grip].reshape(3, 3) @ TIP
        n = float(np.linalg.norm(tip - tgt))
        if n < best:
            best, tip_best, t_best = n, tip.copy(), t
    if tip_best is None:
        return None
    ach = np.array([tip_best[0] - (ctr[0] - half[0]),
                    ctr[1] - tip_best[1],
                    tip_best[2] - (ctr[2] + half[2])])
    return dict(closest_mm=1000 * best, t=t_best, achieved=ach.tolist(),
                tip=tip_best.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    cache, recs = {}, []
    for spec in a.runs:
        label, path = spec.split("=", 1)
        s = os.path.join(path, "summary.csv")
        if not os.path.exists(s):
            continue
        for r in csv.DictReader(open(s)):
            q = (r.get("csv") or "").replace(".csv", ".qpos.csv")
            if not q or not os.path.exists(q):
                continue
            face = float(r["table_h"])
            if face not in cache:
                m = pristine_model(); set_table_height(m, face)
                d = mujoco.MjData(m); mujoco.mj_forward(m, d)
                cache[face] = (m, d,
                    R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision"),
                    R.nid(m, mujoco.mjtObj.mjOBJ_BODY, "right_magpie_gripper"))
            got = slab_rtt(*cache[face], q, r["csv"])
            if got is None:
                continue
            got.update(label=label, h=face, seed=int(r["seed"]),
                       outcome=("fell" if r["fell"] == "1" else
                                "complete" if r["complete"] == "1" else "stalled"))
            recs.append(got)

    print("%d runs reached rung 2\n" % len(recs))
    print("achieved reach_target_table, per height (median over runs)")
    print("  face      n   rtt_x    rtt_y    rtt_z   closest(mm)")
    hs = sorted({r["h"] for r in recs})
    med = {}
    for h in hs:
        g = [r for r in recs if r["h"] == h]
        A = np.array([r["achieved"] for r in g])
        med[h] = np.median(A, axis=0)
        print("  %.3f  %3d  %+.4f  %+.4f  %+.4f   %5.0f"
              % (h, len(g), *med[h], np.median([r["closest_mm"] for r in g])))
    print("\n  demanded  %+.4f  %+.4f  %+.4f" % tuple(RTT))
    print("\nspread within a height (max-min, mm)")
    for h in hs:
        A = np.array([r["achieved"] for r in recs if r["h"] == h])
        print("  %.3f  x %5.0f  y %5.0f  z %5.0f"
              % (h, *(1000 * (A.max(0) - A.min(0)))))
    if a.json:
        json.dump(dict(runs=recs, median={("%.3f" % h): med[h].tolist()
                                          for h in hs}),
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
