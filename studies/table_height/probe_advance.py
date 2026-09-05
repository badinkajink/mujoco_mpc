#!/usr/bin/env python3
"""What the rung-2 advance gate actually sees, replayed from the qpos dumps.

Rung 2 of strategy 25 is the targeting rung: it carries `reach_target_table`
[0.55, 0.04, 0.15], so `lean::TransitionLocked` builds its advance target from
the slab -- near edge + 0.55 in x, table centre - 0.04 in y, face + 0.15 in z --
and advances when the RIGHT gripper jaw tip is within
`target_distance_tolerance` (0.07 m) of it for `success_sustain_time` (2 s).

Every failing height in the A/B reaches rung 2 and never leaves it, so this is
the gate the runs die at. Replaying the logged qpos gives the exact distance the
gate saw, with no extra sim time.

usage: probe_advance.py runs/ab/on runs/lead/on ...
"""
import csv, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height

TIP = np.array([0.2254, -0.0118, -0.1062])   # kGripperTipLocal, lean.cc
RTT = (0.55, 0.04, 0.15)                     # h12_brace_targeting rung 2
TOL = 0.07


def one(qpos_csv, states_csv, face):
    m = pristine_model()
    set_table_height(m, face)
    d = mujoco.MjData(m)
    tgc = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    grip = R.nid(m, mujoco.mjtObj.mjOBJ_BODY, "right_magpie_gripper")
    mujoco.mj_forward(m, d)
    ctr, half = d.geom_xpos[tgc].copy(), m.geom_size[tgc].copy()
    tgt = np.array([ctr[0] - half[0] + RTT[0], ctr[1] - RTT[1],
                    ctr[2] + half[2] + RTT[2]])
    phase = {}
    for r in csv.DictReader(open(states_csv)):
        phase[round(float(r["t"]), 3)] = (r["phase_name"], int(float(r["phase"])))
    best, best_t, best_in2 = 1e9, -1, 1e9
    n2 = 0
    for r in csv.DictReader(open(qpos_csv)):
        t = round(float(r["t"]), 3)
        q = np.array([float(r["q%d" % i]) for i in range(m.nq)])
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        tip = d.xpos[grip] + d.xmat[grip].reshape(3, 3) @ TIP
        dist = float(np.linalg.norm(tip - tgt))
        if dist < best:
            best, best_t = dist, t
        ph = phase.get(t)
        if ph and ph[1] == 2:
            n2 += 1
            best_in2 = min(best_in2, dist)
    return tgt, best, best_t, best_in2, n2


def main():
    print("rung-2 advance: right jaw tip to (near_edge+%.2f, ctr-%.2f, face+%.2f),"
          " tolerance %.0f mm\n" % (RTT[0], RTT[1], RTT[2], 1000 * TOL))
    print("%-24s %-7s %-4s %-9s %-11s %-11s %s"
          % ("arm", "face", "seed", "target z", "closest (mm)", "in rung 2", "verdict"))
    for d0 in sys.argv[1:]:
        s = os.path.join(d0, "summary.csv")
        if not os.path.exists(s):
            continue
        for r in csv.DictReader(open(s)):
            q = (r.get("csv") or "").replace(".csv", ".qpos.csv")
            if not os.path.exists(q):
                continue
            face = float(r["table_h"])
            tgt, best, bt, in2, n2 = one(q, r["csv"], face)
            out = ("fell" if r["fell"] == "1"
                   else "complete" if r["complete"] == "1" else "stall")
            print("%-24s %-7.3f %-4s %-9.3f %-11.0f %-11s %s"
                  % (d0.split("runs/")[-1] + " " + out, face, r["seed"], tgt[2],
                     1000 * best, "--" if n2 == 0 else "%.0f" % (1000 * in2),
                     "reaches" if best <= TOL else "NEVER within tolerance"))


if __name__ == "__main__":
    main()
