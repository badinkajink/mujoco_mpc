#!/usr/bin/env python3
"""What the reach-arm basin lock aims at, per slab.

`reach_arm_posture` (lean.cc, ships 0 = off) replaces the seven right-arm entries
of the posture target with the fixed config `reach_arm_q` on any rung that
carries `reach_target_table` with hover under `reach_arm_hgate`. The XML sets
that gate to 0.16 and strategy 25's rung 2 hovers at 0.15, so the lock is already
aimed at the rung the height sweep dies on.

This reports, from the brace pose re-seated on each slab, how far `reach_arm_q`
puts the jaw tip from the rung-2 gate target, against a right arm solved for that
target directly. The first is what one existing numeric buys; the second is the
ceiling for any per-height version of the same idea.

usage: probe_basin.py [--json out.json]
"""
import argparse, json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height
from probe_reachset import brace_pose, RTT

HEIGHTS = (0.785, 0.885, 0.985, 1.035, 1.085)
ARM_Q0 = 27          # qpos index of the right shoulder; 7 joints follow


def tip_of(m, q, grip):
    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    return d.xpos[grip] + d.xmat[grip].reshape(3, 3) @ R.TIP_LOCAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    m0 = pristine_model()
    nominal = R.face_of(m0)
    nid = R.nid(m0, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_arm_q")
    rq = np.array(m0.numeric_data[m0.numeric_adr[nid]:
                                  m0.numeric_adr[nid] + 7])
    print("reach_arm_q      = %s" % np.array2string(rq, precision=3))
    print("reach_arm_hgate  = %.2f   (rung 2 hover %.2f -> lock applies)"
          % (R.num(m0, "reach_arm_hgate", 0.10), RTT[2]))
    print("reach_arm_posture= %.1f   (0 = off, as shipped)\n"
          % R.num(m0, "reach_arm_posture", 0.0))
    print(" face   reach_arm_q: dx    dy    dz   dist(mm) | solved arm dist(mm)")
    out = {}
    for face in HEIGHTS:
        m = pristine_model(); set_table_height(m, face)
        tgc = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
        d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
        ctr, half = d0.geom_xpos[tgc].copy(), m.geom_size[tgc].copy()
        tgt = np.array([ctr[0] - half[0] + RTT[0], ctr[1] - RTT[1],
                        ctr[2] + half[2] + RTT[2]])
        qb, _, p1, p2 = brace_pose(m, face, nominal)
        grip = R.nid(m, mujoco.mjtObj.mjOBJ_BODY, "right_magpie_gripper")

        qf = qb.copy(); qf[ARM_Q0:ARM_Q0 + 7] = rq
        ef = tip_of(m, qf, grip) - tgt
        qs, _ = R.solve(m, qb, p1, p2, target_tip=tgt, iters=1200)
        es = tip_of(m, qs, grip) - tgt
        print(" %.3f  %14.0f %5.0f %5.0f %8.0f | %13.0f"
              % (face, 1000 * ef[0], 1000 * ef[1], 1000 * ef[2],
                 1000 * np.linalg.norm(ef), 1000 * np.linalg.norm(es)))
        out["%.3f" % face] = dict(
            fixed_mm=1000 * float(np.linalg.norm(ef)),
            fixed_err_mm=(1000 * ef).tolist(),
            solved_mm=1000 * float(np.linalg.norm(es)),
            solved_arm=qs[ARM_Q0:ARM_Q0 + 7].tolist())
    if a.json:
        json.dump(dict(reach_arm_q=rq.tolist(), heights=out),
                  open(a.json, "w"), indent=1)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
