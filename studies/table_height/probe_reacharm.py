#!/usr/bin/env python3
"""Can the reaching arm hit the rung-2 advance target from the brace posture?

Rung 2 of strategy 25 advances when the right gripper jaw tip comes within 70 mm
of (near edge + 0.55, table centre - 0.04, face + 0.15) and holds it for 2 s.
Every failing height in the A/B enters rung 2 and never leaves it, so this is the
gate the runs die at.

With the trunk held at the brace posture -- shipped, or re-solved for the slab --
and the right arm's 7 joints free, how close can the tip get? That bounds what
any weight on any reach cost could buy from that posture.
"""
import argparse, json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model, geometry

RARM = list(range(26, 33))
TIP = np.array([0.2254, -0.0118, -0.1062])
RTT = (0.55, 0.04, 0.15)
TOL = 0.07


def tip_of(m, d, grip):
    return d.xpos[grip] + d.xmat[grip].reshape(3, 3) @ TIP


def closest(m, q0, tgt, iters=900):
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    grip = R.nid(m, mujoco.mjtObj.mjOBJ_BODY, "right_magpie_gripper")
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    best = 1e9
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        r = tip_of(m, d, grip) - tgt
        best = min(best, float(np.linalg.norm(r)))
        mujoco.mj_jacBody(m, d, jp, jr, grip)
        # jaw tip, not the body origin: shift the rotational part onto the offset
        off = d.xmat[grip].reshape(3, 3) @ TIP
        J = jp[:, RARM] + np.cross(jr[:, RARM].T, off).T
        dq = -J.T @ np.linalg.solve(J @ J.T + 3e-4 * np.eye(3), r)
        full = np.zeros(m.nv); full[RARM] = np.clip(dq, -0.08, 0.08)
        mujoco.mj_integratePos(m, d.qpos, full, 1.0)
        for j in range(1, m.njnt):
            if m.jnt_limited[j] and int(m.jnt_dofadr[j]) in RARM:
                a = m.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], m.jnt_range[j][0], m.jnt_range[j][1])
    mujoco.mj_kinematics(m, d)
    return best, d.qpos.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "reacharm.json"))
    a = ap.parse_args()
    base = pristine_model()
    g = geometry(base)
    rows = []
    print("rung-2 target: (near edge + %.2f, centre - %.2f, face + %.2f), "
          "tolerance %.0f mm" % (RTT[0], RTT[1], RTT[2], 1000 * TOL))
    print("\n%-8s %-9s %-13s %-13s %s"
          % ("face", "target z", "shipped (mm)", "re-solved (mm)", "shoulder z"))
    for h in np.round(np.arange(0.785, 1.1351, 0.025), 3):
        m = pristine_model()
        tgc = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
        d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
        ctr, half = d0.geom_xpos[tgc].copy(), m.geom_size[tgc].copy()
        tgt = np.array([ctr[0] - half[0] + RTT[0], ctr[1] - RTT[1], h + RTT[2]])
        t = g["pad0"].copy();  t[2] += h - g["nom"]
        t2 = g["pad20"].copy(); t2[2] += h - g["nom"]
        q_re, _ = R.solve(m, g["q0"], t, t2)
        d_ship, _ = closest(m, g["q0"], tgt)
        d_re, qq = closest(m, q_re, tgt)
        dd = mujoco.MjData(m); dd.qpos[:] = q_re; mujoco.mj_kinematics(m, dd)
        sh = float(dd.xpos[R.nid(m, mujoco.mjtObj.mjOBJ_BODY,
                                 "right_shoulder_yaw_link")][2])
        rows.append(dict(h=float(h), target_z=float(tgt[2]),
                         shipped_mm=1000 * d_ship, resolved_mm=1000 * d_re,
                         shoulder_z=sh))
        print("%-8.3f %-9.3f %-13.0f %-13.0f %.3f"
              % (h, tgt[2], 1000 * d_ship, 1000 * d_re, sh))
    json.dump(dict(tol_mm=1000 * TOL, rtt=RTT, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
