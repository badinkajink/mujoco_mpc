#!/usr/bin/env python3
"""With the trunk held at the shipped brace posture, what slab heights can the
arm alone still seat on?

The brace needs the forearm LEVEL and ON the surface: both pads at face height.
Their x and y are free -- the slab is 1.18 m deep and 0.595 m wide, so the pad may
sit anywhere on it. That leaves the left arm's 7 joints against 2 constraints, and
the question is simply how far the pad can travel in z before the shoulder runs
out of arm.

The answer bounds what any amount of cost tuning could buy without moving the
trunk, which is why it is worth computing before tuning anything.
"""
import os, sys, json
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model, geometry

ARM = list(range(19, 26))


def arm_only_seat(m, q0, z_pad, z_wrist, iters=800):
    """Both pad centres to the given world z; x, y and everything else free."""
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    pad = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, R.PAD)
    pad2 = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, R.PAD2)
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        r = np.array([d.geom_xpos[pad][2] - z_pad, d.geom_xpos[pad2][2] - z_wrist])
        if np.max(np.abs(r)) < 1e-7:
            break
        mujoco.mj_jacGeom(m, d, jp, jr, pad);  J1 = jp[2, ARM].copy()
        mujoco.mj_jacGeom(m, d, jp, jr, pad2); J2 = jp[2, ARM].copy()
        Jm = np.vstack([J1, J2])
        dq = -Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(2), r)
        full = np.zeros(m.nv); full[ARM] = np.clip(dq, -0.1, 0.1)
        mujoco.mj_integratePos(m, d.qpos, full, 1.0)
        for j in range(1, m.njnt):
            if m.jnt_limited[j] and int(m.jnt_dofadr[j]) in ARM:
                a = m.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], m.jnt_range[j][0], m.jnt_range[j][1])
    mujoco.mj_kinematics(m, d)
    return d.qpos.copy(), float(np.max(np.abs(r))), d.geom_xpos[pad].copy()


def main():
    m = pristine_model()
    g = geometry(m)
    pad0, pad20, nom = g["pad0"], g["pad20"], g["nom"]
    tgc = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    ctr, half = d.geom_xpos[tgc], m.geom_size[tgc]
    near, far = ctr[0] - half[0], ctr[0] + half[0]
    print("slab: x %.3f..%.3f m, y %.3f..%.3f m, face %.3f m"
          % (near, far, ctr[1] - half[1], ctr[1] + half[1], nom))
    print("\n%-8s %-9s %-9s %-8s %-8s %s"
          % ("face", "seat err", "pad x", "pad y", "on slab", "verdict"))
    ok = []
    for h in np.round(np.arange(0.835, 1.1401, 0.0125), 4):
        q, err, p = arm_only_seat(m, g["q0"], pad0[2] + (h - nom),
                                  pad20[2] + (h - nom))
        on = (near + 0.046 <= p[0] <= far) and abs(p[1] - ctr[1]) <= half[1]
        good = err < 1e-3 and on
        if good:
            ok.append(h)
        print("%-8.4f %-9.1f %-9.3f %-8.3f %-8s %s"
              % (h, 1000 * err, p[0], p[1], "yes" if on else "no",
                 "seats" if good else "cannot seat"))
    if ok:
        print("\ntrunk-frozen seating window: %.3f .. %.3f m (%.0f mm wide); "
              "compiled slab %.3f m" % (min(ok), max(ok), 1000*(max(ok)-min(ok)), nom))
    json.dump(dict(window=[min(ok), max(ok)] if ok else [],
                   near_edge=float(near), far_edge=float(far), nominal=nom),
              open(os.path.join(HERE, "figs", "armreach.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
