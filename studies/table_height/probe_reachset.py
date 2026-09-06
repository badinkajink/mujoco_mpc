#!/usr/bin/env python3
"""Which rung-2 waypoints the reaching arm can hold while the brace stays seated.

Rung 2 demands the right jaw tip reach `reach_target_table` = [0.55, 0.04, 0.15]:
0.55 m in from the slab's near edge, 0.04 m off the centre line, 0.15 m above the
face. Those three numbers are fixed in the strategy JSON and were fitted at the
compiled 0.985 m face.

This asks the kinematic question directly. At each slab height, seat both brace
pads on the face (the `brace_pose_track` solution), plant the feet, and sweep the
demanded waypoint over a grid of (in-from-edge, above-face). A cell is reachable
when the tip lands within `--tol` of it with the pads and feet still held.

Feet planted and pads seated is a strict subset of what the controller must also
do -- stay upright, stay inside its cost caps -- so an unreachable cell here is
unreachable for any amount of cost tuning. A reachable cell is necessary, not
sufficient.

usage: probe_reachset.py [--json out.json]
"""
import argparse, os, sys, json
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height

HEIGHTS = (0.785, 0.885, 0.985, 1.035, 1.085)
RTT = (0.55, 0.04, 0.15)
XS = np.round(np.arange(0.30, 0.751, 0.025), 4)
ZS = np.round(np.arange(0.00, 0.301, 0.025), 4)


def brace_pose(m, face, nominal):
    """The shipped rung-2 key, with both pads re-seated on this slab.

    `m` already carries the moved slab, so its own face is `face`; the shift the
    pads need is measured against the COMPILED face, which the caller reads off
    the pristine model before moving anything.
    """
    k = R.nid(m, mujoco.mjtObj.mjOBJ_KEY, "forearm_brace_lean")
    q0 = np.array(m.key_qpos).reshape(-1, m.nq)[k].copy()
    d = mujoco.MjData(m); d.qpos[:] = q0
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    pad = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, R.PAD)
    pad2 = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, R.PAD2)
    p1, p2 = d.geom_xpos[pad].copy(), d.geom_xpos[pad2].copy()
    dz = face - nominal               # the slab moved by this much
    p1[2] += dz; p2[2] += dz
    q, err = R.solve(m, q0, p1, p2, iters=800)
    return q, err, p1, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    nominal = R.face_of(pristine_model())
    print("compiled face %.3f m\n" % nominal)
    for face in HEIGHTS:
        m = pristine_model(); set_table_height(m, face)
        tgc = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
        d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
        ctr, half = d0.geom_xpos[tgc].copy(), m.geom_size[tgc].copy()
        near, cy, top = ctr[0] - half[0], ctr[1], ctr[2] + half[2]
        qb, eb, p1, p2 = brace_pose(m, face, nominal)
        grip = R.nid(m, mujoco.mjtObj.mjOBJ_BODY, "right_magpie_gripper")
        d = mujoco.MjData(m)
        grid = np.full((len(ZS), len(XS)), np.nan)
        for iz, rz in enumerate(ZS):
            q = qb.copy()                      # warm start along the row
            for ix, rx in enumerate(XS):
                tgt = np.array([near + rx, cy - RTT[1], top + rz])
                qs, _ = R.solve(m, q, p1, p2, target_tip=tgt, iters=400)
                d.qpos[:] = qs; mujoco.mj_kinematics(m, d)
                tip = d.xpos[grip] + d.xmat[grip].reshape(3, 3) @ R.TIP_LOCAL
                grid[iz, ix] = float(np.linalg.norm(tip - tgt))
                q = qs
        ok = grid < a.tol
        # the shipped cell
        ix = int(np.argmin(abs(XS - RTT[0]))); iz = int(np.argmin(abs(ZS - RTT[2])))
        print("face %.3f  brace seat resid %5.1f mm   shipped cell (%.3f, %.3f): "
              "tip err %6.1f mm  %s" % (face, 1000 * eb, RTT[0], RTT[2],
                                        1000 * grid[iz, ix],
                                        "REACHABLE" if ok[iz, ix] else "OUT"))
        for iz2, rz in enumerate(ZS):
            row = XS[ok[iz2]]
            if len(row):
                print("    z=+%.3f  x reachable %.3f .. %.3f m" %
                      (rz, row.min(), row.max()))
        out["%.3f" % face] = dict(grid=grid.tolist(), xs=XS.tolist(),
                                  zs=ZS.tolist(), seat_resid=eb)
    if a.json:
        json.dump(out, open(a.json, "w"))
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
