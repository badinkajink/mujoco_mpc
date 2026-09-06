#!/usr/bin/env python3
"""How much of the per-height brace retarget lives in the floating base?

The posture cost tracks actuated joint angles.  The floating base (x, z, pitch)
is not an actuator, so any part of the retarget that lives there cannot be
commanded by `brace_pose_track`.  Split the delta and re-seat the pads with the
base pinned to see what the joints alone can do.
"""
import os, sys
import numpy as np, mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height
from probe_reachset import brace_pose

HEIGHTS = (0.785, 0.885, 0.985, 1.035, 1.085)

def pitch_of(q):
    return float(np.arcsin(np.clip(2.0*(q[3]*q[5] - q[6]*q[4]), -1, 1)))

def pad_err(m, q, face):
    d = mujoco.MjData(m); d.qpos[:] = q
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    out = {}
    for nm in (R.PAD, R.PAD2):
        g = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
        out[nm] = float(d.geom_xpos[g][2] - m.geom_size[g][0] - face)   # clearance above face
    return out

m0 = pristine_model()
nominal = R.face_of(m0)
kid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_KEY, "forearm_brace_lean")
q_ship = np.array(m0.key_qpos).reshape(-1, m0.nq)[kid].copy()

print("compiled face %.3f m   key=forearm_brace_lean" % nominal)
print()
print(" face | retarget delta:  base_x    base_z   pitch   | joint |dq| max/rms (rad) | pads with base PINNED (mm above face)")
for face in HEIGHTS:
    m = pristine_model(); set_table_height(m, face)
    qb, eb, p1, p2 = brace_pose(m, face, nominal)
    dbx = qb[0] - q_ship[0]
    dbz = qb[2] - q_ship[2]
    dpi = np.degrees(pitch_of(qb) - pitch_of(q_ship))
    dj  = qb[7:] - q_ship[7:]
    # joints-only pose: shipped base + retargeted joints
    q_joint = q_ship.copy(); q_joint[7:] = qb[7:]
    pe = pad_err(m, q_joint, face)
    print(" %.3f |               %+7.3f  %+7.3f  %+6.1f  |   %.3f / %.3f            |  forearm %+7.1f   wrist %+7.1f  (resid %.1f mm)"
          % (face, dbx, dbz, dpi, np.max(np.abs(dj)), float(np.sqrt(np.mean(dj**2))),
             1000*pe[R.PAD], 1000*pe[R.PAD2], 1000*eb))
