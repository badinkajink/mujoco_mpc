#!/usr/bin/env python3
"""Can the legs HOLD the brace posture a slab requires, before the pad lands?

The brace is reached by pitching the trunk forward, and the pad only starts
carrying load once it touches. Everything before that is the legs' problem. This
asks, for each slab height: with the retargeted pose held still and the table
removed, what joint torques does gravity demand, and does any of them exceed what
the actuator can deliver?

Inverse dynamics at zero velocity and zero acceleration, both feet welded to the
floor, table absent. mj_inverse gives qfrc_inverse; the free-joint rows are the
residual the ground contact must supply, so the ankle rows are the ones that say
whether the pose is holdable.
"""
import argparse, json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model, geometry

ANKLES = ("left_ankle_pitch_joint", "right_ankle_pitch_joint")


def hold_torques(m, q):
    """Static inverse dynamics with both feet on the floor, table out of the way."""
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_forward(m, d)          # fills contacts (feet on floor)
    mujoco.mj_inverse(m, d)
    return d.qfrc_inverse.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "static.json"))
    a = ap.parse_args()
    m = pristine_model()
    # take the table far away so it cannot carry any of the pose
    tb = R.nid(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    m.body_pos[tb][0] += 10.0
    g = geometry(m)
    lim = {}
    for i in range(m.nu):
        j = int(m.actuator_trnid[i][0])
        lim[int(m.jnt_dofadr[j])] = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
            float(m.actuator_forcerange[i][1]) if m.actuator_forcelimited[i] else np.nan,
            float(m.actuator_gear[i][0]))
    rows = []
    hs = np.round(np.arange(0.685, 1.1851, 0.025), 3)
    for h in hs:
        t = g["pad0"].copy();  t[2] += h - g["nom"]
        t2 = g["pad20"].copy(); t2[2] += h - g["nom"]
        q, err = R.solve(m, g["q0"], t, t2)
        tau = hold_torques(m, q)
        rec = dict(h=float(h))
        for dof, (nm, fmax, gear) in lim.items():
            rec[nm] = float(tau[dof])
        rows.append(rec)
    hdr = ["h"] + [lim[d][0] for d in sorted(lim)]
    keys = ["h", "left_ankle_pitch_joint", "left_knee_joint", "left_hip_pitch_joint",
            "torso_joint", "left_shoulder_pitch_joint", "left_elbow_joint"]
    print("static hold torque (N m), table removed, both feet on the floor")
    print("  " + " ".join("%-13s" % k.replace("_joint", "").replace("left_", "L ")
                          for k in keys))
    for r in rows:
        print("  " + " ".join("%-13.1f" % r[k] for k in keys))
    print("\nactuator ceilings (N m):")
    for k in keys[1:]:
        for dof, (nm, fmax, gear) in lim.items():
            if nm == k:
                print("  %-28s %.0f" % (nm, fmax * gear if np.isfinite(fmax) else -1))
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
