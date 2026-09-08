#!/usr/bin/env python3
"""Is there a (table height, standoff) pair that keeps the brace posture valid?

The coauthor's hypothesis, 2026-09-08: "changing the distance between table and
robot should also help. Maybe there is some ratio between them that'd work in a
window." This answers it in kinematics, with no sim time.

For a grid of slab heights and base-x offsets it re-solves the
`forearm_brace_lean` keyframe with BOTH pads seated on the slab at the offsets
past the near edge that the shipped keyframe uses, and the feet pinned where the
shifted base puts them. What comes back per cell:

  resid    IK residual; > 1 mm means the pose cannot be reached at all
  dq       joint-space distance from the shipped keyframe, rad rms over the
           selected dofs, base translation excluded -- how far the posture has
           to move from the one Allen authored
  pitch    base pitch the solution needs (deg)
  sh       left shoulder above the face (mm)
  com      CoM ahead of the foot midpoint (mm); positive is toward the table
  marg     worst joint-limit margin in the selection (rad)

The standoff that minimises `dq` at each height is the ratio, if there is one.

usage: probe_standoff.py [--faces ...] [--dx ...] [--json out.json]
"""
import argparse, json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height

KEY = "forearm_brace_lean"


def near_edge(m):
    g = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return float(d.geom_xpos[g][0] - m.geom_size[g][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default="0.785,0.885,0.985,1.035,1.085")
    ap.add_argument("--dx", default="-0.12,-0.08,-0.04,0.0,0.04,0.063,0.08,0.12")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    faces = [float(x) for x in a.faces.split(",")]
    dxs = [float(x) for x in a.dx.split(",")]

    # Reference geometry, taken from the shipped keyframe at the compiled slab.
    m0 = pristine_model()
    k0 = R.nid(m0, mujoco.mjtObj.mjOBJ_KEY, KEY)
    q_key = m0.key_qpos[k0].copy()
    d0 = mujoco.MjData(m0); d0.qpos[:] = q_key; mujoco.mj_kinematics(m0, d0)
    pid = R.nid(m0, mujoco.mjtObj.mjOBJ_GEOM, R.PAD)
    p2id = R.nid(m0, mujoco.mjtObj.mjOBJ_GEOM, R.PAD2)
    e0 = near_edge(m0)
    pad_off = d0.geom_xpos[pid].copy(); pad_off[0] -= e0
    pad2_off = d0.geom_xpos[p2id].copy(); pad2_off[0] -= e0
    face0 = R.face_of(m0)
    print("shipped keyframe at %.3f m: pad %+.3f m past the near edge, "
          "%+.3f m in z" % (face0, pad_off[0], pad_off[2] - face0))

    rows = []
    print("\n  %6s %7s %8s %7s %7s %7s %7s %6s"
          % ("face", "dx", "resid", "dq_rms", "pitch", "sh", "com", "marg"))
    for face in faces:
        m = pristine_model(); set_table_height(m, face)
        e = near_edge(m)
        tp = np.array([e + pad_off[0], pad_off[1], face + (pad_off[2] - face0)])
        tp2 = np.array([e + pad2_off[0], pad2_off[1], face + (pad2_off[2] - face0)])
        best = None
        for dx in dxs:
            q0 = q_key.copy(); q0[0] += dx
            q, res = R.solve(m, q0, tp, tp2)
            e_ = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, e_, 1.0, q0, q)
            joint = e_[6:33]                     # legs + torso + both arms
            dq = float(np.sqrt(np.mean(joint ** 2)))
            R._MPATH[id(m)] = R._MPATH[id(m)] if id(m) in R._MPATH else None
            dg = R.diagnose(m, q, face)
            w = np.array(q[3:7])
            pitch = np.degrees(np.arcsin(np.clip(
                2 * (w[0] * w[2] - w[3] * w[1]), -1, 1)))
            row = dict(face=face, dx=dx, resid=res, dq=dq, pitch=float(pitch),
                       sh=dg["sh_above"] * 1000, com=dg["com_mid"] * 1000,
                       marg=dg["lim_margin"])
            rows.append(row)
            mark = ""
            if res < 1e-3 and (best is None or dq < best):
                best = dq; mark = "  <- least posture change"
            print("  %6.3f %+7.3f %8.5f %7.3f %+7.1f %+7.0f %+7.0f %6.2f%s"
                  % (face, dx, res, dq, pitch, row["sh"], row["com"],
                     row["marg"], mark))
        print()
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
