#!/usr/bin/env python3
"""How bowed the robot has to be to brace on each slab, against the gates.

To put its forearm on a LOW slab the robot has to pitch further forward. Two
constants in `lean.cc` bound that pitch and neither moves with the table:

  standback_pitch_release   0.50 rad (28.6 deg)  the rung-3 release gate --
                            believed base pitch must be under it to let go
  brace_erect_target        0.38 rad (21.8 deg)  the cost that presses pitch
                            down during forearm_brace_release

Both read the same quantity `lean.cc` reads: asin(2 (qw qy - qz qx)) off the
free-joint quaternion. This solves the pose each slab actually requires -- both
brace pads seated on the face, feet planted -- and reports its pitch, so the
requirement and the gate can be compared in one frame.

usage: probe_pitch.py [--json out.json]
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


def pitch_of(q):
    """asin(2 (qw qy - qz qx)) -- the same read as the release gate."""
    return float(np.arcsin(np.clip(2.0 * (q[3] * q[5] - q[6] * q[4]), -1, 1)))


def caps(m, q):
    d = mujoco.MjData(m); d.qpos[:] = q
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    fb = [R.nid(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in R.FEET]
    mid = 0.5 * (d.xpos[fb[0]][0] + d.xpos[fb[1]][0])
    pel = R.nid(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return dict(pitch=pitch_of(q), base_x=float(q[0]),
                pelvis_fwd=float(d.xpos[pel][0] - mid),
                com_fwd=float(d.subtree_com[pel][0] - mid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    m0 = pristine_model()
    nominal = R.face_of(m0)
    gate = R.num(m0, "standback_pitch_release", 0.50)
    erect = R.num(m0, "brace_erect_target", 0.38)
    pcap = R.num(m0, "pelvis_cap_fwd", 0.13)
    ccap = R.num(m0, "com_cap_fwd", 0.145)
    print("compiled face %.3f m" % nominal)
    print("gates: standback_pitch_release %.2f rad (%.1f deg), "
          "brace_erect_target %.2f rad (%.1f deg)"
          % (gate, np.degrees(gate), erect, np.degrees(erect)))
    print("       pelvis_cap_fwd %.3f m, com_cap_fwd %.3f m\n" % (pcap, ccap))
    print(" face   pose   pitch(deg)  vs release  vs erect | pelvis_fwd  com_fwd  base_x")
    out = {}
    for face in HEIGHTS:
        m = pristine_model(); set_table_height(m, face)
        tgc = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
        d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
        ctr, half = d0.geom_xpos[tgc].copy(), m.geom_size[tgc].copy()
        tgt = np.array([ctr[0] - half[0] + RTT[0], ctr[1] - RTT[1],
                        ctr[2] + half[2] + RTT[2]])
        qb, eb, p1, p2 = brace_pose(m, face, nominal)
        qs, _ = R.solve(m, qb, p1, p2, target_tip=tgt, iters=1200)
        rec = {}
        for tag, q in (("brace", qb), ("reach", qs)):
            c = caps(m, q); rec[tag] = c
            print(" %.3f  %-6s %8.1f  %+9.1f %+8.1f | %+9.3f %+8.3f %+7.3f"
                  % (face, tag, np.degrees(c["pitch"]),
                     np.degrees(c["pitch"] - gate), np.degrees(c["pitch"] - erect),
                     c["pelvis_fwd"], c["com_fwd"], c["base_x"]))
        rec["seat_resid_mm"] = 1000 * eb
        out["%.3f" % face] = rec
    print("\nA height whose brace pitch is above the release gate cannot leave the"
          "\nbrace rung without first un-bowing, which is leaving the brace.")
    if a.json:
        json.dump(dict(gate=gate, erect=erect, pelvis_cap=pcap, com_cap=ccap,
                       heights=out), open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
