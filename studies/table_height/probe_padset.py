#!/usr/bin/env python3
"""Which brace surface actually gets closest to the slab, at every height?

WHY.  The 2026-09-08 report bounds the high end of the height window with ONE
surface: `left_forearm_pad`, the capsule the strategy is named after.  Its
clearance above the face falls 0.862 mm per mm of slab and crosses zero at
1.0765 m, which is the whole high-end argument.

The crocoddyl side says that surface may be a bystander.  At its certified
`elbow+forearm` pose the load arrives through `left_shoulder_yaw_link` (48.3 N)
and `left_wrist_pad` (19.8 N) while the forearm pad carries **0.00 N** sitting
1.4 mm inside the wood, and over a 40 s hold the split is elbow 111 N / forearm
1.8 N / uncommanded wrist 33 N.  Contacts are prescribed kinematic constraints
there and do not allocate force, so a mode's NAME says which sites are
constrained, not which links carry weight.  If the same is true of the MJPC
rollouts, then "the forearm pad never gets above the top face" measures the
wrong link and the high-end bound has to be restated.

WHAT THIS COMPUTES.  Replay only -- `mj_kinematics` on the logged qpos, no
physics stepped, no new runs.  For every shipped run, over the brace rungs and
above the upright cutoff, the clearance of the LOWEST POINT of each candidate
surface above the top face, reported two ways:

  min   closest approach.  "Did this surface ever reach the wood."
  max   highest it ever gets.  "Could this surface clear the near edge to land
        on top of the slab at all" -- which is what the report's figure plots.

The two answer different questions and the report only ever plotted the second.

usage: probe_padset.py [--out figs/padset.json]
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from analyze_pose import pristine_model
from render_video import set_table_height

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
BRACE_PHASES = (1, 2)
UPRIGHT_Z = 0.55
SHIPPED = ("seeded", "ab/off", "edge")

# Every surface on the bracing (left) arm that could plausibly take load, plus
# the two the crocoddyl certification says actually do.  `left_shoulder_yaw_link`
# carries no named pad -- its collision geoms are the upper-arm capsules -- so it
# enters by BODY and every collision geom on it is measured.
SURFACES = ["left_forearm_pad", "left_wrist_pad", "left_gripper_collision"]
BODIES = ["left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link"]


def lowest_z(m, d, g):
    """World z of the lowest point of geom `g`, by geom type."""
    t = m.geom_type[g]
    c = float(d.geom_xpos[g][2])
    R = d.geom_xmat[g].reshape(3, 3)
    s = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return c - float(s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return c - abs(float(s[1]) * R[2, 2]) - float(s[0])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return c - abs(float(s[1]) * R[2, 2]) - abs(float(s[0]) * R[2, 0])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return c - sum(abs(float(s[i]) * R[2, i]) for i in range(3))
    return c                                   # planes/meshes: centre, flagged


def collect():
    """Per run: which arm surface actually contacted the slab, and how hard.

    `mj_forward` rather than `mj_kinematics`, because geometry alone cannot tell
    "on the slab" from "under the overhang": the pads are contype=0 and collide
    only through declared pairs, so a purely geometric test scored an arm hanging
    beside the table at -748 mm as having reached the wood, and restricting to
    the slab footprint then scored the arm swinging UNDER it as -16 mm of
    penetration.  The narrowphase is the arbiter.  No physics is stepped -- this
    is the logged qpos evaluated forward, exactly as `analyze_gate` replays it.

    Table contacts only, and NOT the table's own legs on the floor: floor geoms
    belong to body 0, so "one side is the table" books the table's ~166 N of
    weight as brace load (CLAUDE.md 1a).  The free `object` is skipped for the
    same reason.
    """
    models, out = {}, []
    for rel in SHIPPED:
        d_ = os.path.join(RUNS, rel)
        if not os.path.isdir(d_):
            continue
        for f in sorted(os.listdir(d_)):
            if not f.endswith(".qpos.csv"):
                continue
            tag = f[:-len(".qpos.csv")]
            h = int(tag[1:5]) / 1000.0
            seed = int(tag.split("_s")[1])
            if h not in models:
                m = pristine_model()
                set_table_height(m, h)
                models[h] = m
            m = models[h]
            d = mujoco.MjData(m)
            tbl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
            obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
            tgc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                    "table_top_collision")

            ph = {round(float(r["t"]), 3): int(float(r["phase"]))
                  for r in csv.DictReader(open(os.path.join(d_, tag + ".csv")))}
            peak, frames, nfr = {}, {}, 0
            fv = np.zeros(6)
            for r in csv.DictReader(open(os.path.join(d_, f))):
                try:
                    t = round(float(r["t"]), 3)
                    q = [float(r["q%d" % i]) for i in range(m.nq)]
                except (ValueError, TypeError):
                    continue
                if ph.get(t) not in BRACE_PHASES or q[2] < UPRIGHT_Z:
                    continue
                d.qpos[:] = q
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                nfr += 1
                for c in range(d.ncon):
                    con = d.contact[c]
                    b = [int(m.geom_bodyid[con.geom[0]]),
                         int(m.geom_bodyid[con.geom[1]])]
                    if tbl not in b or 0 in b or obj in b:
                        continue
                    gi = con.geom[1] if b[0] == tbl else con.geom[0]
                    nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
                          or "geom%d" % gi)
                    mujoco.mj_contactForce(m, d, c, fv)
                    N = abs(float(fv[0]))
                    peak[nm] = max(peak.get(nm, 0.0), N)
                    frames[nm] = frames.get(nm, 0) + 1
            if not nfr:
                continue
            out.append(dict(rel=rel, tag=tag, h=h, seed=seed, frames=nfr,
                            peak_N=peak, n_contact=frames))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "padset.json"))
    a = ap.parse_args()
    rows = collect()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)

    keys = sorted({k for r in rows for k in r["peak_N"]})
    hs = sorted({r["h"] for r in rows})
    print("PEAK NORMAL FORCE through each arm surface over the brace rungs, "
          "median across shipped seeds [N]\n")
    print("%-7s %4s %7s  %s"
          % ("face", "n", "frames", "  ".join("%22s" % k for k in keys)))
    for h in hs:
        sel = [r for r in rows if r["h"] == h]
        cells = []
        for k in keys:
            v = [r["peak_N"].get(k, 0.0) for r in sel]
            cells.append("%22s" % ("%.1f" % np.median(v)))
        print("%-7.3f %4d %7d  %s"
              % (h, len(sel), int(np.median([r["frames"] for r in sel])),
                 "  ".join(cells)))
    print("\nFRACTION OF BRACE-RUNG FRAMES in contact, median [%]\n")
    print("%-7s %4s  %s" % ("face", "n", "  ".join("%22s" % k for k in keys)))
    for h in hs:
        sel = [r for r in rows if r["h"] == h]
        cells = []
        for k in keys:
            v = [100.0 * r["n_contact"].get(k, 0) / max(1, r["frames"])
                 for r in sel]
            cells.append("%22s" % ("%.0f" % np.median(v)))
        print("%-7.3f %4d  %s" % (h, len(sel), "  ".join(cells)))
    print("\nwrote %s (%d runs)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
