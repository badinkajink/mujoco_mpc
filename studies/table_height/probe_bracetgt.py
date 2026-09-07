#!/usr/bin/env python3
"""Where `Brace Pos` aims the forearm pad, against where the pad actually gets.

`brace_target_slab` ships 0 in `Lean_H12_Magpie.xml`, so the brace x target is the
legacy expression at `lean.cc:952`:

    brace_x_target = torso_x + 0.4 * (table_centre_x - torso_x)

-- a convex combination of the TORSO and the table_top geom CENTRE, deliberately
short of the near edge and tied to the body rather than the slab.  The z target is
`table_face_z - brace_press_depth` with `brace_target_face` = 1 and
`brace_press_depth` = -0.044, i.e. 44 mm ABOVE the face.

This replays the logged qpos and reports, per run over the brace rungs, the target
that expression produces and the pad's own best position, both measured from the
slab's near edge.  If the pad sits AT the target, the target is the limit; if it
stops short of it, something else is.  No sim time.

usage: probe_bracetgt.py --runs LABEL=dir [...] [--json out.json]
"""
import argparse, csv, json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height

BRACE_PHASES = (1, 2)


def gid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)


def scan(m, d, qcsv, scsv):
    ph = {round(float(r["t"]), 3): int(float(r["phase"]))
          for r in csv.DictReader(open(scsv))}
    pad = gid(m, R.PAD)
    tgc = gid(m, "table_top_collision")
    tgv = gid(m, "table_top")           # the framepos geom lean.cc reads
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position")
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    best = None
    for r in csv.DictReader(open(qcsv)):
        try:
            t = round(float(r["t"]), 3)
            q = [float(r["q%d" % i]) for i in range(m.nq)]
        except ValueError:
            continue
        if ph.get(t) not in BRACE_PHASES:
            continue
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        near = float(d.geom_xpos[tgc][0] - m.geom_size[tgc][0])
        face = float(d.geom_xpos[tgc][2] + m.geom_size[tgc][2])
        ctr = float(d.geom_xpos[tgv][0])
        torso_x = float(d.xpos[tb][0])
        tgt_x = torso_x + 0.4 * (ctr - torso_x)
        pad_x = float(d.geom_xpos[pad][0])
        if best is None or pad_x > best["pad_x_abs"]:
            best = {"t": t, "pad_x_abs": pad_x,
                    "pad_edge": pad_x - near,
                    "tgt_edge": tgt_x - near,
                    "pad_minus_tgt": pad_x - tgt_x,
                    "torso_edge": torso_x - near,
                    "ctr_edge": ctr - near,
                    "pad_clear": float(d.geom_xpos[pad][2] - m.geom_size[pad][0] - face)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = []
    print("all x measured from the slab's NEAR EDGE (+ = past it, over the wood)")
    print("  %-8s %6s %4s %9s %9s %10s %9s %9s"
          % ("arm", "face", "seed", "torso", "target", "pad", "pad-tgt", "pad clear"))
    for spec in a.runs:
        label, d_ = spec.split("=", 1)
        for f in sorted(os.listdir(d_)):
            if not f.endswith(".qpos.csv"):
                continue
            tag = f[:-len(".qpos.csv")]
            h = int(tag[1:5]) / 1000.0
            seed = int(tag.split("_s")[1])
            m = pristine_model(); set_table_height(m, h)
            d = mujoco.MjData(m)
            s = scan(m, d, os.path.join(d_, f), os.path.join(d_, tag + ".csv"))
            if s is None:
                continue
            s.update(arm=label, h=h, seed=seed)
            rows.append(s)
            print("  %-8s %6.3f %4d %+9.3f %+9.3f %+10.3f %+9.3f %+9.3f"
                  % (label, h, seed, s["torso_edge"], s["tgt_edge"],
                     s["pad_edge"], s["pad_minus_tgt"], s["pad_clear"]))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print("-> %s" % a.json)


if __name__ == "__main__":
    main()
