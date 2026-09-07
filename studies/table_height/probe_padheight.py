#!/usr/bin/env python3
"""Can the bracing arm get the pad above the slab, and does it hold it there?

Open item raised 2026-09-06: the slab rises 100 mm between 0.985 m and 1.085 m
while the per-slab retarget asks the base for only +18 mm of z and -8.7 deg of
pitch, so most of the extra height has to come from the arm.  This asks whether
the arm has it.

Two statistics over the brace rungs, and the difference between them is the
finding: the MAXIMUM pad clearance says whether the arm can ever get the pad over
the face, and the MEDIAN says whether it holds it there.  Frames are restricted to
an upright pelvis (z > 0.55 m) because a toppling robot swings the pad through
arbitrary heights and a max over all frames reads a mid-fall pose.

usage: probe_padheight.py --runs LABEL=dir [...] [--heights 0.985,1.085]
"""
import argparse, csv, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height

BRACE_PHASES = (1, 2)
UPRIGHT_Z = 0.55


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL=dir")
    ap.add_argument("--heights", default="0.985,1.085")
    a = ap.parse_args()
    want = [float(x) for x in a.heights.split(",")]

    print("Over the brace rungs, upright frames only (pelvis z > %.2f m)." % UPRIGHT_Z)
    print("pad_clear = pad bottom minus face; sh-face = left shoulder above the face.")
    print("  %-8s %6s %4s %6s %13s %13s %11s"
          % ("arm", "face", "seed", "n", "max pad_clear", "med pad_clear", "med sh-face"))
    for spec in a.runs:
        label, d_ = spec.split("=", 1)
        if not os.path.isdir(d_):
            continue
        for f in sorted(os.listdir(d_)):
            if not f.endswith(".qpos.csv"):
                continue
            tag = f[:-len(".qpos.csv")]
            h = int(tag[1:5]) / 1000.0
            if h not in want:
                continue
            seed = int(tag.split("_s")[1])
            m = pristine_model(); set_table_height(m, h)
            d = mujoco.MjData(m)
            ph = {round(float(r["t"]), 3): int(float(r["phase"]))
                  for r in csv.DictReader(open(os.path.join(d_, tag + ".csv")))}
            sh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
            pad = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, R.PAD)
            tgc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
            clear, shf = [], []
            for r in csv.DictReader(open(os.path.join(d_, f))):
                # a killed run's last row is truncated: missing keys read None
                try:
                    t = round(float(r["t"]), 3)
                    q = [float(r["q%d" % i]) for i in range(m.nq)]
                except (ValueError, TypeError):
                    continue
                if ph.get(t) not in BRACE_PHASES or q[2] < UPRIGHT_Z:
                    continue
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
                face = float(d.geom_xpos[tgc][2] + m.geom_size[tgc][2])
                clear.append(float(d.geom_xpos[pad][2] - m.geom_size[pad][0] - face))
                shf.append(float(d.xpos[sh][2] - face))
            if clear:
                print("  %-8s %6.3f %4d %6d %+13.3f %+13.3f %+11.3f"
                      % (label, h, seed, len(clear), max(clear),
                         float(np.median(clear)), float(np.median(shf))))


if __name__ == "__main__":
    main()
