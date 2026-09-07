#!/usr/bin/env python3
"""`brace_target_slab`: aim the brace at a point that is actually on the slab.

Written before the runs.

WHY.  `probe_bracetgt.py`, 2026-09-06.  `brace_target_slab` ships 0 in
`Lean_H12_Magpie.xml`, so `Brace Pos` aims the forearm pad with the legacy
expression at lean.cc:952:

    brace_x_target = torso_x + 0.4 * (table_centre_x - torso_x)

a convex combination of the TORSO and the table_top geom CENTRE.  It tracks the
body, not the slab, and it is deliberately short of the near edge.  Measured over
the brace rungs, as distance past the near edge (+ = over the wood):

    face    torso    target      pad
    0.985  -0.235    +0.095    +0.140
    1.035  -0.339    +0.033    +0.054
    1.085  -0.444    -0.030    -0.066

At 1.085 m the aim point is 30 mm IN FRONT of the near edge and, with
`brace_press_depth` -0.044 under `brace_target_face` 1, 44 mm above the face --
so the gradient pulls the pad into the slab's vertical front face rather than up
and over the edge.  The pad ends up jammed there, 13 mm below the top.  The
numeric written for this (aim at `near_edge + brace_target_inset`, shipped 0.05)
is off, and its own comment names the 0.785 m drape as the other symptom: "the
body keeps bowing to chase the buried point until the PELVIS lands on the slab",
measured there as 398 N through the torso against 37 N on the forearm.

WHAT.  `--numeric brace_target_slab=1`.  One numeric, already implemented,
0 = OFF = byte-identical.  Top of the minimality ladder: it is an existing model
numeric and it makes the target track the slab, which is the property the whole
study is about.

CONTROL.  `runs/ab/off` (shipped) at these settings -- total_time 75, threads 6,
spp 3 -- 0/3 at 0.785, 3/3 at 0.985, 0/3 at 1.085.

H1.  At 1.085 m the target moves from -0.030 to +0.050 past the near edge and the
pad follows it over the edge: pad_clear goes non-negative and rung-2 forearm load
rises off 0.00 N.

H2.  At 0.785 m the drape weakens -- peak torso force falls well below the 398 N
the shipped controller puts through it -- because the pad is no longer chasing a
point in front of the slab's face.

H3.  At least one failing height completes >= 2 of 3 seeds, with 0.985 m still 3/3.

KILL.  0.985 m losing a completion disqualifies it outright.  Failing that, if the
target lands on the slab at 1.085 m and the pad still stops short of the edge with
pad_clear negative, then nothing about the AIM is what holds the pad, and the next
measurement is the pad's height relative to the shoulder through the brace rungs at
0.985 vs 1.085 -- how much of the slab's extra 100 mm the arm can find and where
the rest has to come from.

NOTE.  The stance arm (`sweep_stance.py`) already moved this target onto the slab
indirectly, by moving the torso, and the pad did not follow -- it opened a 98-185
mm gap to its own target instead.  That is evidence against H1.  It is not
conclusive, because moving the robot changed every other body-referenced term at
the same time; this arm changes the aim alone.

usage: sweep_bracetgt.py --out studies/table_height/runs/bracetgt
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep

KEYS = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
        "t_end", "face_z", "phases", "enter", "csv", "summary"]

ARMS = {
    "slab":   ["--numeric", "brace_target_slab=1"],
    # inset 0.05 is the shipped value; 0.12 pushes the aim further onto the wood
    # if 0.05 leaves the pad on the edge. Only run if `slab` moves the pad.
    "slab12": ["--numeric", "brace_target_slab=1",
               "--numeric", "brace_target_inset=0.12"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="1.085,0.985,0.785")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="slab")
    ap.add_argument("--task", default="Lean H12 Magpie")
    ap.add_argument("--slot", type=int, default=25)
    ap.add_argument("--total_time", type=float, default=75)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--spp", type=int, default=3)
    ap.add_argument("--cpu_quota", type=int, default=700)
    a = ap.parse_args()

    ncpu = os.cpu_count() or 4
    if os.getloadavg()[0] > ncpu / 2:
        sys.exit("refusing: 1-min load %.1f on %d cores" % (os.getloadavg()[0], ncpu))

    arms = a.arms.split(",")
    writers, files = {}, {}
    for arm in arms:
        d = os.path.join(a.out, arm)
        os.makedirs(d, exist_ok=True)
        f = open(os.path.join(d, "summary.csv"), "w", newline="")
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader(); f.flush(); os.fsync(f.fileno())
        writers[arm], files[arm] = w, f

    t0 = time.time()
    for h in [float(x) for x in a.heights.split(",")]:
        for seed in range(a.seeds):
            for arm in arms:
                print("=== face %.3f seed %d arm %s ===" % (h, seed, arm),
                      flush=True)
                rec = sweep.run_one((h, seed, os.path.join(a.out, arm),
                                     a.task, a.slot, a.total_time, a.threads,
                                     True, a.spp, a.cpu_quota, ARMS[arm]))
                writers[arm].writerow(rec)
                files[arm].flush(); os.fsync(files[arm].fileno())
    for f in files.values():
        f.close()
    print("bracetgt sweep done in %.1f min -> %s" % ((time.time() - t0) / 60.0, a.out),
          flush=True)


if __name__ == "__main__":
    main()
