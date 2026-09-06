#!/usr/bin/env python3
"""`pelvis_tilt_max_deg`: cap the free bow so a low slab stops taking the torso.

Written before the runs.

WHY.  At a 0.785 m face the shipped controller bows to 74.9 deg of base pitch
against the 47.1 deg the seated brace pose needs, drapes the torso on the slab --
622 N through torso contact at the end of the run, both feet off the floor -- and
never leaves the lean rung (release entered in 0 of 3 runs).  The pelvis-tilt
residual is `max(0, cos(tmax) - pelvis_up_z)` whenever an arm is on the table,
with tmax = 60 deg, so every degree of that over-bow is free.

WHAT.  `pelvis_tilt_max_deg` is an existing numeric (lean.cc, 2026-08-29) that
sets tmax; it was absent from the model file, so it is now present at 0 -- the
shipped 60 deg, bit for bit -- and reachable by `--numeric`.  50 deg leaves the
47.1 deg brace pose admissible at zero cost and puts a gradient on the 25 deg of
over-bow past it.  One number, no rebuild, low end only: at 0.985 m the pose
needs 25.9 deg, so the cap should be inert there.

H1.  Peak base pitch at 0.785 m falls from 74.9 deg to under 55 deg, and torso
contact force in the lean rungs falls below 100 N.

H2.  0.985 m stays 3/3.  A cost here disqualifies the arm outright.

KILL.  If the bow caps but the pads still do not seat -- rung-2 forearm load
under 30 N at 0.785 m -- then over-bow was a symptom of the brace never catching,
not its cause, and the low end belongs to `brace_pitch_track` alone.

usage: sweep_tilt.py --out studies/table_height/runs/tilt
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep

KEYS = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
        "t_end", "face_z", "phases", "enter", "csv", "summary"]

ARMS = {
    "cap50": ["--numeric", "pelvis_tilt_max_deg=50"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="0.785,0.985,0.885")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="cap50")
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
    print("tilt sweep done in %.1f min -> %s" % ((time.time() - t0) / 60.0, a.out),
          flush=True)


if __name__ == "__main__":
    main()
