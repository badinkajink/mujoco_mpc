#!/usr/bin/env python3
"""`brace_pitch_track`: make the trunk's pitch commandable on the lean rungs.

Written before the runs.

WHY.  The per-slab retarget of the brace keyframe (`brace_pose_track`) seats both
pads to 0.1 mm offline at every height and moved nothing in the runs.  The split
measured on 2026-09-06 (probe_base_split.py) says why: at a 1.085 m face the
retarget is -8.7 deg of BASE PITCH and +18 mm of base z; at 0.785 m it is
+21.3 deg and -47 mm.  Re-apply only the joint half -- all a posture cost can
command -- and the forearm pad lands 75 mm below the face at 1.085 m and 205 mm
above it at 0.785 m.  Meanwhile the pelvis-tilt residual is a one-sided dead band
(free from 0 to 60 deg) whenever an arm is on the table, Torso Forward Tilt is
yaw-only, and Brace Erect is gated to forearm_brace_release.  Base pitch is
uncommanded exactly where the brace has to seat.

WHAT.  `brace_pitch_track` 1 replaces that dead band, on the forearm_brace_lean
rungs only, with two-sided tracking of the base pitch held by the current
keyframe.  It is meaningful only with `brace_pose_track` on, which is what writes
the per-slab pitch into that keyframe, so the arm carries both and the control
arm is `brace_pose_track` alone -- already run at these exact settings
(total_time 75, threads 6, spp 3, quota 700) in runs/ab/on: 0/3, 0/3, 3/3, 1/3,
0/3 over 0.785 .. 1.085 m.

H1.  Rung-2 forearm pad load at 1.085 m rises from 0.0 N median to the 39-136 N
the working heights carry, and the pad clears the face instead of hanging 16 mm
under it.

H2.  At least one failing height completes on at least 2 of 3 seeds, and 0.985 m
stays 3/3.

KILL.  If rung-2 pad load at 1.085 m stays under 30 N with pitch tracking the
commanded target to within 3 deg, the trunk is stopped by contact geometry rather
than by the cost, and the next measurement is where the forearm first touches the
slab edge -- not another weight.

usage: sweep_pitch.py --out studies/table_height/runs/pitch
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep

KEYS = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
        "t_end", "face_z", "phases", "enter", "csv", "summary"]

# gain variants: at gain 1 the term is 0.17% of the rung-2 cost at 1.085 m
# (2.9 against a total of 1740), i.e. below the planner's own run-to-run spread.
# `brace_pitch_gain` is a numeric, so the ladder costs no rebuild.
_BASE = ["--pose_track", "1", "--numeric", "brace_pitch_track=1"]
ARMS = {
    "pitch": _BASE,
    "g20":   _BASE + ["--numeric", "brace_pitch_gain=20"],
    "g100":  _BASE + ["--numeric", "brace_pitch_gain=100"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="1.085,0.985,0.885,1.035,0.785")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="pitch")
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
    print("pitch sweep done in %.1f min -> %s" % ((time.time() - t0) / 60.0, a.out),
          flush=True)


if __name__ == "__main__":
    main()
