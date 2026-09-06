#!/usr/bin/env python3
"""`reach_arm_posture`: Allen's basin lock, switched on for the targeting rung.

Rung 2's posture cost tracks the `forearm_brace_lean` keyframe, whose right arm
is the brace-UP pose. The Cartesian reach pulls the same arm DOWN to
`reach_target_table`. `lean.cc` names the conflict outright -- "Posture (idx
27..33 = right arm) is tracking the brace-UP keyframe here, so it actively fights
Reach DOWN" -- and ships a fix: `reach_arm_posture` overwrites the seven
right-arm entries of the posture target with `reach_arm_q` on rungs that carry a
reach target below `reach_arm_hgate`.

The gate is 0.16 and rung 2's hover is 0.15, so the lock is already aimed at this
rung. `reach_arm_posture` ships at 0.0.

Offline, from the brace pose re-seated on each slab, `reach_arm_q` puts the jaw
tip 98 mm from the gate target at the compiled height and 82-105 mm at 0.885 and
1.085, against 176-216 mm for the arm the posture currently pulls toward.

Two arms, both against the shipped baseline in runs/ab/off, which came off this
same binary:
  basin  -- reach_arm_posture 1 alone, one existing model numeric
  both   -- plus brace_pose_track 1, without which the pads miss a tall slab

usage: sweep_basin.py --out runs/basin
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep

KEYS = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
        "t_end", "face_z", "phases", "enter", "csv", "summary"]

ARMS = {
    "basin": ["--numeric", "reach_arm_posture=1"],
    "both":  ["--numeric", "reach_arm_posture=1", "--pose_track", "1"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="1.085,0.885,0.985,1.035,0.785")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="basin,both")
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
    print("basin sweep done in %.1f min -> %s" % ((time.time() - t0) / 60.0, a.out),
          flush=True)


if __name__ == "__main__":
    main()
