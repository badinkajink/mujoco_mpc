#!/usr/bin/env python3
"""Paired A/B over table height: `brace_pose_track` off vs on, same binary.

Why paired rather than "compare against runs/seeded": the baseline sweep was run
with an earlier binary, and MJPC's sampling planner is not run-to-run
deterministic, so a difference between two binaries cannot be attributed. Here
both arms come from ONE build, and the two arms of a (height, seed) cell run back
to back, so machine state drifts through both equally.

`--pose_track 0` is the shipped controller: the numeric ships 0 and the retarget
block is gated on it, so arm A must reproduce `runs/seeded` up to planner noise.
That reproduction is itself a check on the harness.

Writes <out>/off/ and <out>/on/, each with the summary.csv layout analyze.py reads.
Serial, --threads 6, under a CPUQuota scope -- see the run policy in CLAUDE_wxie.md.
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep

KEYS = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
        "t_end", "face_z", "phases", "enter", "csv", "summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="0.885,1.085,0.785,1.035,0.985")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="off,on")
    ap.add_argument("--task", default="Lean H12 Magpie")
    ap.add_argument("--slot", type=int, default=25)
    ap.add_argument("--total_time", type=float, default=75)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--spp", type=int, default=3)
    ap.add_argument("--cpu_quota", type=int, default=700)
    ap.add_argument("--video_seed", type=int, default=0)
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
                extra = ["--pose_track", "1" if arm == "on" else "0"]
                print("=== face %.3f seed %d  pose_track=%s ==="
                      % (h, seed, arm), flush=True)
                rec = sweep.run_one((h, seed, os.path.join(a.out, arm), a.task,
                                     a.slot, a.total_time, a.threads,
                                     seed == a.video_seed, a.spp, a.cpu_quota,
                                     extra))
                writers[arm].writerow(rec)
                files[arm].flush(); os.fsync(files[arm].fileno())
    for f in files.values():
        f.close()
    print("A/B done in %.1f min -> %s/{%s}/summary.csv"
          % ((time.time() - t0) / 60.0, a.out, ",".join(arms)), flush=True)


if __name__ == "__main__":
    main()
