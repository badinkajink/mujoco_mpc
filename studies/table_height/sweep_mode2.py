#!/usr/bin/env python3
"""`brace_pose_track` 2: the reaching arm's jaw tip tracks the slab as well.

Mode 1 moves the trunk and the bracing arm and freezes the reaching arm at the
authored pose. Every failing height in the mode-1 A/B enters rung 2 -- the
targeting rung -- and never leaves it, and the gate it dies at grades the right
gripper jaw tip against a point at face + 0.15 m. Replaying the logged qpos, the
tip's closest approach in that rung is 8-32 mm at the two working heights and
154-481 mm at the three failing ones, so that gate is the proximate blocker.

Offline, the tip's error to that target at the re-solved brace pose is 142 mm at
the compiled height, 231 mm at 0.885 and 330 mm at 0.785 with the reaching arm
frozen, and 142-146 mm at EVERY height once it is constrained. Mode 2 makes the
reach cost face the same closure job at every slab that it already solves at the
compiled one.

Same paired shape as sweep_ab.py: one binary, arms back to back per cell.
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
    ap.add_argument("--heights", default="0.885,1.085,0.785,0.985")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", default="2")
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

    modes = a.modes.split(",")
    writers, files = {}, {}
    for md in modes:
        d = os.path.join(a.out, "mode" + md)
        os.makedirs(d, exist_ok=True)
        f = open(os.path.join(d, "summary.csv"), "w", newline="")
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader(); f.flush(); os.fsync(f.fileno())
        writers[md], files[md] = w, f

    t0 = time.time()
    for h in [float(x) for x in a.heights.split(",")]:
        for seed in range(a.seeds):
            for md in modes:
                print("=== face %.3f seed %d pose_track=%s ===" % (h, seed, md),
                      flush=True)
                rec = sweep.run_one((h, seed, os.path.join(a.out, "mode" + md),
                                     a.task, a.slot, a.total_time, a.threads,
                                     True, a.spp, a.cpu_quota,
                                     ["--pose_track", md]))
                writers[md].writerow(rec)
                files[md].flush(); os.fsync(files[md].fileno())
    for f in files.values():
        f.close()
    print("mode sweep done in %.1f min -> %s" % ((time.time() - t0) / 60.0, a.out),
          flush=True)


if __name__ == "__main__":
    main()
