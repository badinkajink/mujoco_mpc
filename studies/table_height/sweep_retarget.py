#!/usr/bin/env python3
"""The same table-height sweep, with `brace_pose_track` on.

The only difference from `sweep.py` is `--pose_track 1`, which sets the model
numeric `brace_pose_track` before the first Transition; `lean::TransitionLocked`
then re-solves the brace posture keyframes for whatever slab `Table H` asks for.
Everything else is held: same binary, same strategy JSON, same weights, same
seeds, same thread count, same wall cap, so it compares to `runs/seeded` run for
run.

At the compiled height the retarget never runs at all -- the height block is
gated on the slab actually moving -- so "costs nothing at 0.985 m" is structural
rather than measured. It is measured anyway, because a harness can be wrong.

Serial, --threads 6, under a CPUQuota scope: see the run policy in CLAUDE_wxie.md.
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # run_one, tag_for



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="0.885,1.085,0.785,1.035,0.985,1.135,0.735")
    ap.add_argument("--seeds", type=int, default=3)
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


    os.makedirs(a.out, exist_ok=True)
    keys = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
            "t_end", "face_z", "phases", "enter", "csv", "summary"]
    sf = open(os.path.join(a.out, "summary.csv"), "w", newline="")
    w = csv.DictWriter(sf, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); sf.flush(); os.fsync(sf.fileno())

    t0 = time.time()
    extra = ["--pose_track", "1"]
    for h in [float(x) for x in a.heights.split(",")]:
        for seed in range(a.seeds):
            rec = sweep.run_one((h, seed, a.out, a.task, a.slot, a.total_time,
                                 a.threads, seed == a.video_seed, a.spp,
                                 a.cpu_quota, extra))
            w.writerow(rec); sf.flush(); os.fsync(sf.fileno())
    sf.close()
    print("retarget sweep done in %.1f min -> %s/summary.csv"
          % ((time.time() - t0) / 60.0, a.out), flush=True)


if __name__ == "__main__":
    main()
