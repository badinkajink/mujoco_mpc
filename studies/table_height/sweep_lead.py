#!/usr/bin/env python3
"""Second lever: turn on `Brace Reach Lead`, with and without the pose retarget.

WHY THIS TERM. lean.cc carries a one-sided penalty on the base travelling forward
past `brace_lead_x0` (0.24 m) plus `brace_lead_gain` x the measured brace load --
built, its comment says, for runs where "that forward excursion RUN[s] AWAY to
+0.30..+0.41 and never settle[s]". It is inert below the line, so it cannot
starve the press the way raising Balance did. Its weight is **0.0 in every
strategy JSON in the repo**, so the term has never been on.

MEASURED, on the runs already in hand:

  baseline 1.035 m  complete   peak base_x 0.218   0.0 s past the line
  baseline 0.985 m  complete   peak base_x 0.304   2.0 s
  baseline 0.885 m  fell       peak base_x 0.440  11.2 s
  retarget 0.885 m  fell       peak base_x 0.807  12.3 s
  baseline 0.785 m  stalled    peak base_x 0.584  52.1 s

The completing runs barely cross it and the low-slab failures live past it, so
the term is close to free where the controller already works.

Two arms, interleaved per (height, seed) as in sweep_ab.py: `--pose_track 1` and
`--pose_track 0`, both with the weight on, which says whether either lever is
sufficient alone. The strategy JSON is loaded from SOURCE_DIR at runtime, so the
weight is written into the source file and restored on every exit path.
"""
import argparse, atexit, csv, json, os, shutil, signal, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep

ROOT = os.path.normpath(os.path.join(HERE, "../.."))
JSON = os.path.join(ROOT, "mjpc/tasks/humanoid_bench/lean/strategies/"
                          "h12_brace_targeting.json")
BK = JSON + ".leadbackup"
KEYS = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
        "t_end", "face_z", "phases", "enter", "csv", "summary"]


def restore():
    if os.path.exists(BK):
        shutil.move(BK, JSON)
        print("restored", JSON, flush=True)


def patch(weight):
    shutil.copy2(JSON, BK)
    d = json.load(open(JSON))
    n = 0
    for k in d:
        # the term is gated on `is_forearm_brace`, which is exactly the rungs
        # named forearm_brace_lean; a weight anywhere else would be inert.
        if k.get("name") == "forearm_brace_lean":
            k["weight"]["Brace Reach Lead"] = weight
            n += 1
    json.dump(d, open(JSON, "w"), indent=2)
    chk = json.load(open(JSON))
    got = [k["weight"].get("Brace Reach Lead") for k in chk
           if k.get("name") == "forearm_brace_lean"]
    if not n or any(g != weight for g in got):
        restore()
        sys.exit("JSON patch did not take: %r" % got)
    print("Brace Reach Lead = %g on %d forearm_brace_lean rungs" % (weight, n),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", type=float, default=400.0,
                    help="XML declares max 400 for this cost; MJPC clamps above it")
    ap.add_argument("--heights", default="0.885,0.785,0.985")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="on,off",
                    help="pose_track value paired with the weight")
    ap.add_argument("--wait_for", default="",
                    help="file that must contain --wait_token before starting")
    ap.add_argument("--wait_token", default="POSEDONE")
    ap.add_argument("--task", default="Lean H12 Magpie")
    ap.add_argument("--slot", type=int, default=25)
    ap.add_argument("--total_time", type=float, default=75)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--spp", type=int, default=3)
    ap.add_argument("--cpu_quota", type=int, default=700)
    a = ap.parse_args()

    if a.wait_for:
        for _ in range(1440):
            if os.path.exists(a.wait_for) and a.wait_token in open(a.wait_for).read():
                break
            time.sleep(30)
        print("gate cleared at", time.strftime("%H:%M"), flush=True)

    atexit.register(restore)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: sys.exit(1))
    patch(a.weight)

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
                print("=== face %.3f seed %d  lead=%g pose_track=%s ==="
                      % (h, seed, a.weight, arm), flush=True)
                # qpos for EVERY seed here, not just the video seed: base_x is
                # the quantity this cost acts on and it is only in the qpos dump.
                rec = sweep.run_one((h, seed, os.path.join(a.out, arm), a.task,
                                     a.slot, a.total_time, a.threads, True,
                                     a.spp, a.cpu_quota, extra))
                writers[arm].writerow(rec)
                files[arm].flush(); os.fsync(files[arm].fileno())
    for f in files.values():
        f.close()
    restore()
    print("lead sweep done in %.1f min -> %s" % ((time.time() - t0) / 60.0, a.out),
          flush=True)


if __name__ == "__main__":
    main()
