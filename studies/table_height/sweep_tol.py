#!/usr/bin/env python3
"""Diagnostic, not a fix: open the rung-2 gate and see whether anything else fails.

Replaying the qpos of every run in the study, the right gripper jaw tip comes
within `target_distance_tolerance` (70 mm) of the rung-2 target in every run that
completes the ladder and in none that does not. That is a perfect separation, but
it does not by itself prove the gate is the ONLY thing wrong at the failing
heights -- the rest of the ladder has never been reached there, so it has never
been tested there.

Widening that tolerance tests it. If the failing heights then complete, every
rung after the targeting one works at those heights and table-height
generalisation reduces to making that reach. If they still fail, something else
is broken downstream and the gate was only the first thing to break.

This is deliberately NOT proposed as the fix: the tolerance is the task's success
criterion, and loosening it declares victory rather than earning it.

The strategy JSON is loaded from SOURCE_DIR at runtime, so the value is written
into the source file and restored on every exit path.
"""
import argparse, atexit, csv, json, os, shutil, signal, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep

ROOT = os.path.normpath(os.path.join(HERE, "../.."))
JSON = os.path.join(ROOT, "mjpc/tasks/humanoid_bench/lean/strategies/"
                          "h12_brace_targeting.json")
BK = JSON + ".tolbackup"
KEYS = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
        "t_end", "face_z", "phases", "enter", "csv", "summary"]


def restore():
    if os.path.exists(BK):
        shutil.move(BK, JSON)
        print("restored", JSON, flush=True)


def patch(tol):
    shutil.copy2(JSON, BK)
    d = json.load(open(JSON))
    n = 0
    for k in d:
        # only the rung that carries reach_target_table has a live tolerance
        if len(k.get("reach_target_table", []) or []) == 3:
            k["target_distance_tolerance"] = tol
            n += 1
    json.dump(d, open(JSON, "w"), indent=2)
    got = [k["target_distance_tolerance"] for k in json.load(open(JSON))
           if len(k.get("reach_target_table", []) or []) == 3]
    if n != 1 or got != [tol]:
        restore()
        sys.exit("JSON patch did not take: %r on %d rungs" % (got, n))
    print("target_distance_tolerance = %g on the targeting rung" % tol, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.20)
    ap.add_argument("--heights", default="0.885,1.085,0.785,0.985")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pose_track", default="1")
    ap.add_argument("--wait_for", default="")
    ap.add_argument("--wait_token", default="mode sweep done")
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
    patch(a.tol)

    os.makedirs(a.out, exist_ok=True)
    f = open(os.path.join(a.out, "summary.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
    w.writeheader(); f.flush(); os.fsync(f.fileno())
    t0 = time.time()
    for h in [float(x) for x in a.heights.split(",")]:
        for seed in range(a.seeds):
            print("=== face %.3f seed %d tol=%g pose_track=%s ==="
                  % (h, seed, a.tol, a.pose_track), flush=True)
            rec = sweep.run_one((h, seed, a.out, a.task, a.slot, a.total_time,
                                 a.threads, True, a.spp, a.cpu_quota,
                                 ["--pose_track", a.pose_track]))
            w.writerow(rec); f.flush(); os.fsync(f.fileno())
    f.close()
    restore()
    print("tolerance sweep done in %.1f min -> %s"
          % ((time.time() - t0) / 60.0, a.out), flush=True)


if __name__ == "__main__":
    main()
