#!/usr/bin/env python3
"""Build an outcome-stratified gallery of slalom rollouts.

A success-rate table says 22/100 and stops. What it cannot say is *how* the
other 78 failed, and on this task the failures are not all alike: some never
reach the first bottleneck, some thread one and lose the pendulum before the
next, some clear two. This script turns a batch of dumps into the four rollouts
that show that spectrum, rendered so they can be watched rather than inferred.

Selection is by outcome, not by run index -- the point is coverage of the
failure modes, and run 0 is whatever it happened to be.

Frames are placed at the gap crossings rather than at uniform intervals. A
uniform sample of a 2 s rollout mostly shows the approach; the moments that
decide the outcome are the three instants the cart is level with a bottleneck.

Usage:
  python3 slalom_gallery.py --dumps renders/slalom_gallery/dumps \\
      --out renders/slalom_gallery/render
"""
import argparse
import csv
import glob
import json
import os
import pathlib
import subprocess
import sys

GAPS = (3.0, 6.0, 9.0)
GOAL = 11.0
GOAL_TOL = 0.3
PENETRATION_TOL = 0.0  # any overlap is a collision; see corridor_benchmark.cc

HERE = pathlib.Path(__file__).resolve().parent


def load(path):
    rows = [r for r in csv.reader(open(path)) if not r[0].startswith("#")]
    head, body = rows[0], rows[1:]
    idx = {k: head.index(k) for k in head}
    col = lambda k: [float(r[idx[k]]) for r in body]  # noqa: E731
    return {k: col(k) for k in ("step", "time", "cart", "min_clearance")}


def classify(d):
    """Outcome of one rollout: gaps cleared, and whether it reached the goal.

    "Reached the goal" is *ever* inside the goal box, not "ended near it". A run
    recorded with --early_exit=false keeps driving after it arrives and parks
    against the rail limit at 13 m, so testing the final or maximum cart
    position marks a clean traversal as a failure -- which is exactly what hid
    the only clean three-bottleneck runs in this batch.
    """
    max_cart = max(d["cart"])
    min_clear = min(d["min_clearance"])
    gaps = sum(1 for g in GAPS if max_cart > g + 0.5)
    reached = any(abs(c - GOAL) < GOAL_TOL for c in d["cart"])
    solved = reached and min_clear > -PENETRATION_TOL
    return dict(max_cart=max_cart, min_clear=min_clear, gaps=gaps,
                solved=solved, reached=reached, sim_time=d["time"][-1],
                steps=len(d["step"]))


def frames_at_gaps(d):
    """Step indices at start, each gap crossing, and the end.

    The crossing is the step whose cart position is closest to the gap's x, and
    only counts if the cart actually got there -- a run that stops at 2.2 m has
    no gap-2 frame to show.
    """
    picks = [0]
    for g in GAPS:
        if max(d["cart"]) < g - 0.4:
            continue
        i = min(range(len(d["cart"])), key=lambda k: abs(d["cart"][k] - g))
        picks.append(i)
    goal = [i for i, c in enumerate(d["cart"]) if abs(c - GOAL) < GOAL_TOL]
    picks.append(goal[0] if goal else len(d["cart"]) - 1)
    return sorted(set(picks))


def render(dump, out_png, out_mp4, frames, track, distance, width, height):
    cmd = [
        sys.executable, str(HERE / "filmstrip.py"), "--dump", dump,
        "--out", out_png, "--video", out_mp4,
        "--to-step", str(frames[-1] + 25),
        "--at", ",".join(str(f) for f in frames),
        "--cols", str(len(frames)),
        "--width", str(width), "--height", str(height),
        "--video-stride", "2", "--fps", "30", "--elevation", "-6",
        "--distance", str(distance),
    ]
    if track:
        cmd.append("--track")
    else:
        cmd += ["--lookat", "5.5,0,0"]
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", required=True, nargs="+",
                   help="one or more directories of --dump CSVs")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--field", action="store_true",
                   help="also render the fixed whole-field view, which shows "
                        "progress but renders the pendulum ~1/12 of frame width")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = []
    for dumpdir in args.dumps:
        for f in sorted(glob.glob(os.path.join(dumpdir, "*.csv"))):
            d = load(f)
            runs.append((f, d, classify(d)))
    if not runs:
        sys.exit(f"no dumps in {args.dumps}")

    # One representative per outcome tier. Within a tier take the run that got
    # furthest, so each tile is the best case of its failure mode and the
    # spectrum reads as a progression rather than as noise.
    #
    # The top tier is "cleared all three without touching a disk". That tier is
    # frequently empty: under the correct collision test -- any overlap counts,
    # see corridor_benchmark.cc --penetration_tolerance -- a run that threads
    # three 0.5 m gaps at speed usually brushes one of them. When it is empty
    # the gallery falls back to the best grazing run and says so, rather than
    # promoting a run that touched into a tile labelled "solved".
    tiers = {
        "1_stopped_at_gap1": lambda c: c["gaps"] == 0,
        "2_cleared_one": lambda c: c["gaps"] == 1,
        "3_cleared_two": lambda c: c["gaps"] == 2,
        "4_solved": lambda c: c["solved"],
        "4b_all_three_grazed": lambda c: c["gaps"] == 3 and not c["solved"],
    }
    manifest = []
    have_clean = any(r[2]["solved"] for r in runs)
    print(f"{'tier':20s} {'dump':12s} {'gaps':>5s} {'max_cart':>9s} "
          f"{'min_clear':>10s} {'sim_s':>7s}")
    for tier, pred in tiers.items():
        if tier == "4b_all_three_grazed" and have_clean:
            continue  # a real clean solve is available; do not also show a graze
        pool = [r for r in runs if pred(r[2])]
        if not pool:
            print(f"{tier:20s} (no run in this tier)")
            continue
        # best case of the tier: cleanest for a solve, furthest otherwise
        if tier == "4_solved":
            f, d, c = max(pool, key=lambda r: r[2]["min_clear"])
        else:
            f, d, c = max(pool, key=lambda r: r[2]["max_cart"])
        frames = frames_at_gaps(d)
        png = os.path.join(args.out, f"{tier}.png")
        mp4 = os.path.join(args.out, f"{tier}.mp4")
        r = render(f, png, mp4, frames, track=True, distance=3.2,
                   width=640, height=520)
        if r.returncode != 0:
            print(r.stdout[-800:], r.stderr[-800:])
            sys.exit(f"render failed for {f}")
        if args.field:
            render(f, os.path.join(args.out, f"{tier}_field.png"),
                   os.path.join(args.out, f"{tier}_field.mp4"), frames,
                   track=False, distance=14.0, width=1600, height=460)
        print(f"{tier:20s} {os.path.basename(f):12s} {c['gaps']:5d} "
              f"{c['max_cart']:9.2f} {c['min_clear']:+10.3f} "
              f"{c['sim_time']:7.2f}")
        manifest.append(dict(tier=tier, dump=os.path.basename(f), src=f,
                             png=png, mp4=mp4, frames=frames, **c))

    # Every run's outcome, for the distribution chart. This is the part a
    # success count throws away.
    dist = [dict(dump=os.path.basename(f), **c) for f, _, c in runs]
    with open(os.path.join(args.out, "outcomes.json"), "w") as fh:
        json.dump(dict(representatives=manifest, all_runs=dist), fh, indent=2)
    n = len(runs)
    hist = {g: sum(1 for r in dist if r["gaps"] == g) for g in (0, 1, 2, 3)}
    solved = sum(1 for r in dist if r["solved"])
    print(f"\n{n} runs: gaps cleared 0/1/2/3 = "
          f"{hist[0]}/{hist[1]}/{hist[2]}/{hist[3]}, solved {solved}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
