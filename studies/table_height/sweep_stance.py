#!/usr/bin/env python3
"""Stance shift: start the robot where the brace keyframes assume it stands.

Written before the runs.

WHY.  `probe_stance.py`, 2026-09-06.  The `home` keyframe every entry point
resets to (`lean_bench.cc:150`, `app.cc:423`, `deploy_common.cc:899`) puts the
feet 260 mm behind the slab's near edge.  All three `forearm_brace_*` keyframes
were authored for 197 mm.  Strategy 25 carries `Foot Left Up` / `Foot Right Up`
at weight 2000 on all nine rungs and has no stepping rung, so the 63 mm is
structural: measured foot midpoint over the brace rungs is -0.257 to -0.270 m at
every height tested, flat to within 8 mm.  The consequence is the forearm pad's
best reach past the near edge -- +0.140 m at a 0.985 m face, +0.054 at 1.035,
-0.066 at 1.085 -- monotone, crossing zero exactly where completions stop.  At
1.085 m the pad never gets over the wood, carries 0.00 N in every run of every
arm, and the robot topples backward with the CoM 1.07-1.14 m behind the foot edge.

WHAT.  `--stance_shift_x 0.063`, bench-only, applied to qpos[0] after the reset.
The task, the XMLs and every strategy JSON are untouched.  It lands the feet at
-0.197 m, exactly where `forearm_brace_lean` puts them.

NO CONFOUND IN THIS STRATEGY (checked, and re-check before reusing the flag):
every live forward-x term is measured from midfoot and travels with the feet --
the `Pelvis Forward` band is midfoot+0.05 to midfoot+`pelvis_cap_fwd`,
`com_cap_fwd` and `brace_com_hold` likewise.  The one absolute-world-x constant,
`brace_lead_x0` 0.24 against `data->qpos[0]`, sits behind JSON weight
"Brace Reach Lead" = 0.0 on all nine rungs.

CONTROL.  `runs/ab/off` (shipped) at these exact settings -- total_time 75,
threads 6, spp 3, quota 700 -- 0/3, 0/3, 3/3, 2/3, 0/3 over 0.785 .. 1.085 m.

H1.  At 1.085 m the pad reach goes positive (it is -0.066 m shipped; the shift is
+0.063, so a rigid translation predicts about -0.003 and anything the arm adds on
top puts it over the wood) and rung-2 forearm load rises off 0.00 N.

H2.  At least one failing height completes on at least 2 of 3 seeds, with 0.985 m
still 3/3.

KILL.  0.985 m losing a completion disqualifies the arm outright, under the rule
that a change must cost nothing where the controller already works.  Failing that,
if 1.085 m pad reach goes positive and it still completes 0/3, the stance was
never the binding constraint and the next measurement is what the pad does once
it is over the slab -- contact force and pad clearance through rung 2, not another
placement.

RISK, stated in advance.  A closer stance should make the low end WORSE: at
0.785 m the robot already drapes on the slab with 398 N through the torso, and
starting 63 mm nearer gives it less room to bow.  0.785 m getting worse is
expected and does not by itself kill the arm; 0.985 m getting worse does.

usage: sweep_stance.py --out studies/table_height/runs/stance
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep

KEYS = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
        "t_end", "face_z", "phases", "enter", "csv", "summary"]

# 0.063 m is measured, not tuned: it is the gap between the `home` stance and the
# one all three brace keyframes assume. 0.100 is the dose-response check, run at
# the high slab only and only if 0.063 moves the pad.
ARMS = {
    "s63":  ["--stance_shift_x", "0.063"],
    "s100": ["--stance_shift_x", "0.100"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    # 1.085 first (the target), 0.985 second (the control that can kill the arm),
    # then the marginal upper edge, then the low end the shift should hurt.
    ap.add_argument("--heights", default="1.085,0.985,1.035,0.785")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="s63")
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
    print("stance sweep done in %.1f min -> %s" % ((time.time() - t0) / 60.0, a.out),
          flush=True)


if __name__ == "__main__":
    main()
