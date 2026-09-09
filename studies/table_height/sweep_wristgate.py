#!/usr/bin/env python3
"""Arm the wrist brace gate, and see whether the high end opens.

HYPOTHESIS.  The high end of the height window is not a reach deficit, it is a
gate watching the wrong surface.

`lean.cc`'s brace-contact-gated advance (the `brace_contact_verify` block,
~line 5583) decides the brace is established by scanning MuJoCo's contact list
for `left_forearm_pad` and nothing else.  Replaying the sweep's own qpos through
the narrowphase (`probe_padset.py`, no new runs) shows where the load actually
goes as the slab rises, median across shipped seeds -- peak normal force and the
fraction of brace-rung frames in contact:

    face      left_forearm_pad        left_wrist_roll_link (unnamed mesh)
    0.985     52 N over 19%           0 N
    1.035     89 N over 50%           0 N
    1.050     58 N over 49%           47 N over <1%
    1.060     73 N over 54%           66 N over <1%
    1.085      5 N over 76%           51 N over 52%

At the tall slab the pad is in contact for three quarters of the rung and holds
5 N; the wrist housing holds 51 N for half of it.  The robot IS braced, on a
surface the gate does not look at.

WHAT ARMS IT.  `wrist_brace_gate` [N] -- already implemented in lean.cc since
2026-09-01, whose comment describes exactly this failure ("a WRIST-on-rail brace
never triggers the `left_forearm_pad` contact scan above, so the reach rungs
could only advance by blind TIMEOUT -- the descent fired on a fixed timer even
mid-wobble").  The numeric was simply never declared in `Lean_H12_Magpie.xml`,
so on this task the wrist path could not be turned on.  It is declared now at
data="0.0" = OFF = byte-identical, and `Lean_H12_Magpie_battery_hip.xml` already
carries it at 30.0 -- Allen's own choice of threshold on another variant, and
within the 32-51 N the wrist measurably carries here.

MINIMALITY.  One existing model numeric, moved off its default.  That is the
cheapest class of change on this branch's own scale (an existing model numeric
beats a new weight, a weight beats a new residual, a new residual beats moving
the robot), and it is cheaper than every one of the twelve arms already run.

KILL CONDITION.  Stated before the runs, as house rule: this arm is refuted
unless it completes at least one height above 1.035 m on 2 of 3 seeds, at fixed
`--threads 6`, AND holds 3/3 at the 0.985 m control.  A gate that opens the top
by breaking the compiled height is not a widening.

usage: sweep_wristgate.py --out studies/table_height/runs/wristgate
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep

KEYS = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
        "t_end", "face_z", "phases", "enter", "csv", "summary"]

ARMS = {
    # 30 N is not a guess: it is the value Allen already ships on
    # Lean_H12_Magpie_battery_hip.xml, and the wrist carries 32-51 N at 1.085 m.
    "wg30": ["--numeric", "wrist_brace_gate=30"],
    # A lower bar, in case 30 N is only reached transiently on some seeds. Run
    # this ONLY if wg30 moves nothing -- it is a second look at the same knob,
    # not an independent arm.
    "wg15": ["--numeric", "wrist_brace_gate=15"],
    # The control: the numeric declared and left at its shipped 0. Proves the
    # XML edit alone changed nothing, which is the byte-identical claim.
    "off":  ["--numeric", "wrist_brace_gate=0"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="1.085,1.060,1.050,0.985")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="wg30")
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
    print("wristgate sweep done in %.1f min -> %s"
          % ((time.time() - t0) / 60.0, a.out), flush=True)


if __name__ == "__main__":
    main()
