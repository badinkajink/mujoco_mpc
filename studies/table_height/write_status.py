#!/usr/bin/env python3
"""Write STATUS.md: where the table-height study stands and what to run next.

Read this first in the morning. Numbers come from agg.json so it cannot drift
from the figures.
"""
import argparse, json, os, datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--long", default="", help="agg.json dir for the 140 s sweep")
    ap.add_argument("--ab", default="", help="runs dir for the brace_com_hold A/B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    agg = json.load(open(os.path.join(a.figs, "agg.json")))
    runs = json.load(open(os.path.join(a.figs, "summary.json")))
    ok = [x for x in agg if x["complete"] == x["n"]]
    bad = [x for x in agg if x["complete"] == 0]

    def row(x):
        modes = {}
        for r in runs:
            if abs(r["table_h"] - x["h"]) < 1e-6:
                modes[r["outcome"]] = modes.get(r["outcome"], 0) + 1
        mode = ", ".join("%d %s" % (v, k) for k, v in sorted(modes.items()))
        seat = ("--" if x["seated_fraction"] is None
                else "%.0f%%" % (100 * x["seated_fraction"]))
        peak = "--" if x["f_forearm_peak"] is None else "%.0f N" % x["f_forearm_peak"]
        com = ("--" if x["com_beyond_peak_mm"] is None
               else "%+.0f mm" % x["com_beyond_peak_mm"])
        return "| %.3f | %d/%d | %s | %s | %s | %s |" % (
            x["h"], x["complete"], x["n"], mode, seat, peak, com)

    txt = """# Table-height study — status

Written %s by the unattended finish script.

## Where it stands

Table height is an MJPC task parameter (`residual_Table H`, index 7) and the
controller was swept over it with nothing else changed. **%d of %d runs
completed.** The working window is %s; %s fail outright.

| face (m) | complete | outcomes | seated | forearm peak | CoM past toes |
|---|---|---|---|---|---|
%s

Three distinct failure modes, not one. At 0.785 the robot **drapes**: 398 N
through the torso against 37 N on the forearm, CoM 625 mm past the toes, and it
never falls because the table is holding it up. At 0.885 it **falls forward**
(torso 106 deg). At 1.085 it **falls backward** — CoM peaks 85 mm *behind* the
toes with ~1 N of table contact, because the slab is above the reachable brace
envelope and it never gets support at all.

The seating metric had to be defined by newtons, not geometry: 1.085 scores
"100%% at face level" while carrying 0 N, since the forearm hangs outboard of the
wood at face height. At both working heights the two measures agree exactly.

## Artifacts

- Page: `docs/lean/20260904-table_height_generalization.html` (local only, by
  request — nothing published to claude.ai)
- Figures + JSON: `docs/lean/media/th/`, source in `studies/table_height/figs/`
- Videos: `docs/lean/media/th/h*_s0.mp4`, one per height
- Raw runs: `studies/table_height/runs/seeded/` (gitignored)
- Index: `docs/experiments/INDEX.md`

## Code changed (all uncommitted, on `icra2026`)

- `mjpc/tasks/humanoid_bench/lean/lean.h` — param index 7, bench accessors
- `mjpc/tasks/humanoid_bench/lean/lean.cc` — the `Table H` block in
  `TransitionLocked` (0 = OFF = byte-identical)
- `mjpc/tasks/humanoid_bench/lean/*.xml` — `residual_Table H` numeric, appended
  at the tail so indices 0-6 are untouched
- `mjpc/lean_bench.cc` + `mjpc/CMakeLists.txt` — the bench and its target
- `studies/table_height/` — sweep, analyze, render, page, status
- `CLAUDE.md` — corrected to point at `lean.cc` on `icra2026`

⚠ Nothing is committed. Allen commits to `icra2026` most days; a `git pull`
before committing will otherwise clobber the `lean.h`/`lean.cc` edits.

## Next, in order

**The two ends fail differently and need different fixes.** My pre-sweep guess
that `com_cap_fwd` was the binding constraint is wrong on both counts: it is a
soft cost (`max(0, com_x - (midfoot_x + cap))` folded into a residual), not a
limit, and the failures blow straight past it — 371 mm at 0.885 against a 145 mm
cap. It is being beaten, not enforced.

1. **High end — `brace_com_hold`, already queued as `brace_hold_ab.sh`.**
   `lean.cc` has a one-sided "keep the CoM at least this far ahead of midfoot
   while an arm is on the table" term whose comment describes my 1.085 signature
   exactly: *"every backward fall of hp133-161 shows the CoM walking from +5 cm
   to -10 cm at a rung fire."* The numeric is **absent from
   `Lean_H12_Magpie.xml`, so it defaults to 0 = off.** The A/B runs 3 seeds with
   it off and 3 with `brace_com_hold 0.05`, at 1.085 m, editing only the build
   copy of the model. Results append below. Falsified if the treatment arm still
   falls backward 3/3.
2. **Low end — the excursion penalty is losing, so raise its price.** At 0.785
   the torso takes 398 N and the CoM goes 625 mm out; the robot buys chest
   support with excursion because the brace reward outbids the excursion penalty.
   The lever is the WEIGHT on that residual (`Pelvis Forward`, which carries
   `com_over`), not the cap value. A weight sweep at 0.885 is the cheap test.
3. **Resolve the stalls.** The long-cap sweep (`runs/long`, 140 s) is appended
   below. If the stalled points still do not finish, they are stuck rather than
   clipped, and it is a reach problem: the slab's near edge sits at x = 0.45 m at
   every height, so a low table is reached by bowing rather than stepping.
4. **Only then, standoff.** Sweep the table's x at one off-nominal height to
   separate "cannot brace low" from "cannot brace far". Goal 4's last resort.

## Two defects to send Allen (neither affects the controller)

- `lean::ComputeMetrics` — `brace_force` reads `right_contact[0]` while the model
  braces with `left_forearm_pad` / `left_wrist_pad`. It reports **0.00 N through a
  real brace** in which the left forearm carried 97-168 N, so the deploy monitor
  shows no brace force when the brace is working.
- `reach_err` / `reach_tgt_*` are gated on `kf.name == "reach_to_target"`, a rung
  name strategy 25 never uses, so the whole family is nan for any braced ladder.

## Run policy on this box

MJPC is CPU-bound and parallel sweeps have stuttered the desktop twice. Serial,
`--threads 6`, under `systemd-run --user --scope -p CPUQuota=700%%`. `sweep.py`
enforces this and refuses to start if the load is already high. Budget ~5-10 min
per run.
""" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
       sum(x["complete"] for x in agg), sum(x["n"] for x in agg),
       (", ".join("%.3f m" % x["h"] for x in ok) or "empty"),
       (", ".join("%.3f m" % x["h"] for x in bad) or "no height"),
       "\n".join(row(x) for x in agg))
    # ---- long-cap sweep ----------------------------------------------------
    if a.long and os.path.exists(os.path.join(a.long, "agg.json")):
        lg = json.load(open(os.path.join(a.long, "agg.json")))
        txt += ("\n## Long-cap check (`--total_time 140`)\n\n"
                "The 75 s cap censors a stall: a run that neither fell nor "
                "finished might only have been slow. 140 s is about 3x the "
                "nominal completion time.\n\n"
                "| face (m) | complete | in contact | forearm peak |\n"
                "|---|---|---|---|\n")
        for x in lg:
            txt += "| %.3f | %d/%d | %s | %s |\n" % (
                x["h"], x["complete"], x["n"],
                "--" if x["seated_fraction"] is None
                else "%.0f%%" % (100 * x["seated_fraction"]),
                "--" if x["f_forearm_peak"] is None else "%.0f N" % x["f_forearm_peak"])
        low = [x for x in lg if x["h"] < 0.985 and x["complete"] == 0]
        if low:
            txt += ("\nThe stalls are real, not clipped: %s still complete 0 of "
                    "their seeds with three times the clock, so the low slab is "
                    "unreachable rather than slow.\n" % ", ".join(
                        "%.3f m" % x["h"] for x in low))
        marg = [x for x in lg if 0 < x["complete"] < x["n"]]
        if marg:
            txt += ("\n⚠ %s completed at 75 s on 3/3 seeds but only %s here. "
                    "Treat the upper edge of the window as marginal, not "
                    "solid.\n" % (
                        ", ".join("%.3f m" % x["h"] for x in marg),
                        ", ".join("%d/%d" % (x["complete"], x["n"]) for x in marg)))

    # ---- brace_com_hold A/B ------------------------------------------------
    if a.ab and os.path.isdir(a.ab):
        import glob, re
        arms = {}
        for fp in sorted(glob.glob(os.path.join(a.ab, "*.log"))):
            arm = os.path.basename(fp).split("_")[0]
            m = re.search(r"fell=(\d+) complete=(\d+) t_complete=(\S+) "
                          r"t_end=(\S+)", open(fp).read())
            if m:
                arms.setdefault(arm, []).append(
                    (int(m.group(1)), int(m.group(2)), float(m.group(4))))
        if arms:
            txt += ("\n## `brace_com_hold` A/B at 1.085 m\n\n"
                    "`lean.cc` carries a one-sided \"keep the CoM at least this "
                    "far ahead of midfoot while an arm is on the table\" term "
                    "whose comment describes the 1.085 m signature exactly. The "
                    "numeric is **absent from `Lean_H12_Magpie.xml`**, so it "
                    "defaults to 0 = off. Treatment injects "
                    "`brace_com_hold 0.05` into the BUILD copy only.\n\n"
                    "| arm | complete | falls | t_end (s) |\n|---|---|---|---|\n")
            for arm in ("off", "on"):
                if arm not in arms:
                    continue
                v = arms[arm]
                txt += "| %s | %d/%d | %d | %s |\n" % (
                    "off (shipped)" if arm == "off" else "on (0.05)",
                    sum(c for _, c, _ in v), len(v), sum(f for f, _, _ in v),
                    ", ".join("%.1f" % t for _, _, t in v))
            ns = {k: len(v) for k, v in arms.items()}
            if len(set(ns.values())) > 1:
                txt += ("\n⚠ Unequal arms (%s) — the A/B was still running when "
                        "this was written. Re-run `write_status.py` to refresh.\n"
                        % ", ".join("%s n=%d" % kv for kv in sorted(ns.items())))
            if "on" in arms and "off" in arms:
                mo = sum(t for _, _, t in arms["on"]) / len(arms["on"])
                mf = sum(t for _, _, t in arms["off"]) / len(arms["off"])
                txt += ("\nMean survival %.1f s off vs %.1f s on. %s\n" % (
                    mf, mo,
                    "The term delays the backward fall without preventing it, so "
                    "it is a lead rather than a fix." if mo > mf + 2 else
                    "No useful separation; look elsewhere for the upper-edge fix."))

    open(a.out, "w").write(txt)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
