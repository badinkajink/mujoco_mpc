# Table-height study — status

Written 2026-09-05 00:08 by the unattended finish script.

## Where it stands

Table height is an MJPC task parameter (`residual_Table H`, index 7) and the
controller was swept over it with nothing else changed. **6 of 15 runs
completed.** The working window is 0.985 m, 1.035 m; 0.785 m, 0.885 m, 1.085 m fail outright.

| face (m) | complete | outcomes | seated | forearm peak | CoM past toes |
|---|---|---|---|---|---|
| 0.785 | 0/3 | 3 stalled | 13% | 191 N | +625 mm |
| 0.885 | 0/3 | 3 fell | 11% | 143 N | +371 mm |
| 0.985 | 3/3 | 3 complete | 32% | 234 N | +185 mm |
| 1.035 | 3/3 | 3 complete | 23% | 179 N | +34 mm |
| 1.085 | 0/3 | 3 fell | 0% | 0 N | -85 mm |

Three distinct failure modes, not one. At 0.785 the robot **drapes**: 398 N
through the torso against 37 N on the forearm, CoM 625 mm past the toes, and it
never falls because the table is holding it up. At 0.885 it **falls forward**
(torso 106 deg). At 1.085 it **falls backward** — CoM peaks 85 mm *behind* the
toes with ~1 N of table contact, because the slab is above the reachable brace
envelope and it never gets support at all.

The seating metric had to be defined by newtons, not geometry: 1.085 scores
"100% at face level" while carrying 0 N, since the forearm hangs outboard of the
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
`--threads 6`, under `systemd-run --user --scope -p CPUQuota=700%`. `sweep.py`
enforces this and refuses to start if the load is already high. Budget ~5-10 min
per run.

## Long-cap check (`--total_time 140`)

The 75 s cap censors a stall: a run that neither fell nor finished might only have been slow. 140 s is about 3x the nominal completion time.

| face (m) | complete | in contact | forearm peak |
|---|---|---|---|
| 0.785 | 0/2 | 1% | 20 N |
| 0.885 | 0/2 | 0% | 8 N |
| 0.985 | 2/2 | 35% | 235 N |
| 1.035 | 1/2 | 14% | 202 N |
| 1.085 | 0/2 | 19% | 22 N |

The stalls are real, not clipped: 0.785 m, 0.885 m still complete 0 of their seeds with three times the clock, so the low slab is unreachable rather than slow.

⚠ 1.035 m completed at 75 s on 3/3 seeds but only 1/2 here. Treat the upper edge of the window as marginal, not solid.

## `brace_com_hold` A/B at 1.085 m

`lean.cc` carries a one-sided "keep the CoM at least this far ahead of midfoot while an arm is on the table" term whose comment describes the 1.085 m signature exactly. The numeric is **absent from `Lean_H12_Magpie.xml`**, so it defaults to 0 = off. Treatment injects `brace_com_hold 0.05` into the BUILD copy only.

| arm | complete | falls | t_end (s) |
|---|---|---|---|
| off (shipped) | 0/3 | 3 | 30.3, 35.0, 32.7 |
| on (0.05) | 0/1 | 1 | 49.4 |

Mean survival 32.6 s off vs 49.4 s on. The term delays the backward fall without preventing it, so it is a lead rather than a fix.
