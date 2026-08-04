# Getting the slalom past 65%: an experiment plan

Status: **plan only, nothing here has been run.** Written 2026-08-02 as a
handoff for a separate session.

The technical report (`report/report.tex`) ends at 65% on the three-gap slalom
and lists four directions for closing the gap to real-time reliability. This
document turns those into experiments with pre-specified analyses, and adds a
fifth direction that only became visible after a planner bug was fixed on
2026-08-02.

Read `report/report.tex` §V and §VI-C first; this assumes its notation
(`w_avoid`, `delta`, the hinge residual, the two clocks).

---

## 0. What changed on 2026-08-02, and why it reorders the list

`RandomSamplingPlanner` was a silent no-op. It cleared `policy` to zero
intending to discard the incumbent, but `SamplingPlanner::OptimizePolicyCandidates`
opens with `UpdateNominalPolicy`, which — with the default
`sliding_plan_ = false` — rebuilds `policy.plan` node by node out of
`candidate_policy[winner]`, throwing the zeroing away. `--planner=9` was
therefore bit-identical to `--planner=0`, which is why a 30-cell sweep of the
two agreed in every cell.

Fixed by also clearing `candidate_policy[winner]`
(`mjpc/planners/random_sampling/planner.cc`). After the fix, at
`w=128000, delta=0.04, planner_seed=1000`:

| planner | solved/50 | gaps cleared |
|---|---|---|
| Predictive Sampling | 33 | 2.52 |
| Random Sampling (fixed) | **0** | **0.00** |

Memoryless sampling does not reach the first gap. **The warm start is doing
essentially all of the work.** Before the fix the report argued the opposite —
that the incumbent was not load-bearing, and therefore that better optimizers
were not worth pursuing. That argument is dead, and it promotes anything that
improves how the plan is *carried between iterations* to the top of the list.

---

## 1. Statistical budget — read this before designing any arm

At 150 trials/arm (3 seeds x 50), against a 65% baseline, two-sided alpha=0.05
and 80% power, the **minimum detectable effect is +14.3 points.** Every
"improvement" smaller than that is invisible at the budget every experiment in
the report used.

| effect to detect | n per arm | seeds x 50 |
|---|---|---|
| +20 pts (65 -> 85) | 70 | 2 |
| +15 pts (65 -> 80) | 136 | 3 |
| +10 pts (65 -> 75) | 326 | **7** |
| +5 pts (65 -> 70) | 1374 | 28 |
| +3 pts (65 -> 68) | 3882 | 78 |

Consequences to respect:

- **Screen at 150, confirm at 350.** Use 3 seeds to reject arms that are flat
  or worse, then re-run only the survivors at 7 seeds. Do not report a
  screening number as a result.
- **Pre-register the comparison.** One arm vs. the 65% baseline, stated before
  the run. The landscape grid already burned the multiplicity budget; a second
  round of "best cell of N" reporting is not interpretable.
- **Pooling 3 seeds as one binomial understates spread.** The 54% cell is
  66/48/48 across seeds — wider than its pooled Wilson interval. Report
  per-seed figures whenever the seed spread exceeds ~10 points, and prefer a
  seed-level paired test when comparing two configurations on the same seeds.
- Wall cost: a 50-trial cell at the ridge is ~550 s at 4 threads under 5-way
  contention. A 350-trial arm is ~65 min of one job, ~13 min of the whole box
  at `JOBS=5`.

---

## 2. Direction A — carry the plan better (new, highest priority)

The 0% vs 66% result says the incumbent is load-bearing. Nothing in the report
measures *how* load-bearing, or which part of the carry matters.

**A1. Ablate the warm start by degree, not on/off.** Interpolate between
predictive sampling and the memoryless control: before each iteration, shrink
the incumbent toward zero by a factor `rho`. `rho=1` is predictive sampling,
`rho=0` is the fixed random sampling.
- Arms: `rho in {0, 0.25, 0.5, 0.75, 0.9, 1.0}` at the ridge cell
  (`w=8192000, delta=0.01`), 3 seeds x 50 to screen.
- Reads out: whether the curve is a cliff near `rho=0` (the plan is a *seed*
  and any of it suffices) or roughly linear (the plan is an *asset* that
  accumulates). These imply different fixes.
- Implementation: one scalar on `SamplingPlanner`, applied in the same place
  the fixed `RandomSamplingPlanner` clears `candidate_policy[winner]`.

**A2. Knot count and interpolation.** 12 knots over a 1.0 s horizon is 83 ms
spacing, and the report notes the outcome is decided in ~80 ms around each
crossing — i.e. one knot interval. That is a suspicious coincidence and it has
never been varied at the ridge cell.
- Arms: `spline_points in {8, 12, 18, 24, 36}` at fixed horizon 1.0 s,
  `sampling_representation` linear (current) — 3 seeds x 50.
- Then the best two knot counts x `{zero, linear, cubic}` interpolation.
- Confound to control: more knots at fixed rollout count means noise is spread
  over more dimensions, so this is not a pure resolution test. Report
  rollouts/iteration and ms/iter alongside, and hold both fixed.

**A3. Does the sliding plan help?** `sampling_sliding_plan` is off, and the
task XML never sets it. With it on, `UpdateNominalPolicy` shifts and extends
the existing plan instead of resampling it from the winner. This is a one-line
XML change that alters exactly the mechanism A1 is probing.
- Arms: sliding on/off x the three best ridge cells, 3 seeds x 50.

---

## 3. Direction B — shape the residual (report §VI-C, untested)

The hinge `max(0, delta - d_ij)` is symmetric in approach direction: it charges
the same for a head moving toward a disk and one moving away, which is the
mechanism behind the wide-margin wall (at `w=128000, delta=0.20`: 0% solved,
0% collided, 0.06 gaps — it stops before the first gap).

**B1. Directional hinge.** Charge only on closing approach: multiply the hinge
by `max(0, -d/dt d_ij)` or gate it on the sign. Keeps the barrier for a head
entering a gap, removes it for one leaving.

**B2. Time-to-contact residual.** Replace distance with `d_ij / max(eps, closing
rate)`, hinged at a time threshold `tau` rather than a distance `delta`. This
makes the margin automatically speed-adaptive, which is the report's diagnosis
of the low-weight failure (at 4 cm and this course's approach speeds, the
planner is told about the obstacle less than one knot interval before it
matters).

- Both need a residual change in `triple_pendulum_cartpole.cc` (`Clearance()`
  at ~L76, hinge at ~L144) plus a new task parameter.
- Screen each on the **full margin/threshold axis at the two best weights**, not
  at one cell — the report's central finding is that weight and margin
  interact, so a one-cell comparison of a new residual against a tuned old one
  is not a fair test. That is 2 weights x 5 thresholds x 3 seeds x 50 = 1500
  runs per residual variant.
- Success criterion, pre-specified: beat 65% by +15 at 3 seeds, then confirm at
  7. A residual that merely matches 65% with a *wider* basin in threshold is
  also a real result and should be reported as such — quote the number of cells
  above 50%, not only the peak.

---

## 4. Direction C — spend the budget where the run is decided

Fig. 1 of the report shows the outcome decided in ~80 ms around each of three
crossings; the planner spends the same 4 iterations/control step for the other
11.7 s. At 1.03 ms/iteration and a 20 ms control period at `speed=0.25` there
is ~4x headroom.

**C1. Oracle upper bound first.** Before building an urgency signal, measure
what perfect allocation would buy: give 16 iterations/control step when the
cart is within 1.0 m of any gap centre and 1 elsewhere, keeping the *total*
iteration count matched to the flat-4 baseline. If the oracle does not beat 65%
by more than +15, the whole direction is dead and costs one screening sweep.

**C2. Then a causal signal**, only if C1 pays: minimum predicted clearance over
the current nominal rollout, which the planner already computes.

- Harness change: `corridor_benchmark.cc` currently derives iterations/control
  step from `speed`. Needs a per-step hook. Non-trivial — do C1 as a hardcoded
  positional rule before touching the interface.
- Report iterations actually spent, not requested; a variable-rate loop that
  quietly averages 5 instead of 4 is not a test of allocation.

---

## 5. Direction D — more rollouts, not more sophistication

10 rollouts x 201 steps is 0.49 Msteps/s against ~1.7 Msteps/s sustained, and
both hosts saturate at 3.5x on a workload that cannot fill them.

**D1.** `num_trajectory in {10, 20, 40, 80}` for predictive sampling at the
ridge cell, iteration count held fixed, 3 seeds x 50. This is the cheapest
experiment in this document and the one with the clearest prior: annealed
sampling already spends 40 rollouts and gets nothing for it, but it spends them
on an annealing schedule rather than on breadth at fixed centre.

**D2.** Re-measure ms/iter per arm — 80 rollouts will not fit the 5 ms budget
and the point is to find where it stops fitting. **Re-run
`machine_bench.sh` on an idle box for any arm that changes rollout count or
knot count**, because the report's throughput table is specific to
10 rollouts / 12 knots and will not describe these arms.

---

## 6. What not to do

- **Do not build a better zeroth-order optimizer.** The report's original
  argument for this was wrong (it rested on the broken control), but the
  corridor table still shows CEM at 2% and stock PSO at 18% against 76% for
  plain predictive sampling. Elaboration has consistently lost here.
- **Do not extend the weight/margin grid further.** The last 4x of weight
  bought +2 points at p=0.72. The next 35 points are not on that axis.
- **Do not report a screening arm as a finding.** See §1.

---

## 7. Order of work

1. D1 (rollout count) — cheapest, clearest prior, no code change.
2. A1 + A3 (warm-start ablation, sliding plan) — small code change, directly
   probes the finding that reordered this list.
3. C1 (oracle allocation bound) — one sweep, kills or justifies direction C.
4. A2 (knots/interpolation).
5. B1/B2 (residual shaping) — largest code change, largest sweep, do last
   unless A/C/D all come back flat.

Re-run the machine benchmark (§5 D2) and regenerate `report/report.tex` and the
published artifact after each direction that produces a confirmed result.

---

## 8. Provenance conventions to keep

Established in this project and expected by the user:

- Timestamped run directories under `renders/runs/<ts>_<name>/`.
- A `manifest.txt` per run recording host, load, commit, binary md5, and the
  full varied/fixed parameter split. **Check that it names the planner it
  actually ran** — `landscape_grid.sh` printed "predictive sampling"
  unconditionally until 2026-08-02, which is what let the no-op bug survive a
  30-cell sweep.
- `--planner_seed` on every run. Four planners still lack seeding
  (only the sampling family honours it); an unseeded arm cannot be compared to
  anything.
- Seed replicates, never a single seed.
- Results into the HTML artifact via `benchmark/landscape_report.py`, and into
  the paper via generated `.tex` (`report/make_ridge_table.py`), so the numbers
  in the prose and the numbers in the run directory cannot diverge.
