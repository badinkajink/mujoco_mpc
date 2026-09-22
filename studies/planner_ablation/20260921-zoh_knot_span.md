# Zero-order knot span: the shipped 3-knot CEM plan has two live segments

2026-09-21, branch `wxie/zoh-knot-span-icra2026` (off `wxie/planner-ablation`).
Runs in `runs/zoh_spp15/`; the clean fix for upstream is `fix/zoh-knot-span`.

## Finding

`CrossEntropyPlanner::ResamplePolicy` places n knots at k·T/(n−1), so with
`sampling_spline_points = 3` and `agent_horizon = 1.0` the knots sit at 0,
0.5 and 1.0 s after the plan's state time. The rollout samples the policy at
steps 0..H−2 (`trajectory.cc:140`), the last at 0.99 s; the step-H−1 action is
a copy of the previous one (`trajectory.cc:199`). Under zero-order hold
`TimeSpline::Sample` returns the lower knot, so the knot at 1.0 s is never
read by a rollout, never enters a cost, and is never executed (the plan is
read ≤ 22 ms after its state time, knot 0). The shipped controller is two
0.5 s hold segments, 54 live parameters of 81; the third knot's 27 noise draws
and its elite refit act on nothing.

Evidence from the shipped binary: `studies/alg1_filmstrips/runs/
cem_stdmin01_s10_dense/samples/tick04_t20.012/` has knot times
20.010 / 20.510 / 21.010 and a last policy sample at 21.000.

Whose: upstream DeepMind code. The formula is alberthli's CEM (2024-01-11); the
rollout loop and final-action copy are Howell (2022-12, 2023-01). Upstream
fixed the *sampling* planner on 2024-04-08 (734bda29): T/n for `kZeroSpline`,
T/(n−1) otherwise. CEM, sample-gradient and gradient never received the
branch, on upstream `main` as of today; the fork's iCEM, iCEM-DR and CoVO
copied CEM's formula. MPPI here was derived from the sampling planner and
already has the branch. So in the planner ablation PS-hold and MPPI-hold at
"3 knots" ran three 0.333 s segments while CEM-hold and iCEM-hold ran two
0.5 s segments.

## The fix

The sampling planner's branch, applied to CEM, iCEM, iCEM-DR, CoVO,
sample-gradient and gradient (commit 54323216 here; 08866ef9 on
`fix/zoh-knot-span` for the three upstream planners). Interpolating splines
are unchanged. `lean_bench` now prints the knot times on the first plan:

| `sampling_spline_points` | knots (s) | last rollout sample (s) |
|---|---|---|
| 2 | 0, 0.500 | 0.990 |
| 3 | 0, 0.333, 0.667 | 0.990 |
| 4 | 0, 0.250, 0.500, 0.750 | 0.990 |

On the fixed binary `sampling_spline_points = 2` is the shipped controller
(same two live knots at the same times; only the dead knot's noise draws are
gone), and 3 is the controller the XML always described.

## Test: does a third live segment help at the robot's plan rate?

`Lean H12 Magpie`, strategy 25 at 0.985 m, `--gains deploy`, `--spp 15`
(33 plans/s), 6 threads, 75 s, seeds 0–11, CEM N=20, n_elite 6, σ floor 0.01.

| arm | segments | complete | falls | t_complete (s) |
|---|---|---|---|---|
| `cem_k2` (= shipped) | 2 × 0.500 s | **11/12** | 1 | 44.4 ± 2.3 |
| `cem` (fixed, 3 knots) | 3 × 0.333 s | **11/12** | 1 | 45.3 ± 4.1 |
| `cem_k4` | 4 × 0.250 s | 7/12 | 5 | 44.1 ± 4.8 |
| shipped binary `cem`, `runs/gains_spp15` | 2 × 0.500 s + dead knot | 10/12 | 2 | 41.7 median |

Every fall is the lean-onset backward fall at t = 14.2–15.5 s (phase 1
enters at 12.0 s). Rung entry times of the completing seeds agree to within
1 s between the 2- and 3-segment arms.

The third segment neither helps nor hurts on this task at 33 plans/s. The
0.25 s segment falls 5/12, which matches the persistence result from the
ablation (PS-hold k4 5/6, k5 1/6, k6 2/6 on the correct placement): the
executed perturbation has to stay correlated for roughly 0.3 s of the rollout
to be selected on at this rate, and 0.25 s is below that.

## What this does not settle

- iCEM on the same ladder: `./sweep.py --out runs/zoh_spp15 --batch KI --spp
  15 --seeds 0..11 --gains deploy`, 36 runs. iCEM's AR(1) knot noise gives the
  executed perturbation its persistence independently of the segment length,
  so it may tolerate k4 where CEM does not.
- 167 plans/s (`--spp 3`): at the well-fed rate cubic splines completed, so
  shorter segments may pay there. Same batch K at spp 3.
- The twin. The deploy node reads `sampling_spline_points = 3` from the XML;
  on the fixed planner that is 3 × 0.333 s. Nothing above says the change is
  safe on hardware; run the twin discriminator (`h12_control_node` on the
  sim2sim plant) before the fix lands on `icra2026`.
- The paper's text: "three knots" should read "two 0.5 s zero-order-hold
  segments", and the hold-vs-hold family rows should say which segment count
  each ran.
