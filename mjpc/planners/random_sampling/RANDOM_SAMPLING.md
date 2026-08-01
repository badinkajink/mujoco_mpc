# Memoryless random sampling in MJPC — teardown

The control baseline for every other sampling planner in this tree: draw
candidates around **zero control** every iteration instead of around the
incumbent, and publish the best.

- Implementation: [`planner.cc`](planner.cc) (30 lines), [`planner.h`](planner.h)
- Registered as planner index 9 ([`include.cc`](../include.cc))

---

## Verdict

**Not an algorithm to use — an experiment control, and the corridor task is a
case where it is nearly as good as the real planners.** That is the finding it
exists to produce.

Every sampling planner in MJPC (Predictive Sampling, Cross-Entropy, PSO,
Annealed Sampling) differs from this one in exactly one respect: they perturb
around the previous iteration's winner, this one perturbs around zero. So the
gap between any of them and this baseline is the measured value of *keeping the
incumbent* — of having a warm start at all. If a planner cannot beat it on a
task, whatever that planner does beyond sampling is not what is solving the
task.

On the corridor task it is competitive with Predictive Sampling
([see the numbers](#measured-behaviour)), which says the task is solved
essentially by re-deciding from scratch at 200 Hz rather than by refining a
plan. That is a statement about the task, and it is the reason this planner is
in the tree.

---

## What problem it solves

It answers "would a random number generator have done that?" for every other
row of a planner comparison table. Aggregate cost tables invite the reading that
a fancier planner's advantage comes from its mechanism; the only way to know is
to run the same rollout budget with the mechanism removed. This is that run.

---

## The algorithm

There is no paper. Random sampling (also called "random shooting" or "random
search" in the literature; the `N`-sample degenerate case of MPPI with a
winner-take-all update) is:

```
(1)  u_i ~ pi_0 + N(0, sigma^2),  i = 1..n       candidates from a FIXED prior
(2)  J_i = rollout(u_i)
(3)  u* = argmin_i J_i
(4)  publish u*                                   ...and forget it
```

The single substantive choice is `pi_0`. Predictive Sampling sets
`pi_0 = ` previous winner, which is what makes it an optimizer with memory. Here
`pi_0 = 0` for all time, which makes each iteration an independent draw.

**Restated for this setting.** The implementation does not reimplement any of
this — it inherits `SamplingPlanner` and overrides one method to clear the
incumbent before the base class runs:

```cpp
// planner.cc:37-41
{
  const std::unique_lock<std::shared_mutex> lock(mtx_);
  std::vector<double> zero(model->nu, 0.0);
  policy.Reset(horizon, zero.data());
}
return SamplingPlanner::OptimizePolicyCandidates(ncandidates, horizon, pool);
```

`SamplingPlanner::OptimizePolicyCandidates` begins with `ResamplePolicy`, which
reads `policy` into `parameters_scratch`, and `AddNoiseToPolicy` perturbs around
that. Zeroing the spline first therefore makes the whole sample set centre on
zero control. Candidate 0 is left un-noised by the base class, so **zero control
is always in the sample set** — this planner can never do worse than doing
nothing, which is a meaningful floor on a task where doing nothing is stable.

Because everything else is inherited, the noise scale, spline representation,
rollout budget, thread pool use and clamping are *identical* to Predictive
Sampling by construction. Nothing else can differ, which is what makes it a
clean control.

---

## Equation-to-code map

| Step | Quantity | Code |
|---|---|---|
| (1) `pi_0 = 0` | clear the incumbent each iteration | [planner.cc:37-41](planner.cc#L37-L41) |
| (1) noise | `u_i = pi_0 + N(0, sigma^2)` | [`../sampling/planner.cc`](../sampling/planner.cc) `AddNoiseToPolicy` |
| (2) | rollouts | [`../sampling/planner.cc`](../sampling/planner.cc) `Rollouts` |
| (3), (4) | best candidate published | [`../sampling/planner.cc`](../sampling/planner.cc) `CopyCandidateToPolicy` |

The whole class is the first row. Everything else is inherited unmodified, and
that is the point.

---

## Walkthrough

One iteration, corridor task, 12 knots, 10 candidates:

```
policy (whatever was published last iteration)
  |
  v  planner.cc:40 -- policy.Reset(horizon, 0)  <- the entire algorithm
policy = 0 at every knot
  |
  v  SamplingPlanner::ResamplePolicy -- parameters_scratch = 0
  v  SamplingPlanner::AddNoiseToPolicy -- candidate 0 = 0 (un-noised),
  |                                       candidates 1..9 = N(0, sigma^2)
  v  10 rollouts, 200 steps each
  v  argmin over total_return
published policy = best of {zero control, 9 random splines}
```

Next iteration discards it and draws 10 fresh candidates around zero.

---

## Measured behaviour

Corridor task, obstacle-avoidance objective
`[cart, upright, velocity, control, avoidance] = [1, 0, 0.1, 0.01, 500]`,
100 trials, commit `e22fc9f`:

```bash
mjpc/tasks/triple_pendulum_cartpole/benchmark/avoidance_sweep.sh renders/avoid100 100 0.25
```

The comparison that matters is this planner against Predictive Sampling, which
is identical to it except for the warm start:

| planner | reached goal, no contact | 1 s.e. | collided | median t_solve |
|---|---|---|---|---|
| **Random Sampling** | **83/100** | ±3.8 | 17% | 2.39 s |
| Predictive Sampling | 76/100 | ±4.3 | 24% | 2.33 s |

**The baseline is nominally ahead of the planner it is a control for.** Seven
points, against a combined standard error of about 6 — so the honest reading is
that the two are indistinguishable, not that discarding the incumbent helps.
Either way the warm start buys nothing measurable here: at `--speed=0.25` the
planner gets 800 optimization iterations per simulated second, and at that rate
the corridor is solved by re-deciding from scratch. That is a fact about the
task, not about Predictive Sampling.

(Both numbers are under the corrected collision test — any overlap with a disk
disqualifies. Under the 20 mm penetration tolerance this benchmark used to
apply, the two tied at 93/100. The correction moved Predictive Sampling 17
points and this planner 10, which is itself worth knowing: the warm-started
planner was grazing more often.)

The result does *not* generalize. On stock Cartpole, 10 s, 3 runs
(`benchmark/cartpole_costs.sh 10 3`), average cost per step:

| planner | run 1 | run 2 | run 3 |
|---|---|---|---|
| Predictive Sampling | 0.901 | 0.956 | 0.947 |
| **Random Sampling** | 0.961 | 0.945 | 0.975 |

Close, but consistently behind — and there the task is to hold a balance, where
continuity between iterations is exactly what is being asked for. Two tasks, two
answers: run this baseline per task rather than assuming either result.

**It is also the cheapest planner in the tree, by a hair.** 6000 iterations,
`--early_exit=false`:

| planner | ms/iteration | p95 |
|---|---|---|
| **Random Sampling** | **1.000** | 1.59 |
| PSO | 1.018 | 1.64 |
| Predictive Sampling | 1.031 | 1.62 |

3% below Predictive Sampling, which is what discarding the incumbent is worth on
the clock: `policy.Reset` writes 12 zeros, and skipping the resample of a
non-trivial spline saves marginally more than that costs. The point is the
scale — the difference between having a warm start and not having one is 3% of
an iteration and, on this task, 0 points of success rate.

Full cross-planner table:
[the task README](../../tasks/triple_pendulum_cartpole/README.md#headline-results).

---

## Deviations

No paper, so no deviations. One design decision worth stating: the incumbent is
cleared inside `OptimizePolicyCandidates` rather than by a flag on
`SamplingPlanner`, so `SamplingPlanner` stays untouched and the baseline cannot
drift away from the planner it is a control for.

---

## Failure modes

- **Beats the real planners.** Not a failure of this planner — a finding about
  the task, and usually that the horizon is short enough, or the system stable
  enough, that a warm start buys nothing.
- **Chattering control.** Successive iterations are independent draws, so the
  executed action has no continuity between planning steps. Expected; it is why
  this is a baseline and not a controller.
- **Looks artificially good on tasks where zero control is a good policy.**
  Candidate 0 is always zero control, so on any task whose failure mode is
  "doing something bad", this planner has a free floor the others do not.

---

## Parameters

None of its own. It inherits every `sampling_*` numeric from
[`SamplingPlanner`](../sampling/planner.cc); to keep it a valid control, set
them exactly as they are set for whatever planner it is being compared against.

---

## References

- [`../sampling/planner.cc`](../sampling/planner.cc) — the base class, and the
  planner this is the control for.
- [`../pso/PSO.md`](../pso/PSO.md), [`../annealed_sampling/ANNEALED_SAMPLING.md`](../annealed_sampling/ANNEALED_SAMPLING.md)
- [`../../tasks/triple_pendulum_cartpole/README.md`](../../tasks/triple_pendulum_cartpole/README.md)
