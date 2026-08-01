# DIAL-MPC annealed sampling in MJPC — teardown

DIAL-MPC's diffusion-style annealing as an MJPC `RankedPlanner`. Written to the
[`algorithm-teardown`](../../../.claude/skills/algorithm-teardown/SKILL.md)
format: every equation is anchored to a line verified against the current file,
and every claim has a number produced by running the code.

- Implementation: [`planner.cc`](planner.cc), [`planner.h`](planner.h)
- Registered as planner index 8 ([`include.cc`](../include.cc))
- Paper: Xue et al., *Full-Order Sampling-Based MPC for Torque-Level Locomotion
  Control via Diffusion-Style Annealing* (DIAL-MPC), arXiv:2409.15610.
  Equation numbers below are the paper's.

---

## Verdict

**A faithful implementation of DIAL-MPC's dual-loop annealing on top of MJPC's
MPPI update, with one systematic scaling deviation and one budget hazard.**

1. **The annealing schedules are right.** Both loops of Eq (7) — trajectory-level
   over the `N` refinement iterations and action-level over the horizon — are
   present and correctly oriented ([planner.cc:416-427](planner.cc#L416-L427)):
   noise starts wide and narrows over iterations, and is larger for far-horizon
   knots than near-horizon ones.

2. **It anneals the standard deviation where the paper specifies the
   covariance.** Eq (7) defines `Σ = exp(−(N−i)/(β₁N) − (H−h)/(β₂H))·I`. The code
   uses that expression as a multiplier on σ ([planner.cc:427](planner.cc#L427),
   consumed as the stddev argument at [planner.cc:432](planner.cc#L432)), so the
   implemented covariance is the paper's squared. The schedule shape is
   unchanged; the effective temperatures are halved. Set `β₁, β₂` to twice the
   paper's values to match it exactly.

3. **It costs `annealing_iterations` × the rollouts of every other sampling
   planner.** `OptimizePolicy` runs the full sample-rollout-update loop `N` times
   per planning iteration ([planner.cc:187-214](planner.cc#L187-L214)), so with
   the defaults it does 10 × 4 = 40 rollouts where Predictive Sampling, PSO,
   Cross-Entropy and Random Sampling each do 10. Any comparison that does not
   account for this is comparing budgets, not algorithms. See
   [Measured behaviour](#measured-behaviour).

4. **Unlike [PSO](../pso/PSO.md), the inner loop genuinely refines.**
   `MPPIUpdate` writes the weighted average into `policy`
   ([planner.cc:257-276](planner.cc#L257-L276)) and the next annealing iteration
   re-centres its noise on that updated `policy` via `UpdateNominalPolicy`
   ([planner.cc:193](planner.cc#L193)). State carries across iterations, which is
   what makes the annealing schedule mean anything.

---

## What problem it solves

MJPC's Predictive Sampling picks the single best of `n` perturbed rollouts. That
is one diffusion stage at one noise level, and it forces a choice: wide noise
covers the space but converges poorly, narrow noise converges but misses distant
optima (DIAL-MPC Fig. 4). On a chaotic underactuated system there is no single
noise level that does both — the corridor task needs coarse exploration to find
"lay the pendulum flat and drive" and fine refinement to thread a 0.5 m gap.

DIAL-MPC's answer is to run several stages per control step, from wide noise to
narrow, and to make far-horizon actions noisier than near-horizon ones on the
grounds that they have been refined fewer times.

---

## The algorithm

Algorithm 1 of the paper, per control step `t`:

```
for i = N down to 1:
    Σ^i_{t:t+H} ← Eq (7)
    W_{1:N_W} ~ N(0, Σ^i)
    rollout u + W, evaluate J(u + W)
    estimate score with (3), update u with (2)      # the MPPI update
shift u forward one step                             # receding horizon
```

with the dual-loop covariance design:

```
(5)  det Σ^i_{t:t+H} ∝ exp( −(N−i)/(β₁N) · H·d_u )       trajectory level
(6)  det Σ^i_{t+h}   ∝ exp( −(H−h)/(β₂H) · d_u )         action level
(7)  Σ^i_{t+h} = exp( −(N−i)/(β₁N) − (H−h)/(β₂H) ) · I   isotropic realization
```

**Restated for this setting.** Three translations matter:

*Decision variable.* The paper's `u_{t:t+H}` is a per-timestep action sequence.
Here it is the control points of a spline policy, so `h` indexes
`sampling_spline_points` knots (12 on the corridor task), not the 200 rollout
timesteps. `H` in the code is the knot count ([planner.cc:403-408](planner.cc#L403-L408)).
The action-level schedule therefore spreads over knots; with a 1.0 s horizon and
12 knots each "action" covers ~83 ms.

*Index direction.* The paper counts `i` down from `N` to 1; the code counts
`iter` up from 0 to `N−1` and uses `−iter/(β₁N)` ([planner.cc:417](planner.cc#L417)),
which is the same schedule: full noise at the first iteration, narrowest at the
last. Likewise `(H−1−h)/(β₂H)` ([planner.cc:424](planner.cc#L424)) is Eq (6)'s
`(H−h)/(β₂H)` shifted for zero-based knots.

*Covariance vs standard deviation.* As implemented:

```
Eq 7 (paper):        Σ^i_{t+h} = exp( −(N−i)/(β₁N) − (H−h)/(β₂H) ) I
     (as coded):     σ^i_h     = σ_base · exp( −i/(β₁N) − (H−1−h)/(β₂H) )
```

so `Σ_coded = σ_base²·exp(2·(...))`. See deviation 1.

*The update.* Eq (2)/(3)'s score-function step is realized as the standard MPPI
exponential-weighted average over all rollouts
([planner.cc:225-286](planner.cc#L225-L286)), with the minimum cost subtracted
before exponentiating for stability ([planner.cc:229-235](planner.cc#L229-L235)).
This is the paper's own framing — DIAL-MPC *is* MPPI plus the annealing.

---

## Equation-to-code map

| Paper | Quantity | Code |
|---|---|---|
| Alg 1, line 2-9 | the `N`-stage annealing loop | [planner.cc:187-214](planner.cc#L187-L214) |
| Eq 5 | trajectory-level schedule `exp(−(N−i)/(β₁N))` | [planner.cc:416-418](planner.cc#L416-L418) |
| Eq 6 | action-level schedule `exp(−(H−h)/(β₂H))` | [planner.cc:423-425](planner.cc#L423-L425) |
| Eq 7 | the combined isotropic kernel | [planner.cc:427](planner.cc#L427) |
| Alg 1, line 5 | `W ~ N(0, Σ)` applied to the knots | [planner.cc:432-433](planner.cc#L432-L433) |
| Alg 1, line 6 | rollout and evaluate `J` | [planner.cc:443](planner.cc#L443) |
| Eq 2, 3 | MPPI weights `w_i ∝ exp(−(J_i − J_min)/λ)` | [planner.cc:233-237](planner.cc#L233-L237) |
| Eq 2, 3 | weighted average of the candidates | [planner.cc:263-276](planner.cc#L263-L276) |
| Alg 1, line 10 | receding-horizon shift of `u` | [planner.cc:315-384](planner.cc#L315-L384) |
| — | clamp to actuator limits after noise and after update | [planner.cc:435](planner.cc#L435), [planner.cc:279-281](planner.cc#L279-L281) |

**Not in the paper:** the two-standard-deviation mixture at
[planner.cc:393-397](planner.cc#L393-L397), which draws `σ_base` from
`noise_exploration[1]` with probability 0.2 and from `noise_exploration[0]`
otherwise. This is inherited from MJPC's `SamplingPlanner`, not from DIAL-MPC,
and it perturbs the annealing schedule it sits inside: on 20% of knots the
"annealed" σ is scaled by a different base. Harmless if `noise_exploration[1]`
is 0 (the default), which disables it.

---

## Walkthrough

One `OptimizePolicy` call on the corridor task (`nu=1`, 12 knots, 10 rollouts,
`N=4`, `β₁=β₂=0.5`, horizon 1.0 s). The annealing multipliers below are computed
from [planner.cc:416-427](planner.cc#L416-L427):

```
policy (incumbent spline), 12 knots
  |
  +--> iteration i=0  traj_anneal = exp(-0/(0.5*4)) = 1.000   <- widest
  |      |
  |      v  planner.cc:193 -- UpdateNominalPolicy: resample onto current knots
  |      v  planner.cc:199 -- 10 candidates, each knot h perturbed by
  |      |                    sigma = 0.5*ctrlrange * 1.000 * action_anneal(h)
  |      |                    action_anneal: h=0 -> exp(-11/6) = 0.16
  |      |                                   h=11 -> exp(-0/6) = 1.00
  |      |                    so knot 0 gets 3.2 N of noise, knot 11 gets 20 N
  |      v  planner.cc:207 -- sort by return
  |      v  planner.cc:213 -- MPPIUpdate: policy <- sum_i w_i * candidate_i
  |
  +--> iteration i=1  traj_anneal = exp(-1/2) = 0.607
  +--> iteration i=2  traj_anneal = exp(-2/2) = 0.368
  +--> iteration i=3  traj_anneal = exp(-3/2) = 0.223   <- narrowest
  |
  v  40 rollouts total, policy refined 4 times
published policy
```

Two things to read off this. First, the near-horizon knots are barely perturbed
at all (0.16 × the base even at the widest stage) — they are treated as already
refined by previous control steps, which is exactly the paper's argument, and it
is what makes the executed action stable from step to step. Second, the
iteration cost is four sampling rounds, not one.

---

## Measured behaviour

All numbers from this repository at commit `e22fc9f`, on the
triple-pendulum-cartpole corridor task with the obstacle-avoidance objective
`[cart, upright, velocity, control, avoidance] = [1, 0, 0.1, 0.01, 500]`,
100 trials per planner.

```bash
mjpc/tasks/triple_pendulum_cartpole/benchmark/avoidance_sweep.sh renders/avoid100 100 0.25
```

Full table and discussion: [the task README](../../tasks/triple_pendulum_cartpole/README.md#headline-results).

**Rollout budget per planning iteration** — the number that has to accompany any
comparison of this planner with the others:

| planner | rollouts / planning iteration |
|---|---|
| Predictive Sampling, Cross-Entropy, PSO, Random Sampling | 10 |
| **Annealed Sampling** | **10 × `annealing_iterations` = 40** |

To compare like with like, either set `annealing_iterations=1` (which reduces
this planner to MPPI with a horizon-shaped noise profile) or give the others
4× the trajectories. The benchmark's `--speed` flag holds *iterations* constant,
not rollouts, so the default sweep gives Annealed Sampling 4× the samples.

**Measured per-iteration cost**, 6000 iterations, `--early_exit=false`:

```bash
mjpc/tasks/triple_pendulum_cartpole/benchmark/timing_bench.sh renders/timing 10 3 1.0
```

| planner | ms/iteration | p95 | rollout share |
|---|---|---|---|
| Predictive Sampling | 1.031 | 1.62 | 99.8% |
| **Annealed Sampling** | **4.128** | **5.91** | 99.9% |

4.00× exactly, and 99.9% of it is rollouts — there is no overhead to trim, the
cost *is* `annealing_iterations` × the samples. Two consequences:

1. The 4× ratio is now measured two independent ways (the code reading and the
   clock), so any table that shows this planner ahead has to be read as "ahead
   on 4× the budget".
2. **Its p95 iteration, 5.91 ms, exceeds the 5 ms control period.** The harness
   hands out iterations regardless of what they cost, but a real control loop
   cannot: on the slowest ~10% of iterations this planner would be dropped, and
   those are the iterations where the state is hardest. Success rates measured
   here are therefore optimistic relative to a live run in a way that the 1 ms
   planners' are not. Reducing `annealing_iterations` to 2 is the direct fix and
   costs whatever the annealing was buying.

**Corridor task, 100 trials.** 49/100 solved, 51% collisions — fourth place,
behind two planners that spend a quarter of its rollouts. On this task the
annealing does not pay for itself.

This planner was the largest single casualty of correcting the benchmark's
collision test. Under the 20 mm penetration tolerance the harness used to apply
it scored 81/100 and took second; under "any overlap with a disk" it scores
49/100. A 32-point drop, the largest of any planner, means roughly a third of
its apparent successes were runs that touched an obstacle. The mechanism is
visible in the objective: the MPPI update averages the candidates rather than
taking the best, so the executed policy is a compromise between trajectories
that pass on different sides of a disk — and the average of two clearances is
not itself a clearance. Winner-take-all planners do not have that failure mode,
which is the most likely reason the correction cost them less.

**Cartpole, average cost per step**, 10 s, 3 runs, equal *trajectory* count:

```bash
mjpc/tasks/triple_pendulum_cartpole/benchmark/cartpole_costs.sh 10 3
```

| planner | run 1 | run 2 | run 3 |
|---|---|---|---|
| **Annealed Sampling** | **0.765** | **0.760** | **0.752** |
| PSO | 0.828 | 0.779 | 0.833 |
| Predictive Sampling | 0.901 | 0.956 | 0.947 |
| Random Sampling | 0.961 | 0.945 | 0.975 |
| Cross-Entropy | 8.39 | 11.41 | 15.71 |

Here it wins, and by a margin larger than the run-to-run spread. Read the two
results together: annealing helps on a smooth stabilization task where extra
refinement stages converge on a better local answer, and does not help on a task
whose difficulty is finding a qualitatively different plan through a gap. The
4× rollout cost is the same in both cases; only the payoff changes.

---

## Deviations from the paper

| # | Deviation | Severity |
|---|---|---|
| 1 | Eq (7) is applied to σ, not to Σ ([planner.cc:427](planner.cc#L427) → [planner.cc:432](planner.cc#L432)). The realized covariance is the paper's squared, i.e. both temperatures are effectively halved. | Moderate — schedule shape preserved, magnitude not. Compensate by doubling `β₁, β₂`. |
| 2 | `h` indexes spline knots, not horizon timesteps. | Deliberate adaptation to MJPC's spline policies; means `β₂` is calibrated per-knot, not per-timestep. |
| 3 | The two-σ mixture at [planner.cc:393-397](planner.cc#L393-L397) is an MJPC inheritance with no DIAL-MPC counterpart, and it overrides the annealed base σ 20% of the time when `noise_exploration[1] > 0`. | Low with defaults (feature off), confusing when enabled. |
| 4 | `improvement` compares the best return against `trajectory[0]` ([planner.cc:218-219](planner.cc#L218-L219)), which after the first annealing iteration is no longer the nominal — it is whatever candidate 0 was in the last iteration. Display only. | Cosmetic. |
| 5 | Algorithm 1 samples `N_W` noise vectors per stage from one kernel; here the per-knot σ varies within a candidate (that *is* Eq 6), and each candidate draws independently. Equivalent. | None. |

---

## Failure modes

- **Looks better than every other sampler, because it took 4× the rollouts.**
  Symptom: highest success rate and ~4× the wall time. Check
  `annealing_iterations` before concluding anything.
- **Over-annealed.** With `β₁` small, noise collapses by the second iteration and
  the last stages contribute nothing but cost. Symptom: `improvement` near zero
  for iterations 2..N; wall time unchanged.
- **Under-annealed.** With `β₁` large the schedule is flat and the planner is
  plain MPPI run N times. Symptom: no difference from
  `annealing_iterations=1` other than cost.
- **Near-horizon starvation.** `β₂` small drives `action_anneal(h=0)` toward
  zero, so the knots that produce the *executed* action are never explored and
  the planner cannot correct a bad incumbent. Symptom: the planner tracks a
  wrong plan smoothly and confidently.

---

## Parameters

| Numeric | Default | What it does | How to tell it is wrong |
|---|---|---|---|
| `annealing_iterations` (`N`) | 4 ([planner.cc:69-70](planner.cc#L69-L70)) | Annealing stages per planning iteration. Also the rollout multiplier. | If wall time is ~N× the other samplers and you did not intend to pay that, this is why. |
| `annealing_beta_trajectory` (`β₁`) | 0.5 ([planner.cc:71-72](planner.cc#L71-L72)) | Trajectory-level temperature, Eq (5). Larger = flatter schedule. | Print `exp(−(N−1)/(β₁N))`; if it is below ~0.05 the last stages are dead. |
| `annealing_beta_action` (`β₂`) | 0.5 ([planner.cc:73](planner.cc#L73)) | Action-level temperature, Eq (6). Larger = more noise on near-horizon knots. | If the planner never corrects the executed action, raise it. |
| `sampling_temperature` (`λ`) | 1.0 ([planner.cc:68](planner.cc#L68)) | MPPI softmax temperature. Small = winner-take-all (→ Predictive Sampling), large = uniform average. | Weights all ≈ 1/n means λ is too large for this cost scale. |
| `sampling_exploration` | 0.1 ([planner.cc:51-52](planner.cc#L51-L52)) | Base σ as a fraction of half the control range, before annealing. | This is the top of the schedule; everything else only shrinks it. |
| `sampling_trajectories` | 10 ([planner.cc:61](planner.cc#L61)) | Candidates per stage. | — |

---

## References

- H. Xue, C. Pan, Z. Yi, G. Qu, G. Shi, "Full-Order Sampling-Based MPC for
  Torque-Level Locomotion Control via Diffusion-Style Annealing", arXiv:2409.15610.
- G. Williams et al., "Model Predictive Path Integral Control" — the MPPI update
  that Eq (2)/(3) reduce to here.
- [`../pso/PSO.md`](../pso/PSO.md) — the other swarm/population planner added
  alongside this one, and a contrast: there the inner loop does *not* carry state.
- [`../../tasks/triple_pendulum_cartpole/README.md`](../../tasks/triple_pendulum_cartpole/README.md) — the task and benchmark these numbers come from.
