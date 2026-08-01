# Particle Swarm Optimization in MJPC — teardown

PSO as an MJPC `RankedPlanner`. Written to the [`algorithm-teardown`](../../../.claude/skills/algorithm-teardown/SKILL.md)
format: every equation is anchored to a line verified against the current file,
and every claim has a number produced by running the code.

- Implementation: [`planner.cc`](planner.cc), [`planner.h`](planner.h)
- Registered as planner index 7 ([`include.cc`](../include.cc))
- Benchmarked by [`corridor_benchmark`](../../tasks/triple_pendulum_cartpole/benchmark/corridor_benchmark.cc)

---

## Verdict

**As currently configured this planner is Predictive Sampling wearing a swarm
costume.** The swarm update runs every iteration and its result is thrown away
before anything reads it. The measured behaviour matches that reading exactly.

1. **The swarm update is inert.** `OptimizePolicyCandidates` calls
   `InitializeParticles` unconditionally at the top of every iteration
   ([planner.cc:251](planner.cc#L251)), which rewrites every particle's position
   from the resampled nominal plus fresh Gaussian noise
   ([planner.cc:201-220](planner.cc#L201-L220)) and re-randomizes every particle's
   velocity ([planner.cc:224-231](planner.cc#L224-L231)). `UpdateVelocities` and
   `UpdatePositions` then run at the end of the same iteration
   ([planner.cc:292-293](planner.cc#L292-L293)) and write into
   `candidate_policy` ([planner.cc:422](planner.cc#L422)) — the exact array the
   next iteration overwrites before reading. Positions and velocities therefore
   never persist across iterations, which is the only thing that makes a swarm a
   swarm.

2. **What remains is one Gaussian sampling round.** Perturb the nominal by
   `N(0, 0.1·ctrlrange)` per spline point, roll out 10 particles, publish the
   best. Particle 0 is left un-perturbed ([planner.cc:209](planner.cc#L209)), so
   the incumbent is always in the sample set. That is Predictive Sampling with a
   hard-coded noise scale — and the noise scale is the one thing that differs:
   `0.1·ctrlrange = 4.0 N` here versus `sampling_exploration · 0.5·ctrlrange
   = 8.0 N` for Predictive Sampling on the corridor task.

3. **The measurement confirms it, including the noise-scale difference.**
   100 trials each on the corridor task, obstacle avoidance only, identical
   starts:

   | planner | noise σ | reached goal, no contact |
   |---|---|---|
   | Predictive Sampling, `sampling_exploration=0.4` | 8.0 N | 81/100 ±3.9 |
   | **PSO**, `publish_evaluated=1` | **4.0 N** | **51/100 ±5.0** |
   | Predictive Sampling, `sampling_exploration=0.2` | 4.0 N | 45/100 ±5.0 |

   PSO and Predictive Sampling-at-matched-noise agree to within one standard
   error (51 vs 45, difference 6 against a combined s.e. of about 7). The
   30-point gap to stock Predictive Sampling is therefore *entirely* the
   hard-coded exploration scale, not the swarm: PSO under-explores this task by
   2× and pays for it. Commands in
   [Measured behaviour](#measured-behaviour).

4. **The one thing the swarm update still reaches is the published policy, and
   that was a bug.** With `pso_publish_evaluated=0` the planner publishes the
   *post-update* parameters — parameters whose cost was never evaluated. It
   scores **18/100** against the fixed planner's 51/100, so the fix is worth 33
   points. That path is retained deliberately so the cost of the bug can be
   measured; it is not the default. See
   [Deviations](#deviations-from-the-standard-algorithm).

**Two fixes, in order of value.** Expose the initialization noise scale as a
numeric (it is this planner's only real tuning knob and it is currently a
literal), then stop calling `InitializeParticles` after the first iteration so
the swarm actually swarms. Until both land, prefer Predictive Sampling: same
algorithm, tunable, and 30 points better here.

---

## What problem it solves

MJPC's stock sampling planner ([`sampling/planner.cc`](../sampling/planner.cc))
draws candidates i.i.d. around the incumbent every iteration. It has no memory
of *where* good candidates were found, only of the single best one it kept. PSO
is the cheapest way to add that memory: each particle remembers its own best
position, the swarm remembers the global best, and candidates are drawn by
drifting toward those rather than by re-scattering.

That is the promise. This implementation does not deliver it, for the reason in
the verdict — but the bookkeeping for it (`personal_best`, `global_best_index`)
is present and correct, so the fix is small.

---

## The algorithm

Standard PSO (Kennedy & Eberhart, *Particle Swarm Optimization*, ICNN 1995),
in the inertia-weight form of Shi & Eberhart (1998). For particle `i`, decision
vector `x_i`, velocity `v_i`:

```
(1)  v_i ← w·v_i + c₁·r₁·(p_i − x_i) + c₂·r₂·(g − x_i),   r₁,r₂ ~ U(0,1)
(2)  x_i ← x_i + v_i
(3)  p_i ← x_i   if J(x_i) < J(p_i)          personal best
(4)  g   ← p_j   where j = argmin_i J(p_i)   global best
```

Restated for this setting. The decision vector is not a control sequence but the
**spline control points of a policy**: `nu` values at each of
`sampling_spline_points` knots, so `x_i ∈ R^(nu × num_spline_points)`. `J` is the
MJPC total return over the planning horizon, from a full nonlinear rollout — not
an analytic objective, so there is no gradient and every evaluation costs one
simulated trajectory. Velocity is clamped per-dimension to
`velocity_scale · ctrlrange` ([planner.cc:405](planner.cc#L405)) and position to
the actuator limits ([planner.cc:426](planner.cc#L426)), neither of which is in
the original algorithm; both are necessary here because `x` must be a
realizable control.

The receding horizon adds a step the paper has no concept of: between MPC
iterations the whole problem shifts forward in time, so the incumbent policy is
re-sampled onto the new knot times ([planner.cc:173](planner.cc#L173)) before
anything else happens.

---

## Equation-to-code map

| Source | Quantity | Code |
|---|---|---|
| Eq 1 | `v ← w·v + c₁r₁(p−x) + c₂r₂(g−x)` | [planner.cc:397-399](planner.cc#L397-L399) |
| Eq 1 | cognitive term `c₁r₁(p−x)` | [planner.cc:395](planner.cc#L395) |
| Eq 1 | social term `c₂r₂(g−x)` | [planner.cc:396](planner.cc#L396) |
| Eq 2 | `x ← x + v` | [planner.cc:422](planner.cc#L422) |
| Eq 3 | personal best update | [planner.cc:432-440](planner.cc#L432-L440) |
| Eq 4 | global best update | [planner.cc:443-449](planner.cc#L443-L449) |
| — | velocity clamp to `velocity_scale·ctrlrange` | [planner.cc:401-405](planner.cc#L401-L405) |
| — | position clamp to actuator limits | [planner.cc:426](planner.cc#L426) |
| — | `J(x_i)`, one rollout per particle | [planner.cc:342](planner.cc#L342) |
| — | swarm initialization `x ~ nominal + N(0, 0.1·range)` | [planner.cc:213](planner.cc#L213) |
| — | velocity initialization `v ~ N(0, 0.01·range)` | [planner.cc:228-229](planner.cc#L228-L229) |
| — | receding-horizon resample of the incumbent | [planner.cc:173-189](planner.cc#L173-L189) |
| — | publish the winner | [planner.cc:586-601](planner.cc#L586-L601) |

**Equations with no persistent effect.** Eq 1 and Eq 2 are evaluated but their
output is discarded, for the reason in the verdict. Eq 3 and Eq 4 do persist —
`personal_best` and `global_best_cost` survive across iterations — but the only
consumers are Eq 1 (discarded) and the `improvement` figure shown in the GUI
([planner.cc:310](planner.cc#L310)). So no part of the swarm state reaches the
executed policy.

---

## Walkthrough

One `OptimizePolicyCandidates` call on the corridor task
(`nu=1`, `sampling_spline_points=12`, `pso_num_particles=10`, horizon 1.0 s):

```
policy (incumbent), 12 knots x 1 actuator
  |
  v  planner.cc:247 -- ResamplePolicy: re-evaluate the incumbent spline at the
  |                    new knot times, since t advanced since the last call
parameters_scratch, 12 values
  |
  v  planner.cc:251 -- InitializeParticles
  |                    x_0 = nominal (unperturbed)
  |                    x_1..x_9 = nominal + N(0, 0.1 x 40 N) per knot
  |                    v_0..v_9 = N(0, 0.01 x 40 N) per knot   <-- discards last
  |                                                                iteration's v
x_i, 10 policies
  |
  v  planner.cc:255 -- Rollouts: 10 full nonlinear rollouts, 200 steps each,
  |                    on the thread pool. This is >99% of the iteration cost.
J(x_i), 10 returns
  |
  v  planner.cc:259-260 -- Eq 3, Eq 4: personal and global bests
  |
  v  planner.cc:268 -- partial_sort by total_return -> trajectory_order
  |
  v  planner.cc:282-288 -- snapshot the ranked positions BEFORE mutating them
evaluated_policy[best]
  |
  v  planner.cc:292-293 -- Eq 1, Eq 2 on all 10 particles
  |                        (writes candidate_policy; nothing reads it again)
  |
  v  planner.cc:306 -> 586 -- publish evaluated_policy[best] as the new policy
```

The two arrows that matter are the fifth and the last. The snapshot at
`planner.cc:282` is what makes the published policy the one that earned its
score; the mutation at `planner.cc:292-293` is the swarm step whose result is
overwritten by the next call's `InitializeParticles`.

---

## Measured behaviour

All numbers below are from this repository at commit `e22fc9f`, on the
triple-pendulum-cartpole corridor task with the obstacle-avoidance objective
`[cart, upright, velocity, control, avoidance] = [1, 0, 0.1, 0.01, 500]`.

Reproduce the whole table with:

```bash
mjpc/tasks/triple_pendulum_cartpole/benchmark/avoidance_sweep.sh renders/avoid100 100 0.25
```

See [the task README](../../tasks/triple_pendulum_cartpole/README.md#headline-results)
for the full cross-planner table.

**The controlled experiment behind the verdict.** If PSO is Predictive Sampling
with half the exploration noise, then handing Predictive Sampling half its noise
should reproduce PSO's score. It does:

```bash
# PSO, and Predictive Sampling at its own noise and at PSO's
B=./build/bin/corridor_benchmark
COMMON="--stage=corridor --weights=1,0,0.1,0.01,500 --speed=0.25 --repeats=100 --seed=1 --per_run=false"
$B --planner=7 $COMMON                     # PSO             -> 51/100
$B --planner=0 $COMMON --exploration=0.4   # PS, sigma 8.0 N -> 81/100
$B --planner=0 $COMMON --exploration=0.2   # PS, sigma 4.0 N -> 45/100
```

| run | solved | 1 s.e. |
|---|---|---|
| PSO (σ = 0.1·ctrlrange = 4.0 N) | 51/100 | ±5.0 |
| Predictive Sampling, `--exploration=0.2` (σ = 4.0 N) | 45/100 | ±5.0 |
| Predictive Sampling, `--exploration=0.4` (σ = 8.0 N) | 81/100 | ±3.9 |

Six points apart at matched noise against a combined standard error of about
seven; thirty apart at mismatched noise. The swarm contributes nothing
measurable; the noise scale contributes everything.

(The stock Predictive Sampling row reads 81 here and 76 in the
[headline sweep](../../tasks/triple_pendulum_cartpole/README.md#headline-results).
Same binary, same seeds, same objective — the planners' own sampling is not
seeded, so ±4 points of run-to-run spread at n=100 is the noise floor of this
benchmark. Differences smaller than that are not findings.)

**Cost of the publish bug.** `--pso_publish_evaluated=0` restores the original
behaviour, in which the executed policy is the post-swarm-update iterate:

| `pso_publish_evaluated` | solved | collided |
|---|---|---|
| 1 (fixed, default) | 51/100 | 49% |
| 0 (original) | 18/100 | 77% |

On Cartpole the same bug shows up as an 8-25× cost multiplier (table below),
which is how it was originally isolated: the executed policy differs from the
ranked one by up to `pso_velocity_scale · ctrlrange` of never-evaluated
perturbation at every control step.

**The same knob, opposite sign, on a different task.** Cartpole, 10 s, 3 runs
each, equal budget (10 rollouts for everyone), average cost per step:

```bash
mjpc/tasks/triple_pendulum_cartpole/benchmark/cartpole_costs.sh 10 3
```

| planner | run 1 | run 2 | run 3 |
|---|---|---|---|
| Annealed Sampling *(4× the rollouts)* | 0.765 | 0.760 | 0.752 |
| **PSO** (σ = 0.1·ctrlrange) | **0.828** | **0.779** | **0.833** |
| Predictive Sampling (σ = 0.25·ctrlrange) | 0.901 | 0.956 | 0.947 |
| Random Sampling | 0.961 | 0.945 | 0.975 |
| Cross-Entropy | 8.39 | 11.41 | 15.71 |
| **PSO, stock** (`publish_evaluated=0`) | **20.91** | **6.53** | **15.48** |

Here PSO's narrow noise *beats* Predictive Sampling, where on the corridor it
lost by 25 points. Cartpole is a stabilization task that rewards small
perturbations around a good incumbent; the corridor is a search task that
rewards finding a qualitatively different plan. One hard-coded noise scale
cannot be right for both — which is the argument for making it a numeric, not
for changing the literal.

**Cost of one iteration.** Measured directly over 6000 iterations with
`--early_exit=false`, so both planners run the same iterations from the same
starts:

```bash
mjpc/tasks/triple_pendulum_cartpole/benchmark/timing_bench.sh renders/timing 10 3 1.0
```

| planner | ms/iteration | p95 | where it goes |
|---|---|---|---|
| PSO | 1.018 | 1.64 | rollouts 99.8%, swarm update 0.2% |
| Predictive Sampling | 1.031 | 1.62 | rollouts 99.8%, noise 0.2% |

**Eq 1 and Eq 2 cost 0.2% of an iteration.** That is the verdict measured on the
clock rather than read off the control flow: the swarm update is too cheap to be
doing anything, because a swarm that actually searched would have to move
particles somewhere that changed what got rolled out. It does not, so it is
2 µs of arithmetic on 10 × 12 scalars, thrown away.

Both planners land within 1.3% of each other, and within 8% of the contact-free
floor (0.926 ms on the balance stage) — per-iteration cost is not where any of
the differences in this document live.

---

## Deviations from the standard algorithm

| # | Deviation | Severity |
|---|---|---|
| 0 | The initialization noise scale is the literal `0.1` at [planner.cc:213](planner.cc#L213) rather than `sampling_exploration`, so PSO explores at half the amplitude of every other sampler on the same task. Measured cost: 30 points of success rate. | **Critical in practice** — the largest measured effect in this file. |
| 1 | Particles are re-initialized every iteration ([planner.cc:251](planner.cc#L251)), so Eq 1-2 have no persistent effect and the algorithm degenerates to one-shot Gaussian sampling. | **Critical in principle** — it is not PSO. Currently costs nothing measurable, because #0 dominates. |
| 2 | The comment above that call says "initialize particles if this is first iteration or they need resampling" ([planner.cc:250](planner.cc#L250)); there is no such condition in the code. | Documentation bug that hides #1. |
| 3 | The published policy was the post-update iterate, whose cost was never evaluated. Fixed by the snapshot at [planner.cc:282-288](planner.cc#L282-L288); the old behaviour is retained behind `pso_publish_evaluated=0` for measurement. | Fixed; was **critical**. |
| 4 | `global_best_index` indexes into `candidate_policy` ([planner.cc:384](planner.cc#L384)) rather than into `personal_best`, so the "global best" used by the social term is that particle's *current* position, not its best-ever one. Given #1 the two are the same thing per iteration, so this is currently unobservable — but it becomes a real bug the moment #1 is fixed. | Latent. |
| 5 | Default `pso_num_particles` is 20 while every other sampling planner defaults to 10 rollouts. The corridor task sets both explicitly ([task.xml](../../tasks/triple_pendulum_cartpole/task.xml)) so the comparison is budget-fair; a task that does not will silently give PSO twice the rollouts. | Benchmarking hazard. |

Deviation 1 is the one to fix first, and fixing it makes 4 live. A correct fix
is to guard `InitializeParticles` on a "swarm not yet initialized" flag reset in
`Reset()`, and to change the `gbest_node` lookup to read `personal_best[global_best_index]`.

---

## Failure modes

- **Indistinguishable from Predictive Sampling.** The expected steady state
  today. Symptom: identical success rates and near-identical traces. Cause:
  deviation 1.
- **Executed policy diverges from the ranked one** (`pso_publish_evaluated=0`).
  Symptom: the planner's reported best cost improves while the observed
  behaviour degrades, and the degradation scales with `pso_velocity_scale`.
  Cause: deviation 3.
- **Swarm collapse**, if deviation 1 is fixed without care. All particles
  converge to `g` and the search stops; symptom is `improvement` going to zero
  early and staying there while the task is unsolved. Standard PSO remedies
  apply (velocity floor, re-scatter on stagnation).
- **Budget-unfair comparisons.** Symptom: PSO appears to beat every sampler on a
  task whose XML does not set `pso_num_particles`. Cause: deviation 5.

---

## Parameters

| Numeric | Default | What it does | How to tell it is wrong |
|---|---|---|---|
| `pso_num_particles` | 20 ([planner.cc:58](planner.cc#L58)) | Rollouts per iteration. The dominant cost. | Set it equal to `sampling_trajectories` or comparisons are meaningless. |
| `pso_inertia` (`w`) | 0.7 ([planner.cc:59](planner.cc#L59)) | Momentum in Eq 1. Standard range 0.4-0.9. | No observable effect today (deviation 1). |
| `pso_cognitive` (`c₁`) | 1.5 ([planner.cc:60](planner.cc#L60)) | Pull toward the particle's own best. | As above. |
| `pso_social` (`c₂`) | 1.5 ([planner.cc:61](planner.cc#L61)) | Pull toward the global best. `c₁+c₂ > 4` diverges in the classical analysis. | As above. |
| `pso_velocity_scale` | 0.1 ([planner.cc:62](planner.cc#L62)) | Velocity clamp as a fraction of `ctrlrange`. | With `pso_publish_evaluated=0` this is the size of the unvalidated perturbation injected into the executed policy; setting it to 0 makes that bug disappear, which is how the bug was originally isolated. |
| `pso_publish_evaluated` | 1 ([planner.cc:63](planner.cc#L63)) | 1: publish the parameters that were rolled out. 0: publish the post-update iterate (original behaviour). | Leave at 1 outside of measurement. |
| `sampling_spline_points` | 12 (task) | Dimension of `x` per actuator. | Higher costs nothing in rollouts but makes the search space larger. |

The swarm initialization scales — `0.1·range` for position
([planner.cc:213](planner.cc#L213)) and `0.01·range` for velocity
([planner.cc:229](planner.cc#L229)) — are hard-coded, not exposed as numerics.
Given deviation 1, the position scale is *the* exploration parameter of this
planner, and it is measurably the wrong value on the corridor task: it is half
what every other sampler uses there, and closing that gap is worth 30 points of
success rate ([Measured behaviour](#measured-behaviour)). Today changing it
requires a rebuild. Making it read `sampling_exploration` like the others, or
adding a `pso_exploration` numeric, is the highest-value change in this file.

---

## References

- J. Kennedy and R. Eberhart, "Particle swarm optimization", *Proc. IEEE ICNN*, 1995.
- Y. Shi and R. Eberhart, "A modified particle swarm optimizer", *Proc. IEEE CEC*, 1998. (inertia weight `w`)
- [`../sampling/planner.cc`](../sampling/planner.cc) — Predictive Sampling, which this currently reduces to.
- [`../annealed_sampling/ANNEALED_SAMPLING.md`](../annealed_sampling/ANNEALED_SAMPLING.md) — the other sampling planner added alongside this one.
- [`../../tasks/triple_pendulum_cartpole/README.md`](../../tasks/triple_pendulum_cartpole/README.md) — the task and benchmark these numbers come from.
