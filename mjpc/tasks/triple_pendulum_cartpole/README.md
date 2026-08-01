# Triple pendulum cartpole — task, benchmark, and planner comparison

A 3-link pendulum on a cart that must thread an obstacle corridor. One actuator,
four degrees of freedom, chaotic dynamics, and a gap narrower than the pendulum
is long. It exists to make planners fail in ways that are legible.

- Task: [`task.xml`](task.xml) (one bottleneck, the paper's), [`slalom.xml`](slalom.xml) (three)
- Residual: [`triple_pendulum_cartpole.cc`](triple_pendulum_cartpole.cc)
- Benchmark: [`benchmark/corridor_benchmark.cc`](benchmark/corridor_benchmark.cc)
- Sweeps: [`benchmark/avoidance_sweep.sh`](benchmark/avoidance_sweep.sh), [`benchmark/sweep.sh`](benchmark/sweep.sh)
- Renderer: [`benchmark/filmstrip.py`](benchmark/filmstrip.py)
- Write-up with the rollout videos:
  [*Seven planners, one gap, then three*](https://claude.ai/code/artifact/93fa927b-f82a-4481-8066-70e87be78e16)
  — the failure distributions and the four outcome-tier rollouts, rendered.
  Tables here are the source of truth; that page is where you watch them.

Source: T. M. Caldwell and N. Correll, *Fast Sample-Based Planning for Dynamic
Systems by Zero-Control Linearization-Based Steering*, ISRR 2015, §5.

---

## Headline results

100 trials per planner, obstacle-avoidance objective
`[cart, upright, velocity, control, avoidance] = [1, 0, 0.1, 0.01, 500]`,
`--speed=0.25`, perturbed starts shared across planners.

`--speed=0.25` means control is applied every 5 ms of *simulated* time (200 Hz,
as always) while the wall clock runs four times slower — so this is **50 Hz
control with four optimization iterations per decision**. See
[two clocks](#two-clocks-and-what---speed-actually-buys).

Success means the cart reached the goal box and **never overlapped a disk**; see
[why the collision test has no tolerance](#why-the-collision-test-has-no-tolerance)
— an earlier version of this benchmark allowed 20 mm of penetration and every
number below was 8–32 points higher because of it.

The run-to-run spread at n=100 is about ±4 points (the planners' own sampling is
not seeded), so differences smaller than that are not findings.

Both collision criteria are reported. **Clean** is the constraint as stated —
never overlap a disk. **≤20 mm** is the laxer test this benchmark used
originally, which counts a run as successful if no head penetrated a disk by
more than 20 mm. The difference between the two columns is how much of a
planner's score was grazing.

| planner | clean | 1 s.e. | ≤20 mm | grazing share | median t_solve | rollouts/iter |
|---|---|---|---|---|---|---|
| Random Sampling *(control)* | **83/100** | ±3.8 | 93/100 | 11% | 2.39 s | 10 |
| Predictive Sampling | 76/100 | ±4.3 | 93/100 | 18% | 2.33 s | 10 |
| PSO (publish fixed) | 51/100 | ±5.0 | 66/100 | 23% | 2.41 s | 10 |
| Annealed Sampling (DIAL-MPC) | 49/100 | ±5.0 | 81/100 | **40%** | 2.53 s | **40** |
| iLQG | 35/100 | ±4.8 | 46/100 | 24% | 2.28 s | derivative |
| PSO (stock, publishes unevaluated) | 18/100 | ±3.8 | 26/100 | 31% | 2.29 s | 10 |
| Cross-Entropy | **2/100** | ±1.4 | 21/100 | **90%** | 3.84 s | 10 |

"Grazing share" is the fraction of the lax score that does not survive the clean
test — `1 − clean/lax`. It is a property worth reading on its own: a planner
with a high grazing share is one whose apparent competence is contact it was not
charged for.

Per-iteration cost for each is in [What an iteration costs](#what-an-iteration-costs);
the short version is that the four 10-rollout samplers are within 3% of each
other, and the two that are not — Annealed Sampling and iLQG — are 4× and miss
the per-iteration budget on the tail.

Three readings, in order of how much they should change what you do:

**1. The memoryless control wins.** Random Sampling throws away the incumbent
every iteration and samples around zero control
([RANDOM_SAMPLING.md](../../planners/random_sampling/RANDOM_SAMPLING.md)). It
does not merely keep up with Predictive Sampling, it is nominally ahead of it —
though by 7 points against a combined standard error of about 6, so treat the
two as indistinguishable rather than ranked. Either way the reading is the same:
at 800 planning iterations per simulated second, this task is solved by
re-deciding from scratch, not by refining a plan. Any claim that a planner's
*optimizer* is what solves the corridor has to get past this row first.

**2. Cross-Entropy is not merely worse, it is broken here** — 2/100 with 98%
collisions, against 76/100 for the same sampling machinery with a different
update rule. CEM refits a Gaussian to the elite fraction and samples from it. On
a chaotic system the elite set at one iteration is not predictive of the next,
so the refitted covariance collapses onto a region that the dynamics have already
left; the planner then samples confidently inside a stale mode. The behaviour to
look for in a filmstrip is a smooth, committed, wrong trajectory rather than a
flailing one. This confirms the informal observation that CEM is the one method
that will not work on this task — and under the corrected collision test it is
not "weak", it essentially never produces a clean run.

**3. Annealed Sampling was the biggest beneficiary of the old, lax collision
test.** Under a 20 mm penetration tolerance it scored 81/100 and took second
place; under "any overlap" it scores 49/100 and takes fourth. A 32-point drop —
the largest of any planner — means a third of its apparent successes were runs
that touched a disk. It still costs 4× the rollouts
([ANNEALED_SAMPLING.md](../../planners/annealed_sampling/ANNEALED_SAMPLING.md)):
`annealing_iterations = 4` sample-rollout-update stages per planning iteration,
40 rollouts against everyone else's 10. `--speed` equalizes *iterations*, not
rollouts, so there is no setting of it that makes this row a fair fight.

The lax-criterion logs are in `renders/avoid100_s025_laxtol/`; reproduce them
with `--penetration_tolerance=0.02 --contact_fraction_tolerance=0.02`.

### Three bottlenecks

Same objective and protocol on [`slalom.xml`](slalom.xml): the paper's gap at
x=3 unchanged, two more at x=6 and x=9, goal at x=11. Because the first gap is
the corridor task's exactly, the drop between the two tables is attributable to
what comes *after* it.

| planner | clean | 1 s.e. | ≤20 mm | grazing share | gaps before first contact | corridor (clean) |
|---|---|---|---|---|---|---|
| Predictive Sampling | 4/100 | ±2.0 | 22/100 | 82% | **0.94** | 76/100 |
| Annealed Sampling | 2/100 | ±1.4 | 34/100 | **94%** | 0.49 | 49/100 |
| Random Sampling *(control)* | 1/100 | ±1.0 | 22/100 | 95% | 0.73 | 83/100 |
| PSO stock | 1/100 | ±1.0 | 2/100 | 50% | 0.26 | 18/100 |
| PSO | 0/100 | ±0.0 | 27/100 | 100% | 0.65 | 51/100 |
| iLQG | 0/100 | ±0.0 | 0/100 | — | 0.09 | 35/100 |
| Cross-Entropy | 0/100 | ±0.0 | 1/100 | 100% | 0.05 | 2/100 |

**Nothing solves it.** The best planner clears three bottlenecks without
touching a disk in 4 runs out of 100, and five of the seven never manage it at
all. At these rates the differences between the top rows are one to two standard
errors — this table ranks nothing, it reports a wall. Three chained 0.5 m gaps,
at the speed an 11 m goal induces, is past every planner in this tree.

**Read the partial-credit column instead.** "Gaps before first contact" is how
far the cart got before it first violated the constraint, and it does separate
the planners: 0.94 for Predictive Sampling down to 0.05 for Cross-Entropy. That
ordering is essentially the corridor ordering — the planners that thread one gap
best also get furthest into three. There is no re-ranking, no crossover, and no
planner that "handles chaining better".

> **Correction.** An earlier version of this document read the ≤20 mm column as
> the result, and reported the slalom as a ranking *inversion* — Annealed
> Sampling first at 34/100, Predictive Sampling third at 22/100 — with an
> argument about chaining rewarding different behaviour. The clean column
> retracts it. Grazing shares of 82–100% here mean that on the slalom the lax
> test was measuring almost nothing but contact: every planner's apparent
> competence at three bottlenecks was the tolerance, not the planner. The two
> columns are both kept because the gap between them is the finding.
> Logs: `renders/slalom100_s025_laxtol/` and `renders/slalom100_s025/`.

**Why the first gap gets so much harder.** The same bottleneck at x=3 is cleared
76% of the time when it is the only one and contributes to a 0.94 average here.
Nothing about the geometry changed; the goal moved from 6 m to 11 m, so the Cart
residual is ~3× larger at the start and the planner commits to more speed on
approach. iLQG is the extreme case — 35/100 on the corridor, 0.09 gaps here —
arriving at a gap it can otherwise clear with far too much velocity to lay the
pendulum out.

**This makes the slalom a headroom task, not a benchmark.** As configured it
does not discriminate between planners, because they all fail. To make it
discriminate, weaken it: widen the gaps, shorten the field, or lower the Cart
weight so the approach is slower. That is a task-design question, and the number
to watch while tuning is the partial-credit column, not the success rate.

---

## What an iteration costs

The success tables hold planning *iterations* constant, which is what makes them
a comparison of algorithms rather than of throughput. This is the other half:
what an iteration costs, and therefore whether a planner could deliver those
iterations outside the harness.

```bash
mjpc/tasks/triple_pendulum_cartpole/benchmark/timing_bench.sh renders/timing 10 3 1.0
STAGE=balance mjpc/tasks/triple_pendulum_cartpole/benchmark/timing_bench.sh renders/timing_balance 10 3 1.0
```

6000 planning iterations each, `--early_exit=false` so every planner runs the
same iterations from the same starts, 15 planner threads on 20 cores.

| planner | ms/iter | p95 | worst | % of 5 ms budget | where the iteration goes |
|---|---|---|---|---|---|
| Random Sampling | 1.000 | 1.59 | 6.2 | 20% | rollouts 99.7% |
| PSO | 1.018 | 1.64 | 6.4 | 20% | rollouts 99.8% |
| Predictive Sampling | 1.031 | 1.62 | 4.2 | 21% | rollouts 99.8% |
| Cross-Entropy | 1.326 | 2.27 | 5.4 | 27% | rollouts 99.6% |
| Annealed Sampling | 4.128 | 5.91 | 11.5 | 83% | rollouts 99.9% |
| iLQG | 4.651 | 7.38 | 15.5 | 93% | **derivatives 45%, nominal 27%, rollouts 25%, backward 3%** |

### Two clocks, and what `--speed` actually buys

Control is applied once per timestep of **simulated** time — 200 Hz — at every
speed. What `--speed` changes is the **wall** clock:

| `--speed` | wall-clock control rate | iterations per control step | wall time per step | wall time per iteration |
|---|---|---|---|---|
| 1.0 | 200 Hz | 1 | 5 ms | 5 ms |
| 0.25 *(the tables above)* | **50 Hz** | 4 | 20 ms | 5 ms |
| 0.1 | 20 Hz | 10 | 50 ms | 5 ms |

Slowing down buys iterations per decision and pays for them in control rate. The
last column is constant, and that is the useful invariant: **the planner's
budget per iteration is one timestep, 5 ms, at any speed.** It is *not* the
control period — at `--speed=0.25` the control period in wall time is 20 ms and
the loop is running at 50 Hz.

So the headline tables describe a system controlled at 50 Hz with four
optimization iterations per decision. Every planner above fits the 5 ms
per-iteration budget on the mean. On the tail, iLQG (p95 7.4 ms) and Annealed
Sampling (p95 5.9 ms) do not: a real loop would drop iterations for them
precisely when the state is hardest, so their success rates are optimistic in a
way the 1 ms planners' are not.

### Why not just coarsen the timestep instead?

`--speed=0.25` and `<option timestep="0.020">` both produce a 50 Hz wall-clock
control loop, so it looks as though the second is the simpler way to get there.
It is not the same experiment, and it loses on every axis at once. There are
**three** timesteps in this stack and only the first two are candidates:

| | set by | governs |
|---|---|---|
| `<option timestep>` = 0.005 | [`slalom.xml`](slalom.xml) | physics integration **and** control rate — control is applied once per `mj_step` |
| `agent_timestep` = 0.005 | `<custom>` → [agent.cc:104](../../agent.cc#L104), [agent.cc:288](../../agent.cc#L288) | the **planner's rollout** resolution only, on a separate model copy |
| `--speed` | [corridor_benchmark.cc:713](benchmark/corridor_benchmark.cc#L713) | planner iterations per control step |

1. **It does not buy planner compute.** At `dt=0.005, speed=0.25` a control
   decision arrives every 20 ms of wall time with **four** iterations behind it.
   At `dt=0.020, speed=1.0` a decision arrives every 20 ms of wall time with
   **one**. Same wall clock, a quarter of the compute, a quarter of the control
   rate.
2. **It does not buy wall time either.** Physics is already free: a slalom run
   reports `planning 12.3 s of 12.3 s wall (100%), physics 0.0 s`. The sweeps
   are planner-bound, so making `mj_step` cheaper changes nothing.
3. **It corrupts the dynamics measurably.** With `u=0` and no friction the
   horizontal momentum starts at zero and is conserved, so the centre of mass is
   an *exact* invariant — drift is integrator error, not chaos:

   | integrator | dt | CoM_x drift over 12 s |
   |---|---|---|
   | implicitfast | **0.005** *(the task)* | 0.088 m |
   | implicitfast | 0.010 | 0.326 m |
   | implicitfast | 0.020 | **1.476 m** |
   | rk4 | 0.005 | **0.0002 m** |

   1.48 m of cart travel manufactured from nothing, on a course whose half-gap
   is 0.25 m and whose bottlenecks are 3 m apart.
4. **It silently loosens the collision test** — the same class of error as the
   20 mm penetration tolerance that
   [this benchmark already retracted](#why-the-collision-test-has-no-tolerance).
   `min_clearance` is sampled once per `mj_step`. Across 256 recorded slalom
   rollouts, **46% of the episodes that first disqualify a run last ≤ 4 steps
   (20 ms)**, so a 0.020 timestep steps over a large share of them and reports a
   *higher* success rate for a worse simulation.

The lever that actually trades fidelity for planner budget is the third row of
that table, `agent_timestep`: it sets the planner's rollout resolution alone
([agent.cc:288](../../agent.cc#L288) writes it to the planning model, not the
simulation), so coarsening it to 0.020 quarters the rollout length without
touching the physics being scored. That is untested here and is the obvious next
experiment; coarsening `<option timestep>` is not.

The same table shows `rk4` cutting the momentum error 440× at `dt=0.005`
(0.088 → 0.0002 m). **Do not read that as a recommendation.** It was measured on
the contact-free swing, and this task's obstacles are real collision geoms: RK4
is an explicit integrator that re-evaluates the dynamics at sub-steps, which is
exactly where MuJoCo's contact impulses are discontinuous, so it degrades on
contact rather than improving. The integrator stays `implicitfast`; the row is
kept only to show how much of the drift at `dt=0.005` is the integrator rather
than the timestep, which is the quantity that makes the coarser-timestep numbers
above interpretable.

```bash
# both measurements above
python3 mjpc/tasks/triple_pendulum_cartpole/benchmark/timestep_study.py
```

**The four 10-rollout samplers are within 3% of each other**, and almost all of
that is contact. On the contact-free `balance` stage — obstacles removed, same
dynamics — they converge further:

| planner | corridor | contact-free | difference |
|---|---|---|---|
| Random Sampling | 1.000 | 0.924 | +8% |
| PSO | 1.018 | 0.926 | +10% |
| Predictive Sampling | 1.031 | 0.939 | +10% |
| **Cross-Entropy** | **1.326** | **0.986** | **+34%** |
| Annealed Sampling | 4.128 | 3.779 | +9% |
| iLQG | 4.651 | 4.343 | +7% |

Cross-Entropy's apparent 30% overhead is not the algorithm — it is the 79%
collision rate. MuJoCo charges more for contact-rich rollouts, so a planner that
crashes into disks pays for it twice, once in the score and once on the clock.
Contact-free, all four samplers sit between 0.92 and 0.99 ms. **Per-iteration
cost is essentially identical across the sampling family; only the outcome
differs.**

The two exceptions are structural, not incidental:

- **Annealed Sampling, 4.1 ms = 4.0× Predictive Sampling.** That is its
  `annealing_iterations = 4` rollout multiplier, measured independently of the
  code reading. 99.9% of the iteration is rollouts, so there is no overhead to
  trim: the cost *is* the extra samples.
- **iLQG, 4.7 ms, and only a quarter of it is rollouts.** 45% goes to finite-
  difference model and cost derivatives and 27% to the nominal rollout; the
  Riccati backward pass — the part people expect to dominate — is 2.7%. On a
  4-DOF system the derivatives are already the bill, which is why iLQG's cost
  will scale with model size far worse than any sampler's here.

## Watch the rollouts

Aggregate metrics on this task are actively misleading, so the primary artifact
is the video. `benchmark/slalom_gallery.py` picks one rollout per outcome tier —
by outcome, not by run index — and frames each at its bottleneck crossings
rather than at uniform intervals, because the crossings are the events that
decide the run.

```bash
python3 benchmark/slalom_gallery.py \
    --dumps renders/slalom_gallery/dumps renders/outcome_dist/predictive_sampling \
    --out renders/slalom_gallery/render
python3 benchmark/make_report.py --out renders/report.html   # charts + videos, one file
```

| tier | reached | closest approach | contact steps |
|---|---|---|---|
| stopped at the first bottleneck | 2.21 m | −0.039 m | 1 |
| cleared one | 6.09 m | −0.020 m | 9 |
| cleared two | 9.10 m | −0.027 m | 10 |
| **cleared all three, clean** | goal at 2.23 s | **+0.023 m** | **0** |

The clean run is worth studying, because its posture is not the one the
single-corridor solution uses. It does **not** lay the pendulum out flat and
hold it — it folds the links *under* the cart and keeps them folded through all
three gaps, with clearances of +70, +30 and +53 mm. Laying flat sweeps a 1 m
horizontal arc, which survives one crossing and rarely two; the tucked posture
keeps the swept area small enough to survive three.

Three runs in 56 found it. That is the whole margin this task leaves.

### The distribution behind those four videos

Bottlenecks cleared **before the first contact**, 40 trials per planner:

| planner | 0 | 1 | 2 | 3 | clean solves |
|---|---|---|---|---|---|
| Predictive Sampling | 11 | 23 | 3 | 3 | **3** |
| Random Sampling | 11 | 24 | 2 | 3 | 0 |
| PSO | 17 | 22 | 1 | 0 | 0 |
| Annealed Sampling | 28 | 12 | 0 | 0 | 0 |
| Cross-Entropy | 38 | 2 | 0 | 0 | 0 |
| iLQG | 39 | 1 | 0 | 0 | 0 |

These dumps were recorded with `--early_exit=false`, so each run continues past
its first contact — which is why progress has to be measured *up to* that
contact. Counting the whole rollout marks every trial as "reached all three",
since the cart keeps driving after it hits something and parks at the rail
limit.

## Does more lookahead help?

The bottlenecks are 3 m apart, about a second of travel, so at the default 1 s
horizon a planner can barely see the next gap while committing to the posture
for this one. The obvious fix is a longer horizon. It backfires — and the
controlled version says why.

Success rate is useless as the measure here (every cell is 0–6 runs in 50, which
is noise), so the column below is partial credit: how far the cart got before its
first contact, out of three bottlenecks. 50 trials each.

| horizon | knots | knot spacing | Predictive Sampling | Random Sampling |
|---|---|---|---|---|
| 1.0 s | 12 | 83 ms | **0.96** | **0.88** |
| 2.0 s | 12 | 167 ms | 0.48 | 0.56 |
| 3.0 s | 12 | 250 ms | 0.26 | 0.28 |

Progress collapses to a quarter as the horizon triples. But `sampling_spline_points`
is fixed at 12, and the knots are spread over the horizon — so a 3 s horizon is
also a 3× coarser control signal at t=0, which is the part that actually gets
executed. **The planner was buying lookahead by giving up the resolution of the
action it is about to take.** Scaling the knots with the horizon separates the
two effects (`--spline_points`, 12 knots per second of horizon):

| horizon | knots | knot spacing | Predictive Sampling |
|---|---|---|---|
| 1.0 s | 12 | 83 ms | 0.98 |
| 2.0 s | 24 | 83 ms | 0.80 |
| 3.0 s | 36 | 83 ms | 0.86 |

Flat within noise. Two conclusions, and the second is the one worth keeping:

1. **The cost was resolution, not horizon.** Hold knot spacing fixed and the
   collapse disappears entirely.
2. **More lookahead, paid for honestly, still buys nothing.** The matched-knot
   row does not improve either — it just stops getting worse. On this task the
   binding constraint is not what the planner can see; it is that a chaotic
   3-link pendulum cannot be steered accurately enough over 3 m to arrive at the
   next gap laid out, however far ahead the plan extends.

The practical lesson generalizes past this task: `agent_horizon` and
`sampling_spline_points` are not independent knobs, and sweeping one without the
other measures their product.

## The task

```
cart mass            1.0 kg              control      force on cart, |u| <= 20 N
3 links, total 1.0 m (each 1/3 m, massless rod)      gravity      9.81 m/s^2
head mass            0.1 kg per link     obstacles    disks r=0.6 at (3, +/-0.85)
start                x=0, all links up   goal         x=6, all links up
```

`nu = 1`, `nv = 4`: one actuator for four degrees of freedom. The obstacle gap is
`2·(0.85 − 0.6) = 0.5 m` and the pendulum is 1.0 m, so it cannot pass upright.
The only solution is to swing the pendulum down, lay it out near-horizontal,
drive through, and re-erect it on the far side — while the 3-link pendulum is
chaotic and the cart is the only thing that can be pushed.

Angle convention follows the reference implementation: `θ_i = 0` points link `i`
along `+z`, so a head sits at `x = x_cart + Σ L·sin(θ_j)`, `z = Σ L·cos(θ_j)`.

### Cost terms

| Term | dim | What it asks |
|---|---|---|
| Cart | 1 | `x_cart − goal` |
| Upright | 3 | `cos(θ_i) − 1`, zero exactly at upright and smooth across the ±π wrap |
| Velocity | 4 | joint velocities small |
| Control | 1 | control small |
| Avoidance | 3×N | hinge loss `max(0, margin − clearance)` per head per disk |

The Avoidance residual
([triple_pendulum_cartpole.cc:130-136](triple_pendulum_cartpole.cc#L130-L136))
is exactly zero until a head is within `margin` of a disk *surface*, so it does
not bias the swing-up. Clearance is surface-to-surface — both radii subtracted
([triple_pendulum_cartpole.cc:62-70](triple_pendulum_cartpole.cc#L62-L70)) — so
zero means touching. Measuring from the head *centre* instead would put the
cost's idea of contact a full head radius inside MuJoCo's, and the avoidance
term would still read clearance while the contact solver was already pushing
back.

The obstacles are also real collision geoms. The cost exists to give the sampler
a gradient *before* contact, which contact alone (a discontinuity) does not
provide.

### Margin is not weight

The single most consequential parameter here, and the one most easily set wrong:

- **Weight** sets how much a violation costs.
- **`residual_Clearance` (margin)** sets how early the cost arrives — and only
  that buys the planner time to steer.

The half-gap is 0.25 m, and clearance is measured surface-to-surface. A margin at
or above that makes the hinge active across the *entire* gap — even a head
travelling exactly down the centreline is inside the penalty — so threading the
corridor always costs and the barrier becomes a wall. Measured at 100 trials
each, weight fixed at 500:

| margin | solved |
|---|---|
| 0.05 m | *(see `renders/margin100/`)* |
| 0.08 m (task default) | |
| 0.15 m | |
| 0.25 m (half-gap) | |

`slalom.xml` caps the slider at 0.2 m for this reason. Note that
`--stage=corridor`'s built-in margin of 0.25 m predates the change that made
clearance surface-to-surface (commit `e22fc9f`) and is on the wrong side of this
boundary; runs in this document set the objective explicitly with `--weights`,
which bypasses the stage margin.

---

## The benchmark

`testspeed` reports average cost per step, which cannot distinguish "swung up but
never left the start" from "threaded the corridor and reached the goal". On this
task the second scores *worse* on aggregate cost than driving through with the
pendulum whirling. So the harness reports outcomes instead:

```bash
./build/bin/corridor_benchmark --planner=0 --stage=corridor \
    --weights=1,0,0.1,0.01,500 --speed=0.25 --repeats=100
```

Key flags — full list via `--help`:

| Flag | Why it exists |
|---|---|
| `--weights` | States the objective on the command line, so the table says what was optimized. Overrides the stage's weights *and* its margin. |
| `--speed` | The GUI speed slider. `1/speed` planner iterations per control step: `1.0` = 200 Hz, `0.25` = 800 Hz-equivalent. Holds iterations constant across planners, so the comparison is about the algorithm, not single-thread throughput; `wall_s` is where iteration cost shows up. |
| `--early_exit` | Stops a run once the outcome is decided (solved, or disqualifying contact). Both conditions are monotone, so the verdict is unchanged and a 100-trial sweep gets several times cheaper. Suppresses "held to end". |
| `--init_noise`, `--seed` | Perturbs the start per trial. Without it, a deterministic planner like iLQG reports the same run 100 times and its success rate is 0 or 1 by construction. Trial *k* is the same start for every planner. |
| `--stage` | `corridor` / `balance` / `combined` — isolates the two capabilities the full task confounds. |
| `--task` | `corridor` (one gap) or `slalom` (three). |
| `--horizon`, `--exploration` | Override `agent_horizon` and `sampling_exploration` without editing the XML. |

### What counts as solved

- **Goal**: cart within `--goal_tolerance` (0.3 m) of the goal. The pendulum's
  state is part of the goal set only if the Upright weight is nonzero — a run
  with `upright=0` was never asked to capture anything, so requiring it to end
  upright would score it against a term it was not given.
- **Collided**: any overlap between a link head and a disk, at any step.

### Why the collision test has no tolerance

This used to allow 0.02 m of penetration before calling a run collided, on the
theory that MuJoCo's soft contacts permit a few harmless millimetres on touch.
That theory is wrong, and it cost 17 points of measured success rate.

Soft contacts govern how *deep* an overlap gets once two surfaces meet. They
never report a contact between separated geoms. The dumps show this directly:
across 30 rollouts, `ncon > 0` and `min_clearance < 0` agree on every step —
contact happens exactly when the surfaces overlap, never before. So there is no
grazing band to protect.

What the tolerance did instead: the link heads have radius 0.028 m, so 0.02 m of
allowed penetration let a head sit **71% inside a disk** and still score clean.
And the measured penetrations run continuously from 2 mm to 39 mm with no gap
between "numerical" and "real", so no depth threshold could have been principled
either.

| criterion | Predictive Sampling, corridor, 100 trials |
|---|---|
| penetration < 0.02 m (old) | 93/100 |
| any overlap (current) | **76/100** |

On the slalom the correction is starker: **0 of 16 sampled runs clear all three
bottlenecks without touching a disk.** The runs that scored as solved before
were grazing, and the video shows it. Both flags survive
(`--penetration_tolerance`, `--contact_fraction_tolerance`) so the old numbers
can be reproduced, but zero is the default and the physically correct reading of
an avoidance constraint.
- **`--stage=balance` and `combined`** additionally require `‖qvel‖ <
  --speed_tolerance`. Without it a pendulum swinging *through* vertical at 5
  rad/s scores as solved; the measured median speed in the goal configuration
  was 4.7 rad/s, so the configuration test alone is not a solve.

### Look at the rollout before believing the number

Aggregate metrics on this task are actively misleading, so every sweep dumps the
first runs and there is a renderer for them:

```bash
mjpc/tasks/triple_pendulum_cartpole/benchmark/render_sweep.sh renders/avoid100_s025
```

See the [`planner-eyes`](../../../.claude/skills/planner-eyes/SKILL.md) skill.

---

## The three stages

`--stage` exists because the combined task confounds two capabilities: a planner
that never reaches the goal upright might be failing to thread the corridor or
failing to capture the pendulum afterwards, and one aggregate number cannot say
which.

| Stage | World | Objective | Asks |
|---|---|---|---|
| `corridor` | obstacles, goal x=6 | Upright 0, Velocity 0.5, Avoidance 300 | Can it drive to the goal without putting a head in a disk? |
| `balance` | no obstacles, goal x=0, starts fully hung | Upright 80, Velocity 3 | Can it erect a 3-link underactuated pendulum and hold it? |
| `combined` | the paper's task | `task.xml` defaults | Both, in sequence. |

A planner that passes both isolated stages and fails `combined` is failing at the
*composition*, which is a different diagnosis from failing a component.

Note the stage weights and `--weights` are alternatives, not layers: passing
`--weights` replaces the stage objective entirely, margin included. The results
in this document use `--weights`, not the stage defaults.

### Balance is where everything except iLQG stops

Only iLQG holds the pendulum up indefinitely from the upright start, and it
cannot recover once knocked down. Nothing else comes close. That is a capability
gap, not a tuning problem, and it is the reason the headline table above uses the
avoidance-only objective: the combined task cannot separate "cannot avoid" from
"cannot balance", and on the balance half the answer for every sampler is
already known.

---

## The planners

Each has a teardown binding the published algorithm to its source lines and to
measured results:

| Planner | Idx | Teardown |
|---|---|---|
| Predictive Sampling | 0 | (upstream) [`planners/sampling`](../../planners/sampling) |
| iLQG | 2 | (upstream) [`planners/ilqg`](../../planners/ilqg) |
| Cross-Entropy | 5 | (upstream) [`planners/cross_entropy`](../../planners/cross_entropy) |
| PSO | 7 | [`PSO.md`](../../planners/pso/PSO.md) |
| Annealed Sampling (DIAL-MPC) | 8 | [`ANNEALED_SAMPLING.md`](../../planners/annealed_sampling/ANNEALED_SAMPLING.md) |
| Random Sampling | 9 | [`RANDOM_SAMPLING.md`](../../planners/random_sampling/RANDOM_SAMPLING.md) |
| AgileRRT | — | [`AGILE_RRT.md`](../../planners/agile_rrt/AGILE_RRT.md) (feasibility study, prototype only) |

### Budget parity

Comparisons are only meaningful at equal budget, and the library defaults do not
give it. `task.xml` therefore states the rollout budget explicitly
([task.xml:56-57](task.xml#L56-L57)): `sampling_trajectories = 10` and
`pso_num_particles = 10`, because PSO defaults to 20 while the others default to
10. Two residual inequalities remain, both documented in the tables above:

1. **Annealed Sampling** multiplies its budget by `annealing_iterations` (4).
2. **PSO** hard-codes its exploration noise at `0.1·ctrlrange`, half what
   `sampling_exploration = 0.4` gives the others. This is worth 30 points of
   success rate — see [PSO.md](../../planners/pso/PSO.md#measured-behaviour) for
   the controlled experiment.

---

## Reproducing

```bash
cmake --build build --target corridor_benchmark -j

# headline table (~45 min)
mjpc/tasks/triple_pendulum_cartpole/benchmark/avoidance_sweep.sh renders/avoid100_s025 100 0.25

# three bottlenecks (~60 min)
TASK=slalom mjpc/tasks/triple_pendulum_cartpole/benchmark/avoidance_sweep.sh renders/slalom100_s025 100 0.25

# per-iteration cost, and the contact-free control for it (~2 min each)
mjpc/tasks/triple_pendulum_cartpole/benchmark/timing_bench.sh renders/timing 10 3 1.0
STAGE=balance mjpc/tasks/triple_pendulum_cartpole/benchmark/timing_bench.sh renders/timing_balance 10 3 1.0

# does lookahead help on the slalom?
TOTAL_TIME=12 mjpc/tasks/triple_pendulum_cartpole/benchmark/horizon_sweep.sh renders/horizon_slalom 50 0.25 1.0 2.0 3.0

# average cost per step on stock Cartpole, per planner (sanity check)
mjpc/tasks/triple_pendulum_cartpole/benchmark/cartpole_costs.sh 10 3

# all three stages, all planners (diagnostic rather than headline)
mjpc/tasks/triple_pendulum_cartpole/benchmark/sweep.sh renders/stages 20 3
```

Run `timing_bench.sh` with nothing else on the machine. Timing is the one
measurement here that CPU contention silently corrupts; success rates are not
affected by it, because the harness counts iterations rather than seconds.

Every sweep log records the binary's md5 and the commit, because a rebuild
part-way through a sweep is the easy way to end up with a table whose rows were
measured by different code.
