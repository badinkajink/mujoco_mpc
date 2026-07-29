---
name: planner-eyes
description: Look at what an MJPC planner actually DID before explaining its cost. Use after any benchmark/testspeed run, whenever a planner "fails" or a cost term looks flat, before writing up a planner comparison, and before proposing a cost-weight or planner change - renders the rollout to a labelled filmstrip and MP4 alongside a per-step trace, so behaviour is observed rather than inferred from the cost table.
---

# Planner eyes — watch the rollout before you explain the cost

`testspeed` and `CostValue` say what the planner was **paid for**. They do not say
what it **did**, and on MJPC tasks those come apart constantly.

Concrete case from this repo (triple-pendulum-cartpole corridor task): MPPI
recorded the **lowest** average cost of all four planners — 22 against iLQG's 28
and CEM's 84 — while ending every run with the pendulum hanging upside down. The
cost table ranked it best. The render showed it abandoning half the task.

**Aggregate cost is sign-blind, timing-blind, and — most dangerously — blind to
whether your success criterion means what you think it means.**

## 1. Dump, then look

MJPC planners run in C++, so the loop is: dump a per-step trajectory from the
benchmark, replay it through the model in Python, render.

```bash
cd build && ninja corridor_benchmark
./bin/corridor_benchmark --planner=0 --total_time=30 --repeats=1 \
    --dump=<scratch>/mppi.csv

MUJOCO_GL=egl python3 \
  mjpc/tasks/triple_pendulum_cartpole/benchmark/filmstrip.py \
  --dump <scratch>/mppi.csv --out <scratch>/mppi.png --video <scratch>/mppi.mp4
```

**Then `Read` the PNG.** Do not stop at the printed summary. The script tiles
event-aligned frames — first departure from the start pose, closest approach to
an obstacle, deepest contact, corridor crossing, rightmost extent, end — and
labels each with the quantities that decide success. It also prints the paired
per-step trace. Cross-check them: if the picture looks solved but the trace says
`spd_end 9.38`, the picture is a single frame of something moving fast.

Useful flags: `--at 0,214,508,1000` (explicit steps), `--track` (camera follows
the cart: pendulum legible, obstacles no longer a fixed reference),
`--from-step/--to-step` (video range), `--distance/--azimuth/--lookat`.
Keep `--azimuth 90` for planar tasks so the motion is in the image plane.

For a new task, copy `corridor_benchmark.cc` — the `--dump` block is ~15 lines —
rather than trying to infer behaviour from cost alone.

## 2. Distrust your success criterion. Then distrust it again.

This is the part that actually bites. On the corridor task the *same rollouts*
scored three different ways:

| Criterion | Result | Verdict |
|---|---|---|
| final state: cart at goal ∧ upright | 0/10 solved | right answer, **wrong reason** |
| ever: cart at goal ∧ upright at any step | 6/6 "solved then lost balance" | **wrong answer** |
| ever: cart at goal ∧ upright ∧ `‖qvel‖ < 1` | 0/24 solved | right answer, right reason |

Criterion 2 looked like a discovery — "the planner solves it and then falls
over!" — and was an artifact. Measuring speed inside the goal set showed median
`‖qvel‖ = 4.68 rad/s` across 1517 qualifying steps: the pendulum was **sweeping
through** vertical, not arriving at it. Only 5 of 1517 steps were below
`0.5 rad/s`.

So:

- **A configuration test is not a state test.** For anything with momentum, gate
  on velocity too. "Looks like the goal pose" and "is at the goal state" differ
  by exactly the thing that makes the task hard.
- **Check final-state *and* best-ever.** Final-state-only scores a genuine
  success followed by a late failure as a total failure — which hides *which
  half* is broken (plan vs. terminal controller). Report both.
- **When a metric change flips your conclusion, the metric is the finding.**
  Write down which criterion produced which number before drawing any
  conclusion about the planner.

## 3. Separate "cannot plan it" from "cannot hold it"

Before concluding a planner is too myopic, check whether the trajectory ever
*enters* the goal region and simply fails to stay. Three outcomes, three
different projects:

| enters goal set | stays | the problem is |
|---|---|---|
| never | — | search / horizon / cost shaping — a **planning** problem |
| yes | no | terminal stabilization — a **control** problem; a better global planner will not fix it |
| yes | yes | done; write it up |

On the corridor task this distinction decided a design recommendation: the
failure is that no planner brings the pendulum to *rest* upright, and a global
tree planner addresses that only because its stopping test is over the full
state including velocity — not because it explores better.

## 4. Report distributions, not one run

Sampling planners are stochastic and the spread is large. One corridor run
showed zero contacts and 5.5 cm clearance; ten runs of the same configuration
showed 9/10 grazing an obstacle. `--repeats=6` minimum before any claim.

Also separate *grazing* from *plowing*: MuJoCo's soft contacts always allow a few
mm on touch, so a zero-penetration threshold flags harmless contact. Gate on
penetration depth **and** contact fraction — 0.2% of steps in contact is a graze,
87% is a planner leaning on the obstacle as support (which is how iLQG "reaches"
the goal on this task).

## Gotchas this loop exists to catch

- **Lowest cost can be the worst behaviour.** Cost is a weighted sum over
  residuals the planner can trade against each other. It will abandon an
  expensive term it cannot make progress on. Look for terms that are flat and
  large — that is an abandoned objective, not a converged one.
- **Longer horizon is not a fix for myopia on chaotic systems.** On this task
  raising `agent_horizon` 1.0 → 2.0 → 3.0 s produced
  `Rollout divergence` and then `Nan, Inf or huge value in QACC`. Open-loop
  rollouts stop meaning anything past the Lyapunov time; check for divergence
  warnings before believing a long-horizon result.
- **`agent_planner` lives in the XML.** To sweep planners without editing files,
  write the custom numeric before `Agent::Initialize` —
  `GetCustomNumericData(model, "agent_planner")` returns a writable `double*`.
- **Task-level metrics need model ids, not qpos indices.** Read head/obstacle
  positions from `site_xpos`/`geom_xpos` via `mj_name2id`, cached in
  `ResetLocked`, so the metric survives a model edit.
- **`ncon > 0` is not "collided".** It is "touching". See §4.

## Related

- `algorithm-teardown` — for documenting how the planner works (math ↔ code ↔
  result). This skill produces the "result" layer that teardown requires.
