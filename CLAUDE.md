# CLAUDE.md — read this before touching any lean code in this repo

Verified 2026-09-05 against `icra2026` at `3e876d70`.

**This file holds what is true for anyone touching this repo.** Work specific to
wxie's line — branch `wxie/table-height`, the `Table H` parameter, the
`lean_bench` harness, `studies/table_height/`, the doc pages — is in
**`CLAUDE_wxie.md`**. The two do not repeat each other; if a fact appears in
both, one of them is stale.

## 0. Which lean task to work on: `lean.cc` on `icra2026`

**The paper's controller is `lean.cc` on the `icra2026` branch.** Corrected
2026-09-04 by the user, after three sessions were spent on the wrong one:

> "we really messed up earlier by running lean_simple on the crocoddyl-mpc branch
> on mujoco_mpc. i really misunderstood what was working. ive since cleared things
> up with allen, and icra2026 branch, lean.cc is what we should have been working
> on."

So: **new work branches off `icra2026`.** `lean_simple.cc` on the `crocoddyl-mpc`
branch is a separate line of history (`git merge-base --is-ancestor` is false
between them) and is NOT the study task. An earlier version of this file said the
opposite; it was wrong.

Allen owns `lean.cc` and commits to `icra2026` most days (last checked
2026-09-03, `3e876d70`). Pull before starting, and expect local edits to
`lean.h`/`lean.cc` to be overwritten by a fetch if they are not committed.

### Strategy slots to use

Allen's recommendation (2026-09-04): **strategy 9** (`h12_brace_servo_sweep`,
visual servoing via the magpie eye-in-palm camera) or **strategy 25**
(`h12_brace_targeting`, no servoing).

⚠ **Strategy 9 cannot be evaluated in a headless own-sim bench.** Its servo
corrections arrive on the DDS `rt/object_tag` channel and land in
`mjpc::g_object_cam_*` / `g_object_seq` (see `lean.h`). With no tag bridge
running, `g_object_seq` never changes, the freshness gate never opens, and the
rung runs open-loop on its JSON placeholder targets. Use 9 on the twin or the
real robot; use **25** for sim sweeps.

## 1. The planner is NOT run-to-run deterministic

Measured 2026-09-04. Two `lean_bench` invocations with identical task, strategy,
table height, seed and binary:

| | t_complete | phase-3 entry |
|---|---|---|
| rep a | 45.77 s | 32.15 s |
| rep b | 45.50 s | 28.23 s |

**3.9 s apart on the same input.** MJPC's sampling planner draws noise from a
generator shared across the thread pool, so a rollout is not a function of
(config, seed), and **the thread count changes the stream**: h = 0.785 seed 0
stood past t = 35 s at `--threads 4` and fell at t = 3.4 s at `--threads 6`.

Consequences, and they are not optional:
- **n = 1 per condition proves nothing.** Use >= 3 seeds and report counts.
- **Hold `--threads` fixed across every arm of a comparison**, and say what it was.
- A single dramatic run (a fall, a fast completion) is an anecdote until it
  repeats.

## 1a. Known-wrong monitoring channels in `lean::ComputeMetrics`

Both found 2026-09-04, both reported to Allen rather than patched (his file, and
neither affects the controller):

- **`brace_force` is always 0 for this task.** It reads
  `right_contact ? right_contact[0] : 0.0` under the comment "right arm always
  braces, left arm always reaches", but the model's brace geoms are
  `left_forearm_pad` / `left_wrist_pad`. Measured: 0.00 N for a whole run while
  the left forearm carried 97-168 N. The deploy monitor shows no brace force
  during a real brace. **Sum the contacts yourself.**
- **`reach_err` / `reach_tgt_*` are nan for every braced ladder.** The block is
  gated on `kf.name == "reach_to_target"`; strategy 25's targeting rung is named
  `forearm_brace_lean`. Compute reach error from the gripper body and the
  `target` mocap instead.

Related trap when summing table contacts yourself: **the table's legs stand on
the floor**, and floor geoms belong to body 0, so "one side of the contact is the
table" books the table's own weight (~166 N) as brace load. Skip body 0 and the
free `object`.

## 1b. Our additions live on a branch, not here

Branch **`wxie/table-height`** adds a table-height task parameter
(`residual_Table H`, index 7, default 0 = off = byte-identical), a headless
bench (`mjpc/lean_bench.cc`), and the study scripts under
`studies/table_height/`. All of it is additive and default-off so it rebases
cleanly onto Allen's work. **See `CLAUDE_wxie.md`** for the interface, the run
policy and the results; nothing about it is duplicated here.

## 1c. What `lean.cc` still is

`lean.cc` (~3.6 kloc) is the phase-scheduled pipeline: 8-keyframe strategy JSONs
loaded from `SOURCE_DIR` at runtime (**editing one needs no rebuild**), per-phase
`success_sustain_time` / `target_ramp_sec`, hand-authored target ramps. It is also
the deploy pipeline:

```sh
h12_control_node --task "Lean H12 Magpie" --strategy 6   # documented deploy invocation
```

Measured 2026-08-26 (`docs/lean/2026-08-26_schedule_cost.html`), still true:
- **`target_ramp_sec` is NOT additive wall-clock.** It drives target-pose
  interpolation concurrently with the sustain and does not gate the advance.
- **`target_distance_tolerance` is dead on every lean phase** -- no keyframe
  declares a real contact pair, so `total_distance` is identically 0.
- **`brace_contact_verify` (2.0 s) is what actually gates the brace rungs.**

`lean_simple_gripper.cc` is a red herring: not in `mjpc/CMakeLists.txt`, not
registered in `tasks.cc`, and it defines the same symbols as `lean.cc` so it could
not link alongside it.

## 2. The three checkouts

| Path | What it is | Binaries |
|---|---|---|
| `Humanoid_Simulation/mujoco_mpc` | submodule of the GOLEM superproject, branch `icra2026` | `build_cmake/bin/testspeed` (fresh; `ninja -C build_cmake -j6` works). `build/` is empty and root-owned — a docker artifact, ignore it. |
| `/home/humanoid/Programs/mjpc_icra2026` | standalone split checkout, same commit | `build/bin/{mjpc,testspeed,lean_probe}` — **stale** (Aug 4–5) |
| `Humanoid_Simulation/crocoddyl_mpc` | the research repo where the work happens; gitignored by the superproject and deliberately NOT a submodule | has its own `CLAUDE.md` — **read it**, it carries the working agreement (definition of done, `croco` conda env, `--dt 0.02`, never export `LD_PRELOAD`) |

Study scripts (`simple_lean.py`, `simple_matrix.py`, `simple_video.py`,
`simple_metrics.py`, `simple_page.py`, `mjpc_chain.py`, …) live in
`crocoddyl_mpc/studies/`, not in either MJPC checkout. They resolve the binary
and task tree from the environment:

```bash
export MJPC_BIN=/home/humanoid/Programs/Humanoid_Simulation/mujoco_mpc/build_cmake/bin/testspeed
export LEAN_TASK_DIR=/home/humanoid/Programs/Humanoid_Simulation/crocoddyl_mpc/studies/runs/_stage/mjpc/tasks/humanoid_bench/lean
```

`testspeed` on this branch takes only `--task --planner_thread
--steps_per_planning_iteration --total_time --dump_residual --dump_perturb`.
There is no `--phase_schedule` / `--start_qpos` / `--strategy`; older study
scripts that pass those were written against a tree that is not this one.

## 3. Editing rules

- Editing a strategy JSON needs **no rebuild** — `lean.h` loads them from
  `SOURCE_DIR` at runtime.
- Behaviour changes go behind a model `numeric`, default 0 = byte-identical.
  That is the house pattern; `lean_commit_retry` in `lean.cc` is the model.
- Read the comment above a constant before changing the constant. Almost every
  constant here is an argument backed by a measurement.
