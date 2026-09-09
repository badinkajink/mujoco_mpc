# Controlled table-height robustness study

Read `protocol.md` and the dated HTML report before interpreting results.
Parent: `1eab9af6`; isolated branch `codex/table-height-robustness-20260909`.
No original source checkout edits except append-only coordination notes.

## Exact engine, not just matching version strings

The benchmark's pinned MuJoCo source is revision `088079ef` (full identity
and shared-library SHA-256 in `engine.json`). It labels itself 3.2.3 but its
binary model ABI differs from the released PyPI 3.2.3 package. The evaluator
therefore links to the exact benchmark library. It performs FK/contact-force
reconstruction on logged pre-integration full states with no task callback.
Python only computes distances, intervals, tables, and plots from that CSV.

Build the unchanged controller plus passive recorder using the existing build:

```bash
ninja -C build_cmake -j4 lean_bench
g++ -std=c++17 -O2 \
  -I /home/humanoid/Programs/mjpc_icra2026/build/_deps/mujoco-src/include \
  studies/table_height/robustness/replay.cc \
  -L /home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908/build_cmake/lib \
  -Wl,-rpath,/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908/build_cmake/lib \
  -lmujoco -o studies/table_height/robustness/replay
python3 studies/table_height/robustness/test_protocol.py
```

`--state_out` and `--model_out` are passive benchmark additions. They never
change controls, the sampling distribution, integrator state, or task gates.
They save qpos/qvel/control/solver warm-start at the same pre-step timestamp,
and save plant/planner models after height initialization. The existing live
logger mixes post-step labels with cached pre-step kinematics; new primary
metrics use the aligned replay and retain a cache agreement diagnostic.

## Run and resume

```bash
python3 studies/table_height/robustness/run.py \
  studies/table_height/robustness/ablation.json
python3 studies/table_height/robustness/status.py
python3 studies/table_height/robustness/summarize.py --plots
```

The runner holds an exclusive lock, runs serially under CPUQuota=700% and
MemoryMax=11G, refuses busy/low-memory conditions, saves immutable command,
strategy and binary provenance, and restores the isolated strategy in `finally`.
Do not run another job or modify that strategy while a batch is active.
Creating `studies/table_height/robustness/STOP` stops at the next episode
boundary. A completed evaluation is cached only if its exact job matches.
Incomplete directories cause an error rather than being silently overwritten.
To repeat a condition, use a NEW tag and retain the original result.

Per-run states and actual model snapshots are under
`studies/table_height/runs/robustness_20260909/`. Bench rc=1 is a recorded fall,
not a discarded run. Infrastructure exceptions halt the batch. Replay a
completed run with `python3 studies/table_height/robustness/evaluate.py RUN_DIR`.
The source runner uses the existing croco Python environment for NumPy; it
does not invoke Crocoddyl or use Python MuJoCo for physical evaluation.

## Interpretation

One-factor differences are conditional on the reference combination. Removing
one factor does not establish all interactions or additive contributions.
A shorter target is a task relaxation; a longer hold gate provides more time
for reaching. The legacy-model arm is a diagnostic of benchmark validity,
not a candidate controller. Three initial-state seeds plus uncontrolled sampler
noise are limited repeatability evidence. New confirmation uses fresh seeds
and heights withheld from this session's tuning; prior data are never pooled.
All thresholds, censoring and recipe-selection rules are in `protocol.md`.

The physical diagnostics include unintended table contacts, upward brace load,
joint-range excursions, joint velocities, model-force-limit utilization,
maximum foot displacement and contact penetration. Foot displacement includes
rolling and possible sliding; model torque limits do not certify hardware.
167 Hz is simulated planner frequency; the runs are slower than real time.

The final recipe and adaptive decision log will be added once the frozen
ablation finishes. Until then the HTML report explicitly labels counts partial.
