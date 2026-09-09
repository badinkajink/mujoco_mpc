# Controlled table-height robustness study

**Finished at the user’s request after 51 completed trials.** The uniform
pitch-tracking change was rejected by the predeclared screen rule. No second
adaptive candidate or fresh confirmation trial was run. The queue is stopped
and the baseline strategy is restored. Read the final HTML findings and
`screen1_decision.json`; `results.json` preserves all 51 scores and
`final_integrity.json` records their hash checks. No controller defaults were
promoted.

Read `protocol.md` and the dated HTML report before interpreting results.
The study was run on `codex/table-height-robustness-20260909` from parent
`1eab9af6` and consolidated onto `wxie/table-height` on 2026-09-09.

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
  -L "$PWD/build_cmake/lib" \
  -Wl,-rpath,"$PWD/build_cmake/lib" \
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
strategy and binary provenance, and restores the checked-out strategy in `finally`.
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
noise are limited repeatability evidence. The unused confirmation design reserves
fresh seeds and heights withheld from tuning; prior data must never be pooled.
All thresholds, censoring and recipe-selection rules are in `protocol.md`.

The physical diagnostics include unintended table contacts, upward brace load,
joint-range excursions, joint velocities, model-force-limit utilization,
maximum foot displacement and contact penetration. Foot displacement includes
rolling and possible sliding; model torque limits do not certify hardware.
167 Hz is simulated planner frequency; the runs are slower than real time.

The 33-run ablation is complete. Its hashes and invariant checks are recorded in
`ablation_integrity.json`. Screen 1 tested one existing numeric,
`brace_pitch_track=1`, against fresh reference controls: seeds 20–22 at the
three anchor heights, 18 trials. `screen1.json` fixes the randomized order;
`decisions.md` records the hypothesis and selection rule before the first trial.

`after_screen1.py` applied only the frozen rule after every screen result. It
wrote `screen1_decision.json` and rejected the candidate, so it did not create
or launch the confirmation manifest. Do not rerun that watcher as a new study.
For future batches, `STOP` prevents new episodes; the running episode finishes
and its result is retained.

The report is final for the 51 completed episodes and explicitly states that
fresh confirmation was not performed. `physical_summary.py` computes descriptive diagnostics during
successful precise intervals; it never changes pass/fail scoring. The near-limit
fraction means samples with ANY actuator at >=99.5% of its modeled force limit,
not the fraction of all actuators saturated. Foot displacement in the main
ledger is measured from episode start, not motion within the precise hold.

The benchmark declares completion after three seconds in its final phase.
A trial reaching final standing near the 75-second deadline can therefore miss
completion despite being upright. Such cases are labeled separately from
braced release stalls; the frozen full-task flag remains unchanged.

There is no new CMPC experiment in this study. These MJPC two-second observed
precision holds are not directly comparable to the earlier CMPC 25-second holds.
Differences in initialization, task, engines, and scoring prevent attributing
the old cross-controller counts solely to the optimizers.
