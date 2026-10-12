# MBRL archive: sim-gap, learned residual, parametric identification and planner distillation on the Unitree H1-2 (branch `mbrl`)

Date: 2026-10-11. Workspace: `~/Desktop/h12/mujoco_mpc/mujoco_mpc/mbrl/`. Robot: Unitree H1-2 (27 actuated joints, 12 legs + 1 waist + 14 arms) controlled by MuJoCo MPC (iCEM sampling planner) on the hand-built plant `Lean_H12_Magpie.xml` for the lean-and-brace reaching task (strategy 25 / slot 37).

## 0. What this document is

This is the single surviving record of the model-based-RL effort of 2026-10-04 to 2026-10-05 (plus the 2026-10-11 tree snapshot commit). It was compiled on 2026-10-11 from the two narrative reports (`REPORT_2026-10-04.md`, `RESEARCH_PLAN.md`), the 42-commit git history of branch `mbrl`, the 90 text logs and JSON result files under `mbrl/out`, `mbrl/out_twin` and `mbrl/out_bc`, the two dataset manifests and npz scans of every array file. Immediately after this compilation the following were deleted:

| deleted | size | content |
|---|---|---|
| `mbrl/data/` | 9.4 GB (scan: 10.04 GB, 1387 npz) | real-robot transition corpus `runs/` (493 npz), one-step sim-gap predictions `simgap/` (493 npz), macro-step Jacobians `jac/` (401 npz) |
| `mbrl/data_twin/` | 4.1 GB (scan: 4.35 GB, 380 npz) | twin (sim2sim) corpus `runs/` (190 npz) and `simgap/` (190 npz) |
| `mbrl/out_bc/` | 671 MB (scan: 0.70 GB, 119 npz + 23 logs) | stage-3 BC / DAgger stand-policy weights, teacher label sets, `bc_data.npz` (587 MB) |
| `mbrl/out/*.npz, *.log, *.json`, `mbrl/out_twin/*.npz, *.log, *.json` | 21 MB + 1.7 MB | residual normalisers, rollout error arrays, every experiment log and summary |

What is kept: the python/shell tooling in `mbrl/*.py`, `mbrl/*.sh`, the two reports, this archive, and the C++ planner hook (`mjpc/tasks/humanoid_bench/lean/residual_mlp.{h,cc}`, `lean::ModifyRolloutState`, `H12_ROLLOUT_SUBSTEPS` in `mjpc/trajectory.cc`).

How to rebuild the datasets (all source recordings still exist under `~/Desktop/h12/logs`):

```
cd ~/Desktop/h12/mujoco_mpc/mujoco_mpc/mbrl
# real-robot corpus (493 runs, 6.1 M transitions, ~2 min) -> data/runs/<run>.npz + data/manifest.json
.venv/bin/python build_dataset.py
# twin corpus (190 runs, 3.1 M transitions) -> data_twin/runs + data_twin/manifest.json
MBRL_DATA_ROOT=$PWD/data_twin MBRL_SRC_GLOB='logs/grid37_twin/*/*/*/*.npz' .venv/bin/python build_dataset.py
# one-step sim-gap predictions (~20 min real, ~15 min twin) -> data*/simgap/<run>.npz
MBRL_WORKERS=10 .venv/bin/python simgap.py          # add MBRL_DATA_ROOT=... MBRL_OUT=... for the twin
.venv/bin/python analyze_simgap.py                  # out/simgap_summary.json + figs
# jacobians (2.1 GB, 178 s with 20 workers) -> data/jac/<run>.npz
MBRL_WORKERS=20 .venv/bin/python jacobians.py
```
The exact run lists (run name, sample count, both-feet-loaded count, strategy label, split, source path) are reproduced in section 7 so that the same train/val/test split can be rebuilt. Every heavy job was run under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0` (planner jobs 20G outer / 14G inner), one at a time.

---

## 1. Executive summary and final verdicts

The thesis under test was: "a sampling MPC on a physics plant plus a small learned residual trained from a few hours of real-robot logs closes most of the sim-to-real gap for whole-body contact tasks, and the corrected plant is good enough to distil an MPC policy that runs without online planning." Every stage produced a measurement; the thesis failed at the stage where the correction meets the planner.

### 1.1 Stage 0, sim-gap measurement (solid, reproducible)

- Corpus: 493 real H1-2 recordings (June to September 2026), 6.1 M transitions at 100 Hz (q, dq, tau_est, PD command target/kp/kd/tau_ff, torso IMU, estimator base pose). Split by run 384/54/55 (train/val/test). 4.0 M transitions have both feet loaded; 1.6 M are "kinematically consistent" (both plant soles within 3 mm of the floor after the state reset).
- One-step (10 ms) error of the deployed planner plant vs the trivial hold-velocity predictor, consistent subset: legs 6.3x (upright) / 7.1x (lean), waist 3.8x / 13.8x, arms 2.0x / 4.5x. On the full loaded-feet set the legs are 12.7x off (one foot floats). PD-command and measured-torque models agree within 10 %: the actuator model is not where the gap is. Reconstructed PD torque matches measured torque to 0.1 to 2 Nm RMSE per joint with a +0.7 Nm bias on both ankle pitches.
- Hardware facts exposed by the state-reset protocol: the planner's gripper / wrist collision shapes self-collide at recorded arm poses (arm one-step error 1.52 -> 0.11 rad/s when disabled); right ankle pitch zero error +3.8 deg median (left -0.2); left/right sole height mismatch median 3.9 mm, p90 10.9 mm, p99 19.2 mm; real joints look at least 3x more damped than the plant's 10 Nm s/rad. On the twin corpus the same protocol returns 0.0/0.1/0.0/0.1 deg flattening and 1.1 mm sole gap, proving the protocol invents nothing.
- The planner's own integration step (1 x 10 ms implicitfast) more than doubles the leg error through contact vs 5 x 2 ms (legs 0.174 -> 0.413 rad/s); arms unaffected.

### 1.2 Stage 1, learned residual (open-loop success, closed-loop failure at long horizons)

- 3x512 SiLU MLP on planner-known inputs (q, dq, commanded torque, cmd - q, gains, gravity, pelvis rate), 1.25 M training transitions: one-step test RMSE all-joints 0.112 (plant) -> 0.076 (ridge) -> 0.049 (MLP), i.e. 2.3x; arms to within 1.25x of hold-velocity; holds on 11 of 12 strategy families (fails only on `stand6supp`).
- One-step-trained residual compounds in closed loop: by 250-500 ms worse than the plain plant; ridge diverges in 28 of 339 segments.
- Data-as-Demonstrator (DaD) retraining on its own rollouts removes the compounding: post-step 5 x 2 ms, 250 ms displacement error legs 0.0385 -> 0.0387, waist 0.0564 -> 0.0152, arms 0.0218 -> 0.0100. At the planner step (pre-step injection, 1 x 10 ms) a 2x64 net on arms + waist only: legs 0.0443 -> 0.0417, waist 0.0552 -> 0.0261, arms 0.0220 -> 0.0133. The 2x64 net equals the 3x512 net and costs ~5 % planning time (3x512 cost 4x).
- Two negative results: linearised multi-step training through finite-difference Jacobians diverges (gains near 1e4 in contact directions, loss 8.5e22 at warm start, NaN in epoch 0); one-step error is a degenerate identification objective (Nelder-Mead drives armature to x2273, torque scale to 1e-4: an immovable robot wins).
- Spotting contamination: the all-data residual's waist term is contaminated by the operator's hand during leans; a hands-off (tilt < 12 deg) residual is clean (arms -30 %, legs -10 % upright) and still helps lean-phase arms/waist.

### 1.3 Parametric identification

- With a 250 ms rollout objective (`param_id2.py`) on real data: damping x3.0 (both groups, at the bound), leg armature x1.3, arm armature x0.33, torque x1.0 / x0.91; held-out 250 ms legs 0.0466 -> 0.0256 (45 % cut), waist 0.0367 -> 0.0275, arms 0.0175 -> 0.0152. Hip/knee zero offsets < 0.4 deg once feet are flattened, so the 12 mm sole gap is not a knee/hip zero.
- On the twin corpus the same fit picks damping x3.0 again plus leg armature x3.0, arm armature x1.1, torque legs x1.1 / arms x0.6, and hip-pitch offsets of -3 deg that cannot be a twin zero error. Conclusion: "x3 damping" is an artefact of the planner's coarse single 10 ms step, not a property of the robot.

### 1.4 Stage 2, residual inside the planner (exact integration, no benefit)

- The C++ hook (`H12_RESIDUAL_MLP`, `H12_RESIDUAL_SCALE=legs,waist,arms`, default `0,1,1`) reproduces the Python features to < 1e-5 and the MLP output to 3e-6.
- Pooled honest-twin outcomes (slot 37, B3, d37, `_real_ns`, interleaved pairs, 180 s): baseline 20 runs, 4 full pipelines, 10 reached rung 9, 14 falls; residual variants (real-data, twin-trained, noise-robust, planner-features-only, reduced authority) 22 runs, 1 full pipeline, 2 reached rung 9, 18 falls. The real-data residual on the twin is bias by construction (the twin is a MuJoCo plant close to the planner's) and its waist term collapses the free stand (2/5 runs squat); arms-only is neutral.
- A residual that is correct for the twin by construction (trained on 190 twin recordings, 20-40 % lower open-loop 250 ms error) still made the twin's planner worse: 0/5 pipelines, 0/5 rung 9, 5 falls, longest seat 19 s vs baseline 1/5, 3/5, 3 falls, 115 s.
- Planner imagination error (iCEM nominal trajectory vs what the twin did): raw 3-10x worse with the residual, but phase-matched to the stand (< 45 s) the residual is slightly better on legs and 40-50 % worse on waist/arms. Reduced authority (clip x0.3, scale 0.5) brings imagination error inside the baseline spread but the 5-pair A/B still loses (0 pipelines / 0 rung 9 vs 2 / 3).
- Parametric corrections in the planner include (damping 30; damping 30 + armature 0.3) are neutral to worse on the twin (damping 30: 0 full / 1 rung 9 / 4 falls vs 1 / 2 / 4; damping+armature: 0 full, max rung 5, 5 falls, four before 57 s vs 1 / 1 / 4).
- `H12_ROLLOUT_SUBSTEPS=2` (a finer planner step) is strongly negative: 3/3 lock the knees after release, two collapse into a deep squat, imagination error 10x worse (legs 0.25 rad at 100 ms). The strategy's costs were tuned against the 10 ms single-step numerics.
- Uncertainty-gated 5-seed ensemble (tau 0.05 rad/s, arms + waist, scale 0.5) is the only configuration that broke even: imagination error at or below baseline, A/B 2 seats (72 s no fall, 94 s rung 3) and 3 falls vs baseline 2 seats (71 s rung 9, 114 s rung 6) and 3 falls; no gain, never the final rung.
- Structural reading: a sampling planner optimises over its imagined futures, so a correction is judged where its errors are selected for; open-loop fit at the planner's step is not the objective that matters.

### 1.5 Stage 3, distilling the planner into a policy (closed: imitation fails, RL is the route)

- Behaviour cloning from 93 twin bench runs (1.4 M samples, held-out command RMSE 3.4 deg vs hold 6.1) falls in 1-2 s every time in closed loop: compounding, the demonstrations do not identify the feedback law. Open-loop replay of the recorded commands stands 24 s; a constant mean command falls onto the table.
- The iCEM action is not a function of the robot state: state alone predicts the teacher label no better than its mean (2.8 vs 3.0 deg), state + previous command predicts it to 0.6 deg (warm-started plan). With plan reset before every label (K=16, Markov teacher) the label becomes noise (tick jitter 2.5 deg vs 0.4; state-only 3.4 deg vs mean 2.7).
- Predicting cmd - q makes the posture a random walk (knees lock); predicting cmd - QREF gave the first student that stands at all.
- Reported "distilled stand v1: 0.97 no push, 0.79-0.83 under 60-120 N pushes" was RETRACTED: the fall detector checked only pelvis height and torso tilt and the robot was leaning on the table with wrist pads and grippers (contact 24.0 s of 24 s, 3 kN peak). With the table 100 m away: v1 0.18 / 0.01 / 0.00 (no push / 60 N / 90 N), best-of-8 0.00. Free-standing redo (table out of the teacher model and plant, contact = failure, 36 teacher episodes, 10.7k labels, 8 seeds, closed-loop selection): 0.21 (n=33) / 0.11 (30 N) / 0.03 (60 N) / 0.01 (90 N), 0 table touches. DAgger iterations, DART noise, label smoothing, longer training and wider nets all failed.
- Verdict: the sampling planner cannot be distilled tick-by-tick by imitation. Its closed-loop behaviour should be learned by RL in the twin with the planner's costs as reward, warm-started from the student, with the Stage-0 sim-gap facts as randomisation ranges. The tools (teacher driver, free-standing evaluator with contact-as-failure, python twin plant at ~4x real time) carry over.

### 1.6 What stands on its own

The hardware facts (gripper shapes, right-ankle zero, leg kinematic gap, spotting contamination, planner step error) and the offline learning results are reproducible from the kept tools and the source recordings. The hands-off real-data residual was never tested on the robot; the twin cannot stand in for that test.

---

## 2. Chronological narrative from the git history of branch `mbrl`

All commits are by the same author, 2026-10-04 to 2026-10-05, except the final tree snapshot of 2026-10-11. The commit messages carry subject lines only; the numbers below are those the subjects and the corresponding report sections report.

### 2.1 Night 1 (2026-10-04): measurement and the first residual (Report Part 1)

| commit | subject | what it established |
|---|---|---|
| `eb03741` | dataset builder + one-step sim-gap evaluator for real H1-2 recordings | `build_dataset.py`, `simgap.py`: 493 runs, 6.1 M transitions, per-sample plant reset + 10 ms step, PD vs measured-torque models |
| `d5739d6` | research plan, sim-gap analysis, residual-model fit scripts | `RESEARCH_PLAN.md`, `analyze_simgap.py`, `residual_fit.py` |
| `a7d8a62` | sim-gap state anchoring (feet on floor at equilibrium depth), loaded-feet filter, gripper/keepaway geoms off, ankle zero offsets; param ablation | estimator height unusable (feet 32 cm above floor in one stand run); pelvis height bisected to zero vertical acceleration (contact normal 687 N vs weight 674 N); gripper/keepaway geoms non-colliding (arm error 1.52 -> 0.11 rad/s); `ablate_params.py` |
| `74927a9` | foot flattening, stationary-feet base velocity, sole-gap diagnostic; analysis on consistent subset | each foot rotated flat through its own ankle pitch/roll; base velocity from least squares over both stationary feet; "consistent" = both soles within 3 mm; 1.6 M consistent transitions |
| `4ab6f9d` | sim-gap report, residual figure, final evaluation protocol | Report Part 1: twin 2.3x worse than hold on upright arms, 6-7x legs, 14x waist leaning; MLP 2.3x one-step gain; one-step residual compounds by 250-500 ms |

### 2.2 Night 1 continued: planner step, DaD, identification, planner hook (Report Part 2)

| commit | subject | what it established |
|---|---|---|
| `30152ca` | jacobian generator, linearised multi-step residual trainer, parametric id (per-chunk loading, memory-safe) | `jacobians.py` (4860 segments, 2.1 GB), `multistep_fit.py` (diverges: Jacobian gains ~1e4), `param_id.py` (one-step objective degenerate, armature x2000) |
| `bfbb781` | nonlinear rollout engine, DaD residual training, rollout-objective identification; memory-lean fits | `rollout_engine.py`, `dad_fit.py` (DaD 5 iterations: waist 3.7x, arms 2x better at 250 ms, legs back to plant level), `param_id2.py` (damping x3, leg armature x1.3, arm armature x0.33: 45 % leg error cut) |
| `69f42f7` | lean: learned dynamics residual hook in planner rollouts (ModifyRolloutState, env H12_RESIDUAL_MLP, default off); mbrl: pre-step injection, exporter | C++ hook; `export_mlp.py` binary format `MLP1`; absent env = byte-identical planner |
| `190cfe3` | lean: per-group residual scale (default legs off); mbrl: planner-step DaD, masked eval, hook smoke test, configurable MLP size | `H12_RESIDUAL_SCALE=0,1,1`; `eval_masked.py`; `hook_smoke.py`; `MBRL_H`, `MBRL_NL` |
| `bddd31a` | small planner residual (2x64) trained with DaD at the planner step; masked evals | 2x64 arms+waist: legs 0.0417, waist 0.0261, arms 0.0133 at 250 ms (plant 0.0443 / 0.0552 / 0.0220); ~5 % planning cost vs 4x for 3x512 |
| `9266ea9` | Part 2 report (planner step, DaD, identification, hook smoke) | planner step 1 x 10 ms doubles leg error (0.174 -> 0.413); own-sim smoke 6 s OK, 4-6 deg rms trajectory difference |

### 2.3 Day 2 (2026-10-04, continued): twin A/B rounds

| commit | subject | what it established |
|---|---|---|
| `20fe8de` | honest-twin A/B result (residual off vs on, slot 37): no benefit, stand collapses with waist/arm residual | baseline 1 full pipeline, 4 dive falls; residual 0 full, 2/5 stand collapses (squat at ~57 s), one seat twisted 100 deg yaw |
| `08c9976` | own-sim verdict: twin/own-sim cannot validate a real-robot residual; plan sim2sim validation | own-sim 60 s: residual OFF pelvis z 0.996, ON z 0.224 (fall), joints diverge 6 -> 29 deg rms; a real-robot residual is pure bias on a MuJoCo twin |
| `89219f7` | data-root/source-glob/planner-step parameters; twin (sim2sim) corpus = 190 runs, 3.1 M transitions | `MBRL_DATA_ROOT`, `MBRL_SRC_GLOB`, `MBRL_NSUB`, `MBRL_DT`; `data_twin/` |
| `9324b58` | arms-only twin result | `H12_RESIDUAL_SCALE=0,0,1`: releases 42-45 s in 5/5, 0 full, 4 falls, 1 brief pad load: neutral; the stand collapse was the waist term |
| `7f45148` | sim-gap pool watchdog + 10 workers (0.8 GB each under the cgroup cap) | `MBRL_WORKERS=10`, `MBRL_TASK_TIMEOUT=900` |
| `ae9f36d` | sim2sim twin corpus results (protocol validated: zero phantom offsets on the twin); twin residual | twin flattening 0.0/0.1/0.0/0.1 deg, sole gap 1.1 mm (p90 5.0); twin-vs-planner one-step consistent upright legs 2.4x, waist 4.1x, arms 5.1x; lean 3.6x / 8.4x / 12.2x; twin residual arms+waist: legs 0.0504 -> 0.0491, waist 0.0446 -> 0.0261, arms 0.0308 -> 0.0256 |
| `871bc77` | MBRL_MAX_TILT filter (spotting-free upright subset) for residual training and rollout segments | `MBRL_MAX_TILT=12` |
| `b5c5f4c` | sim2sim A/B: correct twin residual degrades the planner; hypotheses | baseline 1/5 full, 3/5 rung 9, 3 falls, longest seat 115 s; twin residual 0/5, 0/5 (max rung 3), 5 falls, 19 s; hypotheses: distribution shift under planner noise; exploitation of a free velocity kick |
| `63756af` | planner-like command noise option for DaD rollouts | `MBRL_CMD_NOISE=0.08` |
| `eb370e4` | spotting check: upright-only residual is clean; all-data waist term is contaminated | upright-only eval (282 seg): plant waist 0.0057, all-data residual 0.0122, upright residual 0.0084 |
| `7568bb3` | noise-robust twin residual A/B (neutral); bench variance; next metric = planner imagination error | residual-on pad loaded 5/5, one 63 s seat, max rung 3, 5 falls; baseline pad 2/5, rung 9 once (28 s seat), 5 falls; pooled baselines 15 runs: 2 pipelines, 7 rung 9, 12 falls |
| `86a4658` | planner imagination-error scorer (iCEM dump vs twin recording); dump option in twin_ab.sh | `imagination_error.py`, `DUMP=1`, `H12_ROLLOUT_DUMP`, `H12_ROLLOUT_DUMP_EVERY=20` |

### 2.4 Day 2 to 3 (2026-10-05): feature parity, authority, parametric, ensemble, closing

| commit | subject | what it established |
|---|---|---|
| `577a496` | planner imagination error: residual 3-10x worse inside the planner -> feature mismatch suspected | 100 ms base legs 0.009-0.015 vs residual 0.044-0.049; suspects: node feed-forward torque in features, node command transforms |
| `f2f3c4b` | MBRL_ZERO_TAUFF: planner-available features only (no node feed-forward) | zeroing tau_ff offline turns the waist correction harmful (0.045 -> 0.053 vs plant) until retrained |
| `55e7ecb` | planner-feature residual + phase-matched imagination error; C++ feature debug dump | `twinnoff`: offline arms 0.031 -> 0.027, waist 0.045 -> 0.025; stand-phase (t0 < 45 s) imagination error: residual slightly better legs, 40-50 % worse waist/arms; `H12_RESIDUAL_DEBUG` |
| `5f0a751` | C++/Python feature parity exact; residual authority note | 200 hook calls: features < 1e-5, output 3e-6; first call at rest already outputs 1.1 rad/s (the clip) |
| `3f6331e` | reduced-authority residual neutral on imagination error, both runs reach rung 9; 5-pair A/B launched | clip x0.3, scale 0.5: 0.014/0.012/0.028 and 0.017/0.007/0.029 (legs/waist/arms), inside baseline spread |
| `bd9f3da` | reduced-authority 5-pair A/B and the overall twin verdict | baseline 2 full / 3 rung 9 / 2 falls; low-authority 0 / 0 (one 115 s seat at rung 3) / 4 falls; pooled 20 baseline vs 22 residual runs: 4 vs 1 full, 10 vs 2 rung 9, 14 vs 18 falls |
| `b839e01` | damping x3 twin A/B (neutral; real-identified value is bias on the twin) | `twin_damp_ab.sh`: baseline 1 full / 2 rung 9 / 4 falls; damping 30: 0 / 1 (+ 114 s seat at rung 3) / 4 |
| `464cf50` | twin-corpus identification (damping x3 again = planner step artefact); driver supports armature | `out_twin/param_id2.json`: damping x3.0, leg armature x3.0, arm armature x1.1, torque 1.1 / 0.6, hip-pitch offsets -3 deg; `ARMAT` in `twin_damp_ab.sh` |
| `aa54d7c` | uncertainty-gated residual ensemble (python engine + C++ hook, not yet built) | 5 seeds `ens1..ens5`, gate tau on ensemble disagreement, `MBRL_TAU` |
| `d3f1f44` | twin-identified damping+armature A/B worse; open-loop fit is the wrong objective for the planner | damping 30 + armature 0.3: 0 full, max rung 5, 5 falls (four before 57 s) vs baseline 1 full, 1 rung 9, 1 rung 6, 4 falls |
| `c033fc6` | gated ensemble breaks even on the twin; closing assessment | tau 0.05: offline waist 0.045 -> 0.029, arms 0.031 -> 0.028; imagination 0.021/0.009/0.026 and 0.014/0.005/0.027; A/B 2 seats + 3 falls + 1 stand collapse vs 2 seats + 3 falls |

### 2.5 Day 3 (2026-10-05): stage 3 distillation, substeps, retraction, closing

| commit | subject | what it established |
|---|---|---|
| `5b9cbe2` | stage 3: planner distillation (BC from twin recordings fails by compounding; DAgger with own-sim teacher), planner H12_ROLLOUT_SUBSTEPS | `bc_*.py`, `dagger_stand.py`; BC clone falls in 1-2 s; open-loop replay stands 24 s |
| `15d928c` | stage 3: DAgger diagnostics (non-Markov teacher, absolute actions), report sec 30 | state-only linear fit 2.8 vs mean 3.0 deg; state + prev cmd 0.6 deg; cmd - q random walk; cmd - QREF stands 13 % / 18 % |
| `3560149` | stage 3: distilled stand policy v1 (teacher-driven K=8 labels) stands 97 %, 79-83 % under 60-120 N pushes on the twin | later RETRACTED (table-assisted) |
| `568101f` | planner substeps A/B on the twin negative (locked knees, squat collapse, 10x imagination error); report sec 31 | 3 pairs: baseline r1 full ladder, r2 fall 46 s, r3 fall 66 s; n=2: 3/3 locked knees, 2 deep squats (knee 126 deg, pelvis z 0.59) |
| `4e18521` | stage 3: teacher reference, selection, negative variants; report sec 30.4 | teacher K=2 1.00 / 0.55 / 0.55 (0 / 90 / 120 N, 15 s); best-of-8 0.87 / 0.65; ensemble 1.00 / 0.63; DART 0.29-0.58; third collection 0.42; smoothing 0.61-0.67 (all later retracted as table-assisted) |
| `6c204c0` | imagination error for the substep A/B | `out/imagination_error.json`: sub2 runs 100 ms legs 0.11-0.26 vs baseline 0.008-0.039 |
| `5e7fca1` | stage 3: RETRACT stand-policy numbers (robot was leaning on the table); free-standing eval 0.18/0.00 | contact 24.0 s of 24 s, 3 kN peak; v1 0.18 / 0.01 / 0.00; best-of-8 0.00; 0/165 init states touch the table |
| `f004e70` | stage 3: free-standing redo (0.21/0.01), Markov-teacher labels are noise; imitation of the sampling planner closed; RL is the route | 10,723 labels from 36 teacher episodes; 0.21 / 0.11 / 0.03 / 0.01 at 0 / 30 / 60 / 90 N; Markov teacher: jitter 2.5 deg, state-only 3.4 vs mean 2.7 |
| `37645a5` (2026-10-11) | estimator node v3/v4 and Lean_H12_Magpie tweaks as left on 2026-10-11 | tree snapshot before the data deletion |

---

## 3. RESEARCH_PLAN.md (verbatim, 2026-10-04; headings demoted one level)

## Model-based RL for the H1-2 lean-and-brace controller: research plan

Written 2026-10-04 on branch `mbrl`. Grounded in live documentation fetched the same day
(MuJoCo, Unitree, NVIDIA; URLs at the end) and in what already exists in this tree.

### 1. Where we stand

The deployed controller is MuJoCo MPC with sampling (iCEM) over a *hand-built* MuJoCo plant,
`Lean_H12_Magpie.xml`. That is already the "planning" half of model-based RL. What is missing,
and what makes this a research contribution rather than engineering, is the **learning** half:

1. a model that is corrected from real data (the twin is known to be off: forward-mass offset
   ~8 cm, ankle-zero 4-8 deg, actuator chatter, pad/slab contact creep, heat-dependent torque);
2. a policy or value function learned *with* that model so the robot does not need 20-30
   plans/s of CPU at run time, and so the dive, graze and brace can be committed without an
   operator hand.

Everything below is sequenced so each stage produces a publishable measurement on its own.

### 2. Thesis

> A sampling MPC on a physics plant plus a small learned residual, trained from a few hours
> of real-robot logs, closes most of the sim-to-real gap for whole-body contact tasks, and the
> same corrected plant is good enough to distil an MPC policy that runs without online planning.

Falsifiable at every stage: if the residual does not cut one-step and 0.5 s rollout error on
held-out runs, stage 1 fails and we stop there.

### 3. Stages

#### Stage 0: data and sim-gap measurement (started tonight, see `REPORT_*.md`)

Already in hand: 493 real-robot recordings, 6.1 M transitions at 100 Hz, each with joint
q/dq/tau_est, the full PD command (target, kp, kd, tau_ff), torso IMU quat/gyro/accel and the
estimator base pose. Tools: `build_dataset.py`, `simgap.py`, `analyze_simgap.py`.

Metric: one-step (10 ms) prediction error of the plant vs the recording, against the trivial
"hold velocity" predictor, split arms / legs / waist and upright / leaning. Two torque models:
PD-reconstructed command (the twin as deployed) and measured tau_est (isolates rigid-body +
contact error from actuator error).

#### Stage 1: learned residual dynamics (started tonight, `residual_fit.py`)

Target: residual of the plant's next joint velocity. Inputs: only planner-known quantities
(q, dq, commanded torque, gains, gravity direction, pelvis rate). Models: ridge then a 3x512 MLP.
Judge on held-out *runs*, not held-out samples, with one-step RMSE and closed-loop 0.1 / 0.25 /
0.5 s rollouts (the MPC horizon is 1.0 s). Deliverable: a table "plant / plant+ridge /
plant+MLP / hold" per joint group and per strategy.

Second pass, if the residual helps: **parametric** correction first (armature, frictionloss,
damping, per-joint torque scale, pelvis mass offset) with the MuJoCo 3.5+ `mujoco.sysid`
toolbox. The Unitree mjlab community found the shipped armature/friction defaults give 30 %
overshoot in sim vs 2 % on the real R1 (`unitree_rl_mjlab#51`); an H1-2 equivalent is a clean
result on its own. Parametric fixes transfer into the planner XML directly; the MLP needs a
plugin or a post-step hook in `mjpc`.

#### Stage 2: residual inside the planner

Two ways to use the learned residual at 20-30 plans/s:
- **Online**: a per-step velocity correction applied inside the rollout (`mjpc` rollouts call
  `mj_step`; a `user` callback or a tiny actuator plugin can add it). Cost: one MLP eval per
  rollout step; at 27 x 3 x 512 this is ~1 M flops, fine on CPU for 64 samples x 100 steps.
- **Offline**: distil the residual into plant parameters (Stage 1 second pass) and ship XML.

Measure: twin bench parity (does the honest twin with the corrected plant reproduce the real
seat/collapse statistics), then real-robot A/B on strat 25 with the fixed window protocol.

#### Stage 3: policy from the model (the "RL" in MBRL)

Options, in order of risk:
1. **MPC distillation (DAgger)**: run iCEM on the corrected plant from thousands of perturbed
   starts, record (obs, action), train an MLP policy, iterate with on-policy relabelling. This is
   the MuZero/TD-MPC-lite route that keeps the planner as the teacher.
2. **Dyna-style model-based PPO**: model-free PPO on the corrected plant with the learned
   residual in the loop, GPU batched. Needs MJX/Warp; **blocker**: MuJoCo Warp does not support
   the `noslip` solver our `*_real_ns` plants use, and has no autodiff. Use the pyramidal plant
   for training, verify on the noslip twin.
3. **Model-based value expansion** on top of 1 or 2.

Measure: success rate on the lean/brace ladder in the honest twin with no operator assist, then
on hardware; CPU load at run time (the stand needs < 25 plans/s today).

### 4. What the ecosystem offers (fetched 2026-10-04)

| Need | Status | Source |
|---|---|---|
| GPU batched MuJoCo | MuJoCo 3.14 / `mujoco-warp` 3.14; Warp officially released in 3.5 (Feb 2026); no autodiff; no `noslip`, no implicitfast midpoint | mujoco changelog, mujoco_warp README |
| System ID | `mujoco.sysid` Python toolbox since 3.5; maintainers advise in-air leg ID for floating-base humanoids | mujoco discussion #3238 |
| Humanoid RL envs | Playground: H1 and G1 joystick envs, **no H1-2**; Isaac Lab: H1 and G1, no H1-2; mjlab: `Unitree-H1_2-Flat` 27 DoF with deploy yaml | mujoco_playground, IsaacLab, unitree_rl_mjlab |
| Learned dynamics in MJPC | None; "residual" there means cost residual. Direct/batch estimators exist | mujoco_mpc README, PR #402 |
| World models | GR00T Dreams and Cosmos 3 are pixel-space, data generators or action-conditioned video; not a state-space model for control | GR00T-Dreams, Cosmos docs |
| Real actuator ID recipe | mjlab#51: step + sine sweeps, fit armature/frictionloss per joint group | unitree_rl_mjlab#51 |
| Low-level state | `rt/lowstate` hg: q, dq, ddq, tau_est, temps, IMU; no foot force; 500 Hz in SDK examples | H1_developer docs |

### 5. Risks

- Base pose comes from our own estimator; leg one-step error is contaminated by it. Arms are
  the clean signal. Report both.
- Table height varied per run while the plant has one table; brace-phase error partly reflects
  that. Stage 1 features exclude the table on purpose.
- Heat: motor torque constant drifts with winding temperature (logged since 09-03). A residual
  that ignores temperature will show run-to-run bias; check per-run residual bias vs temp.
- No hardware tonight. Everything here is offline; nothing in this branch changes the live tree.

### 6. Sources

- https://mujoco.readthedocs.io/en/stable/changelog.html
- https://github.com/google-deepmind/mujoco_warp
- https://mujoco.readthedocs.io/en/stable/mjx.html
- https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/README.md
- https://github.com/google-deepmind/mujoco/discussions/3238
- https://github.com/google-deepmind/mujoco_playground
- https://github.com/google-deepmind/mujoco_mpc/issues/389
- https://github.com/unitreerobotics/unitree_rl_mjlab (and issues #51, #12, #25)
- https://github.com/unitreerobotics/unitree_rl_gym , https://github.com/unitreerobotics/unitree_rl_lab
- https://github.com/unitreerobotics/unitree_mujoco , https://github.com/unitreerobotics/unitree_ros/issues/130
- https://support.unitree.com/home/en/H1_developer (Basic Services Interface, joint order)
- https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html
- https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html
- https://github.com/NVIDIA/GR00T-Dreams , https://arxiv.org/abs/2505.12705
- https://github.com/NVIDIA/Cosmos/blob/main/docs/reference/models.md
- https://github.com/isaac-sim/IsaacLab/discussions/7749

---

## 4. REPORT_2026-10-04.md (verbatim, Parts 1 and 2 plus stage 3 sections 30-31; headings demoted one level; figure links point at out/*.png which may no longer exist)

## Sim-to-real dynamics gap of the H1-2 planner plant, and a first learned residual

Date 2026-10-04, branch `mbrl`, workspace `mujoco_mpc/mujoco_mpc/mbrl/`. Offline only; nothing here touched
the live tree or the robot. Plan and sources: `RESEARCH_PLAN.md`.

### 1. Question

How far is the deployed planner plant (`Lean_H12_Magpie.xml`, the model iCEM imagines with) from the real
H1-2, measured on the robot's own recordings, and how much of that gap can a small learned residual close?

### 2. Data

493 real-robot recordings in `~/Desktop/h12/logs` (June to September 2026, every strategy family), 100 Hz,
6.1 M transitions with joint position, velocity, measured torque, the full PD command (target, kp, kd,
feed-forward), torso IMU and estimator base pose. Split by run: 80 / 10 / 10 train / val / test.

Filters. 4.0 M transitions have both feet loaded (knee or ankle torque above 5 Nm on each leg; the rest are
gantry-hung or limp-leg recordings). 1.6 M are "kinematically consistent": after the state reset below both
plant soles lie within 3 mm of the floor. All headline numbers use that subset; the rest is reported too.

### 3. Method

For every sample the plant is reset to the recorded state and stepped 10 ms (5 x 2 ms), then compared to the
recorded next sample. Two torque models: the PD command recomputed every substep (the twin as deployed) and
the measured torque held constant (rigid-body and contact model alone). The scale bar is the trivial
predictor "velocity stays the same".

The state reset is the part that took the night to get right, and each fix is a finding about the plant:

1. Pelvis orientation is the torso IMU quaternion composed back through the waist joint (the IMU site is on
   `torso_link`, not the pelvis).
2. The estimator height is not usable (one stand run had the feet 32 cm above the floor). Each foot is rotated
   flat onto the floor through its own ankle pitch and roll, then the pelvis height is bisected to the contact
   depth at which the vertical acceleration under the measured torques is zero. Contact normal force then
   matches the weight (687 N vs 674 N).
3. Base linear and angular velocity come from the stationary-feet constraint (least squares over both feet),
   not from the gyro and estimator, which conflict with the leg velocities and inject a reset impulse.
4. The table is moved away and the free object parked, since the real table height varied per run.
5. The planner-only keepaway spheres and the gripper and wrist collision shapes are made non-colliding: with
   the recorded arm poses they self-collide where the real hands do not. Arm error 1.52 -> 0.11 rad/s.

Scripts: `build_dataset.py`, `simgap.py`, `analyze_simgap.py`, `ablate_params.py`, `residual_fit.py`.

### 4. Results

#### 4.1 One-step error of the plant (RMSE of the next joint velocity, rad/s, 10 ms)

| phase | group | hold velocity | plant + PD command (twin) | plant + measured torque | twin / hold |
|---|---|---|---|---|---|
| all | legs | 0.026 | 0.329 | 0.351 | 12.7 |
| all | waist | 0.013 | 0.193 | 0.192 | 14.6 |
| all | arms | 0.020 | 0.082 | 0.093 | 4.1 |
| upright | legs | 0.025 | 0.155 | 0.163 | 6.3 |
| upright | waist | 0.011 | 0.043 | 0.035 | 3.8 |
| upright | arms | 0.019 | 0.039 | 0.043 | 2.0 |
| lean | legs | 0.029 | 0.202 | 0.201 | 7.1 |
| lean | waist | 0.015 | 0.207 | 0.198 | 13.8 |
| lean | arms | 0.022 | 0.096 | 0.110 | 4.5 |

![per joint](out/fig_simgap_per_joint.png)
![vs tilt](out/fig_simgap_vs_tilt.png)

- Upright arms are the only place the twin is close to the trivial predictor (2x). Legs are 6 to 7x off,
  the waist 4x upright and 14x leaning. Error grows steeply with torso tilt for arms and waist.
- The PD-command and measured-torque columns agree within 10 %: the actuator model is not where the gap is.
  Reconstructed PD torque matches measured torque to 0.1 to 2 Nm RMSE per joint, with a +0.7 Nm bias on both
  ankle pitches.
- On the full loaded-feet set (not only consistent samples) the legs are 12.7x off: one foot floats.

#### 4.2 Plant artifacts found on the way

| artifact | evidence | status |
|---|---|---|
| gripper / wrist collision shapes self-collide | arm one-step 1.52 -> 0.11 rad/s when non-colliding | disabled in this study; needs re-fit of the gripper geometry in the plant |
| right ankle pitch zero | best offsets per day: right +4, +4, +5, +8 deg; left +1, +1, +1, +2 (`out/ankle_offset_sweep.json`); per-sample flattening median right +3.8 deg, left -0.2 (upright +0.6) | matches the known 4 to 8 deg right-ankle zero error; the plant could carry a per-session calibration |
| left vs right sole height | with recorded q the plant soles differ by median 3.9 mm, p90 10.9 mm, p99 19 mm; up to 12 mm during leans, correlation -0.9 with knee asymmetry in two runs | a knee or hip zero / link-length error; a near-straight leg cannot close it through the knee (ill-conditioned), so it must be identified, not patched |
| joint damping, armature, friction loss | lowering any of them makes the error worse; damping 30 instead of 10 lowers it (legs 0.29 -> 0.20) | the real joints behave stiffer and more damped than the plant; a per-joint fit belongs in Stage 1b |

#### 4.3 Learned residual (planner-known inputs only: q, dq, commanded torque, cmd - q, gains, gravity direction, pelvis rate)

Target: recorded next velocity minus the plant's prediction. Ridge regression and a 3 x 512 SiLU MLP,
trained on 1.25 M consistent transitions from the train runs, evaluated on the held-out test runs.

One-step RMSE on test runs (rad/s):

| model | legs | waist | arms | all |
|---|---|---|---|---|
| hold velocity | 0.028 | 0.022 | 0.021 | 0.024 |
| plant (twin) | 0.149 | 0.110 | 0.063 | 0.112 |
| plant + ridge | 0.106 | 0.064 | 0.033 | 0.076 |
| plant + MLP | 0.066 | 0.048 | 0.026 | 0.049 |

The MLP cuts the twin's one-step error 2.3x overall and brings the arms to within 1.25x of the trivial
predictor. The gain holds on every strategy family (3x on 10 of 12), and fails only on `stand6supp`, a June
supported-stand recording outside the training distribution.

Closed-loop rollouts on test runs (recorded PD commands replayed on the plant, residual applied as a velocity
correction after every 10 ms step, clipped at the 99.9th percentile of training targets; error = joint
displacement RMSE in rad; "diverged" = any joint error above 0.5 rad):

upright starts, 238 segments

| model | diverged | 100 ms legs | 100 ms arms | 250 ms legs | 250 ms arms | 500 ms legs | 500 ms arms |
|---|---|---|---|---|---|---|---|
| plant (twin) | 3 | 0.0131 | 0.0054 | 0.0241 | 0.0123 | 0.0331 | 0.0187 |
| plant + ridge | 5 | 0.0162 | 0.0040 | 0.0676 | 0.0174 | 0.1427 | 0.0717 |
| plant + MLP | 3 | 0.0146 | 0.0037 | 0.0343 | 0.0098 | 0.0525 | 0.0178 |

lean starts, 101 segments

| model | diverged | 100 ms legs | 100 ms arms | 250 ms legs | 250 ms arms | 500 ms legs | 500 ms arms |
|---|---|---|---|---|---|---|---|
| plant (twin) | 0 | 0.0166 | 0.0134 | 0.0406 | 0.0297 | 0.0632 | 0.0423 |
| plant + ridge | 23 | 0.0174 | 0.0064 | 0.0594 | 0.0265 | 0.1094 | 0.0608 |
| plant + MLP | 1 | 0.0191 | 0.0071 | 0.0534 | 0.0209 | 0.0790 | 0.0389 |

![residual](out/fig_residual.png)

- Over 100 ms the MLP halves arm and waist error from upright starts. By 250 to 500 ms it compounds and is
  worse than the plain plant, and it never helps the legs. Ridge diverges in 28 of 339 segments.
- This is the expected failure of a one-step loss used in closed loop. It is the reason the next model is
  trained with a multi-step rollout loss, or injected as a generalized force rather than a velocity kick.

### 5. What this means for the MBRL thesis

The deployed planner imagines joint accelerations an order of magnitude off the robot's in the regimes that
matter (leaning, loaded legs), and most of the gap is geometry and calibration (gripper shapes, ankle zero,
leg kinematics), not actuator dynamics. A small residual learned from a few hours of logs recovers a 2.3x
one-step improvement with no hardware time, but one-step training does not survive a 0.5 s horizon, let alone
the planner's 1 s. Both halves of the thesis statement are therefore live: the model can be corrected from
data, and the correction has to be trained on the horizon it will be used at.

### 6. Next (in order)

1. Stage 1b: multi-step residual loss (unroll 25 to 50 plant steps with the residual in the loop, backprop
   through the residual only), and a residual-as-force variant.
2. Parametric identification with `mujoco.sysid` of: knee and ankle zero offsets per session, sole geometry,
   per-joint damping; then re-run `simgap.py` and see how much residual is left.
3. Re-fit the gripper collision shapes in the plant from the real hand geometry (planning needs them back).
4. Stage 2: residual inside `mjpc` rollouts; twin parity gate; then a real-robot A/B on strat 25.

### 7. Reproduce

```
cd ~/Desktop/h12/mujoco_mpc/mujoco_mpc/mbrl
.venv/bin/python build_dataset.py            # ~2 min, data/runs + manifest
MBRL_WORKERS=20 .venv/bin/python simgap.py   # ~20 min, data/simgap
.venv/bin/python analyze_simgap.py           # out/simgap_summary.json + figs
.venv/bin/python ablate_params.py            # out/ablate_params.json
.venv/bin/python residual_fit.py             # out/residual_summary.json, residual_mlp.pt
.venv/bin/python fig_residual.py
```

---

## Part 2 (same day): multi-step residual, identification, and the planner hook

### 8. The planner's step is part of the gap

MJPC rollouts take one 10 ms `implicitfast` step per planning step (`agent_timestep`), not five 2 ms substeps.
Re-running the one-step test with the planner's own step (4 runs, consistent samples):

| step | legs | waist | arms |
|---|---|---|---|
| 5 x 2 ms (Part 1 protocol) | 0.174 | 0.103 | 0.064 |
| 2 x 5 ms | 0.275 | 0.117 | 0.064 |
| 1 x 10 ms (planner) | 0.413 | 0.138 | 0.067 |

The planner's integration step alone more than doubles its leg error through contact. Arms are unaffected.
Everything below is therefore trained and evaluated at the planner's 1 x 10 ms step.

### 9. Two negative results worth keeping

- **Linearised multi-step training fails.** Finite-difference Jacobians of the 10 ms macro step along plant
  rollouts (`jacobians.py`, 4860 segments, 2.1 GB) have gains near 10^4 in the contact directions, and the
  25-step linear recursion diverges from the first iterate (`multistep_fit.py`). Linearising through stiff foot
  contact is not a usable training signal.
- **One-step error is a degenerate identification objective.** Nelder-Mead on damping/armature/torque scales
  against one-step velocity error converged to armature x2000: an immovable robot matches the trivial
  hold-velocity predictor. This also explains why "more damping looks better" in Part 1. Identification needs a
  horizon objective (section 11).

### 10. Residual trained on its own rollouts (Data-as-Demonstrator)

`rollout_engine.py` runs parallel nonlinear rollouts with a pluggable numpy residual; `dad_fit.py` iterates:
roll the plant plus current residual out on 3000 training segments, record the one-step target at every visited
state, aggregate, fine-tune. Validation = closed-loop displacement error on held-out segments (rad).

Post-step injection, 5 x 2 ms (Part 1 protocol), 221 consistent validation segments:

| model | 250 ms legs | 250 ms waist | 250 ms arms |
|---|---|---|---|
| plant | 0.0385 | 0.0564 | 0.0218 |
| one-step MLP (Part 1) | 0.0527 | 0.0467 | 0.0137 |
| DaD, 5 iterations | 0.0387 | 0.0152 | 0.0100 |

DaD removes the leg compounding (legs back to the plant's level) and halves arm error, waist 3.7x.

Pre-step injection at the planner's 1 x 10 ms step (what the planner hook can do), 352 validation segments, with
a per-group mask (`eval_masked.py`, `out/masked_dadpre10*.json`):

| configuration | 250 ms legs | 250 ms waist | 250 ms arms |
|---|---|---|---|
| plant | 0.0443 | 0.0552 | 0.0220 |
| 3x512 residual, all joints | 0.0478 | 0.0318 | 0.0153 |
| 3x512 residual, arms + waist only | 0.0417 | 0.0333 | 0.0155 |
| 2x64 residual, arms + waist only | 0.0417 | 0.0261 | 0.0133 |

A pre-step velocity kick on a planted foot is absorbed by the contact solver and hurts the legs, so the planner
configuration is arms + waist only. The 2x64 network is as good as the 3x512 one.

### 11. Parametric identification with a 250 ms rollout objective (`param_id2.py`)

Objective: plant-only 250 ms displacement error (legs + arms) on a fixed set of consistent training segments;
log-scale parameters bounded to [1/3, 3]; held-out test on 245 fixed segments.

| plant | 100 ms legs | 250 ms legs | 250 ms waist | 250 ms arms |
|---|---|---|---|---|
| nominal | 0.0181 | 0.0466 | 0.0367 | 0.0175 |
| fitted: damping x3.0 (both), leg armature x1.3, arm armature x0.33, torque x1.0 / x0.91 | 0.0106 | 0.0256 | 0.0275 | 0.0152 |

A 45 % cut in leg error from four scalars; damping sits at the upper bound, so the real joints are at least 3x
more damped than the plant's 10 Nm s/rad. Joint zero offsets on hip pitch and knee come out below 0.4 deg once
the feet are flattened, so the 12 mm sole gap of Part 1 is not a knee or hip zero error; sole geometry or link
lengths remain the candidates. These scales are a candidate planner XML change, to be gated on the twin.

### 12. Planner hook (Stage 2), built and smoke-tested

- `mjpc/tasks/humanoid_bench/lean/residual_mlp.{h,cc}`: binary weight loader, SiLU MLP forward, per-thread scratch,
  feature construction from planner quantities (q, dq, actuator force, ctrl - q, gains, gravity in the IMU frame,
  pelvis rate).
- `lean::ModifyRolloutState` applies the correction pre-step inside planner rollouts only (never on the real-robot
  path). Enabled by `H12_RESIDUAL_MLP=<file>` (from `export_mlp.py`), per-group scale `H12_RESIDUAL_SCALE=legs,waist,arms`
  with default `0,1,1`. Absent env = byte-identical behaviour.
- Own-sim smoke (`hook_smoke.py`, agent_server, strategy 25, 6 s): loads, plans, states finite, robot standing;
  trajectory differs 4 to 6 deg rms from the residual-free run.
- Cost: the 3x512 network made planning 4x slower (160 s vs 38 s wall); the 2x64 network costs ~5 % (41 s vs 39 s).
  `out/residual_dadpre10s.bin` is the deployable file.

### 13. Status and next

Done offline: measurement, residual, identification, planner integration, smoke test. Not done: twin bench with
the residual and with the fitted damping/armature (needs the robot NIC down, wire-check), then a real A/B on
strat 25 with the fixed-window protocol. Both are user-gated.

Reproduce Part 2:
```
MBRL_WORKERS=8 .venv/bin/python dad_fit.py                                    # study protocol, post-step
MBRL_H=64 MBRL_NL=2 MBRL_NSUB=1 MBRL_DT=0.01 MBRL_INJECT=pre MBRL_TAG=dadpre10s .venv/bin/python dad_fit.py
.venv/bin/python export_mlp.py dadpre10s; MBRL_H=64 MBRL_NL=2 .venv/bin/python eval_masked.py dadpre10s
.venv/bin/python param_id2.py
../../../h1_mujoco/.venv/bin/python hook_smoke.py ~/Desktop/h12/mujoco_mpc/monitor/proto out/residual_dadpre10s.bin 6
```
Every heavy job runs under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`, one at a time.

### 14. Honest-twin A/B (robot off): residual OFF vs ON, slot 37 (strat 25 + 40 s stand sustain), B3, d37, `_real_ns`

Interleaved, 5 pairs, 180 s, proven honest protocol (rope 0.8, chunked payout, honest-gate release, no brace
assist), `mbrl/twin_ab.sh`, scored with `logs/twin_parity_0925/twin_score.py` (`logs/grid37_twin/d37/ab37_scores.csv`).
Note: plain strat 25 (slot 25) collapses during payout on this twin before any lean rung (base_r1 of the first,
aborted series), which is why slot 37 is the comparator.

| run | release s | pad load s | seat s | rung | done | fall s |
|---|---|---|---|---|---|---|
| base r1 | 42 | - | 0 | 3/9 | 0 | 68 |
| base r2 | 42 | - | 0 | 2/9 | 0 | 46 |
| base r3 | 42 | - | 0 | 2/9 | 0 | 45 |
| base r4 | 42 | 66 | 99 | 9/9 | 1 | - |
| base r5 | 42 | - | 0 | 3/9 | 0 | 64 |
| res r1 | never | - | 0 | 2/9 | 0 | stand collapse 57 s (squat, no fall) |
| res r2 | 78 | - | 0 | 2/9 | 0 | 101 |
| res r3 | never | - | 0 | 2/9 | 0 | stand collapse ~57 s |
| res r4 | 42 | 75 | 98 | 3/9 | 0 | 174 (seat twisted 100 deg yaw) |
| res r5 | 42 | - | 0 | 2/9 | 0 | 59 |

Baseline: 1 full pipeline, 4 pre-contact dive falls (today's twin is harsher than the 09-25 record of 0/4 dive
falls). Residual (arms + waist, 2x64, pre-step): 0 full pipelines, 2 of 5 runs sank into a squat while free-standing
before the honest gate ever released, a failure the baseline never shows; the one seat it reached was twisted.
Verdict: **the residual trained on recordings of the real robot does not transfer as a benefit to the twin's
planner, and the waist/arm pre-step corrections destabilise the planner's stand.** Two readings are possible and
the next two tests separate them: (a) the residual encodes real-robot-vs-plant differences that the twin plant does
not have (the twin IS a MuJoCo plant close to the planner's), so on the twin it is pure bias; (b) the pre-step
velocity kick itself perturbs the planner's stand. Test (a) needs the real robot. Test (b): own-sim 60 s with the
residual on/off, and an arms-only twin arm (`H12_RESIDUAL_SCALE=0,0,1`).

### 15. Own-sim 60 s, residual on vs off: the twin is the wrong oracle for a real-robot residual

`hook_smoke.py` for 60 s of own-sim (agent_server, strategy 25): residual OFF stands (pelvis z 0.996 at 60 s);
residual ON (arms + waist) falls (z 0.224), joint trajectories diverging 6 -> 29 deg rms. This is expected by
construction, not a bug: in own-sim and on the twin the ground truth IS a MuJoCo plant almost identical to the
planner's, so a residual that encodes real-robot-minus-plant differences is pure bias there, and a biased planner
imagination destabilises the stand. Consequences:
1. Sections 14 and 15 say nothing about whether the residual helps on the real robot; the offline evidence
   (section 10: arms/waist 250 ms error halved on held-out real recordings) is the only evidence about that.
2. The method (hook + DaD) can still be validated sim-to-sim: learn a residual from TWIN recordings against the
   planner plant (the twin differs by cone, noslip, PD-per-step, 5 ms step) and A/B it on the twin, where it is
   the correct correction. Today's A/B left 15 twin recordings (~45 min at 100 Hz) for exactly that.
3. A real-robot test of the real residual needs a safer staging than "on at scale 1": scale 0.25 -> 0.5 -> 1 with
   the fixed-window protocol, arms only first. User-gated.

Arms-only follow-up (`H12_RESIDUAL_SCALE=0,0,1`, 5 runs, same protocol): releases 42-45 s in 5/5 (no stand
collapse), 0 full pipelines, 4 falls, 1 brief pad load. The stand collapse of section 14 was the waist term; arms
only is neutral on the twin, as expected for a correction that is bias there.

### 16. Sim-to-sim: the same pipeline on the twin corpus (190 twin recordings, 3.1 M transitions, `data_twin/`, `out_twin/`)

Protocol validation. On the twin the foot-flattening corrections are 0.0 / 0.1 / 0.0 / 0.1 deg (L pitch, L roll,
R pitch, R roll) and the sole gap median is 1.1 mm (p90 5.0). On the real robot they were -0.2 / -0.8 / +3.8 / +1.0 deg
and 3.9 mm (p90 10.9). The reset protocol invents nothing on a clean plant, so the real robot's right-ankle zero
offset and leg kinematic mismatch are hardware facts.

Twin-plant vs planner-plant one-step (planner's 1 x 10 ms step; the twin steps 5 ms with PD per step, pyramidal
cone, noslip): consistent upright legs 2.4x hold, waist 4.1x, arms 5.1x; leaning 3.6x / 8.4x / 12.2x. The twin's
own joint-velocity signal is noisier than the real robot's (hold 0.095 rad/s on legs vs 0.026 real).

Twin-trained residual (DaD, 2x64, pre-step at 1 x 10 ms), 400 twin validation segments, 250 ms displacement RMSE:

| configuration | legs | waist | arms |
|---|---|---|---|
| plant | 0.0504 | 0.0446 | 0.0308 |
| twin residual, arms + waist | 0.0491 | 0.0261 | 0.0256 |
| twin residual, arms only | 0.0489 | 0.0550 | 0.0240 |

This residual is the correct correction for the twin by construction. Its twin A/B (section 17) is therefore the
test of the method: hook, injection point, DaD training, planner behaviour.

### 17. Sim-to-sim twin A/B: a correct residual still degrades the planner

Same protocol as section 14 (slot 37, B3, d37, `_real_ns`, 5 interleaved pairs, 180 s), residual = the twin-trained
2x64 net of section 16, arms + waist, pre-step (`logs/grid37_twin/d37/ab37_twinres_scores.csv`).

| arm | full pipeline | reached rung 9 | falls | longest seat |
|---|---|---|---|---|
| baseline | 1 / 5 | 3 / 5 | 3 | 115 s |
| twin residual, arms + waist | 0 / 5 | 0 / 5 (max rung 3) | 5 | 19 s |

The residual that is correct for the twin by construction, and that lowers the planner plant's 250 ms open-loop
error on twin data by 20-40 %, makes the closed-loop planner worse. Better short-horizon prediction on recorded
trajectories does not transfer to better planning. Two mechanisms, both testable offline:
1. Distribution shift under the planner: the residual was trained on recorded commands, but the sampling planner
   perturbs commands with its exploration noise inside every rollout; the correction is evaluated off its training
   distribution in exactly the rollouts that decide the plan. Fix: train DaD with planner-like command noise.
2. Exploitation: a pre-step velocity kick is motion the optimiser gets without torque; if the residual says "the
   waist drifts forward", the planner can lean on it. Fix: inject as a generalized force (through the inertia matrix
   and the contact solver) instead of a velocity kick, or evaluate the cost on the uncorrected step.
Next: (1) first, then (2) if needed; each is a twin DaD retrain (~5 min) plus a 35 min twin A/B.

### 18. Spotting check: residual trained on hands-off upright data only (`MBRL_MAX_TILT=12`)

The operator's hand (dive damping, catches) is an external force in neither the command nor the plant. Upright
phases (torso tilt < 12 deg) were hands-off. Same DaD recipe at the planner step, 2x64, pre-step; 250 ms
displacement RMSE (rad), held-out real segments:

| evaluated on | model | legs | waist | arms |
|---|---|---|---|---|
| upright only (282 seg) | plant | 0.0374 | 0.0057 | 0.0101 |
| upright only | all-data residual, arms+waist | 0.0372 | 0.0122 | 0.0075 |
| upright only | upright residual, all joints | 0.0336 | 0.0084 | 0.0070 |
| upright only | upright residual, arms only | 0.0369 | 0.0069 | 0.0070 |
| all (352 seg) | plant | 0.0443 | 0.0552 | 0.0220 |
| all | all-data residual, arms+waist | 0.0417 | 0.0261 | 0.0133 |
| all | upright residual, arms+waist | 0.0419 | 0.0243 | 0.0170 |

Findings: (1) upright, the plant's waist is already good and the all-data residual makes it worse: the waist
correction learned from lean phases is contaminated by spotting; (2) the upright-only residual is clean (arms -30 %,
legs -10 % upright, waist unchanged) and still cuts lean-phase waist/arm error, so part of the lean-phase gap is
real dynamics rather than the hand; (3) any residual meant for the robot should be trained hands-off (upright
subset or an external-wrench filter), and its waist term is the one to distrust. File: `out/residual_uprightpre10s.bin`.

### 19. Noise-robust twin residual (DaD with planner-like command noise, sigma 0.08 rad)

Same twin A/B (`ab37_twinnoise_scores.csv`). Residual-on: pad loaded 5/5 (61-67 s), one 63 s seat, max rung 3, 5 falls
(67-143 s). Baseline this round: pad loaded 2/5, reached rung 9 once (28 s seat), 5 falls (53-114 s). Neither arm
completed. Training with planner-like noise removed the clear harm of section 17 but produced no measurable gain.

Bench variance. Pooled baselines of the three rounds (15 runs, same protocol): 2 full pipelines, 7 reached rung 9,
12 falls; per round 1/3/1 reached-rung-9 and 1/1/0 pipelines. Five-pair fall counts cannot resolve anything short
of a large effect. The sensitive metric for a dynamics residual is the planner's own imagination error in closed
loop: the trajectory it planned vs what the twin then did over the next 0.25-1 s (iCEM rollout dump). That is the
next measurement, before any further residual variants.

### 20. Planner imagination error in closed loop (the sensitive metric)

`imagination_error.py`: every 20th iCEM plan dumps the nominal trajectory (the planner's predicted states under its
chosen plan); it is compared with the twin's actual trajectory from the same instant. Displacement RMSE (rad), twin
residual `twinnoise` arms + waist, slot 37, two runs per arm:

| run | 100 ms legs / waist / arms | 250 ms legs / waist / arms | 500 ms legs / waist / arms |
|---|---|---|---|
| base r1 | 0.009 / 0.005 / 0.014 | 0.016 / 0.009 / 0.032 | 0.023 / 0.011 / - |
| base r2 | 0.015 / 0.004 / 0.012 | 0.021 / 0.007 / 0.024 | 0.023 / 0.008 / - |
| res r1 | 0.044 / 0.023 / 0.054 | 0.063 / 0.038 / 0.067 | 0.062 / 0.062 / 0.098 |
| res r2 | 0.049 / 0.030 / 0.021 | 0.086 / 0.121 / 0.048 | 0.101 / 0.203 / 0.061 |

Inside the planner the residual makes the planner's own predictions 3-10x worse, including the legs it never
touches (through the base). Offline, the same network with recorded commands improved the same metric. So the
network is being fed something different in the planner than in training. Candidates: the node's feed-forward
torque (gravity compensation, `--gravity_ff`) is in the training features but absent in the planner; the node's
command transforms (ankle zero, waist fix) sit between planner ctrl and the recorded low-level command. Section 21
tests the first.

### 21. Planner-available features only (`MBRL_ZERO_TAUFF`), and the phase-matched imagination error

The node's feed-forward torque was in the training features and absent in the planner. Offline, zeroing it in the
features turns the waist correction harmful (0.045 -> 0.053 vs plant). Retrained on planner-available inputs only
(`twinnoff`: no feed-forward, planner-like command noise): offline arms 0.031 -> 0.027, waist 0.045 -> 0.025.
PatchActuators writes the node's kp/kv/force limits into the planner model, so gains match the recorded command.

Imagination error restricted to the stand phase (t0 < 45 s, the only phase where every run has comparable
samples; dumps after release are few and dominated by dives/falls), 250 ms displacement RMSE (rad):

| run | legs | waist | arms |
|---|---|---|---|
| base (4 runs) | 0.016 / 0.021 / 0.022 / 0.015 | 0.009 / 0.007 / 0.019 / 0.006 | 0.032 / 0.024 / 0.030 / 0.023 |
| residual `twinnoise` (2) | 0.012 / 0.017 | 0.011 / 0.011 | 0.036 / 0.042 |
| residual `twinnoff` (2) | 0.013 / 0.012 | 0.022 / 0.014 | 0.036 / 0.035 |

The raw "3-10x worse" of section 20 was the dive/fall samples; phase-matched, the residual inside the planner is
slightly better on legs, 40-50 % worse on waist and arms, i.e. still not the offline gain. A C++ vs Python feature
parity test (`H12_RESIDUAL_DEBUG` dump of the hook's feature vector) is the next check before any further training.

### 22. Feature parity C++ vs Python: exact

`feature_parity.py` on 200 hook calls dumped by `H12_RESIDUAL_DEBUG` (own-sim): every feature block (q, dq, tau,
cmd-q, kp, kd, gravity, pelvis rate) agrees to < 1e-5 and the MLP output to 3e-6. The planner integration is
correct. Note: the very first hook call, at rest, already outputs a 1.1 rad/s correction (the clip), i.e. the
residual's authority is large relative to the plant's dynamics; own-sim without PatchActuators also feeds the net
XML gains (arms kp 40) it never saw, which explains the own-sim fall of section 15. Next test: reduced authority
(clip x0.3, group scale 0.5) scored by the stand-phase imagination error.

### 23. Reduced authority: clip x0.3, group scale 0.5 (arms + waist), twin residual `twinnoff`

Stand-phase imagination error, 250 ms, legs / waist / arms (rad):

| configuration | run 1 | run 2 |
|---|---|---|
| baseline (4 runs, range) | 0.015-0.022 / 0.006-0.019 / 0.023-0.032 | |
| residual full authority (`twinnoff`) | 0.013 / 0.022 / 0.036 | 0.012 / 0.014 / 0.035 |
| residual clip x0.3, scale 0.5 | 0.014 / 0.012 / 0.028 | 0.017 / 0.007 / 0.029 |

With reduced authority the planner's imagination is within the baseline's spread on every group, and both runs
reached rung 9 (one full pipeline, one 20 s seat then a fall at 91 s). Pooled baseline: 2 full pipelines in 15.
A 5-pair A/B of this configuration follows (section 24). The lesson so far: the residual's useful content is real
but its raw output (99.9th-percentile clip of training targets, up to ~1 rad/s) is too strong to hand to a sampling
optimiser; the correction must be small relative to the plant's own step.

### 24. Reduced-authority A/B (5 pairs) and the overall twin verdict

`ab37_lowauth_scores.csv`: baseline 2 full pipelines, 3 reached rung 9, 2 falls; residual (clip x0.3, scale 0.5,
arms + waist) 0 full pipelines, 0 reached rung 9 (one 115 s seat at rung 3), 4 falls.

Pooled over every twin round today (slot 37, B3, d37, `_real_ns`, honest protocol):

| arm | runs | full pipeline | reached rung 9 | falls |
|---|---|---|---|---|
| baseline | 20 | 4 | 10 | 14 |
| residual variants (real-data, twin, noise, planner-features, low authority) | 22 | 1 | 2 | 18 |

No residual configuration improved twin outcomes; the best were neutral on the planner's imagination error and still
lost on outcomes. Combined with sections 10, 16, 18 and 22 the conclusion is specific: the learned residual improves
open-loop prediction of recorded trajectories (real and twin), its planner integration is exact, and yet a
sampling planner that imagines with it plans worse. The correction is evaluated on the planner's own sampled
futures, where its errors are not averaged out the way they are on recorded data, and the optimiser selects the
rollouts where those errors look favourable. A residual for MPC therefore needs either an uncertainty estimate
(penalise or discard rollouts where the correction is uncertain), or training against the planner's own rollout
distribution with the cost in the loop, or injection as a conservative parametric change (section 11's damping x3
is the only correction so far that is both validated and planner-safe by construction).

### 25. Parametric correction on the twin: planner joint damping 10 vs 30 (`twin_damp_ab.sh`, 5 interleaved pairs)

Damping x3 was identified against REAL data (section 11). On the twin: baseline 1 full pipeline, 2 reached rung 9,
4 falls; damping 30: 0 full, 1 reached rung 9 (plus a 114 s seat at rung 3), 4 falls. Neutral to slightly worse,
and, as for the real-data residual, expected: the twin's own joints are not 3x more damped than the planner's, so a
real-robot-identified parameter is bias on the twin. The twin can validate the parametric METHOD only with
twin-identified values (section 26); real-robot-identified values can only be validated on the robot.

### 26. Identification against the TWIN corpus (planner step 1 x 10 ms), `out_twin/param_id2.json`

Stage D on twin data picks damping x3.0 (both groups, at the bound), leg armature x3.0, arm armature x1.1, torque
legs x1.1 / arms x0.6; held-out 250 ms legs 0.060 -> about 0.03. The same "x3 damping" the REAL data asked for
(section 11) therefore reflects the planner's coarse single 10 ms step through contact, not a property of the robot:
any finer-stepped plant looks "more damped" to it. Stage K on the twin returns hip-pitch offsets of -3 deg, which
cannot be a twin zero error; it compensates the same step mismatch and is not a calibration. The sim-to-sim test of
the parametric method is damping 30 + armature 0.3 in the planner include on the twin (section 27).

### 27. Twin-identified parametric set on the twin: damping 30 + armature 0.3 (5 interleaved pairs)

`ab37_da_{base,damp}.csv`: baseline 1 full pipeline, 1 reached rung 9, 1 rung 6, 4 falls; treated 0 full, max rung
5, 5 falls, four of them before 57 s (stand phase). The twin-identified parametric correction, which cuts the
planner plant's open-loop 250 ms error on twin data by ~45 %, makes the planner worse on the twin, exactly like
the residual. Offline fit at the planner's step is not the right objective for a sampling planner: a stiffer, more
damped imagined robot under-commands. Any plant change for the planner has to be judged in closed loop (imagination
error and outcomes), never by open-loop fit alone.

### 28. Uncertainty-gated ensemble (5 seeds, tau 0.05 rad/s, arms + waist, scale 0.5)

Offline (twin val, 250 ms): waist 0.045 -> 0.029, arms 0.031 -> 0.028 at tau 0.05 (tau 0.02 withdraws most of
the correction, tau 0.1 approaches the ungated net). Stand-phase imagination error: 0.021 / 0.009 / 0.026 and
0.014 / 0.005 / 0.027 (legs / waist / arms), the best of any residual and at or below the baseline range. Twin A/B
(`ab37_ens_scores.csv`): residual 2 seats of 72 s (no fall) and 94 s at rung 3, 3 falls, 1 stand collapse; baseline
2 seats (71 s at rung 9, 114 s at rung 6), 3 falls. The first residual configuration that is not worse than
baseline; still no gain, and it still does not reach the final rung.

### 29. Where this leaves the project

- Measurement (Stage 0) and offline learning (Stage 1) are solid and reproducible; the hardware facts they exposed
  (gripper shapes, right-ankle zero, leg kinematic gap, spotting contamination) stand on their own.
- Planner integration (Stage 2) is exact and cheap, but on the twin no correction, learned or parametric, improved
  planning; the uncertainty-gated ensemble is the only one that broke even. The reason is structural: a sampling
  planner optimises over its imagined futures, so corrections are judged where their errors are selected for, and
  open-loop fit at the planner's step is not the objective that matters.
- What would move it: train the correction with the planner in the loop (score candidate models by closed-loop
  imagination error, not open-loop fit), or correct the planner's step itself (finer substeps on the legs are 2x
  more accurate than the single 10 ms step), or test the hands-off real residual on the robot, which the twin
  cannot stand in for.

### 30. Stage 3 : distilling the planner into a policy (2026-10-05)

Goal: a 100 Hz MLP that reproduces the iCEM stand (first) and the lean ladder (later) with no planning CPU.
Code: `mbrl/bc_common.py` (observation: q, dq, gravity in IMU frame, gyro, phase/keyframe one-hots, time-in-phase,
optionally the previous command), `bc_dataset.py` + `bc_fit.py` + `bc_eval.py` (behaviour cloning from twin
recordings, closed-loop evaluation on the twin plant), `dagger_stand.py` (DAgger with the own-sim planner as teacher).

#### 30.1 Behaviour cloning from the twin recordings fails by compounding
- Dataset: 93 twin bench runs (strat 25/37 family) that kept their 100 Hz recorder npz; 1.4 M (obs, cmd, tau_ff)
  samples with phase + keyframe labels parsed from node.log. Held-out-run command RMSE 3.4 deg (hold 6.1, mean 4.9).
- Closed loop on the twin plant (dt 5 ms, PD per step with the node's per-joint gains, policy at 100 Hz) from
  free-standing recorded states: the clone falls in 1-2 s every time, as does a noise-regularised and a
  no-velocity variant. At the recorded states it is within 1 deg of the planner; 0.25 deg of state deviation after
  0.1 s already produces 3 deg of command deviation. Classic compounding: the demonstrations do not identify the
  feedback law.
- Evaluator sanity: open-loop replay of the recorded commands keeps the plant up 24 s; a constant mean command does
  not stand but falls forward onto the table and braces there. The planner's stand is an active balance.
- Bug found on the way: the twin's leg gains are NOT 150 across the board (hip pitch, knee, ankle pitch 200,
  ankle roll 80; arms 90/60/40/90/15); the evaluator and the DAgger teacher now take them from the recording.

#### 30.2 DAgger: the planner as an online teacher
- Teacher = `agent_server` (own-sim planner, 12 workers, 10.3 GB RSS) with the node's gains injected through
  `Init(model.xml)` (MjSpec with absolute meshdir), task reset per episode (its internal timers : stand trim/park,
  success-sustain keyframe advance : otherwise fire from the time offset and the teacher itself falls), state set
  per tick, K planner iterations, GetAction. With K=2 at 50 Hz the teacher keeps the twin plant standing 12/12
  episodes incl. 60 N pushes (66 ms/label). Pushes above ~100 N for 0.3 s topple the planner too.
- Finding 1 (the planner is not a state-feedback policy): on 11k teacher labels, a linear map from the robot state
  predicts the command no better than its mean (2.8 vs 3.0 deg); state + the previous command predicts it to
  0.6 deg. The iCEM action is mostly its own persistent plan (warm-started spline). With K=8 iterations per label
  the state-only fit improves (2.9 vs 3.9 deg mean) but the previous command still carries most of it (1.1 deg).
  Consequence: the student must take its last command as input, and labels collected under a beta-mixed executed
  command are inconsistent with the teacher's own history : classic DAgger (teacher once, then pure student) is used.
- Finding 2 (action parametrisation): predicting cmd - q makes the posture a random walk (the knees straightened
  from 31 deg to locked in 1.5 s and the robot toppled); predicting cmd - QREF (fixed reference pose) gave the first
  student that stands at all (13 % / 18 % of 15 s episodes without / with pushes after teacher-only data).

#### 30.3 A distilled stand that works (teacher-driven labels, K=8)
- Recipe that works: teacher = planner with K=8 iterations per 20 ms label (more Markov), task reset per episode,
  **teacher-driven episodes only** (the teacher executes its own commands, so the student's previous-command input
  is the teacher's own history), random 20-70 N torso pushes in 60 % of episodes, student output = cmd - QREF,
  student input includes the last executed command, 3x256 ELU, 30 epochs (longer training overfits the label
  noise and the closed loop collapses: 120 epochs on the same data -> 0-16 % survival).
- Adding labels from student-driven rollouts (classic DAgger iterations) consistently *destroys* the student
  (81 % -> 13 %), because the teacher's label on a trajectory it did not drive is tied to its own plan history,
  not to the command the student executed. One check dataset collected with K=8 in check mode also poisons the fit
  (label std 5.3 vs 3.5 deg); excluded.
- `out_bc/bc_stand_v1.npz` (12.3k labels from 42 teacher episodes): on 31 held-out standing states of the twin,
  30 s episodes, PD at 5 ms, policy at 50 Hz, no planner:

| push (0.3 s at the torso, 4 directions) | survive 30 s | n |
|---|---|---|
| none | 0.97 | 31 |
| 60 N | 0.83 | 124 |
| 90 N | 0.79 | 124 |
| 120 N | 0.79 | 124 |

  The teacher itself topples at ~100-140 N in the same protocol. Seed-to-seed variance of the fit is large with
  8.7k labels (0.55-0.90 survival) and shrinks with 12.3k (0.97); more teacher data is being collected.
- Caveats: stand only (phase 0 / kf 1 of strat 37), one fixed reference pose, twin plant = the deploy plant
  (`_real_ns`), not hardware; the planner's lean ladder is not distilled yet (needs the node's gates as teacher).

### 31. Planner substeps on the twin: negative (2026-10-05)

`H12_ROLLOUT_SUBSTEPS=n` (mjpc/trajectory.cc: each planner step integrates as n substeps of timestep/n via a shallow
model copy) was motivated by Stage 0: the single 10 ms implicitfast step doubles the one-step leg error through
contact. Twin A/B, strat 37 B3 honest twin, 3 interleaved pairs, baseline vs n=2:

| arm | plan/s | outcome |
|---|---|---|
| baseline (1 step) | 21-25 | r1 full ladder kf 1-8, r2 fall at 46 s, r3 fall at 66 s |
| 2 substeps | 20-22 | 3/3 lock the knees right after harness release (knee 31 deg -> 2 / -7 deg, pelvis z 0.96-0.89), then two collapse into a deep squat (knee 126 deg, pelvis z 0.59) and park there; none passes kf 2 |

Planner imagination error (nominal trajectory vs twin) is 10x WORSE with substeps (legs 0.25 rad at 100 ms vs
0.01-0.04), because the planner keeps predicting large leg motions (10-35 deg in 100 ms) that the plant does not
perform. Reading: the strategy's costs were tuned against the 10 ms single-step dynamics, whose extra numerical
damping the planner has been exploiting; a finer, more accurate step changes the planner's equilibrium (locked
knees are suddenly "cheap") and the cost structure, not the physics, breaks. Same lesson as the damping x3 artefact
(sec 27): the planner and the cost tuning are a matched pair; you cannot change the integrator alone. Not pursued.

#### 30.4 Where the distilled stand stands (end of 2026-10-05)
Honest numbers, held-out standing states of the twin, 30 s episodes unless noted, pushes 0.3 s at the torso:

| controller | no push | 90 N (4 dirs) | 120 N | notes |
|---|---|---|---|---|
| planner teacher (K=2 at 50 Hz, 15 s, fwd/back pushes only) | 1.00 (11) | 0.55 (22) | 0.55 (22) | 12 planner workers, 10 GB |
| student v1 `bc_stand_v1.npz` (lucky seed) | 0.97 (31) | 0.79 (124) | 0.79 (124) | 3x256 MLP, 50 Hz, no planner |
| student best-of-8 by closed-loop selection `bc_stand_sel.npz` | 0.87 (31) | 0.65 (124) | - | selection on training states |
| 5-seed ensemble | 1.00 (15 s) | 0.63 (15 s) | - | averaging smooths but does not robustify |

- Things that did NOT help, each tested: DAgger iterations on student rollouts (destroys it: non-Markov teacher),
  DART noise on the executed teacher commands (0.29-0.58), a third teacher collection folded in (0.42), label
  smoothing over 5/11 ticks (0.61-0.67), longer training (collapse), wider/deeper nets.
- Seed-to-seed variance of the fit is the dominant uncertainty (0.32-0.97 stand survival on identical data), so a
  closed-loop selection set is mandatory, and the reported hold numbers are what the recipe delivers, not the max.
- What this shows: the iCEM stand of strat 37 can be replicated by a 50 Hz MLP on the deploy plant at
  teacher-level push robustness, with ~12k labels (42 teacher episodes, ~45 min of own-sim labelling) and zero
  planning CPU. What it does not show yet: the lean ladder (needs the node's gate state machine as part of the
  teacher), hardware behaviour, and a fix for the fit fragility (a value-weighted or RL fine-tuning stage is the
  natural next step; the twin plant in Python runs 8 s episodes in ~2 s, so PPO fine-tuning from this student is feasible).

#### 30.5 RETRACTION of 30.3-30.4 (2026-10-05, caught by the user from the video)
Every stand number in 30.3-30.4 is invalid. The fall detector checked only pelvis height and torso tilt; the
"standing" robot was leaning on the table with wrist pads and grippers from the first seconds (robot-table contact
24.0 s of 24 s, 3 kN peak normal force after the first push). With the table moved 100 m away:

| student | no push | 60 N | 90 N |
|---|---|---|---|
| v1 | 0.18 | 0.01 | 0.00 |
| best-of-8 | 0.00 | 0.00 | 0.00 |

The 165 init states themselves do not touch the table (0/165), so the lean is the policy's doing. The teacher
reference (30.4) was measured with the same detector and is presumed table-assisted too. What survives from 30.1-30.2:
the pipeline, the BC compounding result (measured before any push, falls in 1-2 s either way), the non-Markov-teacher
measurement (a property of the labels), and the action-parametrisation finding. Redo: table out of both the teacher's
model and the plant, robot-table contact counted as failure, contacts and a rendered video checked before any number.

#### 30.6 Free-standing redo and the closing result on imitation (2026-10-05, after the retraction)
Protocol fixed: table moved 100 m in BOTH the teacher's planner model and the plant, block parked, robot-table
contact = failure, contact counters printed with every number.
- Teacher (K=8 warm-started, 50 Hz): stands free 6/6 incl. 60 N pushes. 36 teacher-driven episodes -> 10.7k labels.
- Student, 8 seeds, selected by closed-loop score on training states, held-out 30 s free-standing:

| push | survive 30 s | n | table touch |
|---|---|---|---|
| none | 0.21 | 33 | 0 |
| 30 N | 0.11 | 132 | 0 |
| 60 N | 0.03 | 132 | 0 |
| 90 N | 0.01 | 132 | 0 |

- Markov-teacher test (plan reset before every label, K=16 from scratch, 0.6 s/label): the label becomes NOISE -
  tick jitter 2.5 deg (0.4 warm-started), state-only fit 3.4 deg vs mean-predictor 2.7, previous command does not
  help either (3.4). Warm-started the action is history (state+prev 1.1 deg at K=8); re-planned it is a random
  sample of an unconverged iCEM. Either way it is not a function of the robot state.

**Conclusion:** the sampling planner cannot be distilled tick-by-tick by imitation (BC, DAgger, DART, label
smoothing, Markov teacher all tested). Its closed-loop behaviour is what should be imitated, which is RL: train the
policy in the twin on the planner's own costs as reward, warm-started from the student, with the Stage-0 sim-gap
facts as the randomisation ranges. The tools (teacher, free-standing evaluator with contact-as-failure, python twin
plant at ~4x real time) carry over.

---

## 5. Per-experiment record: `mbrl/out/` (real-robot corpus, 493 runs)

Joint index convention used in every per-joint array below (27 joints, order of `common.JOINT_NAMES`): 0 L_hip_yaw, 1 L_hip_pitch, 2 L_hip_roll, 3 L_knee, 4 L_ankle_pitch, 5 L_ankle_roll, 6 R_hip_yaw, 7 R_hip_pitch, 8 R_hip_roll, 9 R_knee, 10 R_ankle_pitch, 11 R_ankle_roll, 12 torso (waist), 13 L_shoulder_pitch, 14 L_shoulder_roll, 15 L_shoulder_yaw, 16 L_elbow, 17 L_wrist_roll, 18 L_wrist_pitch, 19 L_wrist_yaw, 20 R_shoulder_pitch, 21 R_shoulder_roll, 22 R_shoulder_yaw, 23 R_elbow, 24 R_wrist_roll, 25 R_wrist_pitch, 26 R_wrist_yaw. Groups: legs 0-11, waist 12, arms 13-26. Plant state: qpos[0:3] pelvis pos, [3:7] pelvis quat, [7:34] joints, [34:41] object anchor free joint (nq 41, nv 39, nu 27).

Units: one-step errors are RMSE of the next joint velocity in rad/s over a 10 ms step (`e_zoh` = hold-velocity predictor, `e_pd` = plant + reconstructed PD command, `e_tau` = plant + measured torque); `eq_*` are the matching joint-position errors in rad. Rollout errors are joint-displacement RMSE in rad at 100 / 250 / 500 ms (a 500 ms value of `null` means the segment length T=25 steps did not reach it).

### 5.1 `out/simgap_run.log` (one-step sim-gap driver over the 493 real runs)

Progress log of `simgap.py`: one line per run `[i/493] <run>  <n_samples> ok  <n_ok>  <elapsed s>`; all 493 runs reported `ok`; total wall time `done 1285.07 s` (21.4 min, 10 workers). The sampled index count per run ranges from 0 (very short strat-10 runs such as `10_4_20260908_230105`, `10_5`, `10_6`, `10_7`, `10_3`, `10_8`: 0 samples each; `10_9`: 19) to 31,695 (`table_-5_123_20260911_080951`). Last three runs: `v4lnewstand_20260830_062453` 10,186; `v3stand_20260830_060832` 12,190; `v3newstand_20260830_061824` 14,550 samples (stride 2 of the both-feet-loaded samples).

### 5.2 `out/analyze.log` and `out/simgap_summary.json` (aggregate one-step sim-gap)

`analyze.log` prints `samples 3993845` and the group table as JSON, then `saved figs` and `PIPELINE_DONE`; the file is truncated mid-JSON after the `consistent_upright / arms` entry (the log was overwritten by a later run), the complete table is in `simgap_summary.json`.

Counts: n = 3,993,845 both-feet-loaded transitions; n_lean = 2,446,544 (tilt >= 12 deg); n_consistent = 1,602,684.

Group table (rad/s):

| phase | group | e_zoh | e_pd | e_tau | pd/zoh | tau/zoh |
|---|---|---|---|---|---|---|
| all | legs | 0.0260 | 0.3291 | 0.3507 | 12.651 | 13.482 |
| all | waist | 0.0132 | 0.1926 | 0.1922 | 14.591 | 14.561 |
| all | arms | 0.0200 | 0.0820 | 0.0932 | 4.097 | 4.660 |
| upright | legs | 0.0249 | 0.2625 | 0.2928 | 10.554 | 11.772 |
| upright | waist | 0.0146 | 0.0831 | 0.0852 | 5.692 | 5.836 |
| upright | arms | 0.0199 | 0.0417 | 0.0458 | 2.096 | 2.301 |
| lean | legs | 0.0267 | 0.3650 | 0.3829 | 13.671 | 14.339 |
| lean | waist | 0.0123 | 0.2371 | 0.2361 | 19.276 | 19.195 |
| lean | arms | 0.0201 | 0.0993 | 0.1134 | 4.941 | 5.642 |
| consistent | legs | 0.0264 | 0.1779 | 0.1812 | 6.728 | 6.850 |
| consistent | waist | 0.0131 | 0.1435 | 0.1360 | 10.954 | 10.382 |
| consistent | arms | 0.0205 | 0.0710 | 0.0810 | 3.472 | 3.960 |
| consistent_upright | legs | 0.0246 | 0.1548 | 0.1631 | 6.297 | 6.635 |
| consistent_upright | waist | 0.0112 | 0.0427 | 0.0352 | 3.812 | 3.143 |
| consistent_upright | arms | 0.0195 | 0.0390 | 0.0435 | 2.001 | 2.233 |
| consistent_lean | legs | 0.0285 | 0.2020 | 0.2005 | 7.089 | 7.038 |
| consistent_lean | waist | 0.0150 | 0.2071 | 0.1975 | 13.807 | 13.167 |
| consistent_lean | arms | 0.0216 | 0.0961 | 0.1101 | 4.456 | 5.103 |

Foot / sole diagnostics: sole gap |z_L - z_R| percentiles (p50, p75, p90, p99) = 3.9, 7.0, 10.9, 19.2 mm. Foot-flattening corrections, median over all loaded samples (L pitch, L roll, R pitch, R roll) = -0.18, -0.83, +3.76, +0.98 deg; upright only = +0.61, -0.94, +3.84, +0.77 deg.

Reconstructed PD torque minus measured torque, per joint (Nm): RMSE = [0.523, 0.960, 0.797, 1.987, 1.452, 0.612, 0.485, 1.195, 1.059, 2.039, 1.486, 0.413, 0.461, 0.695, 0.402, 0.309, 0.510, 0.122, 0.115, 0.161, 0.882, 0.406, 0.304, 0.595, 0.141, 0.227, 0.126]; mean = [0.021, 0.030, 0.012, 0.102, 0.683, 0.357, -0.001, 0.001, 0.001, 0.119, 0.761, -0.079, -0.041, 0.280, 0.283, -0.224, 0.275, 0.102, 0.082, 0.144, -0.362, 0.271, -0.230, -0.402, 0.126, -0.205, 0.106] (the +0.68 / +0.76 Nm bias on both ankle pitches, indices 4 and 10).

Per-joint one-step velocity error (rad/s), selected phases:

| joint | all e_zoh | all e_pd | all e_tau | upright e_pd | lean e_pd | consistent e_pd | cons_upright e_pd | cons_lean e_pd |
|---|---|---|---|---|---|---|---|---|
| 0 L_hip_yaw | 0.0125 | 0.1295 | 0.1336 | 0.0765 | 0.1539 | 0.0983 | 0.0365 | 0.1399 |
| 1 L_hip_pitch | 0.0122 | 0.2384 | 0.2467 | 0.1541 | 0.2789 | 0.1923 | 0.1285 | 0.2477 |
| 2 L_hip_roll | 0.0092 | 0.1787 | 0.1837 | 0.0893 | 0.2171 | 0.1877 | 0.0503 | 0.2722 |
| 3 L_knee | 0.0148 | 0.3646 | 0.3729 | 0.3016 | 0.3993 | 0.2701 | 0.2694 | 0.2708 |
| 4 L_ankle_pitch | 0.0451 | 0.6140 | 0.5163 | 0.1850 | 0.7706 | 0.1900 | 0.1585 | 0.2217 |
| 5 L_ankle_roll | 0.0512 | 0.1523 | 0.0985 | 0.1128 | 0.1727 | 0.1250 | 0.1014 | 0.1482 |
| 6 R_hip_yaw | 0.0119 | 0.1749 | 0.1910 | 0.1371 | 0.1951 | 0.1203 | 0.0473 | 0.1703 |
| 7 R_hip_pitch | 0.0121 | 0.2785 | 0.2947 | 0.3027 | 0.2620 | 0.1817 | 0.1583 | 0.2061 |
| 8 R_hip_roll | 0.0089 | 0.1802 | 0.1833 | 0.1167 | 0.2108 | 0.1446 | 0.0606 | 0.2035 |
| 9 R_knee | 0.0160 | 0.4137 | 0.4664 | 0.4881 | 0.3588 | 0.2457 | 0.2914 | 0.1767 |
| 10 R_ankle_pitch | 0.0323 | 0.5290 | 0.7335 | 0.4670 | 0.5646 | 0.1719 | 0.1824 | 0.1585 |
| 11 R_ankle_roll | 0.0345 | 0.2637 | 0.1892 | 0.2663 | 0.2620 | 0.1226 | 0.1003 | 0.1448 |
| 12 torso | 0.0132 | 0.1926 | 0.1922 | 0.0831 | 0.2371 | 0.1435 | 0.0427 | 0.2071 |
| 13 L_shoulder_pitch | 0.0234 | 0.1723 | 0.1867 | 0.0313 | 0.2188 | 0.1344 | 0.0283 | 0.1965 |
| 14 L_shoulder_roll | 0.0167 | 0.1161 | 0.1312 | 0.0343 | 0.1458 | 0.0977 | 0.0276 | 0.1413 |
| 15 L_shoulder_yaw | 0.0235 | 0.0608 | 0.0851 | 0.0312 | 0.0737 | 0.0560 | 0.0340 | 0.0741 |
| 16 L_elbow | 0.0182 | 0.1351 | 0.1627 | 0.0462 | 0.1687 | 0.1147 | 0.0459 | 0.1622 |
| 17 L_wrist_roll | 0.0184 | 0.0389 | 0.0434 | 0.0369 | 0.0401 | 0.0390 | 0.0406 | 0.0370 |
| 18 L_wrist_pitch | 0.0172 | 0.0520 | 0.0590 | 0.0448 | 0.0560 | 0.0514 | 0.0458 | 0.0574 |
| 19 L_wrist_yaw | 0.0215 | 0.0500 | 0.0506 | 0.0644 | 0.0381 | 0.0484 | 0.0554 | 0.0384 |
| 20 R_shoulder_pitch | 0.0211 | 0.0843 | 0.0930 | 0.0377 | 0.1034 | 0.0856 | 0.0343 | 0.1210 |
| 21 R_shoulder_roll | 0.0148 | 0.0698 | 0.0727 | 0.0337 | 0.0850 | 0.0637 | 0.0297 | 0.0885 |
| 22 R_shoulder_yaw | 0.0237 | 0.0416 | 0.0542 | 0.0288 | 0.0479 | 0.0422 | 0.0289 | 0.0539 |
| 23 R_elbow | 0.0198 | 0.0650 | 0.0810 | 0.0629 | 0.0663 | 0.0571 | 0.0531 | 0.0616 |
| 24 R_wrist_roll | 0.0209 | 0.0337 | 0.0351 | 0.0338 | 0.0337 | 0.0345 | 0.0352 | 0.0337 |
| 25 R_wrist_pitch | 0.0194 | 0.0481 | 0.0492 | 0.0481 | 0.0481 | 0.0442 | 0.0455 | 0.0427 |
| 26 R_wrist_yaw | 0.0191 | 0.0288 | 0.0298 | 0.0282 | 0.0292 | 0.0260 | 0.0251 | 0.0270 |

Per-joint position error after 10 ms (rad), `all` phase: eq_zoh = [0.0002, 0.0004, 0.0002, 0.0006, 0.0006, 0.0006, 0.0002, 0.0003, 0.0002, 0.0007, 0.0006, 0.0005, 0.0003, 0.0004, 0.0003, 0.0003, 0.0006, 0.0002, 0.0003, 0.0003, 0.0004, 0.0002, 0.0003, 0.0006, 0.0003, 0.0003, 0.0002]; eq_pd = [0.0010, 0.0017, 0.0013, 0.0027, 0.0366, 0.0428, 0.0013, 0.0020, 0.0013, 0.0031, 0.0684, 0.0397, 0.0014, 0.0012, 0.0008, 0.0005, 0.0011, 0.0003, 0.0005, 0.0004, 0.0007, 0.0005, 0.0003, 0.0008, 0.0003, 0.0004, 0.0002] (the ankle pitch/roll position errors of 0.04-0.07 rad are the floating-foot artefact that the consistent filter removes: consistent eq_pd ankles = 0.0335, 0.0403, 0.0672, 0.0333). Other phases (upright, lean, consistent, consistent_upright, consistent_lean) carry the same pattern; their full arrays were in the JSON and are summarised by the group table above.

Per-strategy one-step error (rad/s, all joints, loaded-feet samples):

| strategy | n | zoh | pd | tau |
|---|---|---|---|---|
| 10 | 45376 | 0.0231 | 0.2693 | 0.3219 |
| 11 | 617047 | 0.0225 | 0.2530 | 0.2616 |
| 12 | 8324 | 0.0180 | 0.1461 | 0.1523 |
| 12new | 26906 | 0.0194 | 0.2167 | 0.2136 |
| 22 | 15799 | 0.0171 | 0.2097 | 0.2233 |
| 22again | 130859 | 0.0219 | 0.2685 | 0.2627 |
| 22new | 8646 | 0.0165 | 0.2292 | 0.2359 |
| 25 | 229134 | 0.0207 | 0.2193 | 0.2528 |
| 25new | 39887 | 0.0192 | 0.1921 | 0.2139 |
| 26 | 92075 | 0.0189 | 0.2234 | 0.2493 |
| 27 | 18794 | 0.0276 | 0.2570 | 0.2591 |
| 28 | 35660 | 0.0238 | 0.2992 | 0.2994 |
| 29 | 549110 | 0.0195 | 0.2110 | 0.2376 |
| 36 | 59119 | 0.0185 | 0.2260 | 0.2311 |
| 9 | 448438 | 0.0209 | 0.2256 | 0.2586 |
| brace22 | 3610 | 0.0163 | 0.1873 | 0.1680 |
| crouch8 | 1834 | 0.0820 | 1.0065 | 1.0557 |
| ghost25 | 672 | 0.0982 | 0.2398 | 0.2511 |
| nobrace0 | 13499 | 0.0362 | 0.3420 | 0.3200 |
| nobrace3 | 23287 | 0.0214 | 0.3056 | 0.3127 |
| real | 967896 | 0.0245 | 0.2033 | 0.2115 |
| realstandsumma3 | 108 | 0.0134 | 0.1122 | 0.1185 |
| s29 | 39424 | 0.0191 | 0.1960 | 0.2292 |
| sq18cold | 4280 | 0.0752 | 0.5829 | 0.6176 |
| sq18harn | 5481 | 0.0736 | 0.7880 | 0.8394 |
| sq18held | 6478 | 0.0492 | 0.1306 | 0.1194 |
| sq18supp | 3542 | 0.0466 | 0.2244 | 0.2374 |
| stand6 | 6691 | 0.0660 | 0.8985 | 0.9424 |
| stand6ready | 4558 | 0.0479 | 0.1673 | 0.1469 |
| stand6ready2 | 3554 | 0.0499 | 0.1573 | 0.1457 |
| stand6supp | 4994 | 0.0719 | 0.1593 | 0.1628 |
| standB2 | 317 | 0.0126 | 0.1784 | 0.2000 |
| table | 541520 | 0.0204 | 0.2144 | 0.2248 |
| v3newstand | 14550 | 0.0170 | 0.2347 | 0.1950 |
| v3stand | 12190 | 0.0166 | 0.2409 | 0.2366 |
| v4lnewstand | 10186 | 0.0168 | 0.2086 | 0.2235 |

### 5.3 `out/ablate.log` and `out/ablate_params.json` (parametric ablation on a few runs, one-step dq RMSE, rad/s)

`ablate_params.py`: plant joint damping / armature / frictionloss varied; `[tau]` = measured-torque mode, `[pd]` = PD-command mode; hold-velocity reference legs 0.019, waist 0.009, arms 0.018 in every row. Plant as-is = damping 10, armature 0.1, frictionloss 0.2.

| configuration | tau legs | tau waist | tau arms | pd legs | pd waist | pd arms |
|---|---|---|---|---|---|---|
| plant as-is (damp10 arm0.1 fl0.2) | 0.288 | 0.177 | 0.094 | 0.305 | 0.166 | 0.083 |
| damp3 | 0.355 | 0.198 | 0.111 | 0.375 | 0.185 | 0.095 |
| damp30 | 0.200 | 0.135 | 0.075 | 0.209 | 0.129 | 0.069 |
| damp1 | 0.382 | 0.206 | 0.118 | 0.403 | 0.191 | 0.100 |
| damp0.1 | 0.396 | 0.209 | 0.122 | 0.418 | 0.194 | 0.102 |
| damp0 | 0.398 | 0.209 | 0.123 | 0.419 | 0.194 | 0.103 |
| damp1 arm0.01 | 1.153 | 0.275 | 0.329 | 0.835 | 0.263 | 0.170 |
| damp0.1 arm0.01 | 1.353 | 0.280 | 0.420 | 0.894 | 0.267 | 0.199 |
| damp0.1 arm0.01 fl0 | 1.383 | 0.294 | 0.485 | 0.905 | 0.276 | 0.229 |
| damp0 arm0.01 fl0 | 1.417 | 0.294 | 0.502 | 0.913 | 0.276 | 0.234 |

Reading: lowering damping, armature or friction makes the one-step error monotonically worse; damping 30 lowers it (legs 0.29 -> 0.20). Part 2 later showed the one-step objective is degenerate (an immovable robot wins), so "more damping is better" here is a step-size artefact of the single 10 ms step rather than a property of the robot.

### 5.4 `out/residual_fit.log`, `out/residual_run.log`, `out/residual_summary.json` (Stage 1 one-step residual, 3x512 MLP, 5 x 2 ms protocol)

Data: train (1,253,218 x 168 features), val (192,714), test (156,752); load 5-6 s. Ridge fit 3 s. MLP training (12 epochs, Adam):

| epoch | train loss | val rmse(all) |
|---|---|---|
| 0 | 0.2282 | 0.0593 |
| 1 | 0.1165 | 0.0547 |
| 2 | 0.1027 | 0.0530 |
| 3 | 0.0960 | 0.0514 |
| 4 | 0.0908 | 0.0506 |
| 5 | 0.0868 | 0.0502 |
| 6 | 0.0836 | 0.0495 |
| 7 | 0.0812 | 0.0491 |
| 8 | 0.0794 | 0.0490 |
| 9 | 0.0781 | 0.0488 |
| 10 | 0.0771 | 0.0487 |
| 11 | 0.0767 | 0.0487 |

MLP fit 17 s. One-step test RMSE (rad/s):

| model | legs | waist | arms | all |
|---|---|---|---|---|
| hold_dq | 0.02803 | 0.02225 | 0.02079 | 0.02432 |
| plant_pd | 0.14936 | 0.11022 | 0.06343 | 0.11158 |
| plant_pd + ridge | 0.10631 | 0.06402 | 0.03253 | 0.07566 |
| plant_pd + mlp | 0.06607 | 0.04777 | 0.02637 | 0.04884 |

Per-strategy one-step test RMSE (all joints):

| strategy | n | hold | plant | plant+mlp |
|---|---|---|---|---|
| 10 | 4351 | 0.0196 | 0.0907 | 0.0363 |
| 11 | 31835 | 0.0240 | 0.1034 | 0.0452 |
| 22again | 12795 | 0.0221 | 0.1269 | 0.0415 |
| 22new | 2324 | 0.0171 | 0.0939 | 0.0331 |
| 25 | 6770 | 0.0205 | 0.1247 | 0.0390 |
| 29 | 22354 | 0.0189 | 0.1132 | 0.0339 |
| 36 | 6021 | 0.0182 | 0.1225 | 0.0422 |
| 9 | 6913 | 0.0198 | 0.1079 | 0.0357 |
| real | 41508 | 0.0235 | 0.1119 | 0.0447 |
| s29 | 3389 | 0.0184 | 0.0936 | 0.0345 |
| stand6supp | 4216 | 0.0707 | 0.1484 | 0.1628 |
| table | 14276 | 0.0191 | 0.1004 | 0.0412 |

Closed-loop rollouts on test runs, 339 segments x 50 steps (recorded PD commands replayed, residual applied post-step after every 10 ms, clipped at the 99.9th percentile of training targets; "diverged" = any joint error > 0.5 rad), displacement RMSE (rad):

| subset (n) | model | diverged | 100 ms legs/waist/arms | 250 ms legs/waist/arms | 500 ms legs/waist/arms |
|---|---|---|---|---|---|
| all (339) | plant | 3 | 0.0143 / 0.0203 / 0.0086 | 0.0301 / 0.0477 / 0.0193 | 0.0443 / 0.0446 / 0.0279 |
| all | ridge | 28 | 0.0165 / 0.0069 / 0.0047 | 0.0656 / 0.0258 / 0.0201 | 0.1351 / 0.0811 / 0.0692 |
| all | mlp | 4 | 0.0161 / 0.0096 / 0.0050 | 0.0410 / 0.0326 / 0.0141 | 0.0616 / 0.0590 / 0.0259 |
| upright (238) | plant | 3 | 0.0131 / 0.0043 / 0.0054 | 0.0241 / 0.0076 / 0.0123 | 0.0331 / 0.0081 / 0.0187 |
| upright | ridge | 5 | 0.0162 / 0.0056 / 0.0040 | 0.0676 / 0.0224 / 0.0174 | 0.1427 / 0.0599 / 0.0717 |
| upright | mlp | 3 | 0.0146 / 0.0067 / 0.0037 | 0.0343 / 0.0204 / 0.0098 | 0.0525 / 0.0229 / 0.0178 |
| lean (101) | plant | 0 | 0.0166 / 0.0365 / 0.0134 | 0.0406 / 0.0862 / 0.0297 | 0.0632 / 0.0804 / 0.0423 |
| lean | ridge | 23 | 0.0174 / 0.0097 / 0.0064 | 0.0594 / 0.0340 / 0.0265 | 0.1094 / 0.1245 / 0.0608 |
| lean | mlp | 1 | 0.0191 / 0.0143 / 0.0071 | 0.0534 / 0.0508 / 0.0209 | 0.0790 / 0.1022 / 0.0389 |

Total 46 s. `residual_run.log` is an earlier run of the same script (MLP test {legs 0.0670, waist 0.0429, arms 0.0258, all 0.0491}; 300 segments; 21 MuJoCo "Nan, Inf or huge value in QACC ... simulation is unstable" warnings at t = 0.30-0.49 s) whose rollout evaluation used an earlier protocol without the divergence filter and reported plant 100/250/500 ms legs 0.0627 / 0.2033 / 0.3480, ridge 0.0424 / 0.1695 / 0.9340, mlp 0.0414 / 0.2688 / 0.4844 (waist ridge 1.77 rad at 500 ms): superseded by `residual_fit.log`. The npz `out/rollout_err.npz` held the per-segment arrays {tilt0 (339,), err_plant / err_ridge / err_mlp (339, 50, 27)} behind the table above; `out/residual_norm.npz` held {mu (168,), sd (168,), ymu (27,), ysd (27,), W_ridge (169, 27)}.

### 5.5 `out/multistep_fit.log`, `out/multistep_run.log`, `out/jac_run.log` (linearised multi-step training: negative)

`jac_run.log`: `jacobians.py` over 404 jobs (train + val runs), 5789 segments requested, up to 15 segments per run (stride 100, T = 25 macro steps), 4860 segments produced in 178 s (20 workers); e.g. `[2/404] 11_+5_101112_20260913_065226 15/15`, `[404/404] v4lnewstand_20260830_062453 15/15 total 4860`. Output `data/jac/<run>.npz` = {run, split, i0 (K,), xn (K, 26, 67), A (K, 25, 66, 66)}, 401 files, 2.15 GB.

`multistep_fit.log`: segments train 4277, val 583, load 11 s, RAM for A 1.9 GB. Val plant-only legs 100/250 ms 0.0186 / 0.0378, arms 0.0048 / 0.0111. Warm start from the one-step MLP through the linear recursion: loss 8.47e22, legs {10 steps: 4257.9, 25 steps: 2.51e11}, arms {10: 276.8, 25: 9.39e9}; epoch 0 train nan, val nan; `done 14s best val 1e9`. `multistep_run.log` is empty (0 bytes). Verdict: finite-difference Jacobians of the 10 ms macro step have gains near 1e4 in the contact directions and the 25-step linear recursion diverges from the first iterate; linearising through stiff foot contact is not a usable training signal.

### 5.6 `out/param_id.log`, `out/param_id.json` (parametric identification v1, one-step objective: degenerate)

`param_id.log`: `samples 3580`. Results:

Stage K (kinematic, sole-gap objective over loaded samples): offsets (L hip pitch, L knee, R hip pitch, R knee) = -3.44, +6.48, -0.31, +5.67 deg; sole gap RMS 10.67 -> 8.53 mm; consistent fraction 0.45 -> 0.50.

Stage D (dynamic, one-step dq error, PD mode, Nelder-Mead): scales damping_legs 212.4, damping_arms 3.8e-5, armature_legs 2273.2, armature_arms 495.9, torque_legs 0.533, torque_arms 8.4e-5. One-step error on the fit set: nominal {legs 0.1201, waist 0.1341, arms 0.0688, all 0.0976, n 72}; with stage-K offsets {0.1998, 0.1168, 0.0780, 0.1463, n 80}; fitted {0.0213, 0.0073, 0.0182, 0.0194, n 80}. Held-out test: nominal {legs 0.1955, waist 0.0989, arms 0.0649, all 0.1398, n 612}; offsets {0.2202, 0.1001, 0.0776, 0.1582, n 772}; fitted {0.0721, 0.0137, 0.0316, 0.0532, n 772}. The "fit" is an immovable robot (armature x2273, arm torque x1e-4) that matches the hold-velocity predictor: the one-step objective is degenerate and this result was discarded.

### 5.7 `out/param_id2.log`, `out/param_id2.json` (parametric identification v2, 250 ms rollout objective, real corpus)

Fixed consistent segment sets: train 324, test 245; 5 x 2 ms substeps. Objective = plant-only 250 ms displacement RMSE (legs + arms); parameters log-bounded [1/3, 3].

| stage | parameters | objective | 100 ms legs/waist/arms | 250 ms legs/waist/arms |
|---|---|---|---|---|
| nominal (train 324) | damping 10, armature 0.1, torque 1 | 0.06944 | 0.0200 / 0.0239 / 0.0096 | 0.0473 / 0.0564 / 0.0222 |
| stage D (48 s, 119 evals) | damping_legs 3.004, damping_arms 3.004, armature_legs 1.302, armature_arms 0.333, torque_legs 0.999, torque_arms 0.909 | 0.04745 | 0.0128 / 0.0163 / 0.0072 | 0.0300 / 0.0401 / 0.0174 |
| stage K (73 s) | offsets (Lhip, Lknee, Rhip, Rknee) = -0.36, +0.08, -0.07, -0.13 deg | 0.04741 | 0.0127 / 0.0163 / 0.0072 | 0.0300 / 0.0401 / 0.0174 |

Held-out test (245 segments):

| plant | 100 ms legs/waist/arms | 250 ms legs/waist/arms |
|---|---|---|
| nominal | 0.0181 / 0.0162 / 0.0078 | 0.0466 / 0.0367 / 0.0175 |
| stage D | 0.0106 / 0.0117 / 0.0068 | 0.0256 / 0.0275 / 0.0152 |
| stage D + K | 0.0106 / 0.0116 / 0.0068 | 0.0257 / 0.0276 / 0.0152 |

### 5.8 DaD residual fits on the real corpus: `out/dad_*.log`, `out/dadpre10*.log`, `out/dadsmoke_fit.log`, `out/uprightpre10s_*.log` and their `*_summary.json`

Common protocol (`dad_fit.py`): 3000 training segments, 500 validation segments (279 skipped as non-consistent or too short, 221 scored), T = 25 steps; one-step warm-start MLP then DaD iterations each aggregating 32,575 on-rollout samples (mixed batch = 3x, i.e. 97,725 per iteration). The `*_run.log` files are the stdout of the same runs including numpy `RuntimeWarning: overflow encountered in exp` lines from `rollout_engine.py` (SiLU on diverging rollouts, harmless) and carry identical numbers to the `*_fit.log` files; the `*_summary.json` files hold the same plant/iters tables plus `params`.

`dad_fit.log` / `dad_run.log` / `dad_summary.json` (tag `dad`, 3x512, post-step injection, nsub 5, dt 0.002, 40 s wall): one-step base val rmse {legs 0.0745, waist 0.0280, arms 0.0218, all 0.0524} (13 s).

| iterate | 100 ms legs/waist/arms | 250 ms legs/waist/arms |
|---|---|---|
| plant only | 0.01743 / 0.02430 / 0.00973 | 0.03850 / 0.05638 / 0.02182 |
| iter 0 (one-step MLP) | 0.01759 / 0.01018 / 0.00502 | 0.05268 / 0.04667 / 0.01366 |
| iter 1 | 0.01340 / 0.00569 / 0.00465 | 0.04049 / 0.01447 / 0.01262 |
| iter 2 | 0.01275 / 0.00615 / 0.00442 | 0.03921 / 0.01423 / 0.01029 |
| iter 3 | 0.01263 / 0.00629 / 0.00445 | 0.03884 / 0.01476 / 0.01029 |
| iter 4 | 0.01252 / 0.00650 / 0.00442 | 0.03888 / 0.01508 / 0.01008 |
| iter 5 | 0.01243 / 0.00667 / 0.00440 | 0.03867 / 0.01518 / 0.01004 |

`dadpre10_fit.log` / `dadpre10_run.log` / `dadpre10_summary.json` (tag `dadpre10`, 3x512, pre-step injection, nsub 1, dt 0.01 = the planner step, 35 s): one-step base {0.0730, 0.0273, 0.0219, 0.0514}.

| iterate | 100 ms legs/waist/arms | 250 ms legs/waist/arms |
|---|---|---|
| plant only | 0.01807 / 0.02565 / 0.01013 | 0.03911 / 0.05651 / 0.02206 |
| iter 0 | 0.01904 / 0.00841 / 0.00669 | 0.04296 / 0.02681 / 0.01526 |
| iter 1 | 0.02279 / 0.00620 / 0.00514 | 0.07229 / 0.03289 / 0.01242 |
| iter 2 | 0.02158 / 0.00579 / 0.00467 | 0.06321 / 0.01394 / 0.01101 |
| iter 3 | 0.02109 / 0.00575 / 0.00447 | 0.06360 / 0.01331 / 0.01016 |
| iter 4 | 0.02074 / 0.00502 / 0.00435 | 0.06390 / 0.01220 / 0.00981 |
| iter 5 | 0.02070 / 0.00497 / 0.00428 | 0.06405 / 0.01169 / 0.00964 |

`dadpre10s_fit.log` / `dadpre10s_run.log` / `dadpre10s_summary.json` (tag `dadpre10s`, the deployable 2x64 net `MBRL_H=64 MBRL_NL=2`, pre-step, 1 x 10 ms, 4 iterates, 20 s): one-step base {0.0785, 0.0317, 0.0249, 0.0557} (10 s).

| iterate | 100 ms legs/waist/arms | 250 ms legs/waist/arms |
|---|---|---|
| plant only | 0.01807 / 0.02565 / 0.01013 | 0.03911 / 0.05651 / 0.02206 |
| iter 0 | 0.01997 / 0.00891 / 0.00674 | 0.05049 / 0.03498 / 0.01726 |
| iter 1 | 0.02053 / 0.00772 / 0.00577 | 0.05339 / 0.02162 / 0.01398 |
| iter 2 | 0.02376 / 0.00533 / 0.00549 | 0.07453 / 0.01401 / 0.01399 |
| iter 3 | 0.02351 / 0.00446 / 0.00512 | 0.07519 / 0.01060 / 0.01184 |

(With all joints corrected the pre-step kick hurts the legs, 0.039 -> 0.075; the masked evaluation in 5.9 is why the planner configuration is arms + waist only.)

`dadsmoke_fit.log` / `dadsmoke_summary.json` (smoke tests of the DaD loop with 120, 40 and 60 training segments, nsub 5): one-step base {0.0705, 0.0255, 0.0210, 0.0496}, {0.0709, 0.0257, 0.0209, 0.0498}, {0.0735, 0.0277, 0.0217, 0.0517}; the 60-segment run aggregated 725 then 1450 on-rollout samples and reached iter 1 = 100 ms 0.01632 / 0.00806 / 0.00490, 250 ms 0.04707 / 0.03977 / 0.01312 (plant 0.01743 / 0.02430 / 0.00973 and 0.03850 / 0.05638 / 0.02182); the 120 and 40 segment runs stopped after iter 0 (0.05631 / 0.06019 / 0.01458 and 0.05627 / 0.06092 / 0.01469 at 250 ms). 21 s.

`uprightpre10s_fit.log` / `uprightpre10s_run.log` / `uprightpre10s_summary.json` (tag `uprightpre10s`, `MBRL_MAX_TILT=12`, hands-off upright subset only, 2x64, pre-step, 1 x 10 ms, 25 s): segments train 1685, val 271 (103 skipped, 168 scored); one-step base {0.0688, 0.0193, 0.0257, 0.0496}; 26,575 on-rollout samples per iteration.

| iterate | 100 ms legs/waist/arms | 250 ms legs/waist/arms |
|---|---|---|
| plant only | 0.01788 / 0.00337 / 0.00436 | 0.03674 / 0.00523 / 0.01014 |
| iter 0 | 0.01714 / 0.00382 / 0.00346 | 0.03617 / 0.01027 / 0.00862 |
| iter 1 | 0.01572 / 0.00335 / 0.00296 | 0.03343 / 0.00859 / 0.00677 |
| iter 2 | 0.01501 / 0.00335 / 0.00291 | 0.03450 / 0.00954 / 0.00678 |
| iter 3 | 0.01518 / 0.00308 / 0.00288 | 0.04179 / 0.00929 / 0.00703 |

Normaliser files `out/residual_norm_{dad,dadpre10,dadpre10s,dadsmoke,uprightpre10s}.npz` each held {mu (168,), sd (168,), ymu (27,), ysd (27,), clip (27,)}; the weights lived in `out/residual_mlp_<tag>.pt` and the exported `out/residual_<tag>.bin`.

### 5.9 Masked evaluations at the planner step: `out/masked_dadpre10.json`, `out/masked_dadpre10s.json`, `out/masked_uprightpre10s.json`

`eval_masked.py` re-scores a trained residual on validation rollouts (pre-step, 1 x 10 ms) with the correction masked to all joints, arms + waist, or arms only. Displacement RMSE (rad), 100 ms and 250 ms, legs / waist / arms.

`masked_dadpre10.json` (3x512 real-data net, 352 segments, all tilts):

| mask | 100 ms | 250 ms |
|---|---|---|
| plant | 0.01863 / 0.02523 / 0.01013 | 0.04426 / 0.05518 / 0.02196 |
| all | 0.01981 / 0.00900 / 0.00665 | 0.04776 / 0.03176 / 0.01527 |
| arms+waist | 0.01797 / 0.00956 / 0.00663 | 0.04165 / 0.03329 / 0.01552 |
| arms | 0.01825 / 0.02932 / 0.00657 | 0.04313 / 0.06386 / 0.01650 |

`masked_dadpre10s.json` (2x64 real-data net evaluated on the 282 upright-only segments; this is the "all-data residual on upright" row of report section 18):

| mask | 100 ms | 250 ms |
|---|---|---|
| plant | 0.01731 / 0.00348 / 0.00448 | 0.03736 / 0.00566 / 0.01014 |
| all | 0.01583 / 0.00540 / 0.00341 | 0.03641 / 0.01320 / 0.00800 |
| arms+waist | 0.01715 / 0.00467 / 0.00331 | 0.03716 / 0.01216 / 0.00748 |
| arms | 0.01713 / 0.00518 / 0.00337 | 0.03697 / 0.00971 / 0.00767 |

`masked_uprightpre10s.json` (2x64 upright-trained net evaluated on the 352 all-tilt segments):

| mask | 100 ms | 250 ms |
|---|---|---|
| plant | 0.01863 / 0.02523 / 0.01013 | 0.04426 / 0.05518 / 0.02196 |
| all | 0.02062 / 0.01229 / 0.00749 | 0.04926 / 0.02834 / 0.01756 |
| arms+waist | 0.01800 / 0.01182 / 0.00737 | 0.04193 / 0.02426 / 0.01696 |
| arms | 0.01815 / 0.02953 / 0.00714 | 0.04244 / 0.06678 / 0.01726 |

(The report's section 18 table rows "upright residual, all joints 0.0336 / 0.0084 / 0.0070" and "upright residual, arms only 0.0369 / 0.0069 / 0.0070" on the upright subset came from an evaluation whose JSON was not kept separately; the numbers are preserved in section 4 above.)

### 5.10 `out/imagination_error.json` (planner imagination error of the SUBSTEP A/B, section 31)

`imagination_error.py` on the six runs of the `H12_ROLLOUT_SUBSTEPS=2` A/B (`base_sub2` = baseline single step, `res_sub2` = 2 substeps; twin_ab.sh reuses the arm names base/res). Displacement RMSE (rad) of the iCEM nominal trajectory vs the twin recording, legs / waist / arms, by phase of the dump time t0.

| run | n_dumps | phase (n) | 100 ms | 250 ms | 500 ms |
|---|---|---|---|---|---|
| base_sub2_r1 | 53 | all | 0.0394 / 0.0638 / 0.0126 | 0.0257 / 0.0188 / 0.0241 | 0.0334 / 0.0314 / 0.0338 |
| | | stand < 45 s (32) | 0.0131 / 0.0047 / 0.0113 | 0.0174 / 0.0075 / 0.0234 | 0.0194 / 0.0086 / 0.0317 |
| | | 45-65 s (1) | 0.0103 / 0.0104 / 0.0116 | 0.0131 / 0.0095 / 0.0239 | 0.0154 / 0.0003 / 0.0317 |
| | | > 65 s (20) | 0.0620 / 0.1036 / 0.0145 | 0.0353 / 0.0290 / 0.0251 | 0.0484 / 0.0500 / 0.0372 |
| base_sub2_r2 | 7 | all = stand (7) | 0.0084 / 0.0028 / 0.0087 | 0.0125 / 0.0040 / 0.0153 | 0.0228 / 0.0029 / 0.0190 |
| base_sub2_r3 | 33 | all = stand (33) | 0.0106 / 0.0038 / 0.0107 | 0.0148 / 0.0061 / 0.0223 | 0.0165 / 0.0075 / 0.0298 |
| res_sub2_r1 | 86 | all | 0.2554 / 0.2393 / 0.1990 | 0.7572 / 0.3588 / 0.2173 | 0.9918 / 0.6562 / 0.2623 |
| | | stand < 45 s (7) | 0.0301 / 0.0077 / 0.0083 | 0.0506 / 0.0391 / 0.0232 | 0.0769 / 0.1008 / 0.0611 |
| | | 45-65 s (1) | 0.0349 / 0.0016 / 0.0080 | 0.0507 / 0.0118 / 0.0171 | 0.0763 / 0.0909 / 0.0552 |
| | | > 65 s (78) | 0.2680 / 0.2512 / 0.2089 | 0.7949 / 0.3766 / 0.2281 | 1.0412 / 0.6883 / 0.2747 |
| res_sub2_r2 | 106 | all | 0.1146 / 0.1447 / 0.0442 | 0.3265 / 0.2483 / 0.1048 | 0.3902 / 0.2364 / 0.1966 |
| | | 45-65 s (22) | 0.1057 / 0.1496 / 0.0451 | 0.3127 / 0.2533 / 0.1015 | 0.3850 / 0.2489 / 0.1922 |
| | | > 65 s (84) | 0.1169 / 0.1434 / 0.0440 | 0.3300 / 0.2470 / 0.1056 | 0.3915 / 0.2330 / 0.1977 |
| res_sub2_r3 | 87 | all | 0.2648 / 0.2226 / 0.0358 | 0.8263 / 0.7242 / 0.0779 | 1.0395 / 1.2245 / 0.1541 |
| | | 45-65 s (9) | 0.2847 / 0.1985 / 0.0391 | 0.8142 / 0.6314 / 0.0866 | 1.0370 / 1.0795 / 0.1544 |
| | | > 65 s (78) | 0.2624 / 0.2252 / 0.0354 | 0.8277 / 0.7341 / 0.0769 | 1.0397 / 1.2401 / 0.1540 |

(res_sub2_r2 and r3 have no dumps before 45 s; r2 and r3 of the substep arm sat in the squat from early on.) The planner with 2 substeps keeps predicting 10-35 deg leg motions in 100 ms that the plant does not perform: 10x the baseline's imagination error.

The imagination-error numbers of the residual rounds (report sections 20, 21, 23, 28) were computed from the run directories under `~/Desktop/h12/logs/grid37_twin/d37/` and are preserved only in section 4 above.

## 6. Per-experiment record: `mbrl/out_twin/` (twin sim2sim corpus, 190 runs)

All twin fits use `MBRL_DATA_ROOT=data_twin MBRL_OUT=out_twin`, the planner step (nsub 1, dt 0.01), pre-step injection, 2x64 networks, 3000 training / 500 validation segments (150 skipped, 350 scored in the DaD validation; the masked evaluations score 400). The twin recordings come from the honest-twin benches of 2026-09-25 (`grid37_twin/d30`, `sweep25`) and the A/B rounds of 2026-10-04/05 (`grid37_twin/d37`), 100 Hz recorder npz with the node's lowcmd. The twin plant steps 5 ms with PD per step, pyramidal cone, noslip (`_real_ns`); the planner plant it is compared against steps 1 x 10 ms implicitfast.

### 6.1 `out_twin/simgap_run.log`

190 runs, all `ok`, `done 924.56 s` (15.4 min). First run `g36_Balancex2_r3_20260925_143108` 6,376 samples; largest `sw_nom_r3_20260925_014454` 16,700; last `sw_nom_r5_20260925_033609` 16,338 samples. (The `done` line appears twice: the log was appended by a re-run of the final summary.)

### 6.2 `out_twin/analyze.log` and `out_twin/simgap_summary.json` (twin-plant vs planner-plant one-step gap)

`analyze.log` prints `samples 2461568` and the group table (truncated after `consistent_upright / arms`, same overwrite artefact as the real log). From the JSON: n = 2,461,568; n_lean = 1,755,155; n_consistent = 1,770,895 (72 % consistent vs 40 % on the real robot).

| phase | group | e_zoh | e_pd | e_tau | pd/zoh | tau/zoh |
|---|---|---|---|---|---|---|
| all | legs | 0.0815 | 0.3069 | 0.2737 | 3.764 | 3.357 |
| all | waist | 0.0137 | 0.1166 | 0.1170 | 8.511 | 8.540 |
| all | arms | 0.0067 | 0.0957 | 0.0965 | 14.322 | 14.433 |
| upright | legs | 0.0949 | 0.2479 | 0.2375 | 2.613 | 2.504 |
| upright | waist | 0.0083 | 0.0363 | 0.0369 | 4.373 | 4.446 |
| upright | arms | 0.0038 | 0.0205 | 0.0207 | 5.416 | 5.470 |
| lean | legs | 0.0755 | 0.3277 | 0.2870 | 4.341 | 3.802 |
| lean | waist | 0.0154 | 0.1362 | 0.1365 | 8.844 | 8.864 |
| lean | arms | 0.0075 | 0.1126 | 0.1135 | 14.941 | 15.061 |
| consistent | legs | 0.0786 | 0.2367 | 0.2247 | 3.011 | 2.858 |
| consistent | waist | 0.0139 | 0.1107 | 0.1115 | 7.964 | 8.022 |
| consistent | arms | 0.0073 | 0.0858 | 0.0865 | 11.720 | 11.814 |
| consistent_upright | legs | 0.0952 | 0.2311 | 0.2264 | 2.427 | 2.377 |
| consistent_upright | waist | 0.0081 | 0.0332 | 0.0334 | 4.099 | 4.123 |
| consistent_upright | arms | 0.0037 | 0.0190 | 0.0190 | 5.068 | 5.082 |
| consistent_lean | legs | 0.0665 | 0.2400 | 0.2236 | 3.612 | 3.364 |
| consistent_lean | waist | 0.0165 | 0.1381 | 0.1391 | 8.370 | 8.430 |
| consistent_lean | arms | 0.0088 | 0.1078 | 0.1087 | 12.224 | 12.321 |

Note the twin's own joint-velocity signal is much noisier on the legs (hold 0.095 rad/s vs 0.026 real) and much cleaner on the arms (0.0038 vs 0.0199), which is why the arm ratios are large although the absolute arm error (0.019-0.021 upright) is half the real robot's.

Sole gap percentiles 1.1, 3.3, 5.0, 13.6 mm (real: 3.9, 7.0, 10.9, 19.2). Foot flattening medians (L pitch, L roll, R pitch, R roll) = -0.01, +0.12, -0.02, +0.11 deg (upright -0.02, +0.10, -0.02, +0.05): the reset protocol invents no offsets on a clean plant.

PD-minus-measured torque RMSE per joint (Nm) = [2.715, 4.577, 5.044, 2.190, 2.610, 0.869, 1.609, 3.316, 3.883, 3.670, 2.245, 0.920, 2.190, 1.239, 1.212, 0.640, 0.489, 0.082, 0.035, 0.034, 1.201, 0.539, 0.165, 0.468, 0.043, 0.086, 0.049]; mean = [-0.216, -0.306, -0.440, 0.101, -0.058, 0.006, 0.085, 0.050, 0.306, 0.334, -0.028, 0.021, 0.182, -0.094, 0.042, -0.039, -0.023, 0.004, 0.002, -0.001, 0.097, 0.050, 0.008, 0.045, 0.000, 0.005, 0.002] (larger leg RMSE than the real robot because the twin's recorded tau is the 5 ms PD-per-step torque, not the 10 ms ZOH reconstruction).

Per-joint e_pd (rad/s): all = [0.0576, 0.2577, 0.2019, 0.3506, 0.5550, 0.3649, 0.0734, 0.2970, 0.1283, 0.3277, 0.3771, 0.3099, 0.1166, 0.1962, 0.1484, 0.1963, 0.1015, 0.0256, 0.0154, 0.0614, 0.0744, 0.0678, 0.0293, 0.0516, 0.0124, 0.0191, 0.0126]; consistent_upright = [0.0485, 0.2430, 0.0927, 0.3222, 0.2473, 0.2864, 0.0573, 0.2485, 0.0873, 0.3233, 0.2666, 0.2752, 0.0332, 0.0236, 0.0244, 0.0150, 0.0361, 0.0040, 0.0073, 0.0061, 0.0308, 0.0152, 0.0043, 0.0318, 0.0017, 0.0050, 0.0019]; consistent_lean = [0.0509, 0.2458, 0.1388, 0.2414, 0.3930, 0.3145, 0.0572, 0.2436, 0.1060, 0.2102, 0.3144, 0.2838, 0.1381, 0.2321, 0.1695, 0.2028, 0.1174, 0.0306, 0.0172, 0.0804, 0.0857, 0.0763, 0.0310, 0.0527, 0.0135, 0.0188, 0.0136]; e_zoh all = [0.0125, 0.0213, 0.0190, 0.0345, 0.0852, 0.1316, 0.0135, 0.0264, 0.0199, 0.0457, 0.1276, 0.1827, 0.0137, 0.0077, 0.0070, 0.0053, 0.0129, 0.0054, 0.0026, 0.0040, 0.0090, 0.0055, 0.0043, 0.0104, 0.0028, 0.0043, 0.0026].

Per-strategy (twin bench families):

| strategy | n | zoh | pd | tau |
|---|---|---|---|---|
| g36 (09-25 weight sweep, graze slot 36) | 585561 | 0.0565 | 0.1919 | 0.1781 |
| s11 | 104364 | 0.0485 | 0.1914 | 0.1782 |
| s12 | 245094 | 0.0479 | 0.1889 | 0.1729 |
| s25 | 139738 | 0.0422 | 0.2923 | 0.2512 |
| s25pb3 | 11016 | 0.0679 | 0.2620 | 0.2223 |
| s25pb5 | 5796 | 0.0686 | 0.2646 | 0.2189 |
| s25s40 | 117432 | 0.0623 | 0.2312 | 0.2054 |
| s36 | 205144 | 0.0502 | 0.2481 | 0.2260 |
| s36gh | 34949 | 0.0505 | 0.1999 | 0.1825 |
| s37 | 37392 | 0.0503 | 0.3179 | 0.2551 |
| s6 | 2667 | 0.0383 | 0.2765 | 0.2669 |
| sw (sweep25) | 972415 | 0.0572 | 0.2138 | 0.1951 |

### 6.3 Twin DaD fits: `out_twin/dad_run.log` = `twinpre10s_fit.log`, `dad_noise_run.log` = `twinnoise_fit.log`, `dad_noff_run.log` = `twinnoff_fit.log`, and `*_summary.json`

Each `dad_*_run.log` is the stdout of the matching `*_fit.log` run (plus overflow warnings); numbers identical. 57,100 on-rollout samples per iteration (mixed batch 171,300). Displacement RMSE (rad), legs / waist / arms.

`twinpre10s` (recorded commands, no noise, 18 s): one-step base {legs 0.1759, waist 0.0510, arms 0.0145, all 0.1182}.

| iterate | 100 ms | 250 ms |
|---|---|---|
| plant only | 0.02046 / 0.02024 / 0.01373 | 0.05394 / 0.04530 / 0.03101 |
| iter 0 | 0.02126 / 0.00960 / 0.00918 | 0.05765 / 0.02409 / 0.02696 |
| iter 1 | 0.02888 / 0.00481 / 0.00627 | 0.09134 / 0.02212 / 0.01772 |
| iter 2 | 0.02885 / 0.00517 / 0.00491 | 0.09143 / 0.01891 / 0.01165 |
| iter 3 | 0.02812 / 0.00533 / 0.00472 | 0.08539 / 0.01457 / 0.01022 |

`twinnoise` (`MBRL_CMD_NOISE=0.08` rad planner-like command noise in the DaD rollouts, 21 s): one-step base {0.1739, 0.0506, 0.0143, 0.1168}. The plant-only row is scored under the same noise, hence slightly different from above.

| iterate | 100 ms | 250 ms |
|---|---|---|
| plant only (noisy cmd) | 0.02331 / 0.02530 / 0.01493 | 0.05605 / 0.04800 / 0.03207 |
| iter 0 | 0.02381 / 0.01935 / 0.01155 | 0.05841 / 0.04151 / 0.02961 |
| iter 1 | 0.03143 / 0.01226 / 0.00917 | 0.09442 / 0.03652 / 0.02300 |
| iter 2 | 0.03230 / 0.01054 / 0.00727 | 0.09738 / 0.02453 / 0.01481 |
| iter 3 | 0.03114 / 0.01101 / 0.00683 | 0.09144 / 0.02363 / 0.01334 |

`twinnoff` (`MBRL_ZERO_TAUFF=1`, planner-available features only, plus command noise 0.08, 20 s): one-step base {0.1772, 0.0516, 0.0141, 0.1190}.

| iterate | 100 ms | 250 ms |
|---|---|---|
| plant only (noisy cmd) | 0.02331 / 0.02530 / 0.01493 | 0.05605 / 0.04800 / 0.03207 |
| iter 0 | 0.02356 / 0.01758 / 0.01072 | 0.05843 / 0.03477 / 0.02710 |
| iter 1 | 0.03139 / 0.01416 / 0.00869 | 0.09511 / 0.04000 / 0.01992 |
| iter 2 | 0.03188 / 0.00967 / 0.00719 | 0.09598 / 0.02233 / 0.01450 |
| iter 3 | 0.03116 / 0.01031 / 0.00678 | 0.09077 / 0.02196 / 0.01305 |

(The legs are deliberately not corrected in the planner, so the leg degradation in these all-joint DaD rows is irrelevant; the masked arms + waist scores in 6.5 are the ones that matter.)

### 6.4 Ensemble members: `out_twin/ens{1..5}_fit.log`, `dad_ens{1..5}.log`, `ens{1..5}_summary.json`

Five seeds of the `twinnoff` recipe (`MBRL_SEED=1..5`, command noise 0.08, zero tau_ff, 2x64, pre-step), used by the uncertainty-gated ensemble (`MBRL_TAU`). `dad_ens*.log` are the stdout copies (4, 6, 9, 5 and 7 overflow warnings respectively). One-step base val rmse and the final (iter 3) rollout scores:

| seed | one-step base legs/waist/arms/all | on-rollout samples per iter | iter 3 100 ms legs/waist/arms | iter 3 250 ms legs/waist/arms | wall |
|---|---|---|---|---|---|
| ens1 | 0.1757 / 0.0504 / 0.0137 / 0.1179 | 56450 | 0.03095 / 0.01028 / 0.00682 | 0.08975 / 0.02274 / 0.01344 | 21 s |
| ens2 | 0.1765 / 0.0513 / 0.0131 / 0.1184 | 56000 | 0.03131 / 0.01067 / 0.00681 | 0.09063 / 0.02479 / 0.01336 | 18 s |
| ens3 | 0.1798 / 0.0508 / 0.0130 / 0.1206 | 57250 | 0.03107 / 0.01033 / 0.00678 | 0.09013 / 0.02481 / 0.01336 | 18 s |
| ens4 | 0.1801 / 0.0509 / 0.0133 / 0.1208 | 56675 | 0.03141 / 0.01081 / 0.00679 | 0.09229 / 0.02359 / 0.01335 | 18 s |
| ens5 | 0.1735 / 0.0505 / 0.0150 / 0.1165 | 57075 | 0.03236 / 0.01148 / 0.00682 | 0.09297 / 0.02613 / 0.01336 | 18 s |

Intermediate iterates (250 ms legs / waist / arms): ens1 iter 0-2 = 0.05401 / 0.03831 / 0.02949, 0.09556 / 0.04073 / 0.02297, 0.09695 / 0.02694 / 0.01472; ens2 = 0.05781 / 0.03552 / 0.03018, 0.09576 / 0.04119 / 0.02149, 0.10128 / 0.02503 / 0.01446; ens3 = 0.05876 / 0.03792 / 0.03191, 0.08937 / 0.02925 / 0.02206, 0.09669 / 0.02515 / 0.01548; ens4 = 0.05781 / 0.04162 / 0.02789, 0.09838 / 0.04264 / 0.02186, 0.10093 / 0.02599 / 0.01517; ens5 = 0.06232 / 0.03897 / 0.02608, 0.10315 / 0.05046 / 0.02094, 0.10195 / 0.02730 / 0.01479. Plant-only (noisy command) reference for all five: 0.05605 / 0.04800 / 0.03207.

### 6.5 Masked twin evaluations: `out_twin/masked_twinpre10s.json`, `masked_twinnoise.json`, `masked_twinnoff.json`, `masked_ens1.json`, `masked_ens1+ens2+ens3+ens4+ens5.json`

400 twin validation segments, pre-step at 1 x 10 ms, recorded commands (no noise at evaluation). Displacement RMSE (rad), legs / waist / arms. The plant row is identical in every file: 100 ms 0.01916 / 0.02001 / 0.01368, 250 ms 0.05042 / 0.04464 / 0.03085.

| residual | mask | 100 ms | 250 ms |
|---|---|---|---|
| twinpre10s | all | 0.02044 / 0.00927 / 0.00892 | 0.05415 / 0.02236 / 0.02624 |
| twinpre10s | arms+waist | 0.01841 / 0.01009 / 0.00897 | 0.04911 / 0.02605 / 0.02562 |
| twinpre10s | arms | 0.01856 / 0.02512 / 0.00833 | 0.04891 / 0.05497 / 0.02395 |
| twinnoise | all | 0.01895 / 0.01654 / 0.00935 | 0.04788 / 0.05227 / 0.02681 |
| twinnoise | arms+waist | 0.01763 / 0.01990 / 0.00924 | 0.04565 / 0.05329 / 0.02668 |
| twinnoise | arms | 0.01771 / 0.02366 / 0.00916 | 0.04523 / 0.04975 / 0.02681 |
| twinnoff | all | 0.02016 / 0.00935 / 0.00851 | 0.05472 / 0.02294 / 0.02569 |
| twinnoff | arms+waist | 0.01821 / 0.01041 / 0.00862 | 0.04882 / 0.02455 / 0.02646 |
| twinnoff | arms | 0.01840 / 0.02436 / 0.00814 | 0.04858 / 0.04785 / 0.02558 |
| ens1 (single seed, gated tau 0.05) | all | 0.01978 / 0.01162 / 0.00838 | 0.05193 / 0.02557 / 0.02621 |
| ens1 | arms+waist | 0.01815 / 0.01153 / 0.00829 | 0.04836 / 0.02724 / 0.02607 |
| ens1 | arms | 0.01833 / 0.02526 / 0.00776 | 0.04813 / 0.05250 / 0.02475 |
| ens1+..+ens5 (gated ensemble, tau 0.05) | all | 0.01965 / 0.00995 / 0.00879 | 0.05175 / 0.02436 / 0.02644 |
| ens1+..+ens5 | arms+waist | 0.01830 / 0.01078 / 0.00892 | 0.04903 / 0.02681 / 0.02674 |
| ens1+..+ens5 | arms | 0.01848 / 0.02493 / 0.00831 | 0.04904 / 0.05078 / 0.02553 |

Readings: `twinnoise` (trained with command noise but with the node's tau_ff still in the features, evaluated on noise-free recordings) loses the waist gain entirely (0.053 vs plant 0.045), which is the offline sign of the feature mismatch that `twinnoff` fixes (waist 0.045 -> 0.025); the gated ensemble gives waist 0.045 -> 0.027 and arms 0.031 -> 0.027 while withdrawing the correction where the five seeds disagree by more than tau = 0.05 rad/s (tau 0.02 withdraws most of the correction, tau 0.1 approaches the ungated net; those two evaluations were not saved as JSON).

### 6.6 `out_twin/param_id2.json` (identification against the twin corpus, planner step)

400 training segments (nominal objective 100 ms 0.02129 / 0.02579 / 0.01489, 250 ms 0.05637 / 0.06059 / 0.03361, legs / waist / arms), 300 test segments, nsub 1, dt 0.01.

| stage | parameters | train 100 ms | train 250 ms |
|---|---|---|---|
| stage D | damping_legs 3.004, damping_arms 3.004, armature_legs 3.004, armature_arms 1.134, torque_legs 1.125, torque_arms 0.603 | 0.01422 / 0.01857 / 0.00789 | 0.03727 / 0.04517 / 0.02003 |
| stage K | offsets (Lhip, Lknee, Rhip, Rknee) = -3.03, +0.86, -3.01, -0.24 deg | 0.01276 / 0.01848 / 0.00842 | 0.03168 / 0.04517 / 0.02051 |

Test (300 segments):

| plant | 100 ms | 250 ms |
|---|---|---|
| nominal | 0.02027 / 0.02020 / 0.01627 | 0.05984 / 0.04632 / 0.03770 |
| stage D | 0.01285 / 0.01394 / 0.00822 | 0.03814 / 0.03505 / 0.02147 |
| stage D + K | 0.01142 / 0.01389 / 0.00872 | 0.03222 / 0.03522 / 0.02189 |

Damping x3 (at the bound) and leg armature x3 are selected again, on a twin whose joints have the planner's own damping 10 and armature 0.1; the -3 deg hip-pitch offsets cannot be a twin zero error. Both compensate the planner's coarse 1 x 10 ms step through contact. Tested in the planner include (damping 30 + armature 0.3, `twin_damp_ab.sh ARMAT=0.3`): worse (report section 27).

Normaliser files `out_twin/residual_norm_{twinpre10s,twinnoise,twinnoff,ens1..ens5}.npz` each held {mu (168,), sd (168,), ymu (27,), ysd (27,), clip (27,)}.

### 6.7 Twin A/B outcome tables (from report sections 14-28; raw scores lived in `~/Desktop/h12/logs/grid37_twin/d37/ab37_*_scores.csv` and `AB_*.md`, outside mbrl/)

All rounds: slot 37 (strat 25 + 40 s stand sustain), target B3, d37, `_real_ns` twin plant, honest protocol (rope 0.8, chunked payout, honest-gate release, no brace assist), 5 interleaved pairs of 180 s unless noted.

| round | treated arm | baseline: full / rung 9 / falls / other | treated: full / rung 9 / falls / other |
|---|---|---|---|
| 1 (sec 14) | real-data residual `dadpre10s`, arms + waist, scale 0,1,1 | 1 / - / 4 (release 42 s in 5/5; r4 pad load 66 s, seat 99 s, 9/9 done) | 0 / - / 3 falls + 2 stand collapses at ~57 s (never released); r4 seat 98 s twisted 100 deg yaw, fall 174 s |
| 1b (sec 15) | real-data residual arms only, 0,0,1 | (same baseline) | 0 full, releases 42-45 s in 5/5, 4 falls, 1 brief pad load: neutral |
| 2 (sec 17) | twin residual `twinpre10s`, arms + waist | 1 / 3 / 3, longest seat 115 s | 0 / 0 (max rung 3) / 5, longest seat 19 s |
| 3 (sec 19) | twin residual `twinnoise` | pad loaded 2/5, rung 9 once (28 s seat), 5 falls (53-114 s) | pad loaded 5/5 (61-67 s), one 63 s seat, max rung 3, 5 falls (67-143 s) |
| 4 (sec 24) | twin residual `twinnoff`, clip x0.3, scale 0.5 | 2 / 3 / 2 | 0 / 0 (one 115 s seat at rung 3) / 4 |
| 5 (sec 25) | planner damping 10 -> 30 (`twin_damp_ab.sh`) | 1 / 2 / 4 | 0 / 1 (+ 114 s seat at rung 3) / 4 |
| 6 (sec 27) | damping 30 + armature 0.3 (twin-identified) | 1 / 1 (+ rung 6) / 4 | 0 / max rung 5 / 5 (four before 57 s) |
| 7 (sec 28) | gated ensemble ens1-5, tau 0.05, scale 0.5 | 2 seats (71 s at rung 9, 114 s at rung 6), 3 falls | 2 seats (72 s no fall, 94 s at rung 3), 3 falls, 1 stand collapse |
| 8 (sec 31, 3 pairs) | `H12_ROLLOUT_SUBSTEPS=2` | r1 full ladder kf 1-8, r2 fall 46 s, r3 fall 66 s; 21-25 plan/s | 3/3 locked knees after release (31 -> 2 / -7 deg, pelvis z 0.96-0.89), 2 deep squats (knee 126 deg, z 0.59); none passes kf 2; 20-22 plan/s |

Pooled (rounds 1-4, sec 24): baseline 20 runs, 4 full, 10 rung 9, 14 falls; residual variants 22 runs, 1 full, 2 rung 9, 18 falls. Pooled baselines of rounds 1-3 (15 runs): 2 full, 7 rung 9, 12 falls; per round 1/3/1 rung 9 and 1/1/0 full: five-pair fall counts cannot resolve anything short of a large effect.

---

## 7. Per-experiment record: `mbrl/out_bc/` (stage 3: BC / DAgger stand policy, 2026-10-05)

Common setup printed by every `dagger_*.log`: init states = recorded free-standing twin states at t = 38 / 41 / 44 / 47 s of the slot-37 bench runs (155 init states from 31 runs in the first series, 165 from 33 runs after the table-free fix); plant gains kp = [150, 200, 200, 200, 200, 80, 150, 200, 200, 200, 200, 80, 200, 90, 60, 40, 90, 15, 15, 15, 90, 60, 40, 90, 15, 15, 15] (hip yaw 150, hip pitch / roll / knee / ankle pitch 200, ankle roll 80, waist 200, shoulder pitch 90, roll 60, yaw 40, elbow 90, wrists 15), taken from the recording (the earlier assumption of 150 across the legs was a bug); teacher = own-sim `agent_server` on 'Lean H12 Magpie' slot 37 with the node's gains injected (`[teacher] up: nq 41 nv 39 t0 0.002`). Student = 3x256 ELU MLP (W0 (256, 101), W1/W2 (256, 256), W3 (54, 256); 101 inputs = q 27 + dq 27 + g_imu 3 + gyro 3 + phase one-hot 4 + kf one-hot 9 + t_phase 1 + previous cmd - q 27; 54 outputs = cmd - QREF 27 + tau_ff 27; the 74-input variants omit the previous command). Episode line format: `it <i> ep <e> beta <b> init <run>@<t0> survived <s>/<T> labels <n> wall <w>s push (<t_push>, <N>)`. `[eval <tag>]` = student alone from held-out standing states, 15 s episodes, with and without pushes.

Every stand/push number in 7.1-7.5 was measured with the fall detector that checked only pelvis height and torso tilt, and the robot was leaning on the table (report section 30.5). They are kept as the record of what was run; only 7.6-7.8 (table 100 m away, contact = failure) are valid.

### 7.1 `out_bc/dagger_dag.log` (classic DAgger, K=2 at 50 Hz, beta 1.0 / 0.5 / 0.25, 8 s episodes, 74-input student without previous command)

- it 0 (beta 1.00): 14 teacher-driven episodes, all survived 8.0 s, 400 labels each (22-25 s wall), pushes between 26 and 63 N at 1.0-6.0 s on 8 of 14; aggregated 5600 samples. `[eval dag0] no-push 0.00, push 0.02, mean t 3.2/15 s`.
- it 1 (beta 0.50): 14 mixed episodes, 12 survived 8.0 s, `base10_da_r4@41` 7.8 s (392 labels), `twinres_r4@41` 6.4 s (319 labels); mean survival 7.9 s; aggregated 11,011. `[eval dag1] 0.00 / 0.00, mean t 3.0 s`.
- it 2 (beta 0.25): 4 episodes logged (all 8.0 s) before the run was stopped. Student `bc_dag0.npz`, `bc_dag1.npz`; data `dagger_dag_data.npz` {X (11011, 74), Y (11011, 27)}.

### 7.2 `out_bc/dagger_daga.log` (same as 7.1 with the previous command as input, 101 features, absolute action cmd - QREF)

- it 0: same 14 episode inits as 7.1 (all 8.0 s, 5600 samples). `[eval daga0] no-push 0.13, push 0.18, mean t 7.2/15 s` (the first student that stands at all, report 30.2 finding 2).
- it 1 (beta 0.50): 14 episodes, `twinres_r4@41` 6.4 s (323 labels), rest 8.0 s; aggregated 11,073. `[eval daga1] 0.03 / 0.03, mean t 2.9 s` (student-driven labels destroy it).
- it 2: 3 episodes logged, then stopped. Weights `bc_daga0.npz`, `bc_daga1.npz`; data `dagger_daga_data.npz` {X (11073, 101), Y (11073, 27)}.

### 7.3 `out_bc/dagger_dagp.log` (same series, cmd - q action parametrisation)

- it 0: 14 episodes all 8.0 s, 5600 samples. `[eval dagp0] 0.00 / 0.00, mean t 1.9 s`.
- it 1: 14 episodes (`twinres_r4@41` 6.4 s, 321 labels), aggregated 11,071. `[eval dagp1] 0.00 / 0.00, mean t 1.5 s`.
- it 2 (beta 0.25): `base10_da_r4@44` 6.3 s (317 labels), `twinnoise_r4@47` 7.2 s (358), `twinres_r1@44` 5.6 s (279): the mixed policy itself now falls. Stopped. This is the "cmd - q makes the posture a random walk, knees straighten 31 deg to locked in 1.5 s" result. Weights `bc_dagp0.npz`, `bc_dagp1.npz`; data `dagger_dagp_data.npz` {X (11071, 101), Y (11071, 27)}.

### 7.4 `out_bc/dagger_dagk8.log` (teacher K=8 iterations per 20 ms label, 6 s episodes, 300 labels each, ~65 s wall per episode)

- it 0 (beta 1.00, teacher-driven): 12 episodes, all 6.0 s, pushes 26-63 N on 8 of 12; aggregated 3600. `[eval dagk80] no-push 0.81, push 0.79, mean t 12.6/15 s`.
- it 1 (beta 0.00, pure student-driven with teacher labels): 12 episodes all 6.0 s (the student drove them to 6 s), aggregated 7200. `[eval dagk81] 0.13 / 0.13, mean t 4.1 s` (81 % -> 13 %: labels on trajectories the teacher did not drive are tied to its own plan history).
- it 2: first episode `lowauth_r3@44` survived 2.4 s (119 labels); stopped. Weights `bc_dagk80.npz`, `bc_dagk81.npz`; data `dagger_dagk8_data.npz` {X (7200, 101), Y (7200, 27)}.

### 7.5 `out_bc/dagger_dagk8b.log`, `dagger_dagk8c.log`, `dagger_dart15.log` (further teacher-driven K=8 collections)

- `dagk8b`: 30 teacher-driven episodes (it 0 only), 28 x 6.0 s, `ens_r5@47` 3.9 s (196 labels), `dumpnoff_r1@38` 4.9 s (244 labels); mean survival 5.9 s; aggregated 8740. `[eval dagk8b0] no-push 0.84, push 0.74, mean t 12.7 s`. Data `dagger_dagk8b_data.npz` {X (8740, 101), Y (8740, 27)}; student `bc_dagk8b0.npz`.
- `dagk8c`: 30 episodes, 29 x 6.0 s, `dumpnoff_r1@47` 4.5 s (227 labels); aggregated 8877. `[eval dagk8c0] 0.26 / 0.42, mean t 8.9 s` (the "check-mode" collection whose label std 5.3 vs 3.5 deg poisoned fits and was excluded). Data `dagger_dagk8c_data.npz` {X (8877, 101)}.
- `dart15`: 30 episodes with DART noise (`DAG_DART`, sigma 1.5 deg) on the executed teacher commands; 28 x 6.0 s, `ens_r1@47` 4.6 s (229), `twinnoise_r4@47` 5.2 s (259); aggregated 8788. `[eval dart150] 0.65 / 0.42, mean t 9.4 s`. Data `dagger_dart15_data.npz` {X (8788, 101)}; student `bc_dart150.npz`.
- Combined teacher label sets: `teacher_stand_K8_abc.npz` {X (21217, 101), Y (21217, 27)} (dagk8 + dagk8b + dagk8c) and `teacher_stand_K8_all.npz` {X (14740, 101)}. The headline v1 student (`bc_stand_v1.npz`, 12.3k labels from 42 episodes) was fit from dagk8 + dagk8b.

### 7.6 `out_bc/teacher_ref.log`, `agent_server_teacherref.log` (teacher reference, table still present)

`[teacher K=2 ref]` from held-out standing states, 15 s episodes, forward/backward pushes only: push 0 N survive 1.00 (n=11, mean t 15.0 s); 90 N 0.55 (n=22, mean t 11.0 s); 120 N 0.55 (n=22, mean t 10.9 s). The agent_server log shows the two posture keyframes (`id=0 z=1.028 knees 0.000`, `stand_up id=1 z=1.000 knees 0.550`) and the trim loop parking the pelvis +0.026 / +0.031 m forward at 5 / 10 s (`armed=1` at 10 s). Presumed table-assisted (section 30.5).

### 7.7 `out_bc/smooth.log` (label smoothing over 5 and 11 ticks, 6 seeds each, table still present)

| variant | seed selection scores (closed-loop on training states) | best-of-6 hold eval (15 s) no-push / push / mean t | stress 30 s: 0 N (n=33) | stress 30 s: 90 N (n=132) |
|---|---|---|---|---|
| smooth5 | 0.70, 0.62, 0.58, 0.45, 0.40, 0.33 | 0.67 / 0.70 / 12.2 s | 0.61 (mean t 22.0 s) | 0.49 (18.1 s) |
| smooth11 | 0.67, 0.58, 0.57, 0.25, 0.30, 0.32 | 0.67 / 0.71 / 12.3 s | 0.61 (21.7 s) | 0.54 (19.1 s) |

Weights `bc_sm5_s1..s6.npz`, `bc_sm11_s1..s6.npz`, `bc_stand_smooth5.npz`, `bc_stand_smooth11.npz`. Retracted with the rest (table-assisted).

### 7.8 `out_bc/check_reset.log`, `teacher_check_notable.log` (Markov-teacher check, table 100 m away)

Both with `NO_TABLE`: plant table moved +100 m, robot-table contact counts as failure, planner table moved +100 m; 165 init states from 33 runs.

`check_reset.log` (`DAG_RESET_PLAN=1`, K=16 from scratch before every label, 4 s episodes, 200 labels):

| init | push | survived | labels | ms/label | mean abs cmd - q |
|---|---|---|---|---|---|
| base10_da_r1@38 | none | 4.0/4.0 s | 200 | 584 | 11.12 deg |
| base10_da_r1@41 | none | 4.0 | 200 | 589 | 11.19 |
| base10_da_r1@44 | none | 4.0 | 200 | 589 | 11.17 |
| base10_da_r1@47 | none | 4.0 | 200 | 594 | 11.14 |
| base10_da_r2@38 | (2.0 s, 60 N, 0) | 3.6 | 181 | 598 | 11.19 |
| base10_da_r2@41 | (2.0 s, 0, 60 N) | 4.0 | 200 | 592 | 11.20 |

`teacher_check_notable.log` (warm-started K=8, 8 s episodes, 400 labels):

| init | push | survived | labels | ms/label | mean abs cmd - q |
|---|---|---|---|---|---|
| base10_da_r1@38 | none | 8.0/8.0 s | 400 | 218 | 6.60 deg |
| base10_da_r1@41 | none | 8.0 | 400 | 216 | 6.66 |
| base10_da_r1@44 | none | 8.0 | 400 | 215 | 6.42 |
| base10_da_r1@47 | none | 8.0 | 400 | 213 | 6.98 |
| base10_da_r2@38 | (2.0, 60, 0) | 8.0 | 400 | 215 | 6.39 |
| base10_da_r2@41 | (2.0, 0, 60) | 8.0 | 400 | 213 | 6.05 |

Label sets `check_K16_0..5.npz` {X (200 or 131, 101), Y (200, 27)}, `check_K8_0..5.npz` and `check_K2_0..5.npz` {X (400, 101), Y (400, 27)}. Result (report 30.6): re-planned labels have tick jitter 2.5 deg (0.4 warm-started), a state-only fit of 3.4 deg vs a mean predictor of 2.7, and the previous command does not help either (3.4): the Markov teacher's label is noise; the warm-started teacher's label is history.

### 7.9 `out_bc/dagger_free1.log`, `free_fit.log`, `agent_server_free1.log` (free-standing redo: the valid stage-3 result)

`dagger_free1.log`: NO_TABLE in both plant and planner; 36 teacher-driven episodes (K=8, 6 s, 300 labels, 63-73 s wall each); 35 x 6.0 s and `lowauth_r2@38` 5.4 s (273 labels); pushes 20-69 N on 22 of 36; aggregated 10,723 labels. `[eval free10] no-push 0.48, push 0.03, mean t 7.3/15 s` (single-seed student `bc_free10.npz`).

`free_fit.log`: 8 seeds (`bc_free_s1..s8.npz`) on the 10,723 free-standing teacher labels, closed-loop selection score on training states: seed 1 0.03, 2 0.18, 3 0.10, 4 0.10, 5 0.28, 6 0.03, 7 0.05, 8 0.07 (ranking 0.28, 0.18, 0.10, 0.10, 0.07, 0.05, 0.03, 0.03; best = seed 5 -> `bc_stand_free_sel.npz`). Held-out 30 s free-standing evaluation of the best-of-8:

| push (0.3 s at the torso, 4 directions) | survive 30 s | n | mean t | table touch at end |
|---|---|---|---|---|
| 0 N | 0.21 | 33 | 11.7 s | 0 |
| 30 N | 0.11 | 132 | 9.6 s | 0 |
| 60 N | 0.03 | 132 | 6.9 s | 0 |
| 90 N | 0.01 | 132 | 5.9 s | 0 |

The teacher itself (K=8 warm-started, 50 Hz) stood free 6/6 including 60 N pushes in the same protocol (report 30.6).

### 7.10 `out_bc/agent_server_*.log` (teacher server logs)

`agent_server_{dag,daga,dagp,dagk8,dagk8b,dagk8c,dart15,free1,teacherref}.log`: each starts with `Server listening on [::]:20292`, the two posture keyframes, and `[trim]` lines (park fwd +0.016 to +0.031 m at 5 s). `agent_server_dag.log` (292 lines) additionally prints the full 29-joint qpos order of the planner model (floating_base_joint at 0; left_hip_yaw..right_ankle_roll at qpos 7-18; torso 19; left arm 20-26; right arm 27-33; object_anchor_joint 34), `[grasp-gate] stand_up entry -> OPEN jaws`, `Keyframe sequence loaded from .../strategies/h12_sweep.json`, and then 127 `[mju_warning] Nan, Inf or huge value in QACC at DOF n ... Time = 0.10-0.54` paired with 126 `Rollout divergence at step` lines: the planner's sampled rollouts diverging when the student-driven states of 7.1 left the stand manifold (harmless to the teacher, which drops those samples).

### 7.11 `out_bc/*.npz` inventory (119 files, 703.6 MB, all deleted)

- `bc_data.npz` (587.2 MB): the BC dataset from 93 twin runs, {X (1415291, 74), Y (1415291, 54), run, phase, t, val (1415291,), dirs (152,)}; `t` spans 169.1 s.
- Student weight files (3x256 unless noted), 0.7 MB each, {mu, sd (nin,), ymu, ysd (54,), nl, mask (nin,), W0..W3, b0..b3}: with 101 inputs `bc_a+b, bc_a_only, bc_ab+dart_s1..s6, bc_ab_s2..s5, bc_all_ep12, bc_all_ep30, bc_b+c, bc_b_ep30_seed1, bc_b_s2, bc_b_s3, bc_c_only, bc_daga0, bc_daga1, bc_dagk80, bc_dagk81, bc_dagk8b0, bc_dagk8c0, bc_dagp0, bc_dagp1, bc_dart150, bc_dart_only_s1..s6, bc_free10, bc_free_s1..s8, bc_refit_s0..s2, bc_sel_s1..s8, bc_sm11_s1..s6, bc_sm5_s1..s6, bc_stand_a, bc_stand_ab+dart, bc_stand_b, bc_stand_dart_only, bc_stand_free_sel, bc_stand_sel, bc_stand_smooth11, bc_stand_smooth5, bc_stand_v1, bc_stand_v2, bc_stand_v2_s1..s3`; wide 3x512 (2.4 MB) `bc_all_ep30_wide, bc_stand_c`; deeper 4x256 (1.0 MB) `bc_stand_d`; 74-input (no previous command) 3x256 `bc_dag0, bc_dag1, bc_dagonly, bc_v1`; 74-input 2x512 `bc_dagonly_wide` (1.3 MB); 74-input 2x128 (0.1 MB) `bc_nodq, bc_noise`; 101-input 2x128 `bc_refit_small` (0.2 MB).
- DAgger label sets {X (n, nin), Y (n, 27)}: `dagger_dag_data` (11011, 74), `dagger_daga_data` (11073, 101), `dagger_dagp_data` (11071, 101), `dagger_dagk8_data` (7200, 101), `dagger_dagk8b_data` (8740, 101), `dagger_dagk8c_data` (8877, 101), `dagger_dart15_data` (8788, 101), `dagger_free1_data` (10723, 101).
- Teacher label pools: `teacher_stand_K8_abc` (21217, 101), `teacher_stand_K8_all` (14740, 101).
- Markov-teacher checks: `check_K2_0..5` (400 each), `check_K8_0..5` (400 each), `check_K16_0..5` (200, 200, 200, 200, 131, 200).

---
## 8. Dataset catalogue

### 8.1 Transition corpus format (`build_dataset.py` output, identical for real and twin)

`data*/runs/<run>.npz`, float32, one row per 100 Hz recorder sample whose row t and t+1 are both finite and 8-12 ms apart: {idx (K,) = indices of valid transitions, t (N,), q (N, 27), dq (N, 27), cmd (N, 27) PD target, tau (N, 27) tau_est, kp (N, 27), kd (N, 27), tau_ff (N, 27), base_quat (N, 4) torso IMU, base_pos (N, 3) estimator, base_vel (N, 6), gyro (N, 3), accel (N, 3), bp_ok (N,) estimator-pose valid flag, tau_lim (27,)}. `data*/manifest.json` lists (run, n = valid transitions, n_bp = transitions with a valid base pose, strategy, split, src). Split by run with seeded hashing: real 384 / 54 / 55, twin 157 / 21 / 12 (train / val / test).

`data*/simgap/<run>.npz` (`simgap.py` output on every 2nd loaded-feet sample, `MBRL_STRIDE=2`): {idx (K,), q_pd / dq_pd (K, 27) plant prediction with the PD command, q_tau / dq_tau (K, 27) with measured torque, tau_pd0 (K, 27) reconstructed torque at step start, w_pd / w_tau (K, 3) predicted pelvis rate, ncon (K,) contacts, tilt (K,) torso tilt deg, ok (K,) consistency flag (both soles within 3 mm of the floor after flattening; `MBRL_GAP_MAX` is the extra gap allowance, default 0.0), w_rec0 / w_rec1 (K, 3) recorded pelvis rate at t and t+1, foot_corr (K, 7) = per-sample foot-flattening / anchoring diagnostics (ankle pitch and roll corrections of each foot in deg, the pelvis height correction and the sole-gap terms; the exact column order is defined in `simgap.py`, which is kept)}.

`data/jac/<run>.npz` (`jacobians.py`): {run (), split (), i0 (K,) segment start indices, xn (K, 26, 67) nominal plant rollout states (qpos 34 + qvel 33) over T+1 = 26 steps, A (K, 25, 66, 66) finite-difference Jacobians of the 10 ms macro step in tangent coordinates (base pos 3, base rotvec 3, joints 27 | vel 33)}. 401 files (train + val runs with at least one valid segment), 4860 segments, 2146.9 MB, 0.4 to 6.6 MB per file.

### 8.2 npz scan summary (sizes as scanned on 2026-10-11, before deletion)

| directory | files | total | per-file range | duration (recorder t) | schema |
|---|---|---|---|---|---|
| `data/runs` | 493 | 5386.2 MB | 0.9 to 76.1 MB | 10.9 s to 900.0 s per run, 18.30 h total, 6,373,266 rows | runs schema above |
| `data/simgap` | 493 | 2505.0 MB | 0.0 to 43.2 MB | - | simgap schema above |
| `data/jac` | 401 | 2146.9 MB | 0.4 to 6.6 MB | - | jac schema above |
| `data/` total | 1387 | 10.04 GB (du 9.4 GB) | | | |
| `data_twin/runs` | 190 | 2813.0 MB | 8.7 to 20.6 MB | 103.3 s to 243.3 s per run, 9.24 h total, 3,326,672 rows | runs schema |
| `data_twin/simgap` | 190 | 1543.6 MB | 1.3 to 14.1 MB | - | simgap schema |
| `data_twin/` total | 380 | 4.35 GB (du 4.1 GB) | | | |
| `out/` | 7 | 5.5 MB | | | `residual_norm*.npz` (5 x {mu, sd (168,), ymu, ysd, clip (27,)}; `residual_norm.npz` has W_ridge (169, 27) instead of clip), `rollout_err.npz` {tilt0 (339,), err_plant / err_ridge / err_mlp (339, 50, 27)} |
| `out_twin/` | 8 | < 1 MB | | | `residual_norm_{twinpre10s, twinnoise, twinnoff, ens1..ens5}.npz`, each {mu, sd (168,), ymu, ysd, clip (27,)} |
| `out_bc/` | 119 | 703.6 MB | 0.1 to 587.2 MB | `bc_data.npz` t spans 169.1 s | see section 7.11 |

Feature vector (168) used by every residual: q 27, dq 27, tau_cmd0 27, cmd - q 27, kp 27, kd 27, gravity direction in the pelvis/IMU frame 3, pelvis angular rate 3. Target 27 = recorded next joint velocity minus the plant's predicted next joint velocity.

### 8.3 Real-robot corpus manifest (`data/manifest.json`, 493 runs, 6,099,684 valid transitions; splits train 384 / val 54 / test 55)

Per-strategy totals (strategy label = prefix of the recording directory name under `~/Desktop/h12/logs`):

| strategy | runs | valid transitions |
|---|---|---|
| 10 | 12 | 96243 |
| 11 | 36 | 840771 |
| 12 | 1 | 13543 |
| 12new | 3 | 41535 |
| 22 | 1 | 20445 |
| 22again | 21 | 247760 |
| 22new | 1 | 12140 |
| 25 | 15 | 287920 |
| 25new | 5 | 60380 |
| 26 | 13 | 175318 |
| 27 | 4 | 31297 |
| 28 | 4 | 49188 |
| 29 | 78 | 855082 |
| 36 | 17 | 223424 |
| 9 | 67 | 696777 |
| brace22 | 1 | 3900 |
| crouch8 | 1 | 3707 |
| ghost25 | 1 | 11873 |
| nobrace0 | 6 | 47037 |
| nobrace3 | 3 | 33688 |
| real | 142 | 1444733 |
| realstandsumma3 | 1 | 113 |
| s29 | 6 | 86544 |
| sq18cold | 1 | 6092 |
| sq18harn | 1 | 8001 |
| sq18held | 1 | 7032 |
| sq18supp | 1 | 4998 |
| stand6 | 1 | 8041 |
| stand6ready | 1 | 4844 |
| stand6ready2 | 1 | 4138 |
| stand6supp | 1 | 7331 |
| standB2 | 1 | 360 |
| table | 42 | 715603 |
| v3newstand | 1 | 18254 |
| v3stand | 1 | 15251 |
| v4lnewstand | 1 | 16321 |

Full run table (493 runs; `n` = valid transitions, `n_bp` = transitions with a valid estimator base pose). Source paths are relative to `~/Desktop/h12/logs/`; each is `<run>/<run>.npz`.

| run | n | n_bp | strategy | split | source (relative to ~/Desktop/h12/logs/) |
|---|---|---|---|---|---|
| 10_10_20260908_230933 | 17859 | 17690 | 10 | train | `10_10_20260908_230933/10_10_20260908_230933.npz` |
| 10_11_20260908_231316 | 11389 | 11167 | 10 | val | `10_11_20260908_231316/10_11_20260908_231316.npz` |
| 10_12_20260908_231539 | 13590 | 13367 | 10 | test | `10_12_20260908_231539/10_12_20260908_231539.npz` |
| 10_1_20260908_225008 | 6456 | 6055 | 10 | train | `10_1_20260908_225008/10_1_20260908_225008.npz` |
| 10_2_20260908_225205 | 16358 | 15453 | 10 | test | `10_2_20260908_225205/10_2_20260908_225205.npz` |
| 10_3_20260908_225925 | 9154 | 8804 | 10 | train | `10_3_20260908_225925/10_3_20260908_225925.npz` |
| 10_4_20260908_230105 | 3769 | 3571 | 10 | train | `10_4_20260908_230105/10_4_20260908_230105.npz` |
| 10_5_20260908_230225 | 3362 | 3164 | 10 | train | `10_5_20260908_230225/10_5_20260908_230225.npz` |
| 10_6_20260908_230319 | 3443 | 3174 | 10 | train | `10_6_20260908_230319/10_6_20260908_230319.npz` |
| 10_7_20260908_230414 | 3555 | 2933 | 10 | val | `10_7_20260908_230414/10_7_20260908_230414.npz` |
| 10_8_20260908_230507 | 3253 | 3010 | 10 | train | `10_8_20260908_230507/10_8_20260908_230507.npz` |
| 10_9_20260908_230831 | 4055 | 3888 | 10 | val | `10_9_20260908_230831/10_9_20260908_230831.npz` |
| 11_+5_101112_20260913_065226 | 26531 | 26272 | 11 | train | `11_+5_101112_20260913_065226/11_+5_101112_20260913_065226.npz` |
| 11_+5_123_20260913_060903 | 36138 | 35582 | 11 | train | `11_+5_123_20260913_060903/11_+5_123_20260913_060903.npz` |
| 11_+5_131415_20260913_070215 | 20448 | 20155 | 11 | train | `11_+5_131415_20260913_070215/11_+5_131415_20260913_070215.npz` |
| 11_+5_161718_20260913_070706 | 89045 | 88621 | 11 | train | `11_+5_161718_20260913_070706/11_+5_161718_20260913_070706.npz` |
| 11_+5_456_20260913_063031 | 33505 | 33173 | 11 | train | `11_+5_456_20260913_063031/11_+5_456_20260913_063031.npz` |
| 11_+5_789_20260913_064321 | 29947 | 29340 | 11 | train | `11_+5_789_20260913_064321/11_+5_789_20260913_064321.npz` |
| 11_-5_101112_20260913_082540 | 33929 | 33624 | 11 | test | `11_-5_101112_20260913_082540/11_-5_101112_20260913_082540.npz` |
| 11_-5_123_20260913_072507 | 30563 | 30563 | 11 | train | `11_-5_123_20260913_072507/11_-5_123_20260913_072507.npz` |
| 11_-5_131415_20260913_083544 | 29935 | 29591 | 11 | train | `11_-5_131415_20260913_083544/11_-5_131415_20260913_083544.npz` |
| 11_-5_161718_20260913_084309 | 30080 | 29808 | 11 | train | `11_-5_161718_20260913_084309/11_-5_161718_20260913_084309.npz` |
| 11_-5_456_20260913_074117 | 29762 | 29366 | 11 | train | `11_-5_456_20260913_074117/11_-5_456_20260913_074117.npz` |
| 11_-5_6_20260913_080811 | 11969 | 11233 | 11 | val | `11_-5_6_20260913_080811/11_-5_6_20260913_080811.npz` |
| 11_-5_789_20260913_081941 | 25192 | 24957 | 11 | train | `11_-5_789_20260913_081941/11_-5_789_20260913_081941.npz` |
| 11_nominal_10_20260913_021335 | 10876 | 10577 | 11 | train | `11_nominal_10_20260913_021335/11_nominal_10_20260913_021335.npz` |
| 11_nominal_11_20260913_023134 | 10618 | 10339 | 11 | train | `11_nominal_11_20260913_023134/11_nominal_11_20260913_023134.npz` |
| 11_nominal_12_20260913_025450 | 13310 | 12732 | 11 | train | `11_nominal_12_20260913_025450/11_nominal_12_20260913_025450.npz` |
| 11_nominal_13_20260913_031243 | 14015 | 13704 | 11 | train | `11_nominal_13_20260913_031243/11_nominal_13_20260913_031243.npz` |
| 11_nominal_14_20260913_032408 | 15006 | 14806 | 11 | train | `11_nominal_14_20260913_032408/11_nominal_14_20260913_032408.npz` |
| 11_nominal_15_20260913_032742 | 16703 | 16338 | 11 | train | `11_nominal_15_20260913_032742/11_nominal_15_20260913_032742.npz` |
| 11_nominal_16_20260913_035014 | 11454 | 11136 | 11 | train | `11_nominal_16_20260913_035014/11_nominal_16_20260913_035014.npz` |
| 11_nominal_17_20260913_043110 | 12680 | 12102 | 11 | train | `11_nominal_17_20260913_043110/11_nominal_17_20260913_043110.npz` |
| 11_nominal_181920_20260913_043559 | 31743 | 31270 | 11 | train | `11_nominal_181920_20260913_043559/11_nominal_181920_20260913_043559.npz` |
| 11_nominal_1_20260912_202906 | 11963 | 11730 | 11 | test | `11_nominal_1_20260912_202906/11_nominal_1_20260912_202906.npz` |
| 11_nominal_212223_20260913_044755 | 32128 | 31680 | 11 | val | `11_nominal_212223_20260913_044755/11_nominal_212223_20260913_044755.npz` |
| 11_nominal_242526_20260913_051650 | 49459 | 49140 | 11 | train | `11_nominal_242526_20260913_051650/11_nominal_242526_20260913_051650.npz` |
| 11_nominal_272829_20260913_053230 | 28452 | 28156 | 11 | train | `11_nominal_272829_20260913_053230/11_nominal_272829_20260913_053230.npz` |
| 11_nominal_2_20260912_203412 | 11478 | 11178 | 11 | test | `11_nominal_2_20260912_203412/11_nominal_2_20260912_203412.npz` |
| 11_nominal_303132_20260913_053927 | 31027 | 30803 | 11 | train | `11_nominal_303132_20260913_053927/11_nominal_303132_20260913_053927.npz` |
| 11_nominal_333435_20260913_055056 | 39622 | 38247 | 11 | test | `11_nominal_333435_20260913_055056/11_nominal_333435_20260913_055056.npz` |
| 11_nominal_3_20260912_204034 | 10397 | 10094 | 11 | train | `11_nominal_3_20260912_204034/11_nominal_3_20260912_204034.npz` |
| 11_nominal_4_20260912_204518 | 9513 | 9264 | 11 | train | `11_nominal_4_20260912_204518/11_nominal_4_20260912_204518.npz` |
| 11_nominal_5_20260912_213622 | 8238 | 7957 | 11 | train | `11_nominal_5_20260912_213622/11_nominal_5_20260912_213622.npz` |
| 11_nominal_6_20260912_220934 | 11740 | 11417 | 11 | train | `11_nominal_6_20260912_220934/11_nominal_6_20260912_220934.npz` |
| 11_nominal_7_20260912_221725 | 9666 | 9210 | 11 | train | `11_nominal_7_20260912_221725/11_nominal_7_20260912_221725.npz` |
| 11_nominal_8_20260913_020441 | 14718 | 13971 | 11 | train | `11_nominal_8_20260913_020441/11_nominal_8_20260913_020441.npz` |
| 11_nominal_9_20260913_020918 | 8921 | 8555 | 11 | train | `11_nominal_9_20260913_020918/11_nominal_9_20260913_020918.npz` |
| 12_nominal_1_20260912_223100 | 13543 | 13283 | 12 | train | `12_nominal_1_20260912_223100/12_nominal_1_20260912_223100.npz` |
| 12new_1_20260912_200955 | 16041 | 15751 | 12new | train | `12new_1_20260912_200955/12new_1_20260912_200955.npz` |
| 12new_2_20260912_201525 | 11771 | 11280 | 12new | train | `12new_2_20260912_201525/12new_2_20260912_201525.npz` |
| 12new_3_20260912_202401 | 13723 | 13064 | 12new | train | `12new_3_20260912_202401/12new_3_20260912_202401.npz` |
| 22_maxreach_1_20260915_054152 | 20445 | 20105 | 22 | train | `22_maxreach_1_20260915_054152/22_maxreach_1_20260915_054152.npz` |
| 22again_10_20260915_080422 | 19739 | 19454 | 22again | train | `22again_10_20260915_080422/22again_10_20260915_080422.npz` |
| 22again_11_20260915_081404 | 10101 | 9873 | 22again | test | `22again_11_20260915_081404/22again_11_20260915_081404.npz` |
| 22again_12_20260915_081822 | 9409 | 9122 | 22again | test | `22again_12_20260915_081822/22again_12_20260915_081822.npz` |
| 22again_13_20260915_082226 | 6673 | 6495 | 22again | test | `22again_13_20260915_082226/22again_13_20260915_082226.npz` |
| 22again_14_20260915_082646 | 8192 | 7877 | 22again | train | `22again_14_20260915_082646/22again_14_20260915_082646.npz` |
| 22again_15_20260915_082948 | 22951 | 22801 | 22again | train | `22again_15_20260915_082948/22again_15_20260915_082948.npz` |
| 22again_16_20260915_083541 | 8377 | 8232 | 22again | test | `22again_16_20260915_083541/22again_16_20260915_083541.npz` |
| 22again_18_20260915_084046 | 14405 | 14014 | 22again | train | `22again_18_20260915_084046/22again_18_20260915_084046.npz` |
| 22again_19_20260915_084613 | 11794 | 11659 | 22again | train | `22again_19_20260915_084613/22again_19_20260915_084613.npz` |
| 22again_1_20260915_070231 | 12144 | 11841 | 22again | train | `22again_1_20260915_070231/22again_1_20260915_070231.npz` |
| 22again_20_20260915_085023 | 9235 | 8972 | 22again | test | `22again_20_20260915_085023/22again_20_20260915_085023.npz` |
| 22again_21_20260915_085246 | 9195 | 8762 | 22again | train | `22again_21_20260915_085246/22again_21_20260915_085246.npz` |
| 22again_22_20260915_090010 | 7969 | 7686 | 22again | test | `22again_22_20260915_090010/22again_22_20260915_090010.npz` |
| 22again_2_20260915_071245 | 16001 | 15613 | 22again | val | `22again_2_20260915_071245/22again_2_20260915_071245.npz` |
| 22again_3_20260915_072021 | 15543 | 15425 | 22again | train | `22again_3_20260915_072021/22again_3_20260915_072021.npz` |
| 22again_4_20260915_072715 | 11558 | 11043 | 22again | train | `22again_4_20260915_072715/22again_4_20260915_072715.npz` |
| 22again_5_20260915_073537 | 10853 | 10477 | 22again | train | `22again_5_20260915_073537/22again_5_20260915_073537.npz` |
| 22again_6_20260915_073909 | 10559 | 10101 | 22again | train | `22again_6_20260915_073909/22again_6_20260915_073909.npz` |
| 22again_7_20260915_074358 | 9870 | 9574 | 22again | train | `22again_7_20260915_074358/22again_7_20260915_074358.npz` |
| 22again_8_20260915_075346 | 13217 | 12923 | 22again | train | `22again_8_20260915_075346/22again_8_20260915_075346.npz` |
| 22again_9_20260915_075737 | 9975 | 9764 | 22again | val | `22again_9_20260915_075737/22again_9_20260915_075737.npz` |
| 22new_1_20260915_023843 | 12140 | 11709 | 22new | test | `22new_1_20260915_023843/22new_1_20260915_023843.npz` |
| 25_icemplan_1_20260825_231240 | 6698 | 5610 | 25 | test | `25_icemplan_1_20260825_231240/25_icemplan_1_20260825_231240.npz` |
| 25_icemplan_2_20260825_232736 | 8058 | 7405 | 25 | test | `25_icemplan_2_20260825_232736/25_icemplan_2_20260825_232736.npz` |
| 25_icemplan_3_20260825_234802 | 14574 | 13855 | 25 | train | `25_icemplan_3_20260825_234802/25_icemplan_3_20260825_234802.npz` |
| 25_icemplan_4_20260825_235929 | 11316 | 10595 | 25 | train | `25_icemplan_4_20260825_235929/25_icemplan_4_20260825_235929.npz` |
| 25_icemplan_5_20260826_004728 | 10613 | 9975 | 25 | train | `25_icemplan_5_20260826_004728/25_icemplan_5_20260826_004728.npz` |
| 25_icemplan_6_20260826_152307 | 4903 | 4218 | 25 | test | `25_icemplan_6_20260826_152307/25_icemplan_6_20260826_152307.npz` |
| 25_icemplan_7_20260826_153109 | 6373 | 5479 | 25 | train | `25_icemplan_7_20260826_153109/25_icemplan_7_20260826_153109.npz` |
| 25_icemplan_8_20260826_153930 | 13638 | 13016 | 25 | train | `25_icemplan_8_20260826_153930/25_icemplan_8_20260826_153930.npz` |
| 25_icemplan_9_20260826_155034 | 12326 | 11426 | 25 | train | `25_icemplan_9_20260826_155034/25_icemplan_9_20260826_155034.npz` |
| 25_loop_1_20260903_215839 | 23050 | 21462 | 25 | test | `25_loop_1_20260903_215839/25_loop_1_20260903_215839.npz` |
| 25_loop_2_20260903_222504 | 4129 | 3903 | 25 | train | `25_loop_2_20260903_222504/25_loop_2_20260903_222504.npz` |
| 25_loop_3_20260903_225800 | 18106 | 17843 | 25 | train | `25_loop_3_20260903_225800/25_loop_3_20260903_225800.npz` |
| 25_loop_4_20260903_230933 | 53869 | 53618 | 25 | train | `25_loop_4_20260903_230933/25_loop_4_20260903_230933.npz` |
| 25_loop_5_20260904_010444 | 25875 | 25694 | 25 | train | `25_loop_5_20260904_010444/25_loop_5_20260904_010444.npz` |
| 25_loop_6_20260904_011314 | 74392 | 74113 | 25 | val | `25_loop_6_20260904_011314/25_loop_6_20260904_011314.npz` |
| 25new_1_20260912_194951 | 6370 | 5803 | 25new | train | `25new_1_20260912_194951/25new_1_20260912_194951.npz` |
| 25new_2_20260912_195231 | 16193 | 15791 | 25new | train | `25new_2_20260912_195231/25new_2_20260912_195231.npz` |
| 25new_3_20260912_200356 | 13976 | 13683 | 25new | train | `25new_3_20260912_200356/25new_3_20260912_200356.npz` |
| 25new_4_20260912_214520 | 15169 | 14735 | 25new | val | `25new_4_20260912_214520/25new_4_20260912_214520.npz` |
| 25new_5_20260912_214911 | 8672 | 8301 | 25new | train | `25new_5_20260912_214911/25new_5_20260912_214911.npz` |
| 26_10_20260915_044311 | 13709 | 11846 | 26 | train | `26_10_20260915_044311/26_10_20260915_044311.npz` |
| 26_11_20260915_044924 | 9753 | 9286 | 26 | train | `26_11_20260915_044924/26_11_20260915_044924.npz` |
| 26_13_20260915_060122 | 13045 | 12677 | 26 | train | `26_13_20260915_060122/26_13_20260915_060122.npz` |
| 26_14_20260915_060930 | 14201 | 13787 | 26 | train | `26_14_20260915_060930/26_14_20260915_060930.npz` |
| 26_1_20260915_024854 | 12850 | 12398 | 26 | val | `26_1_20260915_024854/26_1_20260915_024854.npz` |
| 26_2_20260915_035204 | 26479 | 26227 | 26 | train | `26_2_20260915_035204/26_2_20260915_035204.npz` |
| 26_3_20260915_035925 | 13292 | 12588 | 26 | train | `26_3_20260915_035925/26_3_20260915_035925.npz` |
| 26_4_20260915_040728 | 10258 | 9874 | 26 | train | `26_4_20260915_040728/26_4_20260915_040728.npz` |
| 26_5_20260915_041606 | 5959 | 5690 | 26 | train | `26_5_20260915_041606/26_5_20260915_041606.npz` |
| 26_6_20260915_042356 | 16768 | 16526 | 26 | train | `26_6_20260915_042356/26_6_20260915_042356.npz` |
| 26_7_20260915_042849 | 17470 | 17036 | 26 | val | `26_7_20260915_042849/26_7_20260915_042849.npz` |
| 26_8_20260915_043711 | 8721 | 6335 | 26 | val | `26_8_20260915_043711/26_8_20260915_043711.npz` |
| 26_9_20260915_043959 | 12813 | 12386 | 26 | train | `26_9_20260915_043959/26_9_20260915_043959.npz` |
| 27_1_20260827_001218 | 5273 | 4467 | 27 | train | `27_1_20260827_001218/27_1_20260827_001218.npz` |
| 27_2_20260827_002132 | 6986 | 6217 | 27 | train | `27_2_20260827_002132/27_2_20260827_002132.npz` |
| 27_3_20260827_003235 | 10804 | 10060 | 27 | train | `27_3_20260827_003235/27_3_20260827_003235.npz` |
| 27_4_20260827_005213 | 8234 | 7436 | 27 | val | `27_4_20260827_005213/27_4_20260827_005213.npz` |
| 28_1_20260827_005625 | 11398 | 10645 | 28 | val | `28_1_20260827_005625/28_1_20260827_005625.npz` |
| 28_2_20260827_010537 | 7705 | 6983 | 28 | train | `28_2_20260827_010537/28_2_20260827_010537.npz` |
| 28_3_20260827_012623 | 10610 | 9794 | 28 | train | `28_3_20260827_012623/28_3_20260827_012623.npz` |
| 28_4_20260827_020945 | 19475 | 18731 | 28 | train | `28_4_20260827_020945/28_4_20260827_020945.npz` |
| 29_10_20260828_042924 | 5775 | 5123 | 29 | val | `29_10_20260828_042924/29_10_20260828_042924.npz` |
| 29_11_20260828_043036 | 6705 | 6028 | 29 | train | `29_11_20260828_043036/29_11_20260828_043036.npz` |
| 29_12_20260828_043210 | 7316 | 6701 | 29 | train | `29_12_20260828_043210/29_12_20260828_043210.npz` |
| 29_13_20260828_044749 | 7900 | 7317 | 29 | test | `29_13_20260828_044749/29_13_20260828_044749.npz` |
| 29_14_20260828_211956 | 8512 | 7867 | 29 | train | `29_14_20260828_211956/29_14_20260828_211956.npz` |
| 29_15_20260828_212146 | 7453 | 6790 | 29 | train | `29_15_20260828_212146/29_15_20260828_212146.npz` |
| 29_16_20260828_214027 | 13261 | 12610 | 29 | train | `29_16_20260828_214027/29_16_20260828_214027.npz` |
| 29_17_20260828_214722 | 8100 | 7451 | 29 | test | `29_17_20260828_214722/29_17_20260828_214722.npz` |
| 29_18_20260828_221809 | 9576 | 8867 | 29 | train | `29_18_20260828_221809/29_18_20260828_221809.npz` |
| 29_19_20260828_222025 | 10908 | 10235 | 29 | train | `29_19_20260828_222025/29_19_20260828_222025.npz` |
| 29_1_20260828_005521 | 7099 | 6400 | 29 | test | `29_1_20260828_005521/29_1_20260828_005521.npz` |
| 29_20_20260828_222412 | 8388 | 7806 | 29 | train | `29_20_20260828_222412/29_20_20260828_222412.npz` |
| 29_21_20260828_222927 | 8514 | 7790 | 29 | val | `29_21_20260828_222927/29_21_20260828_222927.npz` |
| 29_22_20260828_223453 | 12269 | 11580 | 29 | train | `29_22_20260828_223453/29_22_20260828_223453.npz` |
| 29_23_20260828_224449 | 6464 | 5849 | 29 | val | `29_23_20260828_224449/29_23_20260828_224449.npz` |
| 29_24_20260828_225619 | 8203 | 7551 | 29 | train | `29_24_20260828_225619/29_24_20260828_225619.npz` |
| 29_25_20260828_234023 | 7264 | 6150 | 29 | train | `29_25_20260828_234023/29_25_20260828_234023.npz` |
| 29_26_20260828_235123 | 7365 | 7365 | 29 | train | `29_26_20260828_235123/29_26_20260828_235123.npz` |
| 29_27_20260829_004941 | 18975 | 18975 | 29 | train | `29_27_20260829_004941/29_27_20260829_004941.npz` |
| 29_28_20260829_015734 | 3181 | 3181 | 29 | val | `29_28_20260829_015734/29_28_20260829_015734.npz` |
| 29_29_20260829_020454 | 14660 | 14306 | 29 | val | `29_29_20260829_020454/29_29_20260829_020454.npz` |
| 29_2_20260828_021007 | 6748 | 6118 | 29 | train | `29_2_20260828_021007/29_2_20260828_021007.npz` |
| 29_30_20260829_021535 | 12113 | 11854 | 29 | train | `29_30_20260829_021535/29_30_20260829_021535.npz` |
| 29_31_20260829_022817 | 9884 | 9716 | 29 | train | `29_31_20260829_022817/29_31_20260829_022817.npz` |
| 29_32_20260829_024438 | 11372 | 11372 | 29 | train | `29_32_20260829_024438/29_32_20260829_024438.npz` |
| 29_33_20260829_030039 | 11589 | 11405 | 29 | train | `29_33_20260829_030039/29_33_20260829_030039.npz` |
| 29_34_20260829_031709 | 12727 | 12579 | 29 | train | `29_34_20260829_031709/29_34_20260829_031709.npz` |
| 29_35_20260829_032455 | 12426 | 12193 | 29 | val | `29_35_20260829_032455/29_35_20260829_032455.npz` |
| 29_36_20260829_032954 | 14491 | 14265 | 29 | train | `29_36_20260829_032954/29_36_20260829_032954.npz` |
| 29_37_20260829_033651 | 12386 | 12212 | 29 | train | `29_37_20260829_033651/29_37_20260829_033651.npz` |
| 29_38_20260829_035315 | 14748 | 14572 | 29 | train | `29_38_20260829_035315/29_38_20260829_035315.npz` |
| 29_39_20260829_040149 | 11119 | 10952 | 29 | train | `29_39_20260829_040149/29_39_20260829_040149.npz` |
| 29_3_20260828_021746 | 7113 | 6469 | 29 | test | `29_3_20260828_021746/29_3_20260828_021746.npz` |
| 29_40_20260829_040630 | 9958 | 9757 | 29 | train | `29_40_20260829_040630/29_40_20260829_040630.npz` |
| 29_41_20260829_041146 | 9928 | 9730 | 29 | test | `29_41_20260829_041146/29_41_20260829_041146.npz` |
| 29_42_20260829_041925 | 14413 | 14184 | 29 | train | `29_42_20260829_041925/29_42_20260829_041925.npz` |
| 29_43_20260829_043307 | 14505 | 14032 | 29 | train | `29_43_20260829_043307/29_43_20260829_043307.npz` |
| 29_44_20260829_044137 | 11488 | 11079 | 29 | train | `29_44_20260829_044137/29_44_20260829_044137.npz` |
| 29_45_20260829_050004 | 9724 | 9557 | 29 | train | `29_45_20260829_050004/29_45_20260829_050004.npz` |
| 29_46_20260829_050450 | 14174 | 13791 | 29 | train | `29_46_20260829_050450/29_46_20260829_050450.npz` |
| 29_47_20260829_051054 | 10772 | 10506 | 29 | train | `29_47_20260829_051054/29_47_20260829_051054.npz` |
| 29_48_20260829_051345 | 12616 | 12392 | 29 | train | `29_48_20260829_051345/29_48_20260829_051345.npz` |
| 29_49_20260829_051911 | 10457 | 10301 | 29 | train | `29_49_20260829_051911/29_49_20260829_051911.npz` |
| 29_4_20260828_021935 | 8286 | 7560 | 29 | train | `29_4_20260828_021935/29_4_20260828_021935.npz` |
| 29_50_20260829_055105 | 12618 | 12298 | 29 | train | `29_50_20260829_055105/29_50_20260829_055105.npz` |
| 29_51_20260829_055901 | 9080 | 8875 | 29 | train | `29_51_20260829_055901/29_51_20260829_055901.npz` |
| 29_52_20260829_060358 | 10849 | 10767 | 29 | train | `29_52_20260829_060358/29_52_20260829_060358.npz` |
| 29_53_20260830_012609 | 11531 | 11298 | 29 | train | `29_53_20260830_012609/29_53_20260830_012609.npz` |
| 29_54_20260830_014640 | 9708 | 9426 | 29 | train | `29_54_20260830_014640/29_54_20260830_014640.npz` |
| 29_55_20260830_015705 | 9496 | 9267 | 29 | train | `29_55_20260830_015705/29_55_20260830_015705.npz` |
| 29_56_20260830_021239 | 9984 | 9781 | 29 | train | `29_56_20260830_021239/29_56_20260830_021239.npz` |
| 29_57_20260830_022511 | 6493 | 6330 | 29 | test | `29_57_20260830_022511/29_57_20260830_022511.npz` |
| 29_58_20260830_023110 | 18318 | 18165 | 29 | train | `29_58_20260830_023110/29_58_20260830_023110.npz` |
| 29_59_20260830_024756 | 9255 | 9090 | 29 | train | `29_59_20260830_024756/29_59_20260830_024756.npz` |
| 29_5_20260828_024119 | 12579 | 11780 | 29 | train | `29_5_20260828_024119/29_5_20260828_024119.npz` |
| 29_60_20260830_030645 | 6893 | 6674 | 29 | train | `29_60_20260830_030645/29_60_20260830_030645.npz` |
| 29_61_20260830_030901 | 14458 | 14261 | 29 | train | `29_61_20260830_030901/29_61_20260830_030901.npz` |
| 29_62_20260830_032147 | 13576 | 13299 | 29 | val | `29_62_20260830_032147/29_62_20260830_032147.npz` |
| 29_63_20260830_034935 | 8974 | 8723 | 29 | test | `29_63_20260830_034935/29_63_20260830_034935.npz` |
| 29_64_20260830_040227 | 6929 | 6758 | 29 | val | `29_64_20260830_040227/29_64_20260830_040227.npz` |
| 29_65_20260830_040608 | 10589 | 9947 | 29 | train | `29_65_20260830_040608/29_65_20260830_040608.npz` |
| 29_66_20260830_041754 | 15307 | 15003 | 29 | val | `29_66_20260830_041754/29_66_20260830_041754.npz` |
| 29_67_20260830_043019 | 36752 | 36583 | 29 | train | `29_67_20260830_043019/29_67_20260830_043019.npz` |
| 29_68_20260830_045818 | 16784 | 16505 | 29 | test | `29_68_20260830_045818/29_68_20260830_045818.npz` |
| 29_69_20260830_050700 | 13196 | 12809 | 29 | train | `29_69_20260830_050700/29_69_20260830_050700.npz` |
| 29_6_20260828_030655 | 6999 | 6253 | 29 | train | `29_6_20260828_030655/29_6_20260828_030655.npz` |
| 29_70_20260830_052021 | 12246 | 11790 | 29 | train | `29_70_20260830_052021/29_70_20260830_052021.npz` |
| 29_71_20260830_052820 | 14528 | 14332 | 29 | train | `29_71_20260830_052820/29_71_20260830_052820.npz` |
| 29_72_20260830_053533 | 10323 | 10089 | 29 | train | `29_72_20260830_053533/29_72_20260830_053533.npz` |
| 29_73_20260830_060434 | 15912 | 15613 | 29 | train | `29_73_20260830_060434/29_73_20260830_060434.npz` |
| 29_74_20260831_014019 | 12744 | 12393 | 29 | train | `29_74_20260831_014019/29_74_20260831_014019.npz` |
| 29_75_20260831_014541 | 15928 | 15694 | 29 | train | `29_75_20260831_014541/29_75_20260831_014541.npz` |
| 29_76_20260831_015401 | 6371 | 6178 | 29 | test | `29_76_20260831_015401/29_76_20260831_015401.npz` |
| 29_77_20260831_015605 | 10254 | 9874 | 29 | val | `29_77_20260831_015605/29_77_20260831_015605.npz` |
| 29_78_20260831_020119 | 14507 | 14285 | 29 | test | `29_78_20260831_020119/29_78_20260831_020119.npz` |
| 29_7_20260828_035539 | 7493 | 6769 | 29 | train | `29_7_20260828_035539/29_7_20260828_035539.npz` |
| 29_8_20260828_040659 | 7263 | 6504 | 29 | train | `29_8_20260828_040659/29_8_20260828_040659.npz` |
| 29_9_20260828_041903 | 6220 | 5578 | 29 | test | `29_9_20260828_041903/29_9_20260828_041903.npz` |
| 36_10_20260925_200512 | 7835 | 5797 | 36 | train | `36_10_20260925_200512/36_10_20260925_200512.npz` |
| 36_11_20260925_200920 | 5230 | 4903 | 36 | train | `36_11_20260925_200920/36_11_20260925_200920.npz` |
| 36_12_20260925_201510 | 5843 | 5461 | 36 | train | `36_12_20260925_201510/36_12_20260925_201510.npz` |
| 36_13_20260925_202329 | 5507 | 5315 | 36 | train | `36_13_20260925_202329/36_13_20260925_202329.npz` |
| 36_14_20260925_202524 | 5379 | 5076 | 36 | train | `36_14_20260925_202524/36_14_20260925_202524.npz` |
| 36_15_20260925_203434 | 13326 | 12191 | 36 | test | `36_15_20260925_203434/36_15_20260925_203434.npz` |
| 36_16_20260925_204249 | 11675 | 11422 | 36 | train | `36_16_20260925_204249/36_16_20260925_204249.npz` |
| 36_17_20260925_204621 | 11601 | 11401 | 36 | test | `36_17_20260925_204621/36_17_20260925_204621.npz` |
| 36_1_20260925_164906 | 51452 | 47716 | 36 | train | `36_1_20260925_164906/36_1_20260925_164906.npz` |
| 36_2_20260925_170010 | 11492 | 7581 | 36 | train | `36_2_20260925_170010/36_2_20260925_170010.npz` |
| 36_3_20260925_191016 | 57288 | 56868 | 36 | train | `36_3_20260925_191016/36_3_20260925_191016.npz` |
| 36_4_20260925_192755 | 4358 | 3707 | 36 | train | `36_4_20260925_192755/36_4_20260925_192755.npz` |
| 36_5_20260925_192854 | 3762 | 3401 | 36 | train | `36_5_20260925_192854/36_5_20260925_192854.npz` |
| 36_6_20260925_193554 | 5110 | 3039 | 36 | train | `36_6_20260925_193554/36_6_20260925_193554.npz` |
| 36_7_20260925_193728 | 10713 | 9919 | 36 | test | `36_7_20260925_193728/36_7_20260925_193728.npz` |
| 36_8_20260925_194719 | 5948 | 5658 | 36 | train | `36_8_20260925_194719/36_8_20260925_194719.npz` |
| 36_9_20260925_195120 | 6905 | 6535 | 36 | train | `36_9_20260925_195120/36_9_20260925_195120.npz` |
| 9_B3_10_20260905_011702 | 3675 | 3044 | 9 | val | `9_B3_10_20260905_011702/9_B3_10_20260905_011702.npz` |
| 9_B3_11_20260905_011807 | 10995 | 10659 | 9 | val | `9_B3_11_20260905_011807/9_B3_11_20260905_011807.npz` |
| 9_B3_12_20260905_012111 | 3334 | 3048 | 9 | train | `9_B3_12_20260905_012111/9_B3_12_20260905_012111.npz` |
| 9_B3_12_20260905_012329 | 13123 | 12802 | 9 | train | `9_B3_12_20260905_012329/9_B3_12_20260905_012329.npz` |
| 9_B3_13_20260905_013124 | 8428 | 6188 | 9 | train | `9_B3_13_20260905_013124/9_B3_13_20260905_013124.npz` |
| 9_B3_14_20260905_013506 | 8853 | 8429 | 9 | train | `9_B3_14_20260905_013506/9_B3_14_20260905_013506.npz` |
| 9_B3_15_20260905_015050 | 4591 | 3961 | 9 | val | `9_B3_15_20260905_015050/9_B3_15_20260905_015050.npz` |
| 9_B3_16_20260905_015202 | 12050 | 11581 | 9 | train | `9_B3_16_20260905_015202/9_B3_16_20260905_015202.npz` |
| 9_B3_17_20260905_015718 | 17186 | 16773 | 9 | train | `9_B3_17_20260905_015718/9_B3_17_20260905_015718.npz` |
| 9_B3_18_20260905_032456 | 3565 | 3077 | 9 | train | `9_B3_18_20260905_032456/9_B3_18_20260905_032456.npz` |
| 9_B3_19_20260905_032554 | 11898 | 11558 | 9 | train | `9_B3_19_20260905_032554/9_B3_19_20260905_032554.npz` |
| 9_B3_1_20260904_000706 | 3600 | 3230 | 9 | train | `9_B3_1_20260904_000706/9_B3_1_20260904_000706.npz` |
| 9_B3_20_20260905_033405 | 8489 | 8154 | 9 | train | `9_B3_20_20260905_033405/9_B3_20_20260905_033405.npz` |
| 9_B3_21_20260905_033902 | 9023 | 8692 | 9 | train | `9_B3_21_20260905_033902/9_B3_21_20260905_033902.npz` |
| 9_B3_22_20260905_035656 | 6844 | 6673 | 9 | train | `9_B3_22_20260905_035656/9_B3_22_20260905_035656.npz` |
| 9_B3_23_20260905_041240 | 11485 | 11143 | 9 | train | `9_B3_23_20260905_041240/9_B3_23_20260905_041240.npz` |
| 9_B3_24_20260905_043126 | 13094 | 12900 | 9 | train | `9_B3_24_20260905_043126/9_B3_24_20260905_043126.npz` |
| 9_B3_25_20260905_044158 | 9510 | 9155 | 9 | val | `9_B3_25_20260905_044158/9_B3_25_20260905_044158.npz` |
| 9_B3_26_20260905_045613 | 13074 | 12811 | 9 | train | `9_B3_26_20260905_045613/9_B3_26_20260905_045613.npz` |
| 9_B3_27_20260905_050853 | 10193 | 9825 | 9 | train | `9_B3_27_20260905_050853/9_B3_27_20260905_050853.npz` |
| 9_B3_28_20260905_052518 | 8970 | 8723 | 9 | train | `9_B3_28_20260905_052518/9_B3_28_20260905_052518.npz` |
| 9_B3_29_20260905_060327 | 11712 | 11353 | 9 | train | `9_B3_29_20260905_060327/9_B3_29_20260905_060327.npz` |
| 9_B3_2_20260904_002143 | 7765 | 7453 | 9 | train | `9_B3_2_20260904_002143/9_B3_2_20260904_002143.npz` |
| 9_B3_30_20260905_061345 | 11605 | 11156 | 9 | train | `9_B3_30_20260905_061345/9_B3_30_20260905_061345.npz` |
| 9_B3_31_20260905_062616 | 10371 | 10072 | 9 | train | `9_B3_31_20260905_062616/9_B3_31_20260905_062616.npz` |
| 9_B3_32_20260905_063021 | 7236 | 6925 | 9 | train | `9_B3_32_20260905_063021/9_B3_32_20260905_063021.npz` |
| 9_B3_33_20260905_063239 | 13563 | 13216 | 9 | train | `9_B3_33_20260905_063239/9_B3_33_20260905_063239.npz` |
| 9_B3_34_20260905_065117 | 9391 | 9219 | 9 | train | `9_B3_34_20260905_065117/9_B3_34_20260905_065117.npz` |
| 9_B3_35_20260905_065830 | 10008 | 9609 | 9 | val | `9_B3_35_20260905_065830/9_B3_35_20260905_065830.npz` |
| 9_B3_36_20260905_070215 | 9204 | 8694 | 9 | val | `9_B3_36_20260905_070215/9_B3_36_20260905_070215.npz` |
| 9_B3_37_20260905_070658 | 8712 | 8379 | 9 | train | `9_B3_37_20260905_070658/9_B3_37_20260905_070658.npz` |
| 9_B3_38_20260905_071037 | 9700 | 9446 | 9 | train | `9_B3_38_20260905_071037/9_B3_38_20260905_071037.npz` |
| 9_B3_3_20260904_002349 | 11675 | 11063 | 9 | train | `9_B3_3_20260904_002349/9_B3_3_20260904_002349.npz` |
| 9_B3_4_20260904_003209 | 14090 | 13799 | 9 | train | `9_B3_4_20260904_003209/9_B3_4_20260904_003209.npz` |
| 9_B3_5_20260904_003838 | 11999 | 11617 | 9 | train | `9_B3_5_20260904_003838/9_B3_5_20260904_003838.npz` |
| 9_B3_6_20260904_004308 | 10287 | 9985 | 9 | train | `9_B3_6_20260904_004308/9_B3_6_20260904_004308.npz` |
| 9_B3_7_20260905_002158 | 15227 | 14913 | 9 | train | `9_B3_7_20260905_002158/9_B3_7_20260905_002158.npz` |
| 9_B3_8_20260905_003154 | 12326 | 12174 | 9 | train | `9_B3_8_20260905_003154/9_B3_8_20260905_003154.npz` |
| 9_B3_9_20260905_005247 | 12136 | 11939 | 9 | train | `9_B3_9_20260905_005247/9_B3_9_20260905_005247.npz` |
| 9_B50_1_20260905_080856 | 14435 | 14149 | 9 | train | `9_B50_1_20260905_080856/9_B50_1_20260905_080856.npz` |
| 9_B50_2_20260905_081650 | 11721 | 11060 | 9 | test | `9_B50_2_20260905_081650/9_B50_2_20260905_081650.npz` |
| 9_B50_3_20260905_082646 | 10658 | 10059 | 9 | train | `9_B50_3_20260905_082646/9_B50_3_20260905_082646.npz` |
| 9_B50_4_20260905_083207 | 8886 | 8554 | 9 | train | `9_B50_4_20260905_083207/9_B50_4_20260905_083207.npz` |
| 9_B50_5_20260905_084447 | 10013 | 9637 | 9 | train | `9_B50_5_20260905_084447/9_B50_5_20260905_084447.npz` |
| 9_B50_6_20260905_084912 | 10465 | 10081 | 9 | train | `9_B50_6_20260905_084912/9_B50_6_20260905_084912.npz` |
| 9_B50_7_20260905_085902 | 10264 | 9980 | 9 | train | `9_B50_7_20260905_085902/9_B50_7_20260905_085902.npz` |
| 9_C3_10_20260905_092423 | 9061 | 8860 | 9 | test | `9_C3_10_20260905_092423/9_C3_10_20260905_092423.npz` |
| 9_C3_11_20260908_210327 | 12674 | 12357 | 9 | train | `9_C3_11_20260908_210327/9_C3_11_20260908_210327.npz` |
| 9_C3_12_20260908_212610 | 11291 | 11026 | 9 | train | `9_C3_12_20260908_212610/9_C3_12_20260908_212610.npz` |
| 9_C3_13_20260908_213413 | 14347 | 13922 | 9 | train | `9_C3_13_20260908_213413/9_C3_13_20260908_213413.npz` |
| 9_C3_14_20260908_214323 | 12732 | 12357 | 9 | train | `9_C3_14_20260908_214323/9_C3_14_20260908_214323.npz` |
| 9_C3_15_20260908_215643 | 11833 | 10439 | 9 | train | `9_C3_15_20260908_215643/9_C3_15_20260908_215643.npz` |
| 9_C3_16_20260908_220230 | 8372 | 7912 | 9 | test | `9_C3_16_20260908_220230/9_C3_16_20260908_220230.npz` |
| 9_C3_17_20260908_220616 | 13723 | 13222 | 9 | train | `9_C3_17_20260908_220616/9_C3_17_20260908_220616.npz` |
| 9_C3_18_20260908_221124 | 11567 | 11237 | 9 | train | `9_C3_18_20260908_221124/9_C3_18_20260908_221124.npz` |
| 9_C3_19_20260908_221911 | 14420 | 14133 | 9 | val | `9_C3_19_20260908_221911/9_C3_19_20260908_221911.npz` |
| 9_C3_1_20260905_072045 | 10623 | 10290 | 9 | train | `9_C3_1_20260905_072045/9_C3_1_20260905_072045.npz` |
| 9_C3_20_20260908_222330 | 9992 | 9580 | 9 | train | `9_C3_20_20260908_222330/9_C3_20_20260908_222330.npz` |
| 9_C3_2_20260905_072625 | 10324 | 9944 | 9 | train | `9_C3_2_20260905_072625/9_C3_2_20260905_072625.npz` |
| 9_C3_3_20260905_073317 | 11610 | 11349 | 9 | val | `9_C3_3_20260905_073317/9_C3_3_20260905_073317.npz` |
| 9_C3_4_20260905_073842 | 9215 | 8733 | 9 | train | `9_C3_4_20260905_073842/9_C3_4_20260905_073842.npz` |
| 9_C3_5_20260905_074529 | 11310 | 11006 | 9 | train | `9_C3_5_20260905_074529/9_C3_5_20260905_074529.npz` |
| 9_C3_6_20260905_075453 | 10378 | 9844 | 9 | train | `9_C3_6_20260905_075453/9_C3_6_20260905_075453.npz` |
| 9_C3_7_20260905_075955 | 12067 | 11820 | 9 | val | `9_C3_7_20260905_075955/9_C3_7_20260905_075955.npz` |
| 9_C3_8_20260905_090848 | 10168 | 9798 | 9 | train | `9_C3_8_20260905_090848/9_C3_8_20260905_090848.npz` |
| 9_C3_8_20260908_205034 | 11848 | 11575 | 9 | train | `9_C3_8_20260908_205034/9_C3_8_20260908_205034.npz` |
| 9_C3_9_20260905_092316 | 3793 | 3515 | 9 | test | `9_C3_9_20260905_092316/9_C3_9_20260905_092316.npz` |
| brace22_capture2_20260806_163845 | 3900 | 3900 | brace22 | train | `brace22_capture2_20260806_163845/brace22_capture2_20260806_163845.npz` |
| crouch8_20260616_004631 | 3707 | 3707 | crouch8 | train | `crouch8_20260616_004631/crouch8_20260616_004631.npz` |
| ghost25_20260922_184455 | 11873 | 11873 | ghost25 | train | `ghost25_20260922_184455/ghost25_20260922_184455.npz` |
| nobrace0_1_20260915_062411 | 21381 | 20855 | nobrace0 | train | `nobrace0_1_20260915_062411/nobrace0_1_20260915_062411.npz` |
| nobrace0_2_20260915_063851 | 8405 | 8053 | nobrace0 | train | `nobrace0_2_20260915_063851/nobrace0_2_20260915_063851.npz` |
| nobrace0_3333_20260915_064733 | 6114 | 5832 | nobrace0 | train | `nobrace0_3333_20260915_064733/nobrace0_3333_20260915_064733.npz` |
| nobrace0_333_20260915_064632 | 3900 | 3604 | nobrace0 | train | `nobrace0_333_20260915_064632/nobrace0_333_20260915_064632.npz` |
| nobrace0_33_20260915_064330 | 3951 | 3678 | nobrace0 | train | `nobrace0_33_20260915_064330/nobrace0_33_20260915_064330.npz` |
| nobrace0_3_20260915_064240 | 3286 | 3176 | nobrace0 | train | `nobrace0_3_20260915_064240/nobrace0_3_20260915_064240.npz` |
| nobrace3_1_20260915_064920 | 13548 | 13350 | nobrace3 | train | `nobrace3_1_20260915_064920/nobrace3_1_20260915_064920.npz` |
| nobrace3_2_20260915_065223 | 11309 | 10918 | nobrace3 | train | `nobrace3_2_20260915_065223/nobrace3_2_20260915_065223.npz` |
| nobrace3_3_20260915_065459 | 8831 | 8708 | nobrace3 | val | `nobrace3_3_20260915_065459/nobrace3_3_20260915_065459.npz` |
| real_v3_lean_4_20260811_213652 | 168 | 168 | real | train | `real_v3_lean_4_20260811_213652/real_v3_lean_4_20260811_213652.npz` |
| real_v4a_full_1_20260818_191104 | 10621 | 9801 | real | train | `real_v4a_full_1_20260818_191104/real_v4a_full_1_20260818_191104.npz` |
| real_v4a_full_2_20260818_194659 | 8335 | 7757 | real | train | `real_v4a_full_2_20260818_194659/real_v4a_full_2_20260818_194659.npz` |
| real_v4a_full_3_20260818_200725 | 8360 | 7750 | real | val | `real_v4a_full_3_20260818_200725/real_v4a_full_3_20260818_200725.npz` |
| real_v4a_full_4_20260818_202422 | 16716 | 15698 | real | train | `real_v4a_full_4_20260818_202422/real_v4a_full_4_20260818_202422.npz` |
| real_v4a_lean_11_20260813_000927 | 9442 | 8774 | real | train | `real_v4a_lean_11_20260813_000927/real_v4a_lean_11_20260813_000927.npz` |
| real_v4a_lean_12_20260813_002028 | 10856 | 10226 | real | train | `real_v4a_lean_12_20260813_002028/real_v4a_lean_12_20260813_002028.npz` |
| real_v4a_lean_13_20260813_002542 | 10260 | 9625 | real | val | `real_v4a_lean_13_20260813_002542/real_v4a_lean_13_20260813_002542.npz` |
| real_v4a_lean_14_20260813_011356 | 10744 | 10032 | real | train | `real_v4a_lean_14_20260813_011356/real_v4a_lean_14_20260813_011356.npz` |
| real_v4a_lean_15_20260813_011805 | 10178 | 9536 | real | val | `real_v4a_lean_15_20260813_011805/real_v4a_lean_15_20260813_011805.npz` |
| real_v4a_lean_23_again_2_20260819_023157 | 2999 | 2322 | real | test | `real_v4a_lean_23_again_2_20260819_023157/real_v4a_lean_23_again_2_20260819_023157.npz` |
| real_v4a_lean_23_final_20_20260819_215417 | 4646 | 3967 | real | train | `real_v4a_lean_23_final_20_20260819_215417/real_v4a_lean_23_final_20_20260819_215417.npz` |
| real_v4a_lean_24_10_20260820_224901 | 17679 | 17031 | real | train | `real_v4a_lean_24_10_20260820_224901/real_v4a_lean_24_10_20260820_224901.npz` |
| real_v4a_lean_24_11_20260820_230603 | 16098 | 15441 | real | val | `real_v4a_lean_24_11_20260820_230603/real_v4a_lean_24_11_20260820_230603.npz` |
| real_v4a_lean_24_12_20260820_231459 | 16487 | 15827 | real | train | `real_v4a_lean_24_12_20260820_231459/real_v4a_lean_24_12_20260820_231459.npz` |
| real_v4a_lean_24_13_20260820_232249 | 3838 | 3085 | real | test | `real_v4a_lean_24_13_20260820_232249/real_v4a_lean_24_13_20260820_232249.npz` |
| real_v4a_lean_24_13_20260820_232621 | 18494 | 17858 | real | test | `real_v4a_lean_24_13_20260820_232621/real_v4a_lean_24_13_20260820_232621.npz` |
| real_v4a_lean_24_14_20260820_233605 | 19383 | 18734 | real | train | `real_v4a_lean_24_14_20260820_233605/real_v4a_lean_24_14_20260820_233605.npz` |
| real_v4a_lean_24_15_20260820_234533 | 6904 | 5955 | real | train | `real_v4a_lean_24_15_20260820_234533/real_v4a_lean_24_15_20260820_234533.npz` |
| real_v4a_lean_24_16_20260820_234803 | 6367 | 5730 | real | train | `real_v4a_lean_24_16_20260820_234803/real_v4a_lean_24_16_20260820_234803.npz` |
| real_v4a_lean_24_17_20260820_235832 | 12105 | 11416 | real | train | `real_v4a_lean_24_17_20260820_235832/real_v4a_lean_24_17_20260820_235832.npz` |
| real_v4a_lean_24_18_20260821_001041 | 5101 | 4440 | real | test | `real_v4a_lean_24_18_20260821_001041/real_v4a_lean_24_18_20260821_001041.npz` |
| real_v4a_lean_24_18_20260821_001247 | 3258 | 2456 | real | train | `real_v4a_lean_24_18_20260821_001247/real_v4a_lean_24_18_20260821_001247.npz` |
| real_v4a_lean_24_18_20260821_001357 | 3442 | 2746 | real | train | `real_v4a_lean_24_18_20260821_001357/real_v4a_lean_24_18_20260821_001357.npz` |
| real_v4a_lean_24_18_20260821_001905 | 6118 | 5420 | real | train | `real_v4a_lean_24_18_20260821_001905/real_v4a_lean_24_18_20260821_001905.npz` |
| real_v4a_lean_24_19_20260821_002057 | 17158 | 16482 | real | train | `real_v4a_lean_24_19_20260821_002057/real_v4a_lean_24_19_20260821_002057.npz` |
| real_v4a_lean_24_1_20260820_142257 | 17903 | 17164 | real | train | `real_v4a_lean_24_1_20260820_142257/real_v4a_lean_24_1_20260820_142257.npz` |
| real_v4a_lean_24_20_20260821_002959 | 14982 | 14396 | real | train | `real_v4a_lean_24_20_20260821_002959/real_v4a_lean_24_20_20260821_002959.npz` |
| real_v4a_lean_24_21_20260821_004238 | 13177 | 12521 | real | test | `real_v4a_lean_24_21_20260821_004238/real_v4a_lean_24_21_20260821_004238.npz` |
| real_v4a_lean_24_22_20260821_005147 | 15485 | 14773 | real | test | `real_v4a_lean_24_22_20260821_005147/real_v4a_lean_24_22_20260821_005147.npz` |
| real_v4a_lean_24_23_20260821_010533 | 18212 | 17514 | real | train | `real_v4a_lean_24_23_20260821_010533/real_v4a_lean_24_23_20260821_010533.npz` |
| real_v4a_lean_24_24_20260821_011327 | 16445 | 15780 | real | train | `real_v4a_lean_24_24_20260821_011327/real_v4a_lean_24_24_20260821_011327.npz` |
| real_v4a_lean_24_25_20260821_012212 | 8360 | 7677 | real | train | `real_v4a_lean_24_25_20260821_012212/real_v4a_lean_24_25_20260821_012212.npz` |
| real_v4a_lean_24_26_20260821_012632 | 14158 | 13528 | real | train | `real_v4a_lean_24_26_20260821_012632/real_v4a_lean_24_26_20260821_012632.npz` |
| real_v4a_lean_24_27_20260821_013420 | 6460 | 5820 | real | train | `real_v4a_lean_24_27_20260821_013420/real_v4a_lean_24_27_20260821_013420.npz` |
| real_v4a_lean_24_28_20260821_013933 | 6546 | 5900 | real | train | `real_v4a_lean_24_28_20260821_013933/real_v4a_lean_24_28_20260821_013933.npz` |
| real_v4a_lean_24_2_20260820_145448 | 6693 | 6011 | real | test | `real_v4a_lean_24_2_20260820_145448/real_v4a_lean_24_2_20260820_145448.npz` |
| real_v4a_lean_24_3_20260820_145715 | 7194 | 6522 | real | train | `real_v4a_lean_24_3_20260820_145715/real_v4a_lean_24_3_20260820_145715.npz` |
| real_v4a_lean_24_4_20260820_182950 | 23291 | 22629 | real | test | `real_v4a_lean_24_4_20260820_182950/real_v4a_lean_24_4_20260820_182950.npz` |
| real_v4a_lean_24_5_20260820_185505 | 14225 | 13478 | real | train | `real_v4a_lean_24_5_20260820_185505/real_v4a_lean_24_5_20260820_185505.npz` |
| real_v4a_lean_24_6_20260820_191325 | 17191 | 16583 | real | train | `real_v4a_lean_24_6_20260820_191325/real_v4a_lean_24_6_20260820_191325.npz` |
| real_v4a_lean_24_7_20260820_205114 | 17383 | 16557 | real | val | `real_v4a_lean_24_7_20260820_205114/real_v4a_lean_24_7_20260820_205114.npz` |
| real_v4a_lean_24_8_20260820_210416 | 19540 | 18875 | real | test | `real_v4a_lean_24_8_20260820_210416/real_v4a_lean_24_8_20260820_210416.npz` |
| real_v4a_lean_24_9_20260820_223123 | 23283 | 22466 | real | train | `real_v4a_lean_24_9_20260820_223123/real_v4a_lean_24_9_20260820_223123.npz` |
| real_v4a_lean_24_again_15_20260821_044720 | 3512 | 2815 | real | train | `real_v4a_lean_24_again_15_20260821_044720/real_v4a_lean_24_again_15_20260821_044720.npz` |
| real_v4a_lean_24_again_16_20260821_045707 | 6809 | 6118 | real | train | `real_v4a_lean_24_again_16_20260821_045707/real_v4a_lean_24_again_16_20260821_045707.npz` |
| real_v4a_lean_24_again_17_20260821_051003 | 4778 | 4055 | real | val | `real_v4a_lean_24_again_17_20260821_051003/real_v4a_lean_24_again_17_20260821_051003.npz` |
| real_v4a_lean_24_again_18_20260821_051856 | 11644 | 10962 | real | train | `real_v4a_lean_24_again_18_20260821_051856/real_v4a_lean_24_again_18_20260821_051856.npz` |
| real_v4a_lean_24_again_18_20260821_052223 | 8725 | 8093 | real | train | `real_v4a_lean_24_again_18_20260821_052223/real_v4a_lean_24_again_18_20260821_052223.npz` |
| real_v4a_lean_24_again_18_20260821_052659 | 10676 | 10022 | real | train | `real_v4a_lean_24_again_18_20260821_052659/real_v4a_lean_24_again_18_20260821_052659.npz` |
| real_v4a_lean_24_again_21_20260821_054509 | 3428 | 2767 | real | train | `real_v4a_lean_24_again_21_20260821_054509/real_v4a_lean_24_again_21_20260821_054509.npz` |
| real_v4a_lean_24_again_21_20260821_054917 | 7128 | 6339 | real | train | `real_v4a_lean_24_again_21_20260821_054917/real_v4a_lean_24_again_21_20260821_054917.npz` |
| real_v4a_lean_24_again_30_20260821_183104 | 3338 | 2493 | real | train | `real_v4a_lean_24_again_30_20260821_183104/real_v4a_lean_24_again_30_20260821_183104.npz` |
| real_v4a_lean_24_again_32_20260821_190102 | 3749 | 3036 | real | train | `real_v4a_lean_24_again_32_20260821_190102/real_v4a_lean_24_again_32_20260821_190102.npz` |
| real_v4a_lean_24_again_33_20260821_192137 | 3226 | 2499 | real | val | `real_v4a_lean_24_again_33_20260821_192137/real_v4a_lean_24_again_33_20260821_192137.npz` |
| real_v4a_lean_24_again_37_20260821_201701 | 8259 | 7507 | real | test | `real_v4a_lean_24_again_37_20260821_201701/real_v4a_lean_24_again_37_20260821_201701.npz` |
| real_v4a_lean_24_again_39_20260821_213051 | 5938 | 5245 | real | train | `real_v4a_lean_24_again_39_20260821_213051/real_v4a_lean_24_again_39_20260821_213051.npz` |
| real_v4a_lean_24_again_40_20260821_224246 | 13217 | 12367 | real | train | `real_v4a_lean_24_again_40_20260821_224246/real_v4a_lean_24_again_40_20260821_224246.npz` |
| real_v4a_lean_24_again_40_20260821_224602 | 6832 | 6107 | real | train | `real_v4a_lean_24_again_40_20260821_224602/real_v4a_lean_24_again_40_20260821_224602.npz` |
| real_v4a_lean_24_again_47_20260822_001917 | 14093 | 13329 | real | train | `real_v4a_lean_24_again_47_20260822_001917/real_v4a_lean_24_again_47_20260822_001917.npz` |
| real_v4a_lean_24_again_47_20260822_002557 | 16027 | 15368 | real | train | `real_v4a_lean_24_again_47_20260822_002557/real_v4a_lean_24_again_47_20260822_002557.npz` |
| real_v4a_lean_24_again_47_20260822_003734 | 10462 | 9472 | real | train | `real_v4a_lean_24_again_47_20260822_003734/real_v4a_lean_24_again_47_20260822_003734.npz` |
| real_v4a_lean_24_again_47_20260822_004949 | 5539 | 4769 | real | train | `real_v4a_lean_24_again_47_20260822_004949/real_v4a_lean_24_again_47_20260822_004949.npz` |
| real_v4a_lean_24_again_48_20260822_005719 | 6374 | 5709 | real | test | `real_v4a_lean_24_again_48_20260822_005719extra/real_v4a_lean_24_again_48_20260822_005719.npz` |
| real_v4a_lean_25_10_20260822_053915 | 9323 | 8649 | real | train | `real_v4a_lean_25_10_20260822_053915/real_v4a_lean_25_10_20260822_053915.npz` |
| real_v4a_lean_25_11_20260822_054142 | 15818 | 15168 | real | train | `real_v4a_lean_25_11_20260822_054142/real_v4a_lean_25_11_20260822_054142.npz` |
| real_v4a_lean_25_12_20260822_055105 | 7948 | 7259 | real | train | `real_v4a_lean_25_12_20260822_055105/real_v4a_lean_25_12_20260822_055105.npz` |
| real_v4a_lean_25_13_20260822_055312 | 14776 | 14131 | real | train | `real_v4a_lean_25_13_20260822_055312/real_v4a_lean_25_13_20260822_055312.npz` |
| real_v4a_lean_25_14_20260822_055833 | 9961 | 9361 | real | train | `real_v4a_lean_25_14_20260822_055833/real_v4a_lean_25_14_20260822_055833.npz` |
| real_v4a_lean_25_15_20260822_060405 | 14041 | 13370 | real | train | `real_v4a_lean_25_15_20260822_060405/real_v4a_lean_25_15_20260822_060405.npz` |
| real_v4a_lean_25_16_20260822_061649 | 8799 | 8067 | real | train | `real_v4a_lean_25_16_20260822_061649/real_v4a_lean_25_16_20260822_061649.npz` |
| real_v4a_lean_25_17_20260822_064807 | 4356 | 3788 | real | train | `real_v4a_lean_25_17_20260822_064807/real_v4a_lean_25_17_20260822_064807.npz` |
| real_v4a_lean_25_18_20260822_064917 | 15202 | 14594 | real | train | `real_v4a_lean_25_18_20260822_064917/real_v4a_lean_25_18_20260822_064917.npz` |
| real_v4a_lean_25_19_20260822_071227 | 10136 | 9461 | real | train | `real_v4a_lean_25_19_20260822_071227/real_v4a_lean_25_19_20260822_071227.npz` |
| real_v4a_lean_25_1_20260822_041323 | 7619 | 6888 | real | train | `real_v4a_lean_25_1_20260822_041323/real_v4a_lean_25_1_20260822_041323.npz` |
| real_v4a_lean_25_20_20260822_071844 | 12254 | 11695 | real | train | `real_v4a_lean_25_20_20260822_071844/real_v4a_lean_25_20_20260822_071844.npz` |
| real_v4a_lean_25_21_20260822_073133 | 15665 | 14976 | real | train | `real_v4a_lean_25_21_20260822_073133/real_v4a_lean_25_21_20260822_073133.npz` |
| real_v4a_lean_25_22_20260822_073543 | 7604 | 6924 | real | train | `real_v4a_lean_25_22_20260822_073543/real_v4a_lean_25_22_20260822_073543.npz` |
| real_v4a_lean_25_23_20260822_073854 | 10885 | 10151 | real | train | `real_v4a_lean_25_23_20260822_073854/real_v4a_lean_25_23_20260822_073854.npz` |
| real_v4a_lean_25_24_20260822_074332 | 6631 | 5952 | real | train | `real_v4a_lean_25_24_20260822_074332/real_v4a_lean_25_24_20260822_074332.npz` |
| real_v4a_lean_25_25_20260822_074632 | 11565 | 10847 | real | train | `real_v4a_lean_25_25_20260822_074632/real_v4a_lean_25_25_20260822_074632.npz` |
| real_v4a_lean_25_26_20260822_075107 | 16577 | 15840 | real | train | `real_v4a_lean_25_26_20260822_075107/real_v4a_lean_25_26_20260822_075107.npz` |
| real_v4a_lean_25_27_20260822_075923 | 12978 | 12107 | real | val | `real_v4a_lean_25_27_20260822_075923/real_v4a_lean_25_27_20260822_075923.npz` |
| real_v4a_lean_25_28_20260822_080407 | 8265 | 7607 | real | train | `real_v4a_lean_25_28_20260822_080407/real_v4a_lean_25_28_20260822_080407.npz` |
| real_v4a_lean_25_29_20260822_081352 | 13878 | 13285 | real | train | `real_v4a_lean_25_29_20260822_081352/real_v4a_lean_25_29_20260822_081352.npz` |
| real_v4a_lean_25_2_20260822_042014 | 11133 | 10401 | real | train | `real_v4a_lean_25_2_20260822_042014/real_v4a_lean_25_2_20260822_042014.npz` |
| real_v4a_lean_25_30_20260822_082608 | 14819 | 14172 | real | train | `real_v4a_lean_25_30_20260822_082608/real_v4a_lean_25_30_20260822_082608.npz` |
| real_v4a_lean_25_31_20260822_084250 | 16658 | 16067 | real | train | `real_v4a_lean_25_31_20260822_084250/real_v4a_lean_25_31_20260822_084250.npz` |
| real_v4a_lean_25_32_20260822_091532 | 12921 | 12257 | real | train | `real_v4a_lean_25_32_20260822_091532/real_v4a_lean_25_32_20260822_091532.npz` |
| real_v4a_lean_25_33_20260822_092244 | 11815 | 11185 | real | train | `real_v4a_lean_25_33_20260822_092244/real_v4a_lean_25_33_20260822_092244.npz` |
| real_v4a_lean_25_3_20260822_043504 | 12800 | 11992 | real | train | `real_v4a_lean_25_3_20260822_043504/real_v4a_lean_25_3_20260822_043504.npz` |
| real_v4a_lean_25_4_20260822_044838 | 14552 | 13903 | real | val | `real_v4a_lean_25_4_20260822_044838/real_v4a_lean_25_4_20260822_044838.npz` |
| real_v4a_lean_25_5_20260822_045401 | 11113 | 10386 | real | train | `real_v4a_lean_25_5_20260822_045401/real_v4a_lean_25_5_20260822_045401.npz` |
| real_v4a_lean_25_6_20260822_050835 | 10040 | 9334 | real | train | `real_v4a_lean_25_6_20260822_050835/real_v4a_lean_25_6_20260822_050835.npz` |
| real_v4a_lean_25_7_20260822_052243 | 11904 | 11254 | real | train | `real_v4a_lean_25_7_20260822_052243/real_v4a_lean_25_7_20260822_052243.npz` |
| real_v4a_lean_25_8_20260822_052732 | 9347 | 8557 | real | train | `real_v4a_lean_25_8_20260822_052732/real_v4a_lean_25_8_20260822_052732.npz` |
| real_v4a_lean_25_9_20260822_053339 | 9506 | 8818 | real | train | `real_v4a_lean_25_9_20260822_053339/real_v4a_lean_25_9_20260822_053339.npz` |
| real_v4a_lean_25_p_10_20260823_070841 | 7111 | 6370 | real | train | `real_v4a_lean_25_p_10_20260823_070841/real_v4a_lean_25_p_10_20260823_070841.npz` |
| real_v4a_lean_25_p_10_20260823_071146 | 8526 | 7859 | real | train | `real_v4a_lean_25_p_10_20260823_071146/real_v4a_lean_25_p_10_20260823_071146.npz` |
| real_v4a_lean_25_p_17_20260823_074514 | 3162 | 2344 | real | val | `real_v4a_lean_25_p_17_20260823_074514/real_v4a_lean_25_p_17_20260823_074514.npz` |
| real_v4a_lean_25_p_17_20260823_074600 | 3789 | 3125 | real | train | `real_v4a_lean_25_p_17_20260823_074600/real_v4a_lean_25_p_17_20260823_074600.npz` |
| real_v4a_lean_25_p_17_20260823_074753 | 3329 | 2520 | real | train | `real_v4a_lean_25_p_17_20260823_074753/real_v4a_lean_25_p_17_20260823_074753.npz` |
| real_v4a_lean_25_p_17_20260823_074853 | 2879 | 2194 | real | test | `real_v4a_lean_25_p_17_20260823_074853/real_v4a_lean_25_p_17_20260823_074853.npz` |
| real_v4a_lean_25_p_17_20260823_075523 | 11139 | 10473 | real | train | `real_v4a_lean_25_p_17_20260823_075523/real_v4a_lean_25_p_17_20260823_075523.npz` |
| real_v4a_lean_25_p_17_20260823_075748 | 10470 | 9812 | real | test | `real_v4a_lean_25_p_17_20260823_075748/real_v4a_lean_25_p_17_20260823_075748.npz` |
| real_v4a_lean_25_p_1_20260823_054912 | 3226 | 2523 | real | train | `real_v4a_lean_25_p_1_20260823_054912/real_v4a_lean_25_p_1_20260823_054912.npz` |
| real_v4a_lean_7_20260812_193046 | 139 | 139 | real | train | `real_v4a_lean_7_20260812_193046/real_v4a_lean_7_20260812_193046.npz` |
| real_v4a_lean_8_20260812_200043 | 175 | 175 | real | train | `real_v4a_lean_8_20260812_200043/real_v4a_lean_8_20260812_200043.npz` |
| real_v4a_lean_flat_10_20260813_204953 | 8478 | 7806 | real | train | `real_v4a_lean_flat_10_20260813_204953/real_v4a_lean_flat_10_20260813_204953.npz` |
| real_v4a_lean_flat_11_20260813_234557 | 8497 | 7586 | real | train | `real_v4a_lean_flat_11_20260813_234557/real_v4a_lean_flat_11_20260813_234557.npz` |
| real_v4a_lean_flat_12_20260813_235538 | 14308 | 13621 | real | val | `real_v4a_lean_flat_12_20260813_235538/real_v4a_lean_flat_12_20260813_235538.npz` |
| real_v4a_lean_flat_13_20260814_002112 | 18907 | 18260 | real | val | `real_v4a_lean_flat_13_20260814_002112/real_v4a_lean_flat_13_20260814_002112.npz` |
| real_v4a_lean_flat_14_20260814_004259 | 11562 | 10875 | real | train | `real_v4a_lean_flat_14_20260814_004259/real_v4a_lean_flat_14_20260814_004259.npz` |
| real_v4a_lean_flat_1_20260813_162135 | 13501 | 11179 | real | train | `real_v4a_lean_flat_1_20260813_162135/real_v4a_lean_flat_1_20260813_162135.npz` |
| real_v4a_lean_flat_2_20260813_164102 | 15469 | 14830 | real | train | `real_v4a_lean_flat_2_20260813_164102/real_v4a_lean_flat_2_20260813_164102.npz` |
| real_v4a_lean_flat_3_20260813_171502 | 18914 | 18252 | real | train | `real_v4a_lean_flat_3_20260813_171502/real_v4a_lean_flat_3_20260813_171502.npz` |
| real_v4a_lean_flat_4_20260813_180356 | 10777 | 10148 | real | train | `real_v4a_lean_flat_4_20260813_180356/real_v4a_lean_flat_4_20260813_180356.npz` |
| real_v4a_lean_flat_5_20260813_182101 | 8928 | 8065 | real | test | `real_v4a_lean_flat_5_20260813_182101/real_v4a_lean_flat_5_20260813_182101.npz` |
| real_v4a_lean_flat_6_20260813_185736 | 18890 | 17535 | real | train | `real_v4a_lean_flat_6_20260813_185736/real_v4a_lean_flat_6_20260813_185736.npz` |
| real_v4a_lean_flat_7_20260813_192711 | 11803 | 10822 | real | train | `real_v4a_lean_flat_7_20260813_192711/real_v4a_lean_flat_7_20260813_192711.npz` |
| real_v4a_lean_flat_8_20260813_195830 | 9258 | 8256 | real | train | `real_v4a_lean_flat_8_20260813_195830/real_v4a_lean_flat_8_20260813_195830.npz` |
| real_v4a_lean_flat_9_20260813_201855 | 14641 | 13987 | real | val | `real_v4a_lean_flat_9_20260813_201855/real_v4a_lean_flat_9_20260813_201855.npz` |
| real_v4a_lean_new_18_20260816_043524 | 1076 | 438 | real | train | `real_v4a_lean_new_18_20260816_043524/real_v4a_lean_new_18_20260816_043524.npz` |
| real_v4a_lean_new_5_20260815_223202 | 2188 | 1310 | real | train | `real_v4a_lean_new_5_20260815_223202/real_v4a_lean_new_5_20260815_223202.npz` |
| real_v4a_lean_newtags_2_20260817_062026 | 5650 | 4904 | real | train | `real_v4a_lean_newtags_2_20260817_062026/real_v4a_lean_newtags_2_20260817_062026.npz` |
| real_v4a_lean_tags_10_20260817_030546 | 9829 | 9099 | real | train | `real_v4a_lean_tags_10_20260817_030546/real_v4a_lean_tags_10_20260817_030546.npz` |
| real_v4a_lean_tags_11_20260817_031557 | 9039 | 8345 | real | train | `real_v4a_lean_tags_11_20260817_031557/real_v4a_lean_tags_11_20260817_031557.npz` |
| real_v4a_lean_tags_12_20260817_033415 | 12207 | 11568 | real | train | `real_v4a_lean_tags_12_20260817_033415/real_v4a_lean_tags_12_20260817_033415.npz` |
| real_v4a_lean_tags_13_20260817_034527 | 9044 | 8422 | real | train | `real_v4a_lean_tags_13_20260817_034527/real_v4a_lean_tags_13_20260817_034527.npz` |
| real_v4a_lean_tags_14_20260817_034958 | 9209 | 8549 | real | train | `real_v4a_lean_tags_14_20260817_034958/real_v4a_lean_tags_14_20260817_034958.npz` |
| real_v4a_lean_tags_15_20260817_040234 | 3644 | 2957 | real | train | `real_v4a_lean_tags_15_20260817_040234/real_v4a_lean_tags_15_20260817_040234.npz` |
| real_v4a_lean_tags_15_20260817_040423 | 13076 | 12300 | real | train | `real_v4a_lean_tags_15_20260817_040423/real_v4a_lean_tags_15_20260817_040423.npz` |
| real_v4a_lean_tags_16_20260817_051856 | 9436 | 8719 | real | train | `real_v4a_lean_tags_16_20260817_051856/real_v4a_lean_tags_16_20260817_051856.npz` |
| real_v4a_lean_tags_17_20260817_052822 | 11560 | 10838 | real | train | `real_v4a_lean_tags_17_20260817_052822/real_v4a_lean_tags_17_20260817_052822.npz` |
| real_v4a_lean_tags_1_20260817_005359 | 5476 | 4772 | real | val | `real_v4a_lean_tags_1_20260817_005359/real_v4a_lean_tags_1_20260817_005359.npz` |
| real_v4a_lean_tags_3_20260817_005750 | 5444 | 4620 | real | train | `real_v4a_lean_tags_3_20260817_005750/real_v4a_lean_tags_3_20260817_005750.npz` |
| real_v4a_lean_tags_4_20260817_011728 | 3992 | 3300 | real | train | `real_v4a_lean_tags_4_20260817_011728/real_v4a_lean_tags_4_20260817_011728.npz` |
| real_v4a_lean_tags_5_20260817_012746 | 3270 | 2554 | real | train | `real_v4a_lean_tags_5_20260817_012746/real_v4a_lean_tags_5_20260817_012746.npz` |
| real_v4a_lean_tags_6_20260817_021939 | 5536 | 4816 | real | train | `real_v4a_lean_tags_6_20260817_021939/real_v4a_lean_tags_6_20260817_021939.npz` |
| real_v4a_lean_tags_7_20260817_022304 | 8977 | 8239 | real | train | `real_v4a_lean_tags_7_20260817_022304/real_v4a_lean_tags_7_20260817_022304.npz` |
| real_v4a_lean_tags_8_20260817_023139 | 17488 | 16775 | real | train | `real_v4a_lean_tags_8_20260817_023139/real_v4a_lean_tags_8_20260817_023139.npz` |
| real_v4a_lean_tags_9_20260817_024739 | 9786 | 9060 | real | train | `real_v4a_lean_tags_9_20260817_024739/real_v4a_lean_tags_9_20260817_024739.npz` |
| realstandsumma3_20260812_191811 | 113 | 112 | realstandsumma3 | train | `realstandsumma3_20260812_191811/realstandsumma3_20260812_191811.npz` |
| s29_1_20260921_221506 | 12998 | 11143 | s29 | train | `s29_1_20260921_221506/s29_1_20260921_221506.npz` |
| s29_2_20260921_222608 | 15326 | 0 | s29 | val | `s29_2_20260921_222608/s29_2_20260921_222608.npz` |
| s29_3_20260921_222852 | 19338 | 18458 | s29 | train | `s29_3_20260921_222852/s29_3_20260921_222852.npz` |
| s29_4_20260921_225421 | 10782 | 10309 | s29 | train | `s29_4_20260921_225421/s29_4_20260921_225421.npz` |
| s29_5_20260921_230324 | 12912 | 11679 | s29 | test | `s29_5_20260921_230324/s29_5_20260921_230324.npz` |
| s29_6_20260921_231215 | 15188 | 14548 | s29 | train | `s29_6_20260921_231215/s29_6_20260921_231215.npz` |
| sq18cold_20260616_004939 | 6092 | 6092 | sq18cold | train | `sq18cold_20260616_004939/sq18cold_20260616_004939.npz` |
| sq18harn_20260616_005118 | 8001 | 8001 | sq18harn | train | `sq18harn_20260616_005118/sq18harn_20260616_005118.npz` |
| sq18held_20260616_013728 | 7032 | 7032 | sq18held | train | `sq18held_20260616_013728/sq18held_20260616_013728.npz` |
| sq18supp_20260616_013449 | 4998 | 4998 | sq18supp | train | `sq18supp_20260616_013449/sq18supp_20260616_013449.npz` |
| stand6_20260616_004238 | 8041 | 8041 | stand6 | train | `stand6_20260616_004238/stand6_20260616_004238.npz` |
| stand6ready2_20260616_021607 | 4138 | 4138 | stand6ready2 | train | `stand6ready2_20260616_021607/stand6ready2_20260616_021607.npz` |
| stand6ready_20260616_021253 | 4844 | 4844 | stand6ready | train | `stand6ready_20260616_021253/stand6ready_20260616_021253.npz` |
| stand6supp_20260616_013311 | 7331 | 7331 | stand6supp | test | `stand6supp_20260616_013311/stand6supp_20260616_013311.npz` |
| standB2_optitrackreal_20260619_192747 | 360 | 360 | standB2 | train | `standB2_optitrackreal_20260619_192747/standB2_optitrackreal_20260619_192747.npz` |
| table_+5_101112_20260911_071526 | 31227 | 30431 | table | val | `table_+5_101112_20260911_071526/table_+5_101112_20260911_071526.npz` |
| table_+5_131415_20260911_072500 | 22054 | 21592 | table | train | `table_+5_131415_20260911_072500/table_+5_131415_20260911_072500.npz` |
| table_+5_161718_20260911_073400 | 23451 | 23245 | table | train | `table_+5_161718_20260911_073400/table_+5_161718_20260911_073400.npz` |
| table_+5_1_20260911_063932 | 10986 | 10394 | table | train | `table_+5_1_20260911_063932/table_+5_1_20260911_063932.npz` |
| table_+5_2_20260911_064229 | 9051 | 8732 | table | train | `table_+5_2_20260911_064229/table_+5_2_20260911_064229.npz` |
| table_+5_3_20260911_064438 | 10031 | 9396 | table | train | `table_+5_3_20260911_064438/table_+5_3_20260911_064438.npz` |
| table_+5_456_20260911_065300 | 34840 | 34595 | table | train | `table_+5_456_20260911_065300/table_+5_456_20260911_065300.npz` |
| table_+5_789_20260911_070730 | 24260 | 23836 | table | train | `table_+5_789_20260911_070730/table_+5_789_20260911_070730.npz` |
| table_-5_101112_20260911_084414 | 33745 | 33256 | table | val | `table_-5_101112_20260911_084414/table_-5_101112_20260911_084414.npz` |
| table_-5_123_20260911_080951 | 36365 | 36040 | table | train | `table_-5_123_20260911_080951/table_-5_123_20260911_080951.npz` |
| table_-5_131415_20260911_085131 | 28853 | 28367 | table | train | `table_-5_131415_20260911_085131/table_-5_131415_20260911_085131.npz` |
| table_-5_161718_20260911_085926 | 32981 | 32454 | table | train | `table_-5_161718_20260911_085926/table_-5_161718_20260911_085926.npz` |
| table_-5_456_20260911_082441 | 33680 | 33369 | table | train | `table_-5_456_20260911_082441/table_-5_456_20260911_082441.npz` |
| table_-5_789_20260911_083755 | 28179 | 27942 | table | val | `table_-5_789_20260911_083755/table_-5_789_20260911_083755.npz` |
| table_nom_10_20260911_040821 | 13808 | 13478 | table | train | `table_nom_10_20260911_040821/table_nom_10_20260911_040821.npz` |
| table_nom_11_20260911_044431 | 10539 | 10030 | table | val | `table_nom_11_20260911_044431/table_nom_11_20260911_044431.npz` |
| table_nom_12_20260911_045704 | 11096 | 9790 | table | train | `table_nom_12_20260911_045704/table_nom_12_20260911_045704.npz` |
| table_nom_13_20260911_051315 | 13728 | 13134 | table | train | `table_nom_13_20260911_051315/table_nom_13_20260911_051315.npz` |
| table_nom_14_20260911_053012 | 11365 | 10942 | table | test | `table_nom_14_20260911_053012/table_nom_14_20260911_053012.npz` |
| table_nom_15_20260911_053235 | 14739 | 14386 | table | train | `table_nom_15_20260911_053235/table_nom_15_20260911_053235.npz` |
| table_nom_16_20260911_053533 | 10202 | 9628 | table | train | `table_nom_16_20260911_053533/table_nom_16_20260911_053533.npz` |
| table_nom_17_20260911_053911 | 15016 | 14743 | table | train | `table_nom_17_20260911_053911/table_nom_17_20260911_053911.npz` |
| table_nom_18_20260911_054223 | 12789 | 12287 | table | train | `table_nom_18_20260911_054223/table_nom_18_20260911_054223.npz` |
| table_nom_19_20260911_054508 | 14672 | 14080 | table | train | `table_nom_19_20260911_054508/table_nom_19_20260911_054508.npz` |
| table_nom_1_20260910_231755 | 14416 | 14096 | table | test | `table_nom_1_20260910_231755/table_nom_1_20260910_231755.npz` |
| table_nom_20_20260911_054915 | 12828 | 12535 | table | test | `table_nom_20_20260911_054915/table_nom_20_20260911_054915.npz` |
| table_nom_21_20260911_055418 | 17830 | 17376 | table | train | `table_nom_21_20260911_055418/table_nom_21_20260911_055418.npz` |
| table_nom_22_20260911_055827 | 16822 | 16511 | table | train | `table_nom_22_20260911_055827/table_nom_22_20260911_055827.npz` |
| table_nom_23_20260911_061426 | 13902 | 13664 | table | train | `table_nom_23_20260911_061426/table_nom_23_20260911_061426.npz` |
| table_nom_24_20260911_061046 | 14238 | 13816 | table | train | `table_nom_24_20260911_061046/table_nom_24_20260911_061046.npz` |
| table_nom_25_20260911_061733 | 14924 | 14021 | table | train | `table_nom_25_20260911_061733/table_nom_25_20260911_061733.npz` |
| table_nom_26_20260911_062247 | 10850 | 10568 | table | train | `table_nom_26_20260911_062247/table_nom_26_20260911_062247.npz` |
| table_nom_27_20260911_062655 | 12035 | 11767 | table | train | `table_nom_27_20260911_062655/table_nom_27_20260911_062655.npz` |
| table_nom_28_20260911_063004 | 13457 | 12874 | table | train | `table_nom_28_20260911_063004/table_nom_28_20260911_063004.npz` |
| table_nom_2_20260910_232752 | 9964 | 9651 | table | train | `table_nom_2_20260910_232752/table_nom_2_20260910_232752.npz` |
| table_nom_3_20260911_023137 | 14309 | 13090 | table | train | `table_nom_3_20260911_023137/table_nom_3_20260911_023137.npz` |
| table_nom_4_20260911_024820 | 10310 | 10094 | table | val | `table_nom_4_20260911_024820/table_nom_4_20260911_024820.npz` |
| table_nom_5_20260911_025359 | 10518 | 10173 | table | test | `table_nom_5_20260911_025359/table_nom_5_20260911_025359.npz` |
| table_nom_6_20260911_030612 | 10540 | 10316 | table | train | `table_nom_6_20260911_030612/table_nom_6_20260911_030612.npz` |
| table_nom_7_20260911_033829 | 8323 | 7885 | table | train | `table_nom_7_20260911_033829/table_nom_7_20260911_033829.npz` |
| table_nom_8_20260911_034238 | 10526 | 10164 | table | train | `table_nom_8_20260911_034238/table_nom_8_20260911_034238.npz` |
| table_nom_9_20260911_035534 | 12154 | 11614 | table | train | `table_nom_9_20260911_035534/table_nom_9_20260911_035534.npz` |
| v3newstand_20260830_061824 | 18254 | 17998 | v3newstand | train | `v3newstand_20260830_061824/v3newstand_20260830_061824.npz` |
| v3stand_20260830_060832 | 15251 | 14991 | v3stand | train | `v3stand_20260830_060832/v3stand_20260830_060832.npz` |
| v4lnewstand_20260830_062453 | 16321 | 14950 | v4lnewstand | train | `v4lnewstand_20260830_062453/v4lnewstand_20260830_062453.npz` |

### 8.4 Twin corpus manifest (`data_twin/manifest.json`, 190 runs, 3,110,177 valid transitions; splits train 157 / val 21 / test 12)

Per-strategy totals (twin bench families of 2026-09-25 and the 2026-10-04/05 A/B rounds):

| strategy | runs | valid transitions |
|---|---|---|
| g36 | 50 | 713719 |
| s11 | 10 | 142388 |
| s12 | 23 | 324568 |
| s25 | 14 | 234120 |
| s25pb3 | 1 | 23185 |
| s25pb5 | 1 | 23115 |
| s25s40 | 6 | 138736 |
| s36 | 13 | 269833 |
| s36gh | 2 | 46280 |
| s37 | 4 | 69056 |
| s6 | 1 | 9336 |
| sw | 65 | 1115841 |

Full run table (190 runs). Source paths relative to `~/Desktop/h12/logs/` (bench directories `grid37_twin/d30`, `grid37_twin/d37`, `sweep25`, ...).

| run | n | n_bp | strategy | split | source (relative to ~/Desktop/h12/logs/) |
|---|---|---|---|---|---|
| g36_Balancex0.5_r1_20260925_131920 | 14382 | 14382 | g36 | train | `grid37_twin/d30/g36_Balancex0.5_r1/g36_Balancex0.5_r1_20260925_131920/g36_Balancex0.5_r1_20260925_131920.npz` |
| g36_Balancex0.5_r2_20260925_135746 | 13788 | 13788 | g36 | train | `grid37_twin/d30/g36_Balancex0.5_r2/g36_Balancex0.5_r2_20260925_135746/g36_Balancex0.5_r2_20260925_135746.npz` |
| g36_Balancex0.5_r4_20260925_151357 | 14275 | 14275 | g36 | train | `grid37_twin/d30/g36_Balancex0.5_r4/g36_Balancex0.5_r4_20260925_151357/g36_Balancex0.5_r4_20260925_151357.npz` |
| g36_Balancex2_r2_20260925_135227 | 14250 | 14250 | g36 | val | `grid37_twin/d30/g36_Balancex2_r2/g36_Balancex2_r2_20260925_135227/g36_Balancex2_r2_20260925_135227.npz` |
| g36_Balancex2_r3_20260925_143108 | 14358 | 14358 | g36 | val | `grid37_twin/d30/g36_Balancex2_r3/g36_Balancex2_r3_20260925_143108/g36_Balancex2_r3_20260925_143108.npz` |
| g36_Balancex2_r4_20260925_150838 | 13965 | 13965 | g36 | train | `grid37_twin/d30/g36_Balancex2_r4/g36_Balancex2_r4_20260925_150838/g36_Balancex2_r4_20260925_150838.npz` |
| g36_BracePosx0.5_r1_20260925_132440 | 14475 | 14475 | g36 | train | `grid37_twin/d30/g36_BracePosx0.5_r1/g36_BracePosx0.5_r1_20260925_132440/g36_BracePosx0.5_r1_20260925_132440.npz` |
| g36_BracePosx0.5_r2_20260925_142105 | 14390 | 14390 | g36 | train | `grid37_twin/d30/g36_BracePosx0.5_r2/g36_BracePosx0.5_r2_20260925_142105/g36_BracePosx0.5_r2_20260925_142105.npz` |
| g36_BracePosx0.5_r3_20260925_150318 | 14405 | 14405 | g36 | train | `grid37_twin/d30/g36_BracePosx0.5_r3/g36_BracePosx0.5_r3_20260925_150318/g36_BracePosx0.5_r3_20260925_150318.npz` |
| g36_BracePosx0.5_r4_20260925_153435 | 14358 | 14358 | g36 | train | `grid37_twin/d30/g36_BracePosx0.5_r4/g36_BracePosx0.5_r4_20260925_153435/g36_BracePosx0.5_r4_20260925_153435.npz` |
| g36_BracePosx2_r1_20260925_133936 | 14357 | 14357 | g36 | train | `grid37_twin/d30/g36_BracePosx2_r1/g36_BracePosx2_r1_20260925_133936/g36_BracePosx2_r1_20260925_133936.npz` |
| g36_CoMVelx0.5_r2_20260925_134947 | 13996 | 13996 | g36 | train | `grid37_twin/d30/g36_CoMVelx0.5_r2/g36_CoMVelx0.5_r2_20260925_134947/g36_CoMVelx0.5_r2_20260925_134947.npz` |
| g36_CoMVelx0.5_r4_20260925_153155 | 14431 | 14431 | g36 | test | `grid37_twin/d30/g36_CoMVelx0.5_r4/g36_CoMVelx0.5_r4_20260925_153155/g36_CoMVelx0.5_r4_20260925_153155.npz` |
| g36_CoMVelx2_r1_20260925_133203 | 14121 | 14121 | g36 | train | `grid37_twin/d30/g36_CoMVelx2_r1/g36_CoMVelx2_r1_20260925_133203/g36_CoMVelx2_r1_20260925_133203.npz` |
| g36_CoMVelx2_r2_20260925_141038 | 14396 | 14396 | g36 | train | `grid37_twin/d30/g36_CoMVelx2_r2/g36_CoMVelx2_r2_20260925_141038/g36_CoMVelx2_r2_20260925_141038.npz` |
| g36_CoMVelx2_r4_20260925_152659 | 14535 | 14535 | g36 | train | `grid37_twin/d30/g36_CoMVelx2_r4/g36_CoMVelx2_r4_20260925_152659/g36_CoMVelx2_r4_20260925_152659.npz` |
| g36_KneesStraightx0.5_r2_20260925_140518 | 14190 | 14190 | g36 | train | `grid37_twin/d30/g36_KneesStraightx0.5_r2/g36_KneesStraightx0.5_r2_20260925_140518/g36_KneesStraightx0.5_r2_20260925_140518.npz` |
| g36_KneesStraightx0.5_r4_20260925_150558 | 13515 | 13515 | g36 | train | `grid37_twin/d30/g36_KneesStraightx0.5_r4/g36_KneesStraightx0.5_r4_20260925_150558/g36_KneesStraightx0.5_r4_20260925_150558.npz` |
| g36_KneesStraightx2_r10_20260925_155814 | 14152 | 14152 | g36 | train | `grid37_twin/d30/g36_KneesStraightx2_r10/g36_KneesStraightx2_r10_20260925_155814/g36_KneesStraightx2_r10_20260925_155814.npz` |
| g36_KneesStraightx2_r11_20260925_160053 | 14386 | 14386 | g36 | train | `grid37_twin/d30/g36_KneesStraightx2_r11/g36_KneesStraightx2_r11_20260925_160053/g36_KneesStraightx2_r11_20260925_160053.npz` |
| g36_KneesStraightx2_r12_20260925_160333 | 14420 | 14420 | g36 | train | `grid37_twin/d30/g36_KneesStraightx2_r12/g36_KneesStraightx2_r12_20260925_160333/g36_KneesStraightx2_r12_20260925_160333.npz` |
| g36_KneesStraightx2_r2_20260925_140757 | 14409 | 14409 | g36 | val | `grid37_twin/d30/g36_KneesStraightx2_r2/g36_KneesStraightx2_r2_20260925_140757/g36_KneesStraightx2_r2_20260925_140757.npz` |
| g36_KneesStraightx2_r3_20260925_144626 | 14483 | 14483 | g36 | train | `grid37_twin/d30/g36_KneesStraightx2_r3/g36_KneesStraightx2_r3_20260925_144626/g36_KneesStraightx2_r3_20260925_144626.npz` |
| g36_KneesStraightx2_r4_20260925_152419 | 14410 | 14410 | g36 | train | `grid37_twin/d30/g36_KneesStraightx2_r4/g36_KneesStraightx2_r4_20260925_152419/g36_KneesStraightx2_r4_20260925_152419.npz` |
| g36_KneesStraightx2_r6_20260925_154757 | 14443 | 14443 | g36 | train | `grid37_twin/d30/g36_KneesStraightx2_r6/g36_KneesStraightx2_r6_20260925_154757/g36_KneesStraightx2_r6_20260925_154757.npz` |
| g36_KneesStraightx2_r8_20260925_155254 | 14384 | 14384 | g36 | train | `grid37_twin/d30/g36_KneesStraightx2_r8/g36_KneesStraightx2_r8_20260925_155254/g36_KneesStraightx2_r8_20260925_155254.npz` |
| g36_KneesStraightx2_r9_20260925_155534 | 14414 | 14414 | g36 | train | `grid37_twin/d30/g36_KneesStraightx2_r9/g36_KneesStraightx2_r9_20260925_155534/g36_KneesStraightx2_r9_20260925_155534.npz` |
| g36_LatCenterSetx150_r1_20260925_160833 | 14233 | 14233 | g36 | train | `grid37_twin/d30/g36_LatCenterSetx150_r1/g36_LatCenterSetx150_r1_20260925_160833/g36_LatCenterSetx150_r1_20260925_160833.npz` |
| g36_LatCenterSetx150_r2_20260925_161113 | 14396 | 14396 | g36 | train | `grid37_twin/d30/g36_LatCenterSetx150_r2/g36_LatCenterSetx150_r2_20260925_161113/g36_LatCenterSetx150_r2_20260925_161113.npz` |
| g36_LatCenterSetx150_r3_20260925_161352 | 14367 | 14367 | g36 | val | `grid37_twin/d30/g36_LatCenterSetx150_r3/g36_LatCenterSetx150_r3_20260925_161352/g36_LatCenterSetx150_r3_20260925_161352.npz` |
| g36_PelvisTiltx0.5_r1_20260925_132200 | 13919 | 13919 | g36 | train | `grid37_twin/d30/g36_PelvisTiltx0.5_r1/g36_PelvisTiltx0.5_r1_20260925_132200/g36_PelvisTiltx0.5_r1_20260925_132200.npz` |
| g36_PelvisTiltx0.5_r2_20260925_140238 | 14222 | 14222 | g36 | train | `grid37_twin/d30/g36_PelvisTiltx0.5_r2/g36_PelvisTiltx0.5_r2_20260925_140238/g36_PelvisTiltx0.5_r2_20260925_140238.npz` |
| g36_PelvisTiltx0.5_r3_20260925_144115 | 14324 | 14324 | g36 | train | `grid37_twin/d30/g36_PelvisTiltx0.5_r3/g36_PelvisTiltx0.5_r3_20260925_144115/g36_PelvisTiltx0.5_r3_20260925_144115.npz` |
| g36_PelvisTiltx0.5_r4_20260925_151859 | 13915 | 13915 | g36 | test | `grid37_twin/d30/g36_PelvisTiltx0.5_r4/g36_PelvisTiltx0.5_r4_20260925_151859/g36_PelvisTiltx0.5_r4_20260925_151859.npz` |
| g36_PelvisTiltx2_r1_20260925_133443 | 14355 | 14355 | g36 | val | `grid37_twin/d30/g36_PelvisTiltx2_r1/g36_PelvisTiltx2_r1_20260925_133443/g36_PelvisTiltx2_r1_20260925_133443.npz` |
| g36_PelvisTiltx2_r3_20260925_145822 | 13721 | 13721 | g36 | train | `grid37_twin/d30/g36_PelvisTiltx2_r3/g36_PelvisTiltx2_r3_20260925_145822/g36_PelvisTiltx2_r3_20260925_145822.npz` |
| g36_Rampx0.5_r2_20260925_141558 | 14388 | 14388 | g36 | train | `grid37_twin/d30/g36_Rampx0.5_r2/g36_Rampx0.5_r2_20260925_141558/g36_Rampx0.5_r2_20260925_141558.npz` |
| g36_Rampx2_r1_20260925_134427 | 14411 | 14411 | g36 | val | `grid37_twin/d30/g36_Rampx2_r1/g36_Rampx2_r1_20260925_134427/g36_Rampx2_r1_20260925_134427.npz` |
| g36_Rampx2_r2_20260925_142345 | 14385 | 14385 | g36 | test | `grid37_twin/d30/g36_Rampx2_r2/g36_Rampx2_r2_20260925_142345/g36_Rampx2_r2_20260925_142345.npz` |
| g36_Rampx2_r4_20260925_153926 | 14380 | 14380 | g36 | train | `grid37_twin/d30/g36_Rampx2_r4/g36_Rampx2_r4_20260925_153926/g36_Rampx2_r4_20260925_153926.npz` |
| g36_TorsoTiltx0.5_r1_20260925_134707 | 14141 | 14141 | g36 | train | `grid37_twin/d30/g36_TorsoTiltx0.5_r1/g36_TorsoTiltx0.5_r1_20260925_134707/g36_TorsoTiltx0.5_r1_20260925_134707.npz` |
| g36_TorsoTiltx0.5_r2_20260925_141318 | 14479 | 14479 | g36 | train | `grid37_twin/d30/g36_TorsoTiltx0.5_r2/g36_TorsoTiltx0.5_r2_20260925_141318/g36_TorsoTiltx0.5_r2_20260925_141318.npz` |
| g36_TorsoTiltx0.5_r3_20260925_143830 | 14068 | 14068 | g36 | train | `grid37_twin/d30/g36_TorsoTiltx0.5_r3/g36_TorsoTiltx0.5_r3_20260925_143830/g36_TorsoTiltx0.5_r3_20260925_143830.npz` |
| g36_TorsoTiltx0.5_r4_20260925_154206 | 14459 | 14459 | g36 | train | `grid37_twin/d30/g36_TorsoTiltx0.5_r4/g36_TorsoTiltx0.5_r4_20260925_154206/g36_TorsoTiltx0.5_r4_20260925_154206.npz` |
| g36_TorsoTiltx2_r2_20260925_135507 | 14129 | 14129 | g36 | val | `grid37_twin/d30/g36_TorsoTiltx2_r2/g36_TorsoTiltx2_r2_20260925_135507/g36_TorsoTiltx2_r2_20260925_135507.npz` |
| g36_TorsoTiltx2_r3_20260925_143348 | 14229 | 14229 | g36 | train | `grid37_twin/d30/g36_TorsoTiltx2_r3/g36_TorsoTiltx2_r3_20260925_143348/g36_TorsoTiltx2_r3_20260925_143348.npz` |
| g36_TorsoTiltx2_r4_20260925_151118 | 14295 | 14295 | g36 | train | `grid37_twin/d30/g36_TorsoTiltx2_r4/g36_TorsoTiltx2_r4_20260925_151118/g36_TorsoTiltx2_r4_20260925_151118.npz` |
| g36_nom_r1_20260925_131423 | 14444 | 14444 | g36 | train | `grid37_twin/d30/g36_nom_r1/g36_nom_r1_20260925_131423/g36_nom_r1_20260925_131423.npz` |
| g36_nom_r3_20260925_144906 | 14368 | 14368 | g36 | train | `grid37_twin/d30/g36_nom_r3/g36_nom_r3_20260925_144906/g36_nom_r3_20260925_144906.npz` |
| g36_nom_r4_20260925_152140 | 14403 | 14403 | g36 | train | `grid37_twin/d30/g36_nom_r4/g36_nom_r4_20260925_152140/g36_nom_r4_20260925_152140.npz` |
| s11_B3_20260912_185757 | 14435 | 14435 | s11 | train | `grid37_twin/d30/s11_B3/s11_B3_20260912_185757/s11_B3_20260912_185757.npz` |
| s11_B3d10_20260912_190612 | 14080 | 14080 | s11 | train | `grid37_twin/d30/s11_B3d10/s11_B3d10_20260912_190612/s11_B3d10_20260912_190612.npz` |
| s11_B3l10_20260912_190327 | 14288 | 14288 | s11 | train | `grid37_twin/d30/s11_B3l10/s11_B3l10_20260912_190327/s11_B3l10_20260912_190327.npz` |
| s11_B3ld_20260912_191141 | 13827 | 13827 | s11 | train | `grid37_twin/d30/s11_B3ld/s11_B3ld_20260912_191141/s11_B3ld_20260912_191141.npz` |
| s11_B3r10_20260912_190042 | 14441 | 14441 | s11 | train | `grid37_twin/d30/s11_B3r10/s11_B3r10_20260912_190042/s11_B3r10_20260912_190042.npz` |
| s11_B3rd_20260912_190856 | 14074 | 14074 | s11 | train | `grid37_twin/d30/s11_B3rd/s11_B3rd_20260912_190856/s11_B3rd_20260912_190856.npz` |
| s12_B3_20260912_184129 | 14411 | 14411 | s12 | val | `grid37_twin/d30/s12_B3/s12_B3_20260912_184129/s12_B3_20260912_184129.npz` |
| s12_B3d10_20260912_184943 | 14273 | 14273 | s12 | train | `grid37_twin/d30/s12_B3d10/s12_B3d10_20260912_184943/s12_B3d10_20260912_184943.npz` |
| s12_B3l10_20260912_184658 | 14140 | 14140 | s12 | train | `grid37_twin/d30/s12_B3l10/s12_B3l10_20260912_184658/s12_B3l10_20260912_184658.npz` |
| s12_B3ld_20260912_185513 | 14403 | 14403 | s12 | train | `grid37_twin/d30/s12_B3ld/s12_B3ld_20260912_185513/s12_B3ld_20260912_185513.npz` |
| s12_B3r10_20260912_184414 | 13538 | 13538 | s12 | train | `grid37_twin/d30/s12_B3r10/s12_B3r10_20260912_184414/s12_B3r10_20260912_184414.npz` |
| s12_B3rd_20260912_185228 | 13662 | 13662 | s12 | train | `grid37_twin/d30/s12_B3rd/s12_B3rd_20260912_185228/s12_B3rd_20260912_185228.npz` |
| s25s40_t30_g1_20260925_060445 | 23157 | 23157 | s25s40 | train | `grid37_twin/d30/s25s40_t30_g1/s25s40_t30_g1_20260925_060445/s25s40_t30_g1_20260925_060445.npz` |
| s25s40_t30_g2_20260925_061234 | 23253 | 23253 | s25s40 | train | `grid37_twin/d30/s25s40_t30_g2/s25s40_t30_g2_20260925_061234/s25s40_t30_g2_20260925_061234.npz` |
| s25s40_t30_g4_20260925_062620 | 22989 | 22989 | s25s40 | train | `grid37_twin/d30/s25s40_t30_g4/s25s40_t30_g4_20260925_062620/s25s40_t30_g4_20260925_062620.npz` |
| s36_t30_g1_20260925_060033 | 22717 | 22717 | s36 | train | `grid37_twin/d30/s36_t30_g1/s36_t30_g1_20260925_060033/s36_t30_g1_20260925_060033.npz` |
| s36_t30_g3_20260925_061645 | 22824 | 22824 | s36 | train | `grid37_twin/d30/s36_t30_g3/s36_t30_g3_20260925_061645/s36_t30_g3_20260925_061645.npz` |
| s36_t30b_g1_20260925_063121 | 23158 | 23158 | s36 | train | `grid37_twin/d30/s36_t30b_g1/s36_t30b_g1_20260925_063121/s36_t30b_g1_20260925_063121.npz` |
| s36_t30b_g2_20260925_063533 | 22793 | 22793 | s36 | train | `grid37_twin/d30/s36_t30b_g2/s36_t30b_g2_20260925_063533/s36_t30b_g2_20260925_063533.npz` |
| s36_t30b_g3_20260925_063945 | 22545 | 22545 | s36 | train | `grid37_twin/d30/s36_t30b_g3/s36_t30b_g3_20260925_063945/s36_t30b_g3_20260925_063945.npz` |
| s11_B3_20260912_174131 | 14387 | 14387 | s11 | train | `grid37_twin/d30_v1_above/s11_B3/s11_B3_20260912_174131/s11_B3_20260912_174131.npz` |
| s11_B3l10_20260912_174701 | 14316 | 14316 | s11 | train | `grid37_twin/d30_v1_above/s11_B3l10/s11_B3l10_20260912_174701/s11_B3l10_20260912_174701.npz` |
| s11_B3r10_20260912_174416 | 14327 | 14327 | s11 | val | `grid37_twin/d30_v1_above/s11_B3r10/s11_B3r10_20260912_174416/s11_B3r10_20260912_174416.npz` |
| s12_B3_20260912_172503 | 14158 | 14158 | s12 | train | `grid37_twin/d30_v1_above/s12_B3/s12_B3_20260912_172503/s12_B3_20260912_172503.npz` |
| s12_B3d10_20260912_173317 | 13791 | 13791 | s12 | train | `grid37_twin/d30_v1_above/s12_B3d10/s12_B3d10_20260912_173317/s12_B3d10_20260912_173317.npz` |
| s12_B3l10_20260912_173032 | 14217 | 14217 | s12 | train | `grid37_twin/d30_v1_above/s12_B3l10/s12_B3l10_20260912_173032/s12_B3l10_20260912_173032.npz` |
| s12_B3ld_20260912_173846 | 14248 | 14248 | s12 | train | `grid37_twin/d30_v1_above/s12_B3ld/s12_B3ld_20260912_173846/s12_B3ld_20260912_173846.npz` |
| s12_B3r10_20260912_172747 | 14371 | 14371 | s12 | train | `grid37_twin/d30_v1_above/s12_B3r10/s12_B3r10_20260912_172747/s12_B3r10_20260912_172747.npz` |
| s12_B3rd_20260912_173602 | 14015 | 14015 | s12 | train | `grid37_twin/d30_v1_above/s12_B3rd/s12_B3rd_20260912_173602/s12_B3rd_20260912_173602.npz` |
| s12_B3_20260912_175231 | 14218 | 14218 | s12 | train | `grid37_twin/d30_v2_aim45/s12_B3/s12_B3_20260912_175231/s12_B3_20260912_175231.npz` |
| s12_B3_20260912_181649 | 14261 | 14261 | s12 | train | `grid37_twin/d30_v3_hover_lunge/s12_B3/s12_B3_20260912_181649/s12_B3_20260912_181649.npz` |
| s12_B3_20260912_182036 | 14030 | 14030 | s12 | train | `grid37_twin/d30_v4_hover300/s12_B3/s12_B3_20260912_182036/s12_B3_20260912_182036.npz` |
| s12_B3_20260912_182359 | 14213 | 14213 | s12 | val | `grid37_twin/d30_v5_hover_above/s12_B3/s12_B3_20260912_182359/s12_B3_20260912_182359.npz` |
| s12_B3_20260912_182909 | 14215 | 14215 | s12 | train | `grid37_twin/d30_v6_front_noaim/s12_B3/s12_B3_20260912_182909/s12_B3_20260912_182909.npz` |
| s12_B3r10_20260912_183154 | 14397 | 14397 | s12 | train | `grid37_twin/d30_v6_front_noaim/s12_B3r10/s12_B3r10_20260912_183154/s12_B3r10_20260912_183154.npz` |
| s12_B3_20260912_183738 | 14350 | 14350 | s12 | train | `grid37_twin/d30_v7_front_yawaim/s12_B3/s12_B3_20260912_183738/s12_B3_20260912_183738.npz` |
| s11_B3_20260912_171217 | 14213 | 14213 | s11 | train | `grid37_twin/d37/s11_B3/s11_B3_20260912_171217/s11_B3_20260912_171217.npz` |
| s12_B3_20260912_170807 | 14074 | 14074 | s12 | train | `grid37_twin/d37/s12_B3/s12_B3_20260912_170807/s12_B3_20260912_170807.npz` |
| s12_B3d10_20260912_172032 | 14278 | 14278 | s12 | train | `grid37_twin/d37/s12_B3d10/s12_B3d10_20260912_172032/s12_B3d10_20260912_172032.npz` |
| s12_B3l10_20260912_171747 | 13102 | 13102 | s12 | train | `grid37_twin/d37/s12_B3l10/s12_B3l10_20260912_171747/s12_B3l10_20260912_171747.npz` |
| s12_B3r10_20260912_171502 | 14203 | 14203 | s12 | test | `grid37_twin/d37/s12_B3r10/s12_B3r10_20260912_171502/s12_B3r10_20260912_171502.npz` |
| s25_B3_20260922_184835 | 14129 | 14129 | s25 | train | `grid37_twin/d37/s25_B3/s25_B3_20260922_184835/s25_B3_20260922_184835.npz` |
| s25_B3_bare_20260924_114824 | 14380 | 14380 | s25 | train | `grid37_twin/d37/s25_B3_bare/s25_B3_bare_20260924_114824/s25_B3_bare_20260924_114824.npz` |
| s25_B3_bare_early_20260924_115938 | 14377 | 14377 | s25 | train | `grid37_twin/d37/s25_B3_bare_early/s25_B3_bare_early_20260924_115938/s25_B3_bare_early_20260924_115938.npz` |
| s25_B3_bare_early_comfwd_20260924_121148 | 14353 | 14353 | s25 | train | `grid37_twin/d37/s25_B3_bare_early_comfwd/s25_B3_bare_early_comfwd_20260924_121148/s25_B3_bare_early_comfwd_20260924_121148.npz` |
| s25_B3_bare_early_foff0_20260924_120844 | 11274 | 11274 | s25 | train | `grid37_twin/d37/s25_B3_bare_early_foff0/s25_B3_bare_early_foff0_20260924_120844/s25_B3_bare_early_foff0_20260924_120844.npz` |
| s25_B3_bare_early_zR43_20260924_120508 | 14237 | 14237 | s25 | train | `grid37_twin/d37/s25_B3_bare_early_zR43/s25_B3_bare_early_zR43_20260924_120508/s25_B3_bare_early_zR43_20260924_120508.npz` |
| s25_B3_noassist_20260924_123720 | 14197 | 14197 | s25 | val | `grid37_twin/d37/s25_B3_noassist/s25_B3_noassist_20260924_123720/s25_B3_noassist_20260924_123720.npz` |
| s25_B3_noassist_base_r1_20261004_095020 | 14412 | 14412 | s25 | train | `grid37_twin/d37/s25_B3_noassist_base_r1/s25_B3_noassist_base_r1_20261004_095020/s25_B3_noassist_base_r1_20261004_095020.npz` |
| s25_B3_noassist_g1_20260924_124831 | 19319 | 19319 | s25 | val | `grid37_twin/d37/s25_B3_noassist_g1/s25_B3_noassist_g1_20260924_124831/s25_B3_noassist_g1_20260924_124831.npz` |
| s25_B3_noassist_g3_20260924_130019 | 19205 | 19205 | s25 | train | `grid37_twin/d37/s25_B3_noassist_g3/s25_B3_noassist_g3_20260924_130019/s25_B3_noassist_g3_20260924_130019.npz` |
| s25_B3_noassist_g4_20260924_130602 | 18933 | 18933 | s25 | train | `grid37_twin/d37/s25_B3_noassist_g4/s25_B3_noassist_g4_20260924_130602/s25_B3_noassist_g4_20260924_130602.npz` |
| s25_B3_ns_g1_20260924_224331 | 23192 | 23192 | s25 | train | `grid37_twin/d37/s25_B3_ns_g1/s25_B3_ns_g1_20260924_224331/s25_B3_ns_g1_20260924_224331.npz` |
| s25_B3_ns_g2_20260924_225010 | 22749 | 22749 | s25 | train | `grid37_twin/d37/s25_B3_ns_g2/s25_B3_ns_g2_20260924_225010/s25_B3_ns_g2_20260924_225010.npz` |
| s25_B3_real_g2_20260924_143357 | 19363 | 19363 | s25 | train | `grid37_twin/d37/s25_B3_real_g2/s25_B3_real_g2_20260924_143357/s25_B3_real_g2_20260924_143357.npz` |
| s25pb3_pbcal_g1_20260925_000404 | 23185 | 23185 | s25pb3 | train | `grid37_twin/d37/s25pb3_pbcal_g1/s25pb3_pbcal_g1_20260925_000404/s25pb3_pbcal_g1_20260925_000404.npz` |
| s25pb5_pbcal_g1_20260925_000815 | 23115 | 23115 | s25pb5 | train | `grid37_twin/d37/s25pb5_pbcal_g1/s25pb5_pbcal_g1_20260925_000815/s25pb5_pbcal_g1_20260925_000815.npz` |
| s25s40_ns2_g1_20260924_231026 | 23102 | 23102 | s25s40 | train | `grid37_twin/d37/s25s40_ns2_g1/s25s40_ns2_g1_20260924_231026/s25s40_ns2_g1_20260924_231026.npz` |
| s25s40_ns2_g2_20260924_232056 | 23055 | 23055 | s25s40 | train | `grid37_twin/d37/s25s40_ns2_g2/s25s40_ns2_g2_20260924_232056/s25s40_ns2_g2_20260924_232056.npz` |
| s25s40_ns2_g3_20260924_233141 | 23180 | 23180 | s25s40 | train | `grid37_twin/d37/s25s40_ns2_g3/s25s40_ns2_g3_20260924_233141/s25s40_ns2_g3_20260924_233141.npz` |
| s36_B3_bare_20260924_115428 | 14223 | 14223 | s36 | train | `grid37_twin/d37/s36_B3_bare/s36_B3_bare_20260924_115428/s36_B3_bare_20260924_115428.npz` |
| s36_B3_bare_early_comfwd_20260924_121503 | 17226 | 17226 | s36 | train | `grid37_twin/d37/s36_B3_bare_early_comfwd/s36_B3_bare_early_comfwd_20260924_121503/s36_B3_bare_early_comfwd_20260924_121503.npz` |
| s36_B3_bare_early_comfwd_v2_20260924_121908 | 16911 | 16911 | s36 | val | `grid37_twin/d37/s36_B3_bare_early_comfwd_v2/s36_B3_bare_early_comfwd_v2_20260924_121908/s36_B3_bare_early_comfwd_v2_20260924_121908.npz` |
| s36_B3_noassist_g2_20260924_125648 | 18895 | 18895 | s36 | train | `grid37_twin/d37/s36_B3_noassist_g2/s36_B3_noassist_g2_20260924_125648/s36_B3_noassist_g2_20260924_125648.npz` |
| s36_B3_noassist_v3_20260924_124352 | 19292 | 19292 | s36 | train | `grid37_twin/d37/s36_B3_noassist_v3/s36_B3_noassist_v3_20260924_124352/s36_B3_noassist_v3_20260924_124352.npz` |
| s36_B3_ns_g2_20260924_225422 | 23033 | 23033 | s36 | val | `grid37_twin/d37/s36_B3_ns_g2/s36_B3_ns_g2_20260924_225422/s36_B3_ns_g2_20260924_225422.npz` |
| s36_ns2_g1_20260924_230202 | 23263 | 23263 | s36 | train | `grid37_twin/d37/s36_ns2_g1/s36_ns2_g1_20260924_230202/s36_ns2_g1_20260924_230202.npz` |
| s36_ns3_g3_20260924_234840 | 22953 | 22953 | s36 | train | `grid37_twin/d37/s36_ns3_g3/s36_ns3_g3_20260924_234840/s36_ns3_g3_20260924_234840.npz` |
| s36gh_ns2_g1_20260924_230614 | 23208 | 23208 | s36gh | train | `grid37_twin/d37/s36gh_ns2_g1/s36gh_ns2_g1_20260924_230614/s36gh_ns2_g1_20260924_230614.npz` |
| s36gh_ns2_g3_20260924_232730 | 23072 | 23072 | s36gh | train | `grid37_twin/d37/s36gh_ns2_g3/s36gh_ns2_g3_20260924_232730/s36gh_ns2_g3_20260924_232730.npz` |
| s37_B3_noassist_base_r4_20261004_101116 | 17284 | 17284 | s37 | train | `grid37_twin/d37/s37_B3_noassist_base_r4/s37_B3_noassist_base_r4_20261004_101116/s37_B3_noassist_base_r4_20261004_101116.npz` |
| s37_B3_noassist_res_r1_20261004_095728 | 17290 | 17290 | s37 | train | `grid37_twin/d37/s37_B3_noassist_res_r1/s37_B3_noassist_res_r1_20261004_095728/s37_B3_noassist_res_r1_20261004_095728.npz` |
| s37_B3_noassist_res_r3_20261004_100801 | 17232 | 17232 | s37 | test | `grid37_twin/d37/s37_B3_noassist_res_r3/s37_B3_noassist_res_r3_20261004_100801/s37_B3_noassist_res_r3_20261004_100801.npz` |
| s37_B3_noassist_res_r4_20261004_101432 | 17250 | 17250 | s37 | test | `grid37_twin/d37/s37_B3_noassist_res_r4/s37_B3_noassist_res_r4_20261004_101432/s37_B3_noassist_res_r4_20261004_101432.npz` |
| s6_B3_bare_early_20260924_123453 | 9336 | 9336 | s6 | train | `grid37_twin/d37/s6_B3_bare_early/s6_B3_bare_early_20260924_123453/s6_B3_bare_early_20260924_123453.npz` |
| sw_Balancex0.5_r1_20260925_005052 | 17407 | 17407 | sw | train | `grid37_twin/d37/sw_Balancex0.5_r1/sw_Balancex0.5_r1_20260925_005052/sw_Balancex0.5_r1_20260925_005052.npz` |
| sw_Balancex0.5_r2_20260925_013522 | 16995 | 16995 | sw | train | `grid37_twin/d37/sw_Balancex0.5_r2/sw_Balancex0.5_r2_20260925_013522/sw_Balancex0.5_r2_20260925_013522.npz` |
| sw_Balancex0.5_r3_20260925_021020 | 16979 | 16979 | sw | train | `grid37_twin/d37/sw_Balancex0.5_r3/sw_Balancex0.5_r3_20260925_021020/sw_Balancex0.5_r3_20260925_021020.npz` |
| sw_Balancex0.5_r4_20260925_022303 | 16964 | 16964 | sw | val | `grid37_twin/d37/sw_Balancex0.5_r4/sw_Balancex0.5_r4_20260925_022303/sw_Balancex0.5_r4_20260925_022303.npz` |
| sw_Balancex0.5_r5_20260925_030422 | 17177 | 17177 | sw | train | `grid37_twin/d37/sw_Balancex0.5_r5/sw_Balancex0.5_r5_20260925_030422/sw_Balancex0.5_r5_20260925_030422.npz` |
| sw_Balancex2_r1_20260925_003458 | 17286 | 17286 | sw | train | `grid37_twin/d37/sw_Balancex2_r1/sw_Balancex2_r1_20260925_003458/sw_Balancex2_r1_20260925_003458.npz` |
| sw_Balancex2_r2_20260925_011617 | 17325 | 17325 | sw | train | `grid37_twin/d37/sw_Balancex2_r2/sw_Balancex2_r2_20260925_011617/sw_Balancex2_r2_20260925_011617.npz` |
| sw_Balancex2_r3_20260925_015737 | 17059 | 17059 | sw | train | `grid37_twin/d37/sw_Balancex2_r3/sw_Balancex2_r3_20260925_015737/sw_Balancex2_r3_20260925_015737.npz` |
| sw_Balancex2_r4_20260925_023857 | 16700 | 16700 | sw | train | `grid37_twin/d37/sw_Balancex2_r4/sw_Balancex2_r4_20260925_023857/sw_Balancex2_r4_20260925_023857.npz` |
| sw_Balancex2_r5_20260925_032015 | 17318 | 17318 | sw | train | `grid37_twin/d37/sw_Balancex2_r5/sw_Balancex2_r5_20260925_032015/sw_Balancex2_r5_20260925_032015.npz` |
| sw_BaseHeightx0.5_r1_20260925_004119 | 17250 | 17250 | sw | train | `grid37_twin/d37/sw_BaseHeightx0.5_r1/sw_BaseHeightx0.5_r1_20260925_004119/sw_BaseHeightx0.5_r1_20260925_004119.npz` |
| sw_BaseHeightx0.5_r2_20260925_010956 | 17338 | 17338 | sw | train | `grid37_twin/d37/sw_BaseHeightx0.5_r2/sw_BaseHeightx0.5_r2_20260925_010956/sw_BaseHeightx0.5_r2_20260925_010956.npz` |
| sw_BaseHeightx0.5_r3_20260925_020359 | 17175 | 17175 | sw | val | `grid37_twin/d37/sw_BaseHeightx0.5_r3/sw_BaseHeightx0.5_r3_20260925_020359/sw_BaseHeightx0.5_r3_20260925_020359.npz` |
| sw_BaseHeightx0.5_r4_20260925_024518 | 17270 | 17270 | sw | train | `grid37_twin/d37/sw_BaseHeightx0.5_r4/sw_BaseHeightx0.5_r4_20260925_024518/sw_BaseHeightx0.5_r4_20260925_024518.npz` |
| sw_BaseHeightx0.5_r5_20260925_032637 | 17377 | 17377 | sw | train | `grid37_twin/d37/sw_BaseHeightx0.5_r5/sw_BaseHeightx0.5_r5_20260925_032637/sw_BaseHeightx0.5_r5_20260925_032637.npz` |
| sw_BaseHeightx2_r1_20260925_001904 | 16881 | 16881 | sw | train | `grid37_twin/d37/sw_BaseHeightx2_r1/sw_BaseHeightx2_r1_20260925_001904/sw_BaseHeightx2_r1_20260925_001904.npz` |
| sw_BaseHeightx2_r2_20260925_011307 | 16837 | 16837 | sw | train | `grid37_twin/d37/sw_BaseHeightx2_r2/sw_BaseHeightx2_r2_20260925_011307/sw_BaseHeightx2_r2_20260925_011307.npz` |
| sw_BaseHeightx2_r3_20260925_015116 | 17315 | 17315 | sw | train | `grid37_twin/d37/sw_BaseHeightx2_r3/sw_BaseHeightx2_r3_20260925_015116/sw_BaseHeightx2_r3_20260925_015116.npz` |
| sw_BaseHeightx2_r4_20260925_023546 | 17321 | 17321 | sw | train | `grid37_twin/d37/sw_BaseHeightx2_r4/sw_BaseHeightx2_r4_20260925_023546/sw_BaseHeightx2_r4_20260925_023546.npz` |
| sw_BaseHeightx2_r5_20260925_031704 | 17270 | 17270 | sw | train | `grid37_twin/d37/sw_BaseHeightx2_r5/sw_BaseHeightx2_r5_20260925_031704/sw_BaseHeightx2_r5_20260925_031704.npz` |
| sw_BraceForcex0.5_r1_20260925_002837 | 16820 | 16820 | sw | train | `grid37_twin/d37/sw_BraceForcex0.5_r1/sw_BraceForcex0.5_r1_20260925_002837/sw_BraceForcex0.5_r1_20260925_002837.npz` |
| sw_BraceForcex0.5_r2_20260925_010645 | 17395 | 17395 | sw | train | `grid37_twin/d37/sw_BraceForcex0.5_r2/sw_BraceForcex0.5_r2_20260925_010645/sw_BraceForcex0.5_r2_20260925_010645.npz` |
| sw_BraceForcex0.5_r3_20260925_014805 | 17332 | 17332 | sw | test | `grid37_twin/d37/sw_BraceForcex0.5_r3/sw_BraceForcex0.5_r3_20260925_014805/sw_BraceForcex0.5_r3_20260925_014805.npz` |
| sw_BraceForcex0.5_r4_20260925_022924 | 17365 | 17365 | sw | train | `grid37_twin/d37/sw_BraceForcex0.5_r4/sw_BraceForcex0.5_r4_20260925_022924/sw_BraceForcex0.5_r4_20260925_022924.npz` |
| sw_BraceForcex0.5_r5_20260925_031044 | 17220 | 17220 | sw | train | `grid37_twin/d37/sw_BraceForcex0.5_r5/sw_BraceForcex0.5_r5_20260925_031044/sw_BraceForcex0.5_r5_20260925_031044.npz` |
| sw_BraceForcex2_r1_20260925_002215 | 17085 | 17085 | sw | train | `grid37_twin/d37/sw_BraceForcex2_r1/sw_BraceForcex2_r1_20260925_002215/sw_BraceForcex2_r1_20260925_002215.npz` |
| sw_BraceForcex2_r2_20260925_010024 | 17450 | 17450 | sw | train | `grid37_twin/d37/sw_BraceForcex2_r2/sw_BraceForcex2_r2_20260925_010024/sw_BraceForcex2_r2_20260925_010024.npz` |
| sw_BraceForcex2_r3_20260925_014144 | 17388 | 17388 | sw | train | `grid37_twin/d37/sw_BraceForcex2_r3/sw_BraceForcex2_r3_20260925_014144/sw_BraceForcex2_r3_20260925_014144.npz` |
| sw_BraceForcex2_r4_20260925_024207 | 16484 | 16484 | sw | train | `grid37_twin/d37/sw_BraceForcex2_r4/sw_BraceForcex2_r4_20260925_024207/sw_BraceForcex2_r4_20260925_024207.npz` |
| sw_BraceForcex2_r5_20260925_033258 | 17327 | 17327 | sw | train | `grid37_twin/d37/sw_BraceForcex2_r5/sw_BraceForcex2_r5_20260925_033258/sw_BraceForcex2_r5_20260925_033258.npz` |
| sw_BracePosx0.5_r1_20260925_004430 | 17187 | 17187 | sw | val | `grid37_twin/d37/sw_BracePosx0.5_r1/sw_BracePosx0.5_r1_20260925_004430/sw_BracePosx0.5_r1_20260925_004430.npz` |
| sw_BracePosx0.5_r2_20260925_012239 | 17301 | 17301 | sw | train | `grid37_twin/d37/sw_BracePosx0.5_r2/sw_BracePosx0.5_r2_20260925_012239/sw_BracePosx0.5_r2_20260925_012239.npz` |
| sw_BracePosx0.5_r3_20260925_015427 | 16901 | 16901 | sw | train | `grid37_twin/d37/sw_BracePosx0.5_r3/sw_BracePosx0.5_r3_20260925_015427/sw_BracePosx0.5_r3_20260925_015427.npz` |
| sw_BracePosx0.5_r4_20260925_025800 | 17302 | 17302 | sw | train | `grid37_twin/d37/sw_BracePosx0.5_r4/sw_BracePosx0.5_r4_20260925_025800/sw_BracePosx0.5_r4_20260925_025800.npz` |
| sw_BracePosx0.5_r5_20260925_030111 | 17093 | 17093 | sw | train | `grid37_twin/d37/sw_BracePosx0.5_r5/sw_BracePosx0.5_r5_20260925_030111/sw_BracePosx0.5_r5_20260925_030111.npz` |
| sw_BracePosx2_r1_20260925_005403 | 17306 | 17306 | sw | val | `grid37_twin/d37/sw_BracePosx2_r1/sw_BracePosx2_r1_20260925_005403/sw_BracePosx2_r1_20260925_005403.npz` |
| sw_BracePosx2_r2_20260925_012901 | 17347 | 17347 | sw | train | `grid37_twin/d37/sw_BracePosx2_r2/sw_BracePosx2_r2_20260925_012901/sw_BracePosx2_r2_20260925_012901.npz` |
| sw_BracePosx2_r3_20260925_021330 | 17137 | 17137 | sw | train | `grid37_twin/d37/sw_BracePosx2_r3/sw_BracePosx2_r3_20260925_021330/sw_BracePosx2_r3_20260925_021330.npz` |
| sw_BracePosx2_r4_20260925_025450 | 17350 | 17350 | sw | train | `grid37_twin/d37/sw_BracePosx2_r4/sw_BracePosx2_r4_20260925_025450/sw_BracePosx2_r4_20260925_025450.npz` |
| sw_BracePosx2_r5_20260925_031354 | 17205 | 17205 | sw | train | `grid37_twin/d37/sw_BracePosx2_r5/sw_BracePosx2_r5_20260925_031354/sw_BracePosx2_r5_20260925_031354.npz` |
| sw_FootUpx0.5_r1_20260925_003809 | 16589 | 16589 | sw | train | `grid37_twin/d37/sw_FootUpx0.5_r1/sw_FootUpx0.5_r1_20260925_003809/sw_FootUpx0.5_r1_20260925_003809.npz` |
| sw_FootUpx0.5_r2_20260925_005714 | 17140 | 17140 | sw | train | `grid37_twin/d37/sw_FootUpx0.5_r2/sw_FootUpx0.5_r2_20260925_005714/sw_FootUpx0.5_r2_20260925_005714.npz` |
| sw_FootUpx0.5_r3_20260925_020048 | 16907 | 16907 | sw | train | `grid37_twin/d37/sw_FootUpx0.5_r3/sw_FootUpx0.5_r3_20260925_020048/sw_FootUpx0.5_r3_20260925_020048.npz` |
| sw_FootUpx0.5_r4_20260925_023235 | 17251 | 17251 | sw | train | `grid37_twin/d37/sw_FootUpx0.5_r4/sw_FootUpx0.5_r4_20260925_023235/sw_FootUpx0.5_r4_20260925_023235.npz` |
| sw_FootUpx0.5_r5_20260925_032326 | 17210 | 17210 | sw | train | `grid37_twin/d37/sw_FootUpx0.5_r5/sw_FootUpx0.5_r5_20260925_032326/sw_FootUpx0.5_r5_20260925_032326.npz` |
| sw_FootUpx2_r1_20260925_002526 | 17159 | 17159 | sw | train | `grid37_twin/d37/sw_FootUpx2_r1/sw_FootUpx2_r1_20260925_002526/sw_FootUpx2_r1_20260925_002526.npz` |
| sw_FootUpx2_r2_20260925_010335 | 17064 | 17064 | sw | val | `grid37_twin/d37/sw_FootUpx2_r2/sw_FootUpx2_r2_20260925_010335/sw_FootUpx2_r2_20260925_010335.npz` |
| sw_FootUpx2_r3_20260925_013833 | 16535 | 16535 | sw | test | `grid37_twin/d37/sw_FootUpx2_r3/sw_FootUpx2_r3_20260925_013833/sw_FootUpx2_r3_20260925_013833.npz` |
| sw_FootUpx2_r4_20260925_022613 | 17287 | 17287 | sw | train | `grid37_twin/d37/sw_FootUpx2_r4/sw_FootUpx2_r4_20260925_022613/sw_FootUpx2_r4_20260925_022613.npz` |
| sw_FootUpx2_r5_20260925_030733 | 17123 | 17123 | sw | test | `grid37_twin/d37/sw_FootUpx2_r5/sw_FootUpx2_r5_20260925_030733/sw_FootUpx2_r5_20260925_030733.npz` |
| sw_ReachDistx0.5_r1_20260925_004741 | 17350 | 17350 | sw | test | `grid37_twin/d37/sw_ReachDistx0.5_r1/sw_ReachDistx0.5_r1_20260925_004741/sw_ReachDistx0.5_r1_20260925_004741.npz` |
| sw_ReachDistx0.5_r2_20260925_013211 | 17346 | 17346 | sw | train | `grid37_twin/d37/sw_ReachDistx0.5_r2/sw_ReachDistx0.5_r2_20260925_013211/sw_ReachDistx0.5_r2_20260925_013211.npz` |
| sw_ReachDistx0.5_r3_20260925_021641 | 17293 | 17293 | sw | test | `grid37_twin/d37/sw_ReachDistx0.5_r3/sw_ReachDistx0.5_r3_20260925_021641/sw_ReachDistx0.5_r3_20260925_021641.npz` |
| sw_ReachDistx0.5_r4_20260925_025139 | 17064 | 17064 | sw | train | `grid37_twin/d37/sw_ReachDistx0.5_r4/sw_ReachDistx0.5_r4_20260925_025139/sw_ReachDistx0.5_r4_20260925_025139.npz` |
| sw_ReachDistx0.5_r5_20260925_033920 | 17316 | 17316 | sw | train | `grid37_twin/d37/sw_ReachDistx0.5_r5/sw_ReachDistx0.5_r5_20260925_033920/sw_ReachDistx0.5_r5_20260925_033920.npz` |
| sw_ReachDistx2_r1_20260925_003147 | 17058 | 17058 | sw | train | `grid37_twin/d37/sw_ReachDistx2_r1/sw_ReachDistx2_r1_20260925_003147/sw_ReachDistx2_r1_20260925_003147.npz` |
| sw_ReachDistx2_r2_20260925_012550 | 17261 | 17261 | sw | test | `grid37_twin/d37/sw_ReachDistx2_r2/sw_ReachDistx2_r2_20260925_012550/sw_ReachDistx2_r2_20260925_012550.npz` |
| sw_ReachDistx2_r3_20260925_020710 | 17348 | 17348 | sw | train | `grid37_twin/d37/sw_ReachDistx2_r3/sw_ReachDistx2_r3_20260925_020710/sw_ReachDistx2_r3_20260925_020710.npz` |
| sw_ReachDistx2_r4_20260925_024829 | 17215 | 17215 | sw | train | `grid37_twin/d37/sw_ReachDistx2_r4/sw_ReachDistx2_r4_20260925_024829/sw_ReachDistx2_r4_20260925_024829.npz` |
| sw_ReachDistx2_r5_20260925_032947 | 17329 | 17329 | sw | train | `grid37_twin/d37/sw_ReachDistx2_r5/sw_ReachDistx2_r5_20260925_032947/sw_ReachDistx2_r5_20260925_032947.npz` |
| sw_nom_r1_20260925_001303 | 16768 | 16768 | sw | train | `grid37_twin/d37/sw_nom_r1/sw_nom_r1_20260925_001303/sw_nom_r1_20260925_001303.npz` |
| sw_nom_r2_20260925_011928 | 17277 | 17277 | sw | train | `grid37_twin/d37/sw_nom_r2/sw_nom_r2_20260925_011928/sw_nom_r2_20260925_011928.npz` |
| sw_nom_r3_20260925_014454 | 17328 | 17328 | sw | train | `grid37_twin/d37/sw_nom_r3/sw_nom_r3_20260925_014454/sw_nom_r3_20260925_014454.npz` |
| sw_nom_r4_20260925_021952 | 17266 | 17266 | sw | val | `grid37_twin/d37/sw_nom_r4/sw_nom_r4_20260925_021952/sw_nom_r4_20260925_021952.npz` |
| sw_nom_r5_20260925_033609 | 17118 | 17118 | sw | val | `grid37_twin/d37/sw_nom_r5/sw_nom_r5_20260925_033609/sw_nom_r5_20260925_033609.npz` |

---

## 9. Tooling index (`mbrl/*.py`, `mbrl/*.sh`, all kept)

Shared environment variables: `MBRL_DATA_ROOT` (default `mbrl/data`; set to `mbrl/data_twin` for the twin corpus), `MBRL_OUT` (default `mbrl/out`; `mbrl/out_twin` for the twin), `MBRL_SRC_GLOB` (recording glob relative to `~/Desktop/h12`, default `logs/*/*.npz`), `MBRL_WORKERS`, `MBRL_NSUB` / `MBRL_DT` (substeps and substep length of the plant step: 5 x 0.002 = study protocol, 1 x 0.01 = planner step), `MBRL_INJECT` (`post` or `pre` step velocity correction), `MBRL_H` / `MBRL_NL` (residual MLP width / depth, default 512 / 3; the planner net is 64 / 2), `MBRL_TAG` (output name suffix), `MBRL_MAX_TILT` (keep only segments / samples with torso tilt below this many degrees, e.g. 12 for the hands-off subset), `MBRL_ZERO_TAUFF` (zero the node feed-forward torque in the features so the net sees only planner-available inputs), `MBRL_CMD_NOISE` (planner-like Gaussian command noise in rad added inside DaD rollouts), `MBRL_TAU` (ensemble disagreement gate, rad/s), `MBRL_SEED`. The plant model is always the build-tree `Lean_H12_Magpie.xml` (`common.MODEL`) with actuators disabled and torque applied through `qfrc_applied`.

- `common.py` (83 lines): shared constants and helpers: model path, data/out roots, 27-joint order and groups (legs 0-11, waist 12, arms 13-26), `load_model` (asserts nq 41, nv 39, nu 27, joint qpos addresses 7..33), quaternion helpers, `imu_to_pelvis` (composes the torso-IMU quaternion back through the waist joint to the pelvis), `pd_torque` (clip(kp(cmd - q) - kd dq + tau_ff, lim)), `tilt_deg`.
- `build_dataset.py` (52 lines): scans `MBRL_SRC_GLOB` recordings, keeps every 100 Hz sample whose t and t+1 are finite and 8-12 ms apart, writes `data/runs/<run>.npz` (float32) and `data/manifest.json` with run, n, n_bp, strategy, split. ~2 min for the real corpus.
- `simgap.py` (219 lines): one-step sim-gap driver. For every 2nd loaded-feet sample (`MBRL_STRIDE=2`): pelvis orientation from the torso IMU through the waist, feet rotated flat through their own ankle pitch/roll (max `MBRL_FLAT_MAX=10` deg), pelvis height bisected to zero vertical acceleration, base velocity from the stationary-feet least squares, table moved 100 m and object parked, gripper/keepaway geoms non-colliding, optional `MBRL_ANKLE_OFF="L,R"` ankle-zero offsets; steps 10 ms with the PD command recomputed per substep and with the measured torque held; multiprocessing pool with watchdog (`MBRL_WORKERS=10`, `MBRL_TASK_TIMEOUT=900` s); writes `data/simgap/<run>.npz`.
- `analyze_simgap.py` (95 lines): aggregates the simgap files into `out/simgap_summary.json` (per joint, per group x phase {all, upright, lean, consistent, consistent_upright, consistent_lean}, per strategy, sole-gap percentiles, foot-flattening medians, PD-minus-measured torque statistics) and the figures `fig_simgap_per_joint.png`, `fig_simgap_vs_tilt.png`.
- `ablate_params.py` (41 lines): quick damping / armature / frictionloss ablation on a few runs, one-step dq RMSE per group in tau and pd modes vs hold velocity; writes `out/ablate_params.json`.
- `residual_fit.py` (184 lines): Stage 1 residual: builds the 168-feature / 27-target arrays from `data/runs` + `data/simgap` (consistent samples, `MBRL_MAXTRAIN=4000000` cap, optional `MBRL_MAX_TILT`, `MBRL_ZERO_TAUFF`), fits ridge and a torch MLP (`MBRL_H`, `MBRL_NL`), reports one-step RMSE per group and per strategy on test runs, and runs the 0.5 s closed-loop rollout benchmark (plant / ridge / mlp, upright vs lean starts); also exposes `rollout_eval` used by other scripts. Writes `out/residual_summary.json`, `out/residual_mlp.pt`, `out/residual_norm.npz`, `out/rollout_err.npz`.
- `fig_residual.py` (16 lines): plots `out/fig_residual.png` from `rollout_err.npz`.
- `jacobians.py` (96 lines): nominal plant rollouts and finite-difference Jacobians of the 10 ms macro step along `MBRL_SEG_PER_RUN=15` segments per run (stride `MBRL_SEG_STRIDE=100`, `MBRL_T=25` steps), 20 workers; writes `data/jac/<run>.npz`.
- `multistep_fit.py` (103 lines): linearised multi-step residual training through the Jacobian recursion e_{t+1} = A_t e_t + S r_theta(.), loss = closed-loop displacement error over T, warm start from the one-step MLP, `MBRL_MS_EPOCHS=30`; evaluated with `residual_fit.rollout_eval` on the nonlinear plant. Negative result (diverges).
- `rollout_engine.py` (140 lines): memory-safe parallel nonlinear rollouts of the plant on compact recorded segments with a pluggable numpy residual MLP (post- or pre-step injection, per-group mask, clip, optional ensemble gate) and pluggable plant parameters (damping / armature / torque scales, joint offsets); returns q_sim, dq_sim, dq_plant (before correction) and the 168-feature matrix at each step. Honors `MBRL_MAX_TILT`, `MBRL_ZERO_TAUFF`.
- `dad_fit.py` (58 lines): Data-as-Demonstrator training: roll the plant + current residual on `MBRL_DAD_SEGS=3000` training segments (`MBRL_T=25`), record the one-step target at every visited state, aggregate with `MBRL_DAD_BASE=600000` original one-step samples, fine-tune, `MBRL_DAD_ITERS=5`; validation = rollout displacement error on held-out segments, best iterate kept. `MBRL_CMD_NOISE`, `MBRL_INJECT`, `MBRL_NSUB`, `MBRL_DT`, `MBRL_SEED`, `MBRL_TAG`. Writes `out/residual_mlp_<tag>.pt`, `out/residual_norm_<tag>.npz`, `out/<tag>_summary.json`.
- `eval_masked.py` (17 lines): re-scores a trained residual on validation rollouts at the planner step with the correction masked to all / arms+waist / arms (`MBRL_TAU` for the gated ensemble when several tags are given joined by `+`); writes `out/masked_<tag>.json`.
- `param_id.py` (109 lines): identification v1: stage K joint zero offsets (hip pitch / knee L,R) minimising the sole gap, stage D damping / armature / torque scales minimising one-step dq error (Nelder-Mead, `MBRL_ID_SAMPLES=4000`); writes `out/param_id.json`. Degenerate objective, superseded.
- `param_id2.py` (46 lines): identification v2 with the 250 ms open-loop rollout objective on `MBRL_ID_SEGS=400` fixed consistent segments, log-bounded [1/3, 3] scales for damping / armature / torque (legs, arms) plus stage K offsets bounded +-4 deg; held-out test; writes `out/param_id2.json`. Supports the twin via `MBRL_DATA_ROOT` / `MBRL_OUT`, and the planner step via `MBRL_NSUB=1 MBRL_DT=0.01`.
- `export_mlp.py` (17 lines): exports a residual MLP + standardisation + clip (optionally scaled by `MBRL_CLIP_SCALE`) to the flat little-endian binary the C++ hook loads: magic `MLP1` (0x4d4c5031), nin, nout, nlayers, per layer rows / cols / W row-major / b, then mu, sd, ymu, ysd, clip; SiLU between layers. Writes `out/residual_<tag>.bin`.
- `hook_smoke.py` (46 lines): smoke test of the planner hook: runs the own-sim `agent_server` on 'Lean H12 Magpie' strategy 25 with and without `H12_RESIDUAL_MLP` for a few seconds and reports the ON log line, finite states and the rms divergence of the two trajectories. Usage `hook_smoke.py <proto dir> <residual.bin> <seconds>`.
- `feature_parity.py` (30 lines): reads the `H12_RESIDUAL_DEBUG` rows (time, x[168], y[27], qpos[34], qvel[33], ctrl[27]) dumped by the C++ hook, rebuilds the feature vector in Python with the model's actuator gains and compares block by block, plus the MLP output vs the numpy forward.
- `imagination_error.py` (61 lines): planner imagination error in closed loop: for each iCEM dump (`dump/tick_*.bin`, nominal trajectory of the chosen plan) compares the predicted joint positions over 100 / 250 / 500 ms with the twin recording from the same instant; displacement RMSE per group, split by phase of t0 (stand < 45 s, 45-65 s, > 65 s). Usage `imagination_error.py <run_dir> ...`; writes `out/imagination_error.json`.
- `twin_ab.sh` (29 lines): interleaved honest-twin A/B (residual OFF vs ON): `twin_ab.sh [N=5] [DUR=150]` with `STRAT` (25), `TGT` (B3), `RES_W` (residual bin), `RES_SCALE` (0,1,1), `RES_ENV` (arbitrary env for the treated arm, e.g. `H12_ROLLOUT_SUBSTEPS=2`), `ARM_SUFFIX`, `DUMP=1` + `DUMP_EVERY` (iCEM nominal-trajectory dumps for the imagination-error metric); each run under `systemd-run --scope -p MemoryMax=16G`, `SCENE_VAR=_real_ns`, calls `scratch_bench/grid37_twin_noassist.sh`; appends a summary row (residual line, plan/s, gate tail, twin end) to `logs/grid37_twin/d37/AB_residual_<stamp>.md`.
- `twin_damp_ab.sh` (24 lines): interleaved A/B of the parametric correction: patches `<joint damping=".." armature=".."/>` in the build-tree include `h1_2_modified_magpie.xml` (baseline 10 / 0.1 vs `DAMP` (30) / `ARMAT` (0.1)), restores it on exit; `STRAT` 37, `TGT` B3, `DUR` 180; summary `AB_damp<DAMP>_<stamp>.md`.
- `bc_common.py` (29 lines): stage-3 observation builder (q, dq, g_imu, gyro, phase one-hot 4, kf one-hot 9, t_phase/30, plus previous cmd - q when `MBRL_BC_PREV=1`) and action (cmd - q or cmd - QREF 27, tau_ff 27).
- `bc_dataset.py` (48 lines): builds `out_bc/bc_data.npz` from twin bench run directories that kept the 100 Hz recorder npz; phase and keyframe labels parsed from `node.log` (`[BIO t=.. phase=..]`, `[STRAT n:.. | kf k/9]`), node time = recorder time minus t(first kp > 1), samples after (fall - 2 s) dropped. Usage `bc_dataset.py 'glob1' 'glob2' ...`.
- `bc_fit.py` (41 lines): behaviour cloning on the GPU: `MBRL_BC_H` (256), `MBRL_BC_NL` (3), `MBRL_BC_EPOCHS` (40), `MBRL_BC_TAG`, `MBRL_BC_PHASES` (e.g. "0" = stand only), `MBRL_BC_WTAU` (0.1 weight on tau_ff), `MBRL_BC_NOISE`, `MBRL_BC_WD`, `MBRL_BC_MASK`; saves `out_bc/bc_<tag>.npz` numpy weights + norms.
- `bc_policy.py` (12 lines): numpy forward pass of a `bc_<tag>.npz` policy.
- `bc_eval.py` (62 lines): closed-loop evaluation on the twin plant (scene `*_37_real_ns_grid`, dt 5 ms, PD per step with the node's per-joint gains, 100 Hz policy); modes `replay` (open-loop recorded commands), `hold` (PD hold), `policy`; env `MBRL_EVAL_T` (30 s), `MBRL_EVAL_PUSH` ("t,fx,fy,dur;..."), `MBRL_EVAL_SCHED` ("t:phase:kf;..."), `MBRL_EVAL_INIT` (run dir, default `grid37_twin/d37/s37_B3_noassist_base_r4`), `MBRL_EVAL_T0` (40.0 s).
- `dagger_stand.py` (202 lines): DAgger / teacher-driven distillation of the iCEM stand: teacher = own-sim `agent_server` (slot 37, node gains injected via `Init(model)`, task reset per episode, K planner iterations per label), student env = python twin plant (dt 5 ms, PD per step, node gains, tau_ff = 0.85 qfrc_bias); episodes from recorded standing states with random torso pushes (`DAG_PUSH_LO` 20 to `DAG_PUSH_HI` 70 N), action = beta teacher + (1 - beta) student, labels (obs, teacher cmd - q), refit, evaluate. Env: `DAG_ITERS` 6, `DAG_EPIS` 12, `DAG_T` 8 s, `DAG_K` 2, `DAG_HZ` 50, `DAG_PORT` 20292, `DAG_SMOKE`, `DAG_TAG`, `DAG_BETA_DECAY`, `DAG_CHECK` (label-check mode), `DAG_DART` (DART noise, deg), `DAG_NO_DEMOS`, `DAG_NO_TABLE` (move the table 100 m in plant and planner; robot-table contact = failure), `DAG_RESET_PLAN` (Markov teacher: plan reset before every label), `DAG_SEED`. Writes `out_bc/dagger_<tag>_data.npz`, `out_bc/bc_<tag><it>.npz`, logs `out_bc/dagger_<tag>.log`, `out_bc/agent_server_<tag>.log`.

C++ side (kept in the tree): `mjpc/tasks/humanoid_bench/lean/residual_mlp.{h,cc}` (binary loader, SiLU forward, per-thread scratch, feature construction from q, dq, actuator force, ctrl - q, gains, gravity in the IMU frame, pelvis rate), `lean::ModifyRolloutState` (pre-step correction inside planner rollouts only; env `H12_RESIDUAL_MLP=<bin>`, `H12_RESIDUAL_SCALE=legs,waist,arms` default `0,1,1`, `H12_RESIDUAL_DEBUG=<file>` feature dump, ensemble gate added 2026-10-05 "not yet built"), `H12_ROLLOUT_DUMP=<dir>` / `H12_ROLLOUT_DUMP_EVERY=n` (iCEM nominal-trajectory dumps), `H12_ROLLOUT_SUBSTEPS=n` in `mjpc/trajectory.cc` (each planner step as n substeps of timestep/n via a shallow model copy). Absent env = byte-identical planner behaviour.

---

## 10. Open questions and what was not resolved

1. The hands-off (upright-trained, `MBRL_MAX_TILT=12`) real-data residual `out/residual_uprightpre10s.bin` was never tested on the robot. Every closed-loop test ran on the twin or own-sim, where a real-robot residual is bias by construction. The staged protocol proposed (arms only first, scale 0.25 -> 0.5 -> 1, fixed-window A/B on strat 25) remains untried and user-gated.
2. The 3.9 mm median / 19 mm p99 left-right sole height mismatch on the real robot is not a hip or knee zero error (stage K offsets < 0.4 deg once the feet are flattened). Sole geometry or link lengths remain the candidates; it has to be identified, not patched.
3. The right ankle pitch zero error (+3.8 deg median, day-to-day +4 to +8 deg in the earlier `ankle_offset_sweep`) is a hardware fact that the plant could carry as a per-session calibration; not done.
4. The planner's gripper / wrist collision shapes self-collide at recorded arm poses. They were disabled for the study; planning needs them back, re-fitted from the real hand geometry. Not done.
5. Why a residual that lowers open-loop error (real and twin, 20-45 % at 250 ms) makes a sampling planner plan worse is explained structurally (errors selected for in the planner's own sampled futures, free velocity kicks exploitable by the optimiser) but not resolved. Untested fixes: train the correction with the planner in the loop (score candidates by closed-loop imagination error, not open-loop fit); inject as a generalized force through the inertia matrix and contact solver instead of a velocity kick; evaluate the cost on the uncorrected step; build the uncertainty gate into the C++ hook (python only as of `aa54d7c`).
6. The "x3 damping" identified from real data is an artefact of the planner's single 10 ms step, but a finer planner step (`H12_ROLLOUT_SUBSTEPS=2`) breaks the strategy's cost tuning (locked knees, squat). The planner and its cost weights are a matched pair; re-tuning the costs for a finer step was not attempted.
7. The twin's bench variance (per-round baselines 1/3/1 rung 9 and 1/1/0 pipelines) means five-pair A/B counts cannot detect anything short of a large effect; the imagination-error metric is more sensitive but was only applied to 2-4 runs per arm.
8. Heat: motor torque constant drifts with winding temperature (logged since 09-03); per-run residual bias vs temperature was never checked.
9. Stage 3: the sampling planner cannot be imitated tick-by-tick (BC, DAgger, DART, label smoothing, Markov teacher all fail; best free-standing student 0.21 survival at 30 s). The proposed route, RL in the twin with the planner's own costs as reward, warm-started from the student, with the Stage-0 sim-gap facts as domain-randomisation ranges, was not started. The lean ladder was never distilled (needs the node's gate state machine as part of the teacher).
10. The teacher reference numbers of report 30.4 (K=2, 1.00 / 0.55 / 0.55) are presumed table-assisted and were not re-measured with contact-as-failure except for the 6/6 free-standing check of 30.6.
11. Seed-to-seed variance of the student fit (0.03 to 0.28 closed-loop selection score on identical data) dominates every stage-3 number; a value-weighted or RL fine-tuning stage was proposed, not built.
12. The 2026-10-11 commit `37645a5` (estimator node v3/v4 and Lean_H12_Magpie tweaks) snapshots the tree as left; its content is outside the scope of this archive.

End of archive.
