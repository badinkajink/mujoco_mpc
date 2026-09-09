# Table-height simulation handoff — 2026-09-08

Status: completed local investigation. All simulation processes have ended.

Report: [Table-height generalization with synchronized planning models](20260908-table_height_skunkworks.html).

Worktrees:

- MJPC: `/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908`
- Crocoddyl: `/home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908`

Both use branch `codex/table-height-skunkworks-20260908`. Parents are MJPC
`355a57ea` (newer than baseline report `299f54b7`) and CMPC `aba958e`.
MJPC implementation is committed at `4398d8a8`; CMPC hooks at `51d604e`
and `b1040fd`. The report and study scripts are committed separately.
Nothing was pushed or uploaded. Original checkouts were never reset or switched.
Original `video.mp4` and modified `studies/bvs_plots.py` remain untouched.
Only append-only coordination notes were written in the original MJPC
`CLAUDE_wxie.md`.

## Measured result

Fixed actual-jaw target: `(0.85, -0.04, face + 0.15)` m. Six sampling threads,
167 Hz simulated planning, home reset, 0.003 xorshift qpos/qvel perturbation.
DLS brace keys at low/high; nominal is the identity case and uses authored keys.
A loaded reach requires at least two continuous seconds with left-arm normal
load >=20 N, pelvis >0.8 m, torso/pelvis table load <10 N, and whole-run pelvis
>=0.65 m without a bench fall. Recorded reaching-arm/unattributed table load is
zero during scored successful intervals.

| Face m | Reach <=70 mm | Reach <=30 mm | Full recovery |
|---|---|---|---|
| 0.785 | 2/3 | 2/3 | 0/3 |
| 0.985 | 3/3 | 2/3 | 3/3 |
| 1.085 | 2/3 | 2/3 | 2/3 |

Low successful loaded dwells: 2.82 and 5.18 s; strict dwells 2.10 and 4.28 s.
Its failed reach stayed upright but missed by >=346 mm. Successful low runs
stall in release. The high failure falls during approach. Nominal's third run
completes but has only 0.94 s strict dwell. Do not call this uniform reliability.
The internal five-second hand gate is not five seconds of simultaneous load.

Single-trial intermediate screens at .885 (authored), .935 and 1.035 (DLS)
complete. Loaded dwells at .935/1.035: 2.46/4.98 s; strict: 1.86/3.68 s.
These are samples of a vertical/depth section, not a certified envelope or
hemisphere. No lateral sweep, disturbance campaign or hardware test was done.

## Findings and code

- **Old height bench planned against the nominal model.** Agent::Initialize
  copies mjModel before the first Table H transition. At requested 1.085,
  planner face remained .985 and keyframe edits missed planning. The bench now
  synchronizes after its first transition. `--sync_planning_model 0` reproduces
  the old behavior. The correction is scoped to this fixed-height bench.
- Nominal legacy baseline reproduced, and synchronized original far target at
  1.085 completed (one trial). Prior off-height failures cannot establish an
  optimizer-only height limit. Historical high shipped seeds had four zero
  peak forearm loads and two brief 20.9/31.8 N peaks.
- `brace_base_z_gain` defaults to zero. Gain .5 + DLS delivered brace base
  z=1.0142 in both models and completed one high far-target trial. No separate
  advantage is established. The initial gain=1 failure used the historical
  planner copy and is not a valid rejection of a delivered full-gain command.
- Synchronized +60 mm low stance: upright 75 s, no reach (>=341 mm miss).
  Initial +120 mm trials used the historical model. No offset grid followed.
- `spp=3` means 166.7 Hz because plant dt=.002; the old 33 Hz statement used
  the prediction timestep. At 50 Hz, nominal/high short-target screens fell;
  low authored keys passed one screen. Do not pool planning rates.
- MJPC seed controls reset state only. Abseil sampling noise remains random.
  Reproduction commands recreate conditions, not bit-identical paths.

## Crocoddyl comparison

Existing original-model trajectories were audited from qpos+qvel+ctrl using
MuJoCo 3.10: 3/3 loaded holds at each sampled height .885–1.085, 1/3 at .785.
They use a different collision/joint-limit model, wrist point and target,
stand reset, and Gaussian perturbations. Keep this valid evidence separate.

Five verified-jaw current-model probes all fell: home/x=.85 at .785/.985/1.085
(low uses forearm-only), nominal home/x=1.15, and nominal stand_up/x=1.15.
The paired 50 Hz MJPC standing-start trial also fell. This bounded transfer
branch was stopped. Costs, approach timing and control delivery differ, so
these results do not rank optimizers.

Four earlier `common_h*_x1000_*` CMPC trials used a wrist-frame point. They are
calibration failures, not matched jaw tests. The current gripper is rotated
90 degrees about x. Opt-in `REACH_BODY` / `RECONCILE_MJ_FRAMES` compose its
welded transform. Reference-pose parity: jaw ~1 micrometre, gravity ~1.5e-5 Nm,
actuated mass matrix ~8.3e-7 kg m². All five mapped recorded initial states
match the intended benchmark qpos/qvel exactly. The current physical model
remained byte-identical after rebuilding.

## Reproduction and data

Read `studies/table_height/skunkworks/README.md` and `protocol.md`.
`final_recipe.json` contains the nine endpoint repeat conditions, including
failures. `study_manifest.json` freezes the 25 MJPC / 9 CMPC episode selection
and hashes 442 raw files. Raw output is in
`studies/table_height/runs/codex_skunkworks/`. Commands, strategy snapshots,
model provenance, failed attempts, plots and replay media are linked from the
report. The raw data is gitignored; preserve it locally.

```sh
cd /home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908
# Inspect processes before starting any new simulations.
ps -eo pid,etime,args | rg 'lean_bench|run_mjpc.py|croco_batch.py|run_croco.py'
# Existing IDs are cached. Copy the manifest and assign fresh tags for new trials.
python3 studies/table_height/skunkworks/run_mjpc.py studies/table_height/skunkworks/final_recipe.json
PYTHONPATH=$PWD/studies/table_height/skunkworks/vendor_mujoco323 \
 /home/humanoid/miniconda3/envs/croco/bin/python studies/table_height/skunkworks/analyze_mjpc.py
python3 studies/table_height/skunkworks/make_report.py
```

Build, Python syntax, reference/initial-state parity and local report links were
checked. No quota/auth/tooling blocker occurred. Scheduler logs and final-message
output live in `/home/humanoid/.local/state/codex-table-height-20260908/`.
The remaining practical work is robustness and low-table recovery, including
whether phase advance should require actual sustained brace load; none of
those prospective changes is claimed tested here.
