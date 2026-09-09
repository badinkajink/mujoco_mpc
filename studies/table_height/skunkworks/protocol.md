# Table-height simulation protocol, 2026-09-08

Worktrees: `/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908`
and `/home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908`.
Main parents: MJPC 355a57ea, Crocoddyl aba958e. Original checkouts preserved;
only an append-only coordination note is added to the original CLAUDE_wxie.md.

## Dynamic endpoints fixed before new Crocoddyl trials

MJPC: 75 s maximum, six threads, spp=3, same 0.003 xorshift reset
perturbation as shipped bench. First nominal baseline passed 43.504 s ladder
and 2.04 s joint reach/load dwell. New target arms use five-second rung sustain.

Primary lean+reach: at least 2 continuous s with Euclidean endpoint error
<=70 mm, left-arm table normal force >=20 N, pelvis >0.8 m and torso/pelvis
table normal force <10 N. Whole-run pelvis must stay >=0.65 m and no bench fall.
Strict reach threshold: 30 mm, reported separately. MJPC successful transient
reach may then release; Crocoddyl holds for 25 s and must satisfy the joint
criterion on >=95% of samples in the final 5 s, including 2 continuous s.
Completion of MJPC's return ladder is a separate endpoint. These protocols
have different durations and release requirements; do not equate their counts.

Forces: new MJPC and CMPC traces record actual simulated contact forces,
exclude world/table/object, sum all LEFT arm bodies (including wrist housings),
and record torso separately. CMPC recorder samples before mj_step, so cached
force/kinematics can lag its qpos timestamp by one 2 ms physics step. Prior
CMPC trajectories reconstruct forces from qpos+qvel+ctrl with MuJoCo 3.10,
not from qpos-only snapshots. Prior endpoint summaries are not used as dwell.

Matched comparison: new CMPC uses the current MJPC assembled model, MuJoCo
3.2.3, same jaw point (0.2254,-0.0118,-0.1062) in the right gripper
frame (including its welded transform into the wrist frame), same fixed table-frame target, home reset and xorshift perturbation.
Sampling vs BoxFDDP still differ in costs, internal models, contact schedule,
planning horizon/rate, and PD+feedforward delivery. This cannot isolate the
optimizer as the sole causal variable.

## Adaptive decisions

- Base-height gain 1 at 1.085 m: a direct test of the user's still-open
  command hypothesis, despite latest static notes predicting failure. Target
  is shipped base z + gain*(face-.985), optionally after DLS. This reference
  edit is not a certificate of a feasible whole-body pose. Stop the arm if
  it does not buy dynamic reach/contact. A half-gain trial can distinguish
  an excessively high command from a useful smaller one.
- Low stance: screen +120 mm at .785 and .885. If the drape/fall persists,
  do not run a full offset grid. Repeat only promising cases.
- Short target: x=.85 (400 mm from edge), y=-.04, z=face+.15, against
  the shipped x=1.0. Test nominal and high; repeat only stable candidates.
- Contact-mode screen: 3 heights x 2 targets x the existing curated 8 modes,
  common model and jaw endpoint. Use it to select a bounded dynamic mode
  alternative; static success is not a stable workspace label.
- Seeds: >=3 for a repeatability claim. One-seed screens are reported as
  anecdotes. Cached failures are retained. No blind factorial sweep.

Raw per-run commands, strategy snapshots, solver logs, initial conditions and
metrics remain local. No hardware, DDS processes, external messages or pushes.

## Runtime finding, 20:33

First corrected high run logs plant_face=1.0850 / planner_face=0.9850
before synchronization, and 1.0850 / 1.0850 after it. The first four
pilot jobs used the historical unsynchronized bench. `short_h1085_s0`
and subsequent jobs use the corrected default. Per-run stderr proves
the setting; later commands pass the flag explicitly. Prior height
sweeps cannot establish an optimizer-only height limit.

## Timing audit

The historical notes call spp=3 a 33 Hz replan rate. The bench loop actually
steps the PLANT at 0.002 s and plans every spp steps, hence spp=3 is
166.7 Hz in simulated time. Agent timestep=0.010 is the prediction integration
step, not the plant step. Crocoddyl replans at 50 Hz with a 35-node (0.7 s)
horizon; MJPC predicts 1.0 s. Additional spp=10 trials match the 50 Hz planning
period. They are a separate arm, never pooled with spp=3 seeds.

## Calibration correction and seed scope

The four `common_h*_x1000_*` CMPC runs used a wrist-frame endpoint and are
calibration failures, not matched jaw trials. `jaw_*` runs set REACH_BODY to
right_magpie_gripper and RECONCILE_MJ_FRAMES=1; the bridge composes the actual
90-degree welded transform. Runtime site/gravity/mass-matrix parity is saved
in each run. The corrected static screen is `common_static_jaw`, not
`common_static`.

MJPC `--seed` controls only reset-state perturbations. Sampling uses fresh
Abseil BitGen noise inside AddNoiseToPolicy, so commands reproduce conditions,
not bit-identical trajectories. Counts include controller sampling variability
as well as three different initial states. No confidence interval or population
success rate is inferred from three trials.

A +60 mm low stance and low pose-track trial followed the model-copy fix;
the +120 mm historical-model failures alone did not justify rejecting stance.
At 50 Hz nominal/high short-target screens fell, so the planned 50 Hz
base/stance/pose factorial was stopped. Low 50 Hz and nominal/high 167 Hz
successes are being repeated separately.

## Adaptive priority, 21:44

The synchronized low-table pose_track=1 screen reached and held the short
jaw target at 167 Hz, then entered release. Pending repeats were redirected
to this common-rate candidate: high pose_track=1 pilot, low pose_track=1
seeds 1/2, and nominal original-key seeds 1/2. The initially planned default
high and 50 Hz low repeats are preserved in *_initial_pending.json and are
not results. High retargeting gets repeats only if its pilot is promising.
At nominal height the table does not shift, so pose_track performs no edit.

Corrected CMPC screen also includes one nominal x=1.15 elbow–forearm
trial, because its static jaw screen predicts 38.4 N versus zero load at
the close nominal point. This tests whether a farther, load-bearing target
rescues the approach/hold. It is not matched to an MJPC x=.85 trial and
will not be pooled across targets. Stop this direction if it also falls.

## Matched standing-start check, 22:05

All three corrected close-jaw CMPC home-start screens fell. Its original
height study uses stand_up, whose bent knees put the feet about63 mm
closer than home without changing base x. One nominal common-model
stand_up/xorshift check is therefore added for each controller at50 Hz.
Bench --start_key defaults to home and validates the requested key.
This checks a concrete approach/contact-reference mismatch; it is not a
repeat of stance_shift_x as a high-table strategy. Keep these results
separate from home-start counts. Stop this direction if CMPC still falls.

The shared-start check uses the already-screened x=1.15 jaw target,
whose nominal elbow–forearm candidate carries38.4 N, rather than the
near-point zero-load candidate. Both controllers use x=1.15. The preceding
home/x=1.15 CMPC trial converged offline but fell dynamically; this makes
the next comparison a targeted start-state check. No stand/x=.85 run was
executed. Pending manifest names are generic; inspect their contents.
