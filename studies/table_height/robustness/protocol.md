# Table-height robustness protocol — 2026-09-09

Frozen before the first new dynamic trial. Parent 1eab9af6; isolated branch
codex/table-height-robustness-20260909, same physical model and strategy 25.
No Crocoddyl reruns. Simulation only; no claim of hardware feasibility.

## Question and minimality
Find the smallest change set that supports precise, loaded lean+reach across
0.785–1.085 m, including the nominal 0.985 m control. Recovery is evaluated
separately and cannot be replaced by an easier phase gate. Count controller
scalars changed, strategy fields changed, existing retargeter activation, and
new controller code separately from benchmark correctness/measurement code.
Target relaxation is a task change, not a controller improvement.

## Common conditions
Six sampling threads, CPUQuota=700%, MemoryMax=11G, serial episodes; home reset,
0.003 initial perturbation, 2 ms physics, spp=3 (166.7 simulated Hz), 1 s
prediction horizon. 75 s episode or existing fall/final-phase stop. Target jaw
uses full local offset (0.2254,-0.0118,-0.1062); target is near edge + depth,
table center -0.04, face +0.15. Model snapshots saved AFTER height/retarget.
No changing binary or baseline strategy during a block. Exact command,
strategy, build/source identities, model snapshots and all outcomes retained.

## Frozen outcomes
Primary: at least 2 continuous observed seconds in reach rung 2, jaw error
<=30 mm, summed upward normal support from LEFT arm/table contacts >=20 N,
torso/pelvis table normal load <10 N, other robot/table normal load <10 N,
pelvis >0.8 m. No episode fall and whole-episode pelvis >=0.65 m. Secondary:
the same criterion at 70 mm (prior study tolerance). Dwell is last minus first
passing timestamp, never sample-count times dt. Missing samples break dwell.
A 50 Hz trace establishes sampled behavior, not unsampled-time guarantees.
Full-task success additionally requires ladder completion and final 2 seconds
pelvis >0.8 m, torso tilt <=15 degrees, all left-arm/table normal load <10 N.
Record ladder completion separately so this stricter definition stays visible.
Record falls, approach failures, reach misses, release stalls, maximum joint
velocity/limit violation, actuator-force utilization and unintended contacts.
MuJoCo force clamps are model limits, not independently verified hardware limits.

New --state_out is passive: pre-integration time, qpos, qvel, ctrl and solver
warm-start. Independent MuJoCo 3.2.3 forward replay produces synchronized FK
and contact forces without touching control. Legacy live logger timestamps
mix post-integration qpos and pre-integration derived quantities, about 2 ms;
new primary counts use the synchronized independent audit, never interpolation
of decimated qpos. Save signed/normal load and force reconstruction comparison.

## One-factor ablation, exploratory
Reference is yesterday's candidate: sync=1, pose_track=1, depth=.40, dwell=5.
Each alternative removes exactly ONE factor:
- no_pose: pose_track=0, low/high. Nominal retarget is a structural no-op,
  verified from model snapshots rather than spending redundant runs.
- legacy_model: sync=0, high only. A deliberately invalid-model diagnostic,
  not a deployable controller candidate; geometry/keyframes audited directly.
- far_target: depth=.55, all three anchor heights; no change to hold time.
- short_hold: dwell=2, low/high; no change to target/tolerance.
Reference runs all three anchor heights. Reset seeds 10–12; randomize order
within each seed block using Random(20260909+seed). Sampler randomness is NOT
seeded, so this is a blocked, randomized condition comparison, not a paired
random-number experiment. Report counts/differences per height; retain failures.
Prior 0–2 seed data are hypothesis-generating only and will not be pooled.

## Adaptive optimization and confirmation
Inspect failure traces after the ablation. At most two additional existing
numeric/strategy-scalar hypotheses, selected by recorded failure evidence,
with exact alteration and stop rule written BEFORE their runs. Compare each
to its immediate parent at fixed target, hold, rate and endpoints, at >=3
trials per changed-height condition and nominal controls. Never loosen success
criteria or fall guards to claim a gain. Reject a screen with no gain in the
limiting-height success count or worse nominal count; small samples cannot
prove absence of regression. A promising change is provisional, not confirmed.

Freeze the smallest surviving recipe before fresh confirmation. Test seeds
100–102 at .785,.885,.985,1.035,1.085 (15 trials); .885/1.035 are withheld from
this session's tuning. Report all counts and Wilson 95% intervals, including
worst-height performance. Three successes cannot certify high reliability.
A robustness claim covers only tested perturbations/heights/target and rate;
no interpolated continuous height window, arbitrary target hemisphere, repeated
cycles, disturbance rejection or realtime deployment claim. If anchor behavior
remains poor, finish the same confirmation of the best available recipe and
report its limit rather than tuning on confirmation seeds.

## Reporting
Local HTML with one-factor table, fresh validation separate from exploration,
run ledger, exact configuration diffs, failure classification, representative
success AND failure videos, raw-data links, source/model hashes, recipe and
handoff. Frozen endpoint definitions cannot be rewritten to favor results.

## Preflight correction: matching a version string is insufficient
The C++ library uses MuJoCo git revision 088079ef, labelled 3.2.3, while the
PyPI 3.2.3 package uses an incompatible model layout (76 vs 78 mjModel integer
fields; 378 vs 382 pointers). The first 0.08 s nonexperimental smoke test
exposed this when loading the actual C++ binary model. Therefore evaluation
uses a small standalone C++ replay linked to the EXACT SAME libmujoco and
headers as the bench, not the Python package. Python only aggregates its CSV.
The previous report's Python-version matching did not establish engine identity.
All old runs remain separate and unmodified. No ablation began before this fix.
The earlier nominal hash comparison was of Python-compiled models; it establishes
that XML input is unchanged but cannot certify C++/Python engine equivalence.
