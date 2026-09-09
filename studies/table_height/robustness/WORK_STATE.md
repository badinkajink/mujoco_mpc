# Active work state (2026-09-09)

User requested real, methodical one-alteration-at-a-time robustness work with
minimal engineering. Complete the experiment, adaptive improvement if warranted,
fresh confirmation, and report. Do not stop at a plan or partial counts.
No subagents. No robot/DDS/push. Original checkout untouched except appended notes.

## Locations and running processes

Workdir `/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908`.
Branch `codex/table-height-robustness-20260909`, initial study commit bd682089,
parent 1eab9af6. Existing build uses this worktree path, so do NOT relocate it.
Main randomized batch is running in exec session 50331:
`python3 studies/table_height/robustness/run.py studies/table_height/robustness/ablation.json`
stdout redirected to robustness/ablation.log. Check processes/log before starting
anything else. The runner TEMPORARILY modifies the isolated strategy JSON;
never commit/reset/edit that file while the batch runs. Restores original in
finally. It uses six threads serially, CPUQuota700%, MemoryMax11G.
Native evaluation is run automatically after each episode. Check status.py.

## Frozen plan

33 ablation episodes, seeds10/11/12 randomized within each seed block:
reference (sync1, pose1, depth.4, dwell5) at .785/.985/1.085;
no_pose (pose0) at .785/1.085; legacy_model(sync0) at1.085;
far_target(depth.55) allthree; short_hold(dwell2) low/high.
All75s, home, .003 initial perturbation, spp3=167Hz. Sampling RNG unseeded.
Existing prior 0–2 seed data NEVER pooled into this experiment.
Read protocol.md for endpoints and selection rules. After ablation, at most
TWO evidence-driven existing scalar candidates, predeclare before running.
Then freeze lowest-engineering surviving recipe and run fresh confirmation
seeds100–102 at .785,.885,.985,1.035,1.085 (15 episodes). Do NOT tune on these.
No assertion of high population reliability from 3/3; Wilson95% intervals.

## Important audit discoveries

Bench native MuJoCo revision088079ef labels itself3.2.3 but has ABI DIFFERENT
from pip3.2.3 (76vs78 ints,378vs382 pointers). Native MJB cannot be loaded by
pip3.2.3. New replay.cc uses EXACT bench library/header. evaluate.py only
aggregates native CSV, no Python MuJoCo physics. engine.json records hashes.
Passive bench additions state_out (pre-step qpos,qvel,ctrl,warmstart) and
model_out (plant/planner snapshots) change no control logic. Native cache
agreement so far submicrometre FK and <0.001N p95 brace-force differences.
Python package remains visualization ONLY: FK parity checked at175 samples,
maxjawdifference5e-9m. Geometry renders correct table face from logged qpos.

Primary:2 observed continuous sec phase2,error<=30mm,upward left-arm normal
support>=20N,trunk&other tableloads<10N,pelvis>.8; whole runpelvis>=.65/no fall.
Secondary70mm. Full success additionally ends upright<=15deg,unloaded<10N
for2sec. Ladder completion is separate. A2sec phase gate can yield1.98sec
between passing50Hz samples; report one-sample sensitivity separately, never
change frozen flags. report_generator labels terminal contact vs true stall.
recovery scorer uses longest elapsed duration, fixed beforefirst result.

## Early observations, NOT conclusions

First low reference seed10: strict3.9s and full recovery69.85s, no jointlimit
violations, successful intervalmax jointvelocity.326rad/s, worst-actuator
force fractionmedian.912/p95.986. Footmovement44mmduringhold,103mmwholecycle.
Low short_hold seed10:strict1.1s,70mm1.98s,ladder&uprightrecovery completed.
High no_pose seed10:strict4.32s,laddercompleted,upright3–4deg,but leftarm
briefly16Nfinal -> strict unloaded recovery fails (NOT falling/inability tostand).
Low no_pose seed10:75s stable but stuckrung2,error~336mm,brace~103N.
More runs must land before selecting changes; fixed order MUST finish.

## Candidate ideas (not yet selected/tested)

Existing pose_track=2 includes reaching arm in DLS, unlike1. This is ONE
selector value/no new algorithm, potentially useful if low reach stalls repeat.
Old unsynchronized mode2 failures are not decisive. See lean.cc4168ff.
Old low recovery stalls also waited on standback_stall_full_sec40; testing20
would alter timing without increasing pitch limits. However newseed10 recovers,
so choose only if repeated failures justify it. At most2 scalars per protocol.
Do not reflexively relax pitch bounds or success tolerance.

## Artifact/scripts

New page docs/lean/20260909-table_height_robustness.html (currently partial),
regenerated with `python3 studies/table_height/robustness/summarize.py --plots`.
Counts/ledger from evaluation.json files; full-state/model artifacts are in
studies/table_height/runs/robustness_20260909/ (gitignored large files).
All new scripts in robustness/:run.py,evaluate.py,replay.cc,test_protocol.py,
status.py,summarize.py,plot_traces.py,README.md,decisions.md.
Do not overwrite yesterday's frozen report/data.
Completed low_reference_seed10.mp4; low_no_pose_seed10 render was started in
session78741, check render_failure.log. render.py is old skunkworks renderer,
using absolute assembled XML, new state.csv, new metrics.csv, actualface/target.
Need final representative success+failure videos (notall), final findings.html,
final recipe manifest, raw hashes, README and originalCLAUDE_wxie append,
INDEX entry, commit only owned files once batch restores strategy, local-only.

Ongoing scripts/report edits after bd682089 are uncommitted. run.py resume
logic was strengthened to read frozen baseline_strategy.json and refuse external
or interrupted strategy edits, without affecting current in-memory runner.
Source/controller binary MUST remain unchanged during ablation. Do not rebuild.
