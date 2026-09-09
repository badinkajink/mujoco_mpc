# Table-height simulation investigation

All scripts are local simulation/replay tools. No DDS endpoints are created.
Read `protocol.md` and the report before interpreting success counts.

Paths used in this run:

- MJPC worktree: `/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908`
- Crocoddyl worktree: `/home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908`
- Crocoddyl interpreter: `/home/humanoid/miniconda3/envs/croco/bin/python`
- Original source/data: `/home/humanoid/Programs/Humanoid_Simulation/{mujoco_mpc,crocoddyl_mpc}`

Build the bench with `codex_build_command.json` in the parent directory, also
passing `-DCMAKE_CXX_FLAGS=-Wno-error=unused-result` (the existing checkout's
setting for an unrelated `avoid.cc` warning). Then:

```sh
ninja -C build_cmake -j4 lean_bench
python3 studies/table_height/skunkworks/run_mjpc.py studies/table_height/skunkworks/sync_pilot.json
```

The runner saves strategy and command JSON for every run, executes serially
under CPUQuota=700%, MemoryMax=11G, and restores the worktree strategy in
`finally`. It refuses a busy/low-memory machine. Creating `skunkworks/STOP`
stops either batch runner at its next run boundary; remove it to resume.
Completed run IDs are cached; use a new ID to change a condition.

Historical first four pilot runs used the old unsynchronized planner copy.
To reproduce those with the fixed binary, add `--sync_planning_model 0` to
the saved command. All runs beginning with `short_h1085_s0` synchronize the
planning model, and log both faces before/after. `pilot.json` documents the
chronological exploratory batch and must NOT be rerun as one uniform arm.

The Python environment is isolated without changing any installed packages:

```sh
/home/humanoid/miniconda3/envs/croco/bin/python -m pip install --no-deps \
  --target studies/table_height/skunkworks/vendor_mujoco323 mujoco==3.2.3
python3 studies/table_height/skunkworks/croco_batch.py studies/table_height/skunkworks/croco_jaw.json
PYTHONPATH=$PWD/studies/table_height/skunkworks/vendor_mujoco323 \
  /home/humanoid/miniconda3/envs/croco/bin/python studies/table_height/skunkworks/analyze_mjpc.py
```

Crocoddyl's opt-in comparison hooks live in its isolated worktree:
`REACH_BODY` and `REACH_OFFSET` choose the exact reaching point;
`RECONCILE_MJ_FRAMES=1` composes the welded gripper transform; `MATCH_MJPC_TABLE=1` mirrors
MJPC's leg geometry after table-height changes. The episode accepts explicit
initial qpos/qvel to match the bench's xorshift perturbation. Default existing
CMPC behavior is retained when these are unused. The driver saves solver logs,
plan convergence status, full state/control trajectories, live contact loads,
per-seed JSON and exact commands. It does not hide failed certification or
failed dynamic episodes.

`audit_croco.py` audits prior Claude episodes with their original MuJoCo 3.10
package, endpoint and staged model. It reconstructs contacts from qpos, qvel
AND control, and excludes table-floor contacts. Do not put the MuJoCo 3.2.3
vendor directory on PYTHONPATH for that historical audit.

`render.py` renders logged states at the recorded table height and overlays
measured errors/loads. Orange marks the commanded endpoint. It uses GLFW on
DISPLAY=:1 and never opens a controller or middleware connection.

The four historical `common_h*_x1000_*` CMPC experiments are calibration
trials with a wrist-frame point. Do not rerun `croco_pilot.json` with the
corrected driver under the same IDs, or interpret those trials as jaw tests.
Use `croco_jaw.json` for the verified mapping and inspect per-run parity.json.

MJPC adaptive repeats: `short_repeats.json` (167 Hz low/high DLS, nominal
identity-at-nominal references, then .935/1.035 intermediate screens).
`low_repeats.json` is intentionally empty; the initial proposal is preserved
in `low_repeats_initial_pending.json`. Scripts skip completed IDs. The earlier
`hz50_pilot.json` remainder was deliberately stopped after nominal/high falls;
its unexecuted parameter combinations are not results.

Corrected static contact-mode screen (candidate generation only):

```sh
cd /home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908
systemd-run --user --scope --quiet -p CPUQuota=600% -p MemoryMax=8G \
 env PYTHONPATH=/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908/studies/table_height/skunkworks/vendor_mujoco323:$PWD \
 CL_ASSETS_DIR=/home/humanoid/Programs/Humanoid_Simulation/CL_Assets \
 LEAN_TASK_DIR=/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908/build_cmake/mjpc/tasks/humanoid_bench/lean \
 REACH_BODY=right_magpie_gripper REACH_OFFSET=0.2254,-0.0118,-0.1062 \
 RECONCILE_MJ_FRAMES=1 MATCH_MJPC_TABLE=1 OPENBLAS_NUM_THREADS=1 \
 /home/humanoid/miniconda3/envs/croco/bin/python studies/height_reach.py \
 --faces=.785,.985,1.085 --tx=.85,1.0 --ty=-.04 --dz=.15 \
 --out=/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908/studies/table_height/runs/codex_skunkworks/common_static_jaw_repeat
```

Use `--tx=1.15` and a distinct output for the farther-point screen. The original
static `cells.json` commit field recorded the MJPC working-directory HEAD,
355a57ea, because the upstream script queries cwd. It is not the Crocoddyl
source revision; the corrected jaw hooks correspond to b1040fd. Reproducing
from the Crocoddyl cwd above fixes that provenance ambiguity without rewriting
the original raw files.

`verify_initial_states.py`, with the 3.2.3 vendor PYTHONPATH, compares each
recorded corrected-jaw CMPC initial qpos/qvel to the bench keyframe and xorshift
formula. Results are in `initial_state_parity.json`.

The bounded shared-start check uses `croco_stand_check.json` and
`mjpc_stand_check.json`: stand_up, x=1.15, 50 Hz, same current model and
xorshift perturbation. Both failed; no standing-start height sweep follows.
`--start_key` defaults to home; the C++ bench prints and validates it.

To reconstruct the four wrist-point calibration conditions with the final
driver, copy `croco_pilot.json`, give each job a fresh tag, and add
`"extra": ["--reach_body", "right_wrist_yaw_link"]`. This preserves the
original mistaken wrist-local endpoint while retaining the final bridge's
parity checks. Do not overwrite or pool the original calibration outputs.

`final_recipe.json` contains the nine low/nominal/high repeat conditions,
including failures. It is a reproduction manifest; the measured counts in
the report describe its limits. To run fresh trials, copy it and assign
new unique tags. Sampling noise is not controlled by the reset seed.

The final dated report is frozen by `study_manifest.json`, including raw-data
SHA-256 checksums. Its selected run IDs remain fixed when new raw folders are
added. `report_findings.html` is the final narrative for that frozen dataset.
The dynamic criterion, strict count and recovery count must remain separate.

Prior-study audit reproduction uses the original environment (MuJoCo 3.10),
without the vendor path:

```sh
env PYTHONPATH=/home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908 \
 LEAN_TASK_DIR=/home/humanoid/Programs/Humanoid_Simulation/crocoddyl_mpc/studies/runs/_stage/mjpc/tasks/humanoid_bench/lean \
 CL_ASSETS_DIR=/home/humanoid/Programs/Humanoid_Simulation/CL_Assets \
 /home/humanoid/miniconda3/envs/croco/bin/python studies/table_height/skunkworks/audit_croco.py \
 --croco /home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908 \
 --runs /home/humanoid/Programs/Humanoid_Simulation/crocoddyl_mpc/studies/runs/2026-09-08_height_dyn \
 --out studies/table_height/runs/codex_skunkworks/prior_croco_audit_repeat
```
