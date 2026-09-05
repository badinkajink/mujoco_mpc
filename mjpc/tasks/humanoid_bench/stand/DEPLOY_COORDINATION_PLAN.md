# Deploy coordination plan — mujoco_mpc ↔ Humanoid_Simulation (2026-07-17)

Goal: run `Stand H12` (full, nu=27) and `Stand H12 Lower` (eq-lock, nu=12) through the
HAMS deploy pipeline against the MuJoCo twin, and fill the 4-way comparison table.
Decision (this session): **eq-lock for both MJPC-only and deploy** — one model, so the only
difference between MJPC-only and sim2sim is the pipeline, not the robot.

## How the two repos couple (analysis)

- `Humanoid_Simulation/mujoco_mpc` is a **git submodule** of the same repo
  (`badinkajink/mujoco_mpc`, branch `extended_hw`) as the working checkout
  `/home/humanoid/Programs/mujoco_mpc`. The submodule is pinned at `7bc6eb1`; my HEAD
  `8b22805` is a clean **14-commit fast-forward** ahead, plus the uncommitted Stand tasks.
- `docker/docker-compose.yml` service **`ros`** bind-mounts that submodule **rw** to
  `/home/code/mujoco_mpc`, and `container_cache/mjpc_build` → `/home/code/mujoco_mpc/build`.
- `docker/scripts/launch_ros.sh` auto-hydrates, on first launch: the MJPC build cache (from a
  baked seed, back-dating source mtimes so the rebuild is incremental) **and** `unitree_sdk2`
  into `/opt/unitree_install`. The container toolchain is **clang-13 + LTO + libc++**, which
  matches the SDK build — this is exactly the ABI mismatch that blocks a host deploy build, so
  it only works in-container.
- Deploy controller `mjpc_lowerbody_core` (fork's `mjpc/deploy/h12_lower_body_controller.cc`
  + `deploy_common.cc`) is built by `core_ws/src/h12_deploy_mjpc` via colcon; it links
  `libmjpc.a` from the warm build tree, so **the MJPC tasks are compiled in**. It's driven by
  `controller_launcher.py`, which maps ROS params `task` / `strategy` → the core's CLI flags.

**Net:** to deploy the new tasks, my changes must reach the submodule checkout, then MJPC and
the colcon package get rebuilt in-container. No docker-compose edit needed (the mount already
points at the submodule); the user's "point to the right mjpc location" is really "bump the
submodule to my commit."

## mjpc side — DONE this session (in /home/humanoid/Programs/mujoco_mpc, branch extended_hw)

- `Stand H12` (full, nu=27) and `Stand H12 Lower` (eq-lock, nu=12, neq=15, nq=34) registered,
  first two in the GUI dropdown. Real robot constants baked in (gains from
  `h12_ros2_controller/config/safety_full.yaml`, forcerange from `h12_safety_layer` estop
  ratios × URDF torque, upstream joint damping 0).
- `Stand H12 Lower` uses the **eq-locked** model (reuses `stabilize/_gen_stabilize_model.py`
  on `h1_2_stand.xml`) so the deploy legs-only node can inject the 15 measured arm angles.
- `stand.cc` Control residual is per-actuator (handles nu=12); byte-identical for full-body.
- Verified MJPC-only, 30 s, iLQG: full **0.0401**, lower **0.0424** (~1.2x faster wall-time),
  both stand the full 30 s.

**Remaining (needs your OK — outward/durable):**
1. `git add` the 5 changed + 1 new task files, commit on `extended_hw`.
2. `git push origin extended_hw`.

## Humanoid_Simulation side — for the new conversation

1. **Bump the submodule** to the pushed commit:
   `cd mujoco_mpc && git fetch origin && git checkout extended_hw && git pull` (submodule
   tracks `extended_hw`). Optionally `git add mujoco_mpc && git commit` in the superproject to
   pin the gitlink so the next image bake (`MJPC_REF`) captures it.
2. **Launch the ros container**: `docker/scripts/docker_run.sh` (or `docker compose --profile
   ros up`). First launch hydrates the build cache + unitree_sdk2 (seconds).
3. **Rebuild in-container** (order matters):
   - `docker exec -it hams_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh --install`
     (`--install` because task **XMLs/assets** changed, not just C++).
   - `colcon build --packages-select h12_deploy_mjpc` (relinks `mjpc_lowerbody_core` against
     the fresh `libmjpc`).
4. **Run the 4-way benchmark** (3 processes; export `ROS_DOMAIN_ID` in every shell):
   - twin: `h1_mujoco/h12_mujoco.py --handless` (or `h1_sim_bringup.launch.py`)
   - safety: `safety_layer_main.py --config default_safety_full.yaml`
   - controller — lower: `ros2 launch h12_deploy_mjpc ... task:="Stand H12 Lower"`;
     full: the full-body node (`h12_control_node --task "Stand H12"`) — see decision (1).
   - Log pelvis-z / torso-upright / survival + plan rate from `rt/lowstate`.

## Decision points / integration risks to resolve in the new conversation

1. **Full-body deploy path.** `h12_deploy_mjpc`'s ROS launcher wires only the **lower** node
   (`mjpc_lowerbody_core`). For full-body stand *through the pipeline*, either add a launcher
   for `h12_control_node` (nu=27) or run it natively over DDS (README_EMBED direct invocation).
   Simplest: run the full node natively for config (3), lower via ROS for config (4).
2. **Lower-node start pose.** `h12_lower_body_controller.cc` hardcodes `kLowerStartPose`
   (hip −0.15 / knee 0.35 / ankle −0.28, the stabilize stand). `Stand H12 Lower` home is
   hip −0.4 / knee 0.8 / ankle −0.4, so there's a handover jump. Fix: either set the task home
   to `kLowerStartPose`, or pass `align_pose` / `--align_start` so the node drives to the stand
   home before releasing MJPC.
3. **Strategy coupling.** The lower node's `--strategy` (transitions, align, bring-up) is
   written around the stabilize task. Confirm it drives a plain Stand task cleanly, or use
   `strategy` = a benign value. If it fights the stand, fall back to the validated
   `Stabilize H12 Magpie` for config (4) and note the residual difference.
4. **Arm KP parity (sim2real hazard).** `h12_control_node.cc` hardcodes arm KP 30/20/15, but
   `safety_full.yaml` says 240/200/150 — an 8x gap. Reconcile before trusting full-body arm
   behavior (the planner model now uses the real 240 via `_gen_stand_model.py`; the node's
   commanded KP must match, or watch commanded-vs-actual).
5. **Twin vs real offsets.** Twin: `gravity_ff 0`, `imu_pitch_offset_deg 0`. Real: 0.85 / 1.6.

## 4-way results table (fill (3) and (4))

| # | config              | model         | path      | avg cost | survives | plan Hz |
|---|---------------------|---------------|-----------|----------|----------|---------|
| 1 | full  MJPC-only     | Stand H12     | testspeed | 0.0401   | 30 s ✓   | 45      |
| 2 | lower MJPC-only     | Stand H12 Lower (eq) | testspeed | 0.0424 | 30 s ✓  | ~50     |
| 3 | full  sim2sim       | Stand H12     | twin+ROS  | TBD      | TBD      | TBD     |
| 4 | lower sim2sim       | Stand H12 Lower (eq) | twin+ROS | TBD    | TBD      | TBD     |

Reference (not shipped): a **welded** lower model plans 2.17x faster than full but is
deploy-incompatible (no arm state to inject); see git history + `STAND_H12_LOWER_RESULTS.md`.
