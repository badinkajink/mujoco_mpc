# H1-2 MJPC deploy layer

Embedded MJPC controllers for the Unitree H1-2: each binary embeds `mjpc::Agent`,
reads `rt/lowstate` over Unitree DDS, plans on a twin model, and publishes
`LowCmd` setpoints into the HAMS safety layer. One shared implementation
(`deploy_common.cc`) + four thin mains:

| Main (.cc) | nu | Publishes | colcon binary | fork binary |
|---|---|---|---|---|
| `h12_split_controller` | 27 | `rt/safety/lowcmd_lower_in` + `rt/safety/lowcmd_upper_in` (arms pause-gated) | `mjpc_split_core` | — |
| `h12_lower_body_controller` | 12 | `rt/safety/lowcmd_lower_in` | `mjpc_lowerbody_core` | `h12_lower_body_controller` |
| `h12_control_node` | 27 | `rt/safety/lowcmd_in` | `mjpc_fullbody_core` | `h12_control_node` |
| `h12_upper_body_controller` | 15 | `rt/safety/lowcmd_upper_in` | — | `h12_upper_body_controller` |

**The production chain** (what `h1_bringup` launches, sim and real): the RW-EKF
base estimator (`helper_scripts/base_estimator_node.py`, publishes
`rt/sportmodestate_est`) + `mjpc_split_core` with task **"Lean H12 Magpie Split"**
(strategy 6), spawned by `core_ws/src/h12_deploy_mjpc`'s
`split_body_controller.py`. The split core boots with the arm channel paused
(`--pause_upper_init`); the launcher performs the frame-task handshake and
unpauses via the `rt/mjpc/pause_upperbody` DDS String topic (5 Hz keep-alive).
The lower-body chain (`mjpc_lowerbody_core`, task "Stabilize H12 Magpie") is the
retained legacy legs-only path; the full-body and upper cores are bench binaries.

## Two build routes

- **colcon (production)**: `core_ws/src/h12_deploy_mjpc/CMakeLists.txt` compiles
  the mains + `deploy_common.cc` straight from this submodule against the
  hydrated fork build tree (`libmjpc.a` is clang-13 LTO → the package pins
  clang-13 + ld.lld-13, and must use the build tree's `_deps` MuJoCo headers,
  never pip headers). After changing this directory: run
  `rebuild_mjpc.sh` in the container, then colcon. See root `CLAUDE.md` §8.
- **fork cmake (dev)**: `-DMJPC_BUILD_DEPLOY=ON` (needs unitree_sdk2 at
  `~/unitree_install`) builds full/lower/upper; `-DMJPC_BUILD_GRPC_SERVICE=ON`
  additionally compiles the in-process gRPC monitor (`H12_NODE_GRPC`).

**gRPC availability:** only fork-built binaries have the monitor/goal-ingest
service (full/lower on :10000, upper on :10001 — port 0 disables). The colcon
cores compile it out; `--grpc_port` is parsed but inert there, and the teleop
clients (`helper_scripts/wss_cmd_bridge.py`, `wasd_teleop.py` — strategy-24
drive) only work against fork builds.

## Wire surface (all mains)

Consumes `rt/lowstate` (+ complement rows for the X-aware mains) and
`rt/sportmodestate` / `rt/sportmodestate_est` (site pose for the twin state).
Optional debug: `--plan_topic rt/mjpc/plan` publishes the planner's best
trajectory as JSON (consumed by `mjpc_debug_visualizer` /
`helper_scripts/plan_visualizer.py`). Safe-hold (damping stop) publishes on the
main lowcmd topic on stale-state or fatal MuJoCo error; the safety layer's estop
is the torque backstop (the node emits the raw planner command — the former
torque-budget clamp is a monitor only).

## Settled compiled-in constants (`deploy_common.h`)

200 Hz control; warmup 1 s → ramp 5 s → hold 3 s → policy blend 4.5 s; stale
watchdog 0.05 s (loosen via `--stale_sec` for RoboCasa); planner threads AUTO
(hardware − reserve, `--plan_threads` to override); `kImuOffset` **must equal**
the base estimator's `IMU_OFFSET` (both ends of the IMU-site → pelvis
reconstruction).

## helper_scripts/

Production workers (runpy'd by `core_ws` — path is external ABI, do not move):
`base_estimator_node.py`, `plan_visualizer.py`. Operator tools (fork-gRPC only):
`wasd_teleop.py`, `wss_cmd_bridge.py`. `Command_Sheet_h12.html` is the
historical pre-HAMS native runbook.

Tuning/decision history for this layer: see `HISTORY.md`.
