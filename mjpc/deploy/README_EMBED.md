# MJPC Embedded DDS Control Node (`h12_control_node`)

Real-robot-deployable controller: embeds the MJPC C++ planner **in-process**,
running continuously in a background thread, reading robot state from **CycloneDDS**
(`rt/lowstate` + `rt/sportmodestate`) and writing joint commands to DDS
(`rt/safety/lowcmd_in` → safety layer → `rt/lowcmd` → robot/twin).

This replaces the Python gRPC bridge for deployment. The gRPC bridge proved the
control logic (capability test: every free-standing simple task balances the
full-weight twin with CEM once the planning model is `gravcomp=0` + gravity is
supplied as joint tau). But gRPC round-trips starve the planner (~65× slower than
realtime). Embedding the planner in-thread removes that — it runs as fast as
`app.cc`'s GUI loop.

## Why this exists / what it solves
- **Twin AND real robot**: speaks DDS, so the same binary drives the MuJoCo twin
  (`h1_mujoco/h12_mujoco.py`, which sub/pubs the same DDS topics, domain 0) and the
  real H1-2 (point `--network_interface` at the robot NIC).
- **Throughput**: planner `Plan()` runs continuously in a background thread
  (mirrors `mjpc/app.cc`), NOT one PlannerStep per gRPC round-trip.
- **gravcomp**: the planning model now has `body_gravcomp=0` (full weight, matches
  the real robot). This node adds gravity back as **joint feedforward torque** in
  the outgoing `LowCmd_.tau` (the deployable mechanism; `tau = gravity_ff *
  qfrc_bias` evaluated at qvel=0). Real H1-2 onboard PD closes the loop at 500 Hz+.

## Architecture (2 threads, mirrors app.cc)
```
 [DDS sub rt/lowstate + rt/sportmodestate]  --on msg-->  RobotState (mutex)
        |                                                      |
 [Planner thread: agent.Plan(exit, uiload)]                    | snapshot
   replans forever on the latest SetState                      v
                                          [Control thread @ ctrl_hz]
   SetState(reconstructed qpos/qvel) -> ActionFromPolicy(action) ->
   q*=action[0:27] ; tau=gravity_ff*qfrc_bias(q) ;
   unitree_hg LowCmd_{mode_pr,mode_machine,per-motor q*/dq0/tau/kp/kd} ; CRC ;
   publish rt/safety/lowcmd_in
```
Feed + control share one thread (like app.cc's physics thread) so we never race a
half-written `agent.state` between `SetState` and `ActionFromPolicy`.

**State reconstruction** (pelvis from IMU-site) — identical math to the validated
Python bridge (`mjpc_dds_bridge.py:pelvis_from_site`): the twin/robot reports the
IMU-site pose; back out the free-joint (pelvis):
`base_p = site_p − R(quat)·IMU_OFFSET`,
`base_v = site_v − (R·gyro) × (R·IMU_OFFSET)`, IMU_OFFSET=(-0.04452,-0.01891,0.27756).
`qpos=[base_p(3),quat(4,wxyz),joints(27)] (+ task object slots @home)`;
`qvel=[base_v(3),gyro(3),dq(27)]`.

**Warmup**: for `--warmup_sec` (default 1 s) the node holds the *measured* joint
pose (so the twin/robot stays grounded) while the planner converges, then releases
the policy — "controller live before release".

**Gains** (from `h1_2_modified` actuator classes == real LowCmd kp/kd; full table in
the `.cc`): KP legs/torso 150–200, ankle 80, arms 40; KV 5/4/10/2.

## Build prerequisites — **installed at `~/unitree_install`**
`unitree_sdk2` (C++) bundles its own prebuilt CycloneDDS (`thirdparty/lib`) and the
`unitree_hg` / `unitree_go` IDL, so **no separate CycloneDDS / cyclonedds-cxx build
is needed** (and no `sudo` — it installs to a user prefix):
```sh
git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2.git ~/unitree_sdk2
cmake -S ~/unitree_sdk2 -B ~/unitree_sdk2/build \
      -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$HOME/unitree_install
cmake --build ~/unitree_sdk2/build --target install -j"$(nproc)"
```
This populates `~/unitree_install/{include,lib,lib/cmake/unitree_sdk2}` (libs are
prebuilt → the "build" is just a fast file-copy, no compilation).

## Build the node
```sh
cmake -S ~/Desktop/h12/mujoco_mpc/mujoco_mpc \
      -B ~/Desktop/h12/mujoco_mpc/mujoco_mpc/build -DMJPC_BUILD_DEPLOY=ON
ninja -C ~/Desktop/h12/mujoco_mpc/mujoco_mpc/build h12_control_node
```
`MJPC_BUILD_DEPLOY` defaults OFF, so the normal build never needs unitree_sdk2.
The CMake auto-locates the SDK at `~/unitree_install` and bakes its `lib/` into the
binary's rpath (so the bundled `libddsc`/`libddscxx` resolve at runtime).

## Run (3 terminals — twin path)
```sh
# 1) twin (publishes rt/lowstate + rt/sportmodestate, subscribes rt/lowcmd):
cd ~/Desktop/h12/h1_mujoco && uv run python h12_mujoco.py --handless
# 2) safety layer:
cd ~/Desktop/h12/h12_safety_layer && \
  uv run h12_safety_layer/script/safety_layer_main.py --config default_safety_full.yaml
# 3) the embedded control node (strategy 6=stand 8=crouch 11=arms_overhead 13=lean_left):
~/Desktop/h12/mujoco_mpc/mujoco_mpc/build/bin/h12_control_node \
  --task "Lean H12 Magpie" --strategy 6 --gravity_ff 0   # twin: gravity_ff 0 + --imu_pitch_offset_deg 0
```
For the **real robot**, add `--network_interface <nic>` (e.g. `eth0`) so DDS binds
the robot link instead of loopback.

## Status
**Finalized 2026-06-01.** DDS path uses the verified `unitree_sdk2` C++ API
(`ChannelFactory::Init`, `ChannelPublisher/Subscriber<...>`, `unitree_hg::LowCmd_`
with `mode_pr`/`mode_machine`/`motor_cmd().at(i)` + CRC, `unitree_go::SportModeState_`)
copied from `example/h1/low_level/h1_2_ankle_track.cpp` and the validated Python
bridge. MJPC side uses the verified Agent API (`agent.h`) and mirrors
`grpc/agent_service.cc::Init` (incl. `mjcb_sensor = residual_sensor_callback`, which
the humanoid_bench/lean residuals require).
