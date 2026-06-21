// MJPC embedded DDS control node for the Unitree H1-2 (digital twin + real robot).
// See README_EMBED.md.  Build: cmake .. -DMJPC_BUILD_DEPLOY=ON  (needs unitree_sdk2).
//
// WHY THIS EXISTS
//   The Python gRPC bridge proved the control logic (every free-standing simple
//   task balances the full-weight torque twin with CEM once the planning model is
//   gravcomp=0 and gravity is fed back as joint tau).  But each gRPC GetAction
//   round-trips and starves the planner (~65x slower than realtime).  Embedding
//   mjpc::Agent in-process and running its planner continuously in a background
//   thread removes that wall -- exactly how mjpc/app.cc's GUI runs full-rate.
//
// ARCHITECTURE (mirrors app.cc: a continuous planner thread + a control thread)
//   planner thread : g_agent.Plan(exit, uiload)   -- replans forever on the latest state
//   control thread : @ctrl_hz  read DDS state -> SetState -> ActionFromPolicy
//                    -> q* + gravity-FF tau -> unitree_hg LowCmd_ -> rt/safety/lowcmd_in
//   (feed + control share one thread, like app.cc's physics thread, so we never
//    race a half-written g_agent.state between SetState and ActionFromPolicy.)
//
// STATE  pelvis (free-joint, qpos[0:7]) is backed out of the reported IMU-site
//   pose:  base_p = site_p - R(quat)*IMU_OFFSET ;  base_v = site_v - (R*gyro) x roff.
//   Identical math to mjpc_dds_bridge.py:pelvis_from_site (unit-tested vs ground truth).

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// POSIX networking -- auto-pin the wired robot-subnet NIC (see AutoDetectRobotInterface).
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/planners/cross_entropy/planner.h"
#include "mjpc/task.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

// Unitree SDK2 (C++): DDS channel API + unitree_hg / unitree_go IDL.
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

#ifdef H12_NODE_GRPC
// Optional in-process MJPC gRPC server so the existing monitor can attach
// (display + live Strategy switch). Enabled by the CMake when the grpc service
// is built. Reuses the exact helpers agent_server uses.
#include <absl/strings/str_cat.h>
#include <grpc/grpc_security_constants.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include "mjpc/grpc/agent.grpc.pb.h"
#include "mjpc/grpc/agent.pb.h"
#include "mjpc/grpc/grpc_agent_util.h"
#endif

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using SportState = unitree_go::msg::dds_::SportModeState_;

// Auto-detect the interface holding a 192.168.123.x address (the wired H1-2
// robot subnet), so an EMPTY --network_interface binds the robot link instead
// of CycloneDDS autodetermine grabbing WiFi/Tailscale -- the trap that silently
// makes the node hear the same-host twin but never the real robot. Mirrors
// dds_tools/dds_topic_check.py. Returns "" when no robot-subnet NIC is present
// (-> caller keeps autodetermine/loopback, the right default for the twin).
static std::string AutoDetectRobotInterface() {
  const char kRobotPrefix[] = "192.168.123.";
  std::string result;
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) return result;
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_addr->sa_family != AF_INET) continue;
    char host[INET_ADDRSTRLEN] = {0};
    const void* sin_addr =
        &reinterpret_cast<const struct sockaddr_in*>(ifa->ifa_addr)->sin_addr;
    if (inet_ntop(AF_INET, sin_addr, host, sizeof(host)) == nullptr) continue;
    if (std::strncmp(host, kRobotPrefix, sizeof(kRobotPrefix) - 1) == 0) {
      result = ifa->ifa_name;
      break;
    }
  }
  freeifaddrs(ifaddr);
  return result;
}

ABSL_FLAG(std::string, task, "Lean H12", "MJPC task id");
ABSL_FLAG(int, strategy, 6,
          "Lean Strategy parameter (6=stand 8=crouch 11=arms_overhead 13=lean_left "
          "18=squatter 19=jab 20=stumble ...)");
ABSL_FLAG(double, gravity_ff, 0.85,
          "joint gravity feedforward scale (tau = scale * qfrc_bias); 0 disables");
ABSL_FLAG(int, sync_plan, 0,
          "if >0, plan SYNCHRONOUSLY in the control thread: N PlanIteration per tick "
          "(SetState->plan->action, the cartpole/discriminator pattern) instead of the async "
          "background Plan() thread. Try 8.");
ABSL_FLAG(double, plan_rate_hz, 0.0,
          "if >0, plan SYNCHRONOUSLY in the control thread at this many PlanIteration PER SECOND, "
          "FRACTIONALLY accumulated: most ticks do 0 iters so the ctrl-rate stays high, and ~every "
          "few ticks 1 iter runs on the FRESH state. This is the discriminator's proven "
          "--plan-rate-hz mode (it HOLDS balance where async and integer --sync_plan do not, because "
          "the policy is always planned on the current state without collapsing ctrl-rate). Overrides "
          "--sync_plan and disables the async Plan() thread. Try 80.");
ABSL_FLAG(double, ctrl_hz, 200.0, "control / publish rate (Hz)");
ABSL_FLAG(bool, use_twin_time, false,
          "time the planner by the TWIN's sim-clock (rt/lowstate tick * twin_dt) instead of wall-clock. "
          "The in-process discriminator (which HOLDS balance) uses sim-time; if the twin runs SLOWER "
          "than real-time, wall races ahead of the twin so the policy/phase is mis-timed -> persistent "
          "lean. The status line prints t=wall(twin=...) so you can see the divergence. Try true.");
ABSL_FLAG(double, twin_dt, 0.005,
          "twin physics timestep (s) -- MUST equal the twin's model.opt.timestep (mujoco_env.py=0.005; "
          "the twin sets rt/lowstate tick = sim_time/0.005). Used for the --use_twin_time sim-clock AND "
          "the single-stream velocity finite-diff (Δt = Δtick*twin_dt). WAS 0.002 = WRONG: twin_time read "
          "0.4x real (the bogus '0.40x') and the velocity FD over-scaled 2.5x. Set to the real plant's "
          "tick period on hardware.");
ABSL_FLAG(int, plan_threads, 0,
          "planner ThreadPool size (rollout parallelism). 0 = all hardware threads (default). On a "
          "single machine the twin shares the CPU; all-threads STARVES the Python twin to ~0.37x "
          "real-time (see the twin= readout) -> planner/plant rate mismatch. Cap this (e.g. half your "
          "cores) so the twin gets CPU and runs near real-time; watch twin= climb toward wall.");
ABSL_FLAG(double, warmup_sec, 1.0,
          "seconds to converge the planner while HOLDING the measured pose before releasing the policy");
ABSL_FLAG(double, start_ramp_sec, 5.0,
          "ALL-joint bring-up ramp: blend every joint target from the measured power-on pose to "
          "the home/policy target over this many seconds. Prevents the warmup->policy SNAP when "
          "the robot starts away from home (e.g. crouched on the real robot -- the violent leg "
          "outstretch). No-op when the robot already starts at home. 0 = disable (legacy arm-only "
          "ramp via --arm_ramp_sec still applies).");
ABSL_FLAG(double, ramp_hold_sec, 3.0,
          "after the bring-up ramp reaches the stance, HOLD it scripted this many more "
          "seconds before the policy takes over -- lets the planner converge at the "
          "static operating pose (removes the hand-off tug). 0 = hand over at ramp end.");
ABSL_FLAG(double, policy_blend_sec, 0.0,
          "after the ramp+hold, EASE the commanded target from the scripted stance to the LIVE "
          "policy target over this many seconds (0 = legacy hard switch). The ramp always rises to "
          "'stand', so COLD-STARTING a non-stand task (crouch/squat) snaps at the scripted->policy "
          "handoff; this blend descends smoothly into the task pose instead. Stand is ~unaffected "
          "(its policy target == the stance). Try ~2.0-3.0 for crouch/squat cold starts.");
ABSL_FLAG(double, switch_settle_sec, 2.0,
          "after a LIVE strategy switch (stdin, e.g. type 6 then 8), HOLD the pre-switch pose this "
          "many seconds so the planner re-CONVERGES on the new strategy BEFORE the blend eases into "
          "it (mirrors ramp_hold for cold-start). Without it the robot descends mid-replan and sinks "
          "past the target. The total live-switch transition = switch_settle_sec + the blend.");
ABSL_FLAG(double, ankle_roll_offset_l_deg, 0.0,
          "LEFT ankle-roll zero-offset calibration (deg): SUBTRACTED from the perceived roll AND "
          "ADDED to the command, so a foot the encoder reports rolled is both reasoned about and "
          "driven to its true angle. 0 = off. Dial to flatten an edge-rolled foot (free-hang L/R asym).");
ABSL_FLAG(double, ankle_roll_offset_r_deg, 0.0,
          "RIGHT ankle-roll zero-offset calibration (deg); see --ankle_roll_offset_l_deg. The "
          "free-hang showed the RIGHT foot resting ~6 deg more rolled (edge) than the left.");
// NOTE: strategy 18 ("Squatter") is now a NATIVE 2-phase MJPC strategy (h12_simple_squatter.json,
// stand_up<->crouch), cycled by the task's own phase machine via ActiveTask()->Transition() below.
// The old node-side sequencer (g_squatter + squatter_*_hold/blend flags) was removed -- the planner
// now anticipates the transition instead of being surprised by a mid-flight cost switch.
ABSL_FLAG(double, arm_ramp_sec, 2.0,
          "seconds to linearly ramp ARM joint targets (idx 13..26) from the measured power-on pose to "
          "the home/policy target, so the bent-arm-vs-straight-command elbow PD spike stays under the "
          "safety-layer arm estop (0.80*18=14.4 Nm). 0 = disabled (old instant switch). Arms only; "
          "legs/torso untouched.");
ABSL_FLAG(bool, target_clamp, true,
          "torque-aware target clamp: bound every emitted joint target so the PD demand "
          "|kp*(target-q)| stays <= 0.9x the safety layer's tau-ESTOP threshold. Makes "
          "command-side tau estops impossible for ANY planner output (the recurring kill: "
          "limp bent elbows at engagement -> target ~2 rad away -> kp*err blows the 14.4 Nm "
          "arm estop; ankle-tau on a hard release). Also the slew-limit the safety layer "
          "lacks. false = raw targets (old behavior).");
ABSL_FLAG(bool, execute_best, false,
          "CEM only: emit the single lowest-cost ELITE's action instead of the elite MEAN "
          "(arXiv:2511.19204), so the plant never receives an averaged action no rollout flew. "
          "OFF = current (mean) behavior. Targets forward-drift from a mushy bimodal elite mean.");
// ---- latency compensation (Path B Stage 2) ------------------------------------------------
// The command we emit now lands on the plant Δ later (DDS transport + node compute + twin ZOH).
// For a STATIC stand Δ is harmless; for DYNAMIC balance the plant has moved by the time the
// stale action arrives -> lag -> topple. Fix (the standard mjpc2real move, arXiv 2511.19204):
// roll the planner model forward by Δ under the in-flight command, plan from THAT predicted
// state, and read the policy at the landing time. Pure node-side, DDS-observable state only ->
// deploys to the real robot unchanged. Default OFF = byte-identical to the current node.
// Pairs best with --plan_rate_hz (synchronous fresh-state planning; then measured Δ is exact).
ABSL_FLAG(bool, latency_comp, false,
          "predict state forward by the loop delay before planning (latency compensation). OFF = "
          "current behavior. Best paired with --plan_rate_hz (synchronous, so measured Δ is exact).");
ABSL_FLAG(double, latency_ms, 0.0,
          "fixed loop delay Δ (ms) to compensate. 0 = AUTO: EWMA of the node's measured per-tick "
          "compute time + --latency_extra_ms. Pin a value once you've measured it (e.g. dds_trace).");
ABSL_FLAG(double, latency_extra_ms, 4.0,
          "AUTO mode only: transport + twin zero-order-hold added to the measured compute time "
          "(DDS state->node + cmd->twin + ~one twin step). Loopback ~4ms; real robot: re-measure.");
ABSL_FLAG(double, latency_max_ms, 40.0,
          "hard cap on the predicted-forward horizon, so a mis-estimated Δ can't roll the "
          "prediction arbitrarily far and destabilize it.");
ABSL_FLAG(std::string, lowcmd_topic, "rt/safety/lowcmd_in",
          "DDS LowCmd output topic (through the safety layer)");
ABSL_FLAG(std::string, lowstate_topic, "rt/lowstate", "DDS LowState input topic");
ABSL_FLAG(std::string, sportstate_topic, "rt/sportmodestate",
          "DDS SportModeState input topic (IMU-site world pose)");
// ---- base linear-velocity reconstruction (single-stream FD, the mjpc2real estimation fix) -------
// The base linvel was built as (site_v from rt/sportmodestate) - (gyro from rt/lowstate) x r, which
// FUSES TWO UNSYNCHRONIZED 200Hz DDS streams (separate publisher threads, latest-wins, no timestamp
// align) -> a phantom base velocity that is exactly 0 at rest but grows with motion -> the planner
// perceives drift and leans to chase it. Every proven legged deploy fuses pose AND velocity in ONE
// time-consistent estimator (Bloesch RSS2012; DIAL-MPC / MPPI mocap+EKF). Fix here, sampling-only,
// node-side, DDS-observable-only (deploys to the real robot unchanged): derive base linvel by
// finite-differencing the reconstructed pelvis POSITION over the twin sim-clock + a 1st-order LPF.
// This drops the instantaneous cross-stream gyro x r term (the fast phantom) entirely.
ABSL_FLAG(double, vel_lpf_ms, 30.0,
          "base linear velocity = LPF(d/dt pelvis_pos) over the sim-clock, time constant in ms "
          "(single consistent stream). <=0 = OLD cross-stream (site_v - gyro x r) behavior, for A/B.");
ABSL_FLAG(int, domain_id, 0, "DDS domain id");
ABSL_FLAG(std::string, network_interface, "",
          "DDS network interface (empty = local/loopback for the twin; e.g. 'eth0' for the robot)");
ABSL_FLAG(bool, require_sportstate, true,
          "Require rt/sportmodestate at startup (true = twin/high-level mode -> real world base "
          "pose). false = DEBUG MODE (no high-level estimator publishes it): run on rt/lowstate "
          "alone -- hold a nominal standing base (xy=0, z=home keyframe height) + zero base "
          "linvel, balance off live IMU orientation + joints.");
ABSL_FLAG(double, imu_pitch_offset_deg, 1.6,
          "IMU pitch zero-offset CALIBRATION (deg) -- DEFAULT 1.6 = the MEASURED offset for THIS "
          "H1-2 (body verified truly vertical reads -1.6 deg on the IMU; +1.6 zeroes it). The "
          "perceived base orientation is rotated by this about the body pitch axis before planning. "
          "Validated on real (most-upright stand to date). Pass --imu_pitch_offset_deg 0 for the "
          "TWIN (no IMU mounting offset there) or to A/B against the raw IMU.");
ABSL_FLAG(double, imu_roll_offset_deg, 0.0,
          "IMU roll zero-offset calibration (deg), same idea as --imu_pitch_offset_deg (body roll axis).");
#ifdef H12_NODE_GRPC
ABSL_FLAG(int, grpc_port, 10000,
          "if >0, host an MJPC gRPC server on this port so the monitor can attach "
          "(view state + switch Strategy live); 0 disables");
#endif

namespace {
constexpr int kNU = 27;  // actuated joints on the handless H1-2
// Per-joint gains == h1_2_modified actuator classes == real LowCmd kp/kd.
// (Must match mjpc_dds_bridge.py / _lockstep_capability.py.)
// Safety-layer arm-gain fix (2026-06-09): ARMS ONLY -- arm kp 40 -> 30/20/15 (shoulder_p/r,
// shoulder_yaw+elbow, wrist) so the onboard arm PD torque stays under the arm estop bound. LEG/TORSO/
// hip_yaw kp LEFT AT ORIGINAL: lowering torso/hipY kp + tightening leg forceranges REGRESSED the hold
// (fell ~15s vs ~30s); the strut/lean is a SEPARATE balance problem, not a gain issue. Patched into the
// planner+latency model (PatchActuators) so node KP[]/KV[] == planner kp/kv == twin PD.
const double KP[kNU] = {150, 200, 200, 200, 80, 80,  150, 200, 200, 200, 80, 80,  200,
                        30, 30, 20, 20, 15, 15, 15,   30, 30, 20, 20, 15, 15, 15};
const double KV[kNU] = {5, 5, 5, 5, 4, 4,  5, 5, 5, 5, 4, 4,  5,
                        10, 10, 10, 10, 2, 2, 2,  10, 10, 10, 10, 2, 2, 2};
// SAFETY-LAYER TAU-ESTOP thresholds (estop torque_ratio x URDF torque limit, from
// default_safety_full.yaml + h12_safety_layer/core/joint_limits.py). Used by
// --target_clamp: |KP*(tgt-q)| is capped at 0.9x these, so a commanded position can
// never demand estop-level torque. kv*dq transients are not bounded here -- the 10%
// margin plus the planner's velocity-limit cost cover those.
const double TAU_ESTOP[kNU] = {60, 130, 200, 300, 54, 36,  60, 130, 200, 300, 54, 36,  40,
                               32, 32, 14.4, 14.4, 9.5, 9.5, 9.5,
                               32, 32, 14.4, 14.4, 9.5, 9.5, 9.5};
// ARM actuator force limit = OPERATIONAL URDF (= twin actuatorfrcrange). Patched into the planner+latency
// model for the ARMS ONLY (idx 13..26) so the planner caps arm torque at the real 18/40/19 Nm instead of
// the motor-peak 120 it otherwise assumes. LEG/TORSO forceranges are LEFT at the model default -- tightening
// them to the estop bound clamped the planner's hip/ankle balance authority and regressed the hold.
const double FRC_LIMIT[kNU] = {200, 200, 200, 300, 60, 40,  200, 200, 200, 300, 60, 40,  200,
                               40, 40, 18, 18, 19, 19, 19,   40, 40, 18, 18, 19, 19, 19};
// Patch a freshly-loaded model's <position> actuators to the node's authoritative gains + estop-bound
// forceranges (single source of truth; keeps planner + latency rollout == node KP[]/KV[] == twin PD,
// and bounds planned torque to the estop envelope). <position>: gainprm[0]=kp, biasprm[1]=-kp,
// biasprm[2]=-kv. Call on the loaded model BEFORE Agent::Initialize (it is const after GetModel()).
void PatchActuators(mjModel* m) {
  for (int i = 0; i < kNU && i < m->nu; i++) {
    m->actuator_gainprm[i * mjNGAIN + 0] = KP[i];
    m->actuator_biasprm[i * mjNBIAS + 1] = -KP[i];
    m->actuator_biasprm[i * mjNBIAS + 2] = -KV[i];
    if (i >= 13) {                       // ARM joints only: cap force to operational (18/40/19) so the
      m->actuator_forcelimited[i] = 1;   // planner can't plan a motor-peak (120) arm torque. LEG/TORSO
      m->actuator_forcerange[i * 2 + 0] = -FRC_LIMIT[i];   // forceranges LEFT at the model default --
      m->actuator_forcerange[i * 2 + 1] = FRC_LIMIT[i];    // tightening them regressed the hold.
    }
  }
}
// short joint names (qpos[7..33] order) for the Title-5 baseline (B0) report.
const char* const JOINT_NAMES[kNU] = {
    "LhipY", "LhipP", "LhipR", "Lknee", "LankP", "LankR",
    "RhipY", "RhipP", "RhipR", "Rknee", "RankP", "RankR", "torso",
    "LshP", "LshR", "LshY", "Lelb", "LwrR", "LwrP", "LwrY",
    "RshP", "RshR", "RshY", "Relb", "RwrR", "RwrP", "RwrY"};
// OPERATIONAL H1-2 joint torque limits (Nm) = Unitree URDF actuatorfrcrange == the twin's
// actuatorfrcrange == the safety-layer URDF_TORQUE_LIMITS. NOT the motor-PEAK from specs.md (which is
// ~6.7x higher on the arms: that table lists motor stall torque, e.g. elbow MOTOR 120 Nm, but the
// operational/control limit is 18). The safety estop enforces ratio*THIS, so [B0] must grade against
// THIS basis to predict trips (the old motor-peak table reported the arms 'torque-safe' while they
// tripped the estop). qpos[7..33] order.
//   legs: hipY/hipP/hipR 200, knee 300, ankP 60, ankR 40; torso 200;
//   arms: shoulderP/R 40, shoulderY 18, elbow 18, wrist 19.
const double TAU_LIMIT[kNU] = {200, 200, 200, 300, 60, 40,
                               200, 200, 200, 300, 60, 40, 200,
                               40, 40, 18, 18, 19, 19, 19,
                               40, 40, 18, 18, 19, 19, 19};
// IMU site position in the pelvis (free-joint) frame, from h1_2_handless.xml.
const double IMU_OFFSET[3] = {-0.04452, -0.01891, 0.27756};

// rotate vec v by quaternion q (wxyz) -- matches mjpc_dds_bridge.py:_quat_rot.
void QuatRot(const double q[4], const double v[3], double out[3]) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double tx = 2 * (y * v[2] - z * v[1]);
  double ty = 2 * (z * v[0] - x * v[2]);
  double tz = 2 * (x * v[1] - y * v[0]);
  out[0] = v[0] + w * tx + (y * tz - z * ty);
  out[1] = v[1] + w * ty + (z * tx - x * tz);
  out[2] = v[2] + w * tz + (x * ty - y * tx);
}

// Unitree CRC (matches example/h1/low_level + unitree_sdk2py.utils.crc).
uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    uint32_t xbit = 1u << 31;
    uint32_t data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
      if (data & xbit) CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }
  return CRC32;
}

// Plain, copyable snapshot of the latest robot state.
struct StateData {
  bool have_ls = false, have_ss = false;
  double q[kNU] = {0}, dq[kNU] = {0};
  double quat[4] = {1, 0, 0, 0}, gyro[3] = {0};  // rt/lowstate IMU (wxyz, body gyro)
  double site_p[3] = {0}, site_v[3] = {0};       // rt/sportmodestate (IMU-site world pose)
  uint8_t mode_machine = 0;
  uint32_t tick = 0;          // rt/lowstate tick = twin sim-step count (twin sim_time = tick * twin_dt)
};
// Mutex-guarded holder (std::mutex isn't copyable, so it stays out of the snapshot).
struct RobotState {
  std::mutex mu;
  StateData d;
};

// Globals for the MuJoCo sensor callback (mirror grpc/agent_service.cc).
mjpc::Agent g_agent;
const mjModel* g_agent_model = nullptr;
mjModel* g_model = nullptr;
mjpc::Task* g_task = nullptr;
std::atomic<bool> g_exit{false};
// --- LIVE STRATEGY-SWITCH BLEND: ease the pre-switch pose -> the new policy target over
//     g_switch_blend sec (same idea as the cold-start policy-blend, but re-armed on EVERY
//     live switch, so pressing 8 mid-run descends smoothly instead of snapping). g_switch_from
//     is written only by the main loop; the stdin thread just raises g_switch_pending and sets
//     the duration.
std::atomic<bool> g_switch_pending{false};
std::atomic<double> g_switch_wall{-1e9};   // wall time the active switch-blend started
std::atomic<double> g_switch_blend{0.0};   // duration of the active switch-blend (sec)
double g_switch_from[kNU] = {0};           // captured pre-switch commanded pose

void residual_sensor_callback(const mjModel* m, mjData* d, int stage) {
  if (m == g_agent_model || m == g_model) {
    if (stage == mjSTAGE_ACC) {
      g_task->Residual(m, d, d->sensordata);
    }
  }
}

void on_signal(int) { g_exit.store(true); }

// Headless MuJoCo error/warning handlers (the defaults block on a "Press Enter" prompt).
void FatalMjuError(const char* msg) {
  std::fprintf(stderr, "[mju_error] %s\n", msg);
  std::exit(1);
}
void LogMjuWarning(const char* msg) { std::fprintf(stderr, "[mju_warning] %s\n", msg); }

#ifdef H12_NODE_GRPC
// gRPC service over the node's LIVE agent so the existing MJPC monitor can
// attach. READ-ONLY reflection (state/action/residuals/costs/metrics/params)
// delegates to the same grpc_agent_util helpers agent_server uses; the only
// mutator exposed is SetTaskParameters (the live Strategy switch). All
// state/planner-driving RPCs are inert no-ops so the monitor (or a stray
// button) can never disturb the node's own control loop. A mutex serialises
// concurrent gRPC calls on the service's scratch mjData.
class NodeAgentService final : public agent::Agent::Service {
 public:
  NodeAgentService(mjpc::Agent* ag, const mjModel* m)
      : agent_(ag), model_(m), data_(mj_makeData(m)), rollout_data_(mj_makeData(m)) {}
  ~NodeAgentService() override {
    if (data_) mj_deleteData(data_);
    if (rollout_data_) mj_deleteData(rollout_data_);
  }
  grpc::Status GetState(grpc::ServerContext*, const agent::GetStateRequest*,
                        agent::GetStateResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    agent_->state.CopyTo(model_, data_);
    return grpc_agent_util::GetState(model_, data_, response);
  }
  grpc::Status GetAction(grpc::ServerContext*, const agent::GetActionRequest* request,
                         agent::GetActionResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    return grpc_agent_util::GetAction(request, agent_, model_, rollout_data_,
                                      &rollout_state_, response);
  }
  grpc::Status GetResiduals(grpc::ServerContext*, const agent::GetResidualsRequest* request,
                            agent::GetResidualsResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    agent_->state.CopyTo(model_, data_);
    mj_forward(model_, data_);
    return grpc_agent_util::GetResiduals(request, agent_, model_, data_, response);
  }
  grpc::Status GetCostValuesAndWeights(
      grpc::ServerContext*, const agent::GetCostValuesAndWeightsRequest* request,
      agent::GetCostValuesAndWeightsResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    agent_->state.CopyTo(model_, data_);
    mj_forward(model_, data_);
    return grpc_agent_util::GetCostValuesAndWeights(request, agent_, model_, data_, response);
  }
  grpc::Status GetMetrics(grpc::ServerContext*, const agent::GetMetricsRequest* request,
                          agent::GetMetricsResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    agent_->state.CopyTo(model_, data_);
    mj_forward(model_, data_);
    return grpc_agent_util::GetMetrics(request, agent_, model_, data_, response);
  }
  grpc::Status GetTaskParameters(grpc::ServerContext*, const agent::GetTaskParametersRequest* request,
                                 agent::GetTaskParametersResponse* response) override {
    return grpc_agent_util::GetTaskParameters(request, agent_, response);
  }
  grpc::Status SetTaskParameters(grpc::ServerContext*, const agent::SetTaskParametersRequest* request,
                                 agent::SetTaskParametersResponse*) override {
    return grpc_agent_util::SetTaskParameters(request, agent_);  // <-- live Strategy switch
  }
  grpc::Status GetMode(grpc::ServerContext*, const agent::GetModeRequest* request,
                       agent::GetModeResponse* response) override {
    return grpc_agent_util::GetMode(request, agent_, response);
  }
  grpc::Status GetAllModes(grpc::ServerContext*, const agent::GetAllModesRequest* request,
                           agent::GetAllModesResponse* response) override {
    return grpc_agent_util::GetAllModes(request, agent_, response);
  }
  // inert: the node owns state + planning; never let a client drive/reset them.
  grpc::Status Init(grpc::ServerContext*, const agent::InitRequest*,
                    agent::InitResponse*) override { return grpc::Status::OK; }
  grpc::Status SetState(grpc::ServerContext*, const agent::SetStateRequest*,
                        agent::SetStateResponse*) override { return grpc::Status::OK; }
  grpc::Status PlannerStep(grpc::ServerContext*, const agent::PlannerStepRequest*,
                           agent::PlannerStepResponse*) override { return grpc::Status::OK; }
  grpc::Status Step(grpc::ServerContext*, const agent::StepRequest*,
                    agent::StepResponse*) override { return grpc::Status::OK; }
  grpc::Status Reset(grpc::ServerContext*, const agent::ResetRequest*,
                     agent::ResetResponse*) override { return grpc::Status::OK; }

 private:
  mjpc::Agent* agent_;
  const mjModel* model_;
  mjData* data_;
  mjData* rollout_data_;
  mjpc::State rollout_state_;
  std::mutex mu_;
};
#endif  // H12_NODE_GRPC

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string task_id = absl::GetFlag(FLAGS_task);
  const double gff = absl::GetFlag(FLAGS_gravity_ff);
  const double ctrl_hz = absl::GetFlag(FLAGS_ctrl_hz);
  const double ctrl_dt = 1.0 / ctrl_hz;
  const double warmup_sec = absl::GetFlag(FLAGS_warmup_sec);
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  mju_user_error = FatalMjuError;      // headless: log + exit instead of blocking on getchar
  mju_user_warning = LogMjuWarning;

  // ---- MJPC Agent init (mirrors AgentService::Init) ----
  g_agent.SetTaskList(mjpc::GetTasks());
  int task_index = g_agent.GetTaskIdByName(task_id);
  if (task_index < 0) {
    std::fprintf(stderr, "[node] unknown task '%s'\n", task_id.c_str());
    return 1;
  }
  g_agent.gui_task_id = task_index;   // LoadModel() + Initialize() key off gui_task_id
  g_agent.SetTaskByIndex(task_index);
  auto lm = g_agent.LoadModel();
  if (!lm.model) {
    std::fprintf(stderr, "[node] LoadModel failed: %s\n", lm.error.c_str());
    return 1;
  }
  // ---- Per-strategy planner overrides (GENERIC -- no strategy hardcoded here) --
  // A strategy may need different planner model numerics than the stock stand --
  // e.g. "Stumble" needs a higher control bandwidth (sampling_spline_points 3->5)
  // so its gait clock can lift the swing foot; the stand-tuned 3 knots cannot
  // represent the leg oscillation (8 over-actuates and destabilises even a plain
  // stand -- 5 is the twin-validated sweet spot, 2026-06-18). The node stays
  // strategy-AGNOSTIC: it asks the TASK what numerics this strategy wants
  // (Task::PlannerNumericOverrides, default empty; the lean task overrides it per
  // strategy name) and applies them blindly. Adding a strategy with custom
  // bandwidth is a fork-side edit to the task only -- this deploy node never
  // changes. Applied HERE -- before Agent::Initialize, when the sampling policy
  // reads these numerics (planners/sampling/policy.cc). Keyed off the STARTUP
  // --strategy (the planner policy size is fixed at Initialize; live-switching
  // INTO another strategy keeps the startup bandwidth).
  {
    mjModel* sm = lm.model.get();
    const int strat = absl::GetFlag(FLAGS_strategy);
    for (const auto& kv : g_agent.ActiveTask()->PlannerNumericOverrides(strat)) {
      int id = mj_name2id(sm, mjOBJ_NUMERIC, kv.first.c_str());
      if (id >= 0) {
        sm->numeric_data[sm->numeric_adr[id]] = kv.second;
        std::fprintf(stderr, "[node] planner override: %s = %.4g (strategy %d)\n",
                     kv.first.c_str(), kv.second, strat);
      } else {
        std::fprintf(stderr,
                     "[node] planner override NUMERIC MISSING: %s (strategy %d)\n",
                     kv.first.c_str(), strat);
      }
    }
  }
  PatchActuators(lm.model.get());   // safety-operational gains + estop forceranges BEFORE the agent uses it
  g_agent.Initialize(lm.model.get());
  g_agent.Allocate();
  g_agent.Reset();
  g_task = g_agent.ActiveTask();
  g_agent_model = g_agent.GetModel();
  g_model = mj_copyModel(nullptr, g_agent_model);
  PatchActuators(g_model);   // latency-comp rollout model (the planner model was patched at LoadModel above)
  mjData* data = mj_makeData(g_model);
  int home = mj_name2id(g_model, mjOBJ_KEY, "home");
  if (home >= 0) mj_resetDataKeyframe(g_model, data, home);
  mjcb_sensor = residual_sensor_callback;
  g_agent.SetState(data);
  g_agent.plan_enabled = true;
  g_agent.action_enabled = true;
  // Strategy 18 ("Squatter") is a native multi-phase MJPC strategy now -- it loads like any other
  // index and the task's phase machine (ActiveTask()->Transition() in the loop) cycles stand_up<->
  // crouch on its own. The cold-start ramp + policy_blend eases into phase 0 (stand) just like
  // strat 6, then the first crouch fires once stand is achieved + sustained (success_sustain_time).
  g_agent.SetParamByName("residual_Strategy",
                         static_cast<double>(absl::GetFlag(FLAGS_strategy)));

  const int nq = g_model->nq, nv = g_model->nv, nu = g_model->nu;
  const int nact = nu < kNU ? nu : kNU;
  std::fprintf(stderr,
               "[node] task='%s' nq=%d nv=%d nu=%d strategy=%d gravity_ff=%.2f ctrl_hz=%.0f\n",
               task_id.c_str(), nq, nv, nu, absl::GetFlag(FLAGS_strategy), gff, ctrl_hz);

#ifdef H12_NODE_GRPC
  // ---- MJPC gRPC server (started early so the monitor can attach anytime) ----
  std::unique_ptr<NodeAgentService> grpc_service;
  std::unique_ptr<grpc::Server> grpc_server;
  const int grpc_port = absl::GetFlag(FLAGS_grpc_port);
  if (grpc_port > 0) {
    grpc_service = std::make_unique<NodeAgentService>(&g_agent, g_model);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(absl::StrCat("[::]:", grpc_port),
                             grpc::experimental::LocalServerCredentials(LOCAL_TCP));
    builder.SetMaxReceiveMessageSize(50 * 1024 * 1024);
    builder.RegisterService(grpc_service.get());
    grpc_server = builder.BuildAndStart();
    if (grpc_server) {
      std::fprintf(stderr,
                   "[node] MJPC gRPC server on :%d -- attach the monitor to this "
                   "port to view state + switch Strategy live\n", grpc_port);
    } else {
      std::fprintf(stderr,
                   "[node] WARNING: gRPC server failed on :%d (port in use?) -- "
                   "continuing without the monitor\n", grpc_port);
    }
  }
#endif

  // ---- DDS: subscribe rt/lowstate + rt/sportmodestate, publish lowcmd ----
  // Resolve the interface: explicit --network_interface wins; otherwise auto-pin
  // the 192.168.123.x robot-subnet NIC (so NO flag is needed on the real robot);
  // fall back to autodetermine/loopback for the same-host twin.
  std::string net_if = absl::GetFlag(FLAGS_network_interface);
  if (net_if.empty()) {
    std::string auto_if = AutoDetectRobotInterface();
    if (!auto_if.empty()) {
      net_if = auto_if;
      std::fprintf(stderr,
                   "[node] auto-pinned DDS interface '%s' (192.168.123.x robot "
                   "subnet); pass --network_interface to override\n",
                   net_if.c_str());
    } else {
      std::fprintf(stderr,
                   "[node] no robot-subnet NIC found -> DDS autodetermine "
                   "(twin/loopback); pass --network_interface for the robot\n");
    }
  } else {
    std::fprintf(stderr,
                 "[node] DDS interface '%s' (explicit --network_interface)\n",
                 net_if.c_str());
  }
  ChannelFactory::Instance()->Init(absl::GetFlag(FLAGS_domain_id), net_if);
  RobotState rs;

  ChannelSubscriberPtr<LowState> ls_sub(
      new ChannelSubscriber<LowState>(absl::GetFlag(FLAGS_lowstate_topic)));
  ls_sub->InitChannel(
      [&rs](const void* msg) {
        const LowState* s = static_cast<const LowState*>(msg);
        std::lock_guard<std::mutex> lk(rs.mu);
        for (int i = 0; i < kNU; i++) {
          rs.d.q[i] = s->motor_state().at(i).q();
          rs.d.dq[i] = s->motor_state().at(i).dq();
        }
        for (int k = 0; k < 4; k++) rs.d.quat[k] = s->imu_state().quaternion().at(k);
        for (int k = 0; k < 3; k++) rs.d.gyro[k] = s->imu_state().gyroscope().at(k);
        rs.d.mode_machine = s->mode_machine();
        rs.d.tick = s->tick();
        rs.d.have_ls = true;
      },
      10);

  ChannelSubscriberPtr<SportState> ss_sub(
      new ChannelSubscriber<SportState>(absl::GetFlag(FLAGS_sportstate_topic)));
  ss_sub->InitChannel(
      [&rs](const void* msg) {
        const SportState* s = static_cast<const SportState*>(msg);
        std::lock_guard<std::mutex> lk(rs.mu);
        for (int k = 0; k < 3; k++) {
          rs.d.site_p[k] = s->position().at(k);
          rs.d.site_v[k] = s->velocity().at(k);
        }
        rs.d.have_ss = true;
      },
      10);

  ChannelPublisherPtr<LowCmd> cmd_pub(
      new ChannelPublisher<LowCmd>(absl::GetFlag(FLAGS_lowcmd_topic)));
  cmd_pub->InitChannel();

  // ---- wait for the first full state ----
  // Debug mode (no high-level estimator) never publishes rt/sportmodestate, so
  // --require_sportstate=false gates on rt/lowstate alone and fill_state holds a
  // nominal standing base instead. Default true = twin/high-level (real base pose).
  const bool require_sport = absl::GetFlag(FLAGS_require_sportstate);
  double nominal_base_p[3] = {0, 0, 0};
  if (home >= 0) {
    for (int k = 0; k < 3; k++) nominal_base_p[k] = g_model->key_qpos[home * nq + k];
  } else {
    for (int k = 0; k < 3; k++) nominal_base_p[k] = g_model->qpos0[k];
  }
  std::fprintf(stderr, "[node] waiting for %s%s ...\n",
               absl::GetFlag(FLAGS_lowstate_topic).c_str(),
               require_sport ? " + rt/sportmodestate"
                             : " (debug mode: sportstate optional, nominal base)");
  if (!require_sport) {
    std::fprintf(stderr,
                 "[node] DEBUG MODE: rt/sportmodestate NOT required -> nominal base "
                 "(xy=0, z=%.3f) + zero base linvel; balance off IMU + joints\n",
                 nominal_base_p[2]);
  }
  while (!g_exit.load()) {
    {
      std::lock_guard<std::mutex> lk(rs.mu);
      if (rs.d.have_ls && (rs.d.have_ss || !require_sport)) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  std::fprintf(stderr, "[node] state stream up -> starting continuous planner.\n");

  // ---- planner: pick ONE of three modes ----
  //   plan_rate_hz>0 : SYNCHRONOUS, FRACTIONAL iters/tick on FRESH state (discriminator's proven
  //                    --plan-rate-hz mode; HOLDS balance, ctrl-rate stays high).  <-- the fix
  //   sync_plan>0    : SYNCHRONOUS, INTEGER iters/tick (collapses ctrl-rate; kept for comparison).
  //   else (default) : ASYNC background Plan() thread (refutes balance: policy planned on stale state).
  const int sync_plan = absl::GetFlag(FLAGS_sync_plan);
  const double plan_rate_hz = absl::GetFlag(FLAGS_plan_rate_hz);
  const bool sync_in_ctrl = (plan_rate_hz > 0.0) || (sync_plan > 0);
  std::atomic<bool> plan_exit{false};
  std::atomic<long> plan_count{0};  // REAL planner-iteration counter (PlanSteps() is the HORIZON, not iters)
  const int plan_threads = absl::GetFlag(FLAGS_plan_threads);
  const int n_plan_threads = plan_threads > 0 ? plan_threads : mjpc::NumAvailableHardwareThreads();
  std::fprintf(stderr, "[node] planner ThreadPool: %d threads (%d hw available; cap --plan_threads "
                       "to free CPU for the twin so it runs nearer real-time)\n",
               n_plan_threads, mjpc::NumAvailableHardwareThreads());

  // ---- execute-best (CEM only): hand the plant the lowest-cost elite, not the elite mean ----
  if (absl::GetFlag(FLAGS_execute_best)) {
    if (auto* cem =
            dynamic_cast<mjpc::CrossEntropyPlanner*>(&g_agent.ActivePlanner())) {
      cem->SetBestAction(true);
      std::fprintf(stderr, "[node] execute-best ON: plant gets the single lowest-cost ELITE "
                           "(not the elite mean) -- arXiv:2511.19204\n");
    } else {
      std::fprintf(stderr, "[node] WARNING: --execute_best ignored -- active planner is not "
                           "CrossEntropy (need agent_planner=5). Emitting default policy action.\n");
    }
  } else {
    std::fprintf(stderr, "[node] execute-best OFF (CEM emits the elite MEAN; pass --execute_best)\n");
  }
  if (absl::GetFlag(FLAGS_target_clamp))
    std::fprintf(stderr,
                 "[node] torque-aware target clamp ON: |kp*(tgt-q)| <= 0.9 x tau-estop per "
                 "joint (elbow/wrist/ankle command-side estops impossible; --notarget_clamp "
                 "to disable)\n");

  mjpc::ThreadPool plan_pool(n_plan_threads);
  std::thread planner;
  if (!sync_in_ctrl) {
    planner = std::thread([&] {
      while (!plan_exit.load()) { g_agent.PlanIteration(&plan_pool); plan_count.fetch_add(1); }
    });
  } else if (plan_rate_hz > 0.0) {
    std::fprintf(stderr, "[node] SYNCHRONOUS planning: %.0f PlanIteration/sec FRACTIONALLY in the "
                         "control thread on FRESH state (discriminator's proven --plan-rate-hz mode; "
                         "no async Plan thread; ctrl-rate stays high)\n", plan_rate_hz);
  } else {
    std::fprintf(stderr, "[node] SYNCHRONOUS planning: %d PlanIteration/tick in the control "
                         "thread (cartpole/discriminator pattern; no background Plan thread)\n",
                 sync_plan);
  }

  // ---- live strategy switch via stdin: type a number 0-20 (+Enter), q=quit ----
  std::fprintf(stderr, "[node] live switch ready: type a strategy number 0-20 + Enter (q=quit)\n");
  std::thread stdin_thread([&] {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line == "q" || line == "quit") { g_exit.store(true); break; }
      if (line.empty()) continue;
      try {
        int s = std::stoi(line);
        // Any strategy (incl. 18 = native Squatter, 19 = native Jab) loads the same way: set the
        // residual strategy and arm the live-switch settle+blend so the target eases from the current
        // pose into the new policy target instead of snapping. The phase machine handles the internal
        // cycling of multi-phase strategies (18 squat cycle, 19 jab guard<->extend).
        g_agent.SetParamByName("residual_Strategy", static_cast<double>(s));
        g_switch_blend.store(absl::GetFlag(FLAGS_policy_blend_sec));
        g_switch_pending.store(true);   // main loop captures the from-pose + arms the blend
        std::fprintf(stderr, "[node] >>> Strategy -> %d (eases into the new pose over %.1fs)\n",
                     s, absl::GetFlag(FLAGS_policy_blend_sec));
      } catch (...) {
        std::fprintf(stderr, "[node] (enter a strategy number 0-20, or q to quit)\n");
      }
    }
  });
  stdin_thread.detach();

  // scratch mjData: sd = real state for SetState; gd = qvel-zeroed for gravity FF.
  // Object/task slots beyond the robot (qpos[34:], qvel[33:]) keep home defaults.
  mjData* sd = mj_makeData(g_model);
  mjData* gd = mj_makeData(g_model);
  mjData* pdat = mj_makeData(g_model);   // latency-comp: forward-prediction rollout scratch
  if (home >= 0) {
    mj_resetDataKeyframe(g_model, sd, home);
    mj_resetDataKeyframe(g_model, gd, home);
    mj_resetDataKeyframe(g_model, pdat, home);
  }

  // IMU zero-offset calibration (deg->rad): a measured constant mounting/zero bias in the
  // base orientation feeds straight into the planner (qpos[3:7] below) and makes it balance
  // around a false vertical -> a steady lean (sim-confirmed). Cancel it here.
  const double imu_pitch_off = absl::GetFlag(FLAGS_imu_pitch_offset_deg) * M_PI / 180.0;
  const double imu_roll_off = absl::GetFlag(FLAGS_imu_roll_offset_deg) * M_PI / 180.0;
  if (imu_pitch_off != 0.0 || imu_roll_off != 0.0)
    std::printf("[node] IMU zero-offset calibration ON: perceived base orientation rotated "
                "pitch%+.2f roll%+.2f deg before planning\n",
                absl::GetFlag(FLAGS_imu_pitch_offset_deg), absl::GetFlag(FLAGS_imu_roll_offset_deg));

  auto fill_state = [&](const StateData& cur) {
    double roff[3];
    QuatRot(cur.quat, IMU_OFFSET, roff);
    double ww[3];
    QuatRot(cur.quat, cur.gyro, ww);
    double cr[3] = {ww[1] * roff[2] - ww[2] * roff[1], ww[2] * roff[0] - ww[0] * roff[2],
                    ww[0] * roff[1] - ww[1] * roff[0]};
    if (require_sport && cur.have_ss) {
      for (int k = 0; k < 3; k++) sd->qpos[k] = cur.site_p[k] - roff[k];  // pelvis = site - R*offset
      for (int k = 0; k < 3; k++) sd->qvel[k] = cur.site_v[k] - cr[k];    // pelvis world linvel
    } else {
      // DEBUG MODE / no sportmodestate: no world base estimate. Hold a nominal
      // standing base (xy=0, z=home height) with zero base linvel; balance is
      // driven by the live IMU orientation + joints set below.
      for (int k = 0; k < 3; k++) { sd->qpos[k] = nominal_base_p[k]; sd->qvel[k] = 0.0; }
    }
    // base orientation = IMU quat, with the measured zero-offset cancelled (body-frame
    // post-multiply by the small pitch/roll correction; wxyz, MuJoCo convention).
    double bq[4];
    mju_copy4(bq, cur.quat);
    if (imu_pitch_off != 0.0) {
      double d[4], ax[3] = {0, 1, 0}, t[4];
      mju_axisAngle2Quat(d, ax, imu_pitch_off);
      mju_mulQuat(t, bq, d); mju_copy4(bq, t);
    }
    if (imu_roll_off != 0.0) {
      double d[4], ax[3] = {1, 0, 0}, t[4];
      mju_axisAngle2Quat(d, ax, imu_roll_off);
      mju_mulQuat(t, bq, d); mju_copy4(bq, t);
    }
    mju_normalize4(bq);
    for (int k = 0; k < 4; k++) sd->qpos[3 + k] = bq[k];
    for (int i = 0; i < kNU; i++) sd->qpos[7 + i] = cur.q[i];
    // ankle-roll zero-offset calibration: the planner sees the CORRECTED roll (idx 5=L, 11=R
    // ankle_roll) so it doesn't chase a foot only the encoder thinks is rolled. Pairs with the
    // command shift before publish below. 0 = no-op.
    sd->qpos[7 + 5]  -= absl::GetFlag(FLAGS_ankle_roll_offset_l_deg) * M_PI / 180.0;
    sd->qpos[7 + 11] -= absl::GetFlag(FLAGS_ankle_roll_offset_r_deg) * M_PI / 180.0;
    for (int k = 0; k < 3; k++) sd->qvel[3 + k] = cur.gyro[k];        // free-joint angvel == body gyro
    for (int i = 0; i < kNU; i++) sd->qvel[6 + i] = cur.dq[i];
  };

  // ---- latency compensation: roll the planner model forward by Δ under the in-flight command ----
  // The planner model's <position> actuators have kp/kv == the node's KP[]/KV[] == the twin's PD,
  // so this rollout reproduces the twin's joint law; we only add the gravity tau_ff the twin also
  // applies. Object/task slots are left at home (we copy only the robot dofs back). NaN-guarded.
  const bool   lat_comp  = absl::GetFlag(FLAGS_latency_comp);
  const double lat_fixed = absl::GetFlag(FLAGS_latency_ms) * 1e-3;
  const double lat_extra = absl::GetFlag(FLAGS_latency_extra_ms) * 1e-3;
  const double lat_max   = absl::GetFlag(FLAGS_latency_max_ms) * 1e-3;
  const double pred_dt   = g_model->opt.timestep;     // native model step (0.002) == twin granularity
  double ewma_comp = 1.0 / ctrl_hz;                   // measured per-tick compute time (EWMA), seeded
  double last_cmd_q[kNU];
  for (int i = 0; i < kNU; i++) last_cmd_q[i] = sd->qpos[7 + i];   // home target until first publish
  // ---- arm-target ramp state (safety arm-estop fix): blend ONLY arm joints (idx 13..26) from the
  //      measured power-on pose to the home/policy target over arm_ramp_sec so the elbow PD never
  //      saturates at the warmup->policy switch. home_q = the straight destination during warmup.
  const double arm_ramp_sec = absl::GetFlag(FLAGS_arm_ramp_sec);
  const double start_ramp_sec = absl::GetFlag(FLAGS_start_ramp_sec);
  if (start_ramp_sec > 0.0) {
    std::fprintf(stderr, "[node] bring-up ramp ON: ALL joints measured->home/policy over %.1fs "
                         "(no-op if already at home; --start_ramp_sec 0 to disable)\n",
                 start_ramp_sec);
  }
  double home_q[kNU];
  { // bring-up destination = the OPERATING stance ("stand", bent-knee), NOT the singular
    // straight-knee "home": ramping to knee=0 hands the policy the hyperextension basin
    // (knees snapped to the -0.12 stop within 1 s of ramp end -- live-verified).
    int hk = mj_name2id(g_model, mjOBJ_KEY, "stand");
    if (hk < 0) hk = mj_name2id(g_model, mjOBJ_KEY, "home");
    if (hk < 0) hk = 0;
    std::fprintf(stderr, "[node] bring-up ramp destination keyframe: '%s'\n",
                 mj_id2name(g_model, mjOBJ_KEY, hk));
    for (int i = 0; i < kNU; i++) home_q[i] = g_model->key_qpos[hk * nq + 7 + i]; }
  double arm_q_init[kNU];
  for (int i = 0; i < kNU; i++) arm_q_init[i] = sd->qpos[7 + i];   // refined to measured pose on first live state
  bool arm_init_set = false;
  double ramp_eff = start_ramp_sec;   // rescaled at latch by distance-from-home (see loop)

  auto predict_forward = [&](mjData* s, double dt_ahead, const double* cmd_q) {
    int nsub = static_cast<int>(std::lround(dt_ahead / pred_dt));
    if (nsub < 1) return;
    mju_copy(pdat->qpos, s->qpos, nq);
    mju_copy(pdat->qvel, s->qvel, nv);
    if (g_model->na > 0) mju_zero(pdat->act, g_model->na);
    mju_zero(pdat->qfrc_applied, nv);
    if (gff != 0.0) {                                   // same constant tau_ff the twin's LowCmd carries
      mju_copy(gd->qpos, s->qpos, nq);
      mju_zero(gd->qvel, nv);
      mj_forward(g_model, gd);
      for (int i = 0; i < kNU; i++) pdat->qfrc_applied[6 + i] = gff * gd->qfrc_bias[6 + i];
    }
    for (int k = 0; k < nsub; k++) {
      for (int i = 0; i < nact; i++) pdat->ctrl[i] = cmd_q[i];   // position-actuator target = in-flight cmd
      mj_step(g_model, pdat);
    }
    bool ok = true;                                     // discard a blown-up rollout, keep measured state
    for (int k = 0; ok && k < 7 + kNU; k++) if (!std::isfinite(pdat->qpos[k])) ok = false;
    for (int k = 0; ok && k < 6 + kNU; k++) if (!std::isfinite(pdat->qvel[k])) ok = false;
    if (!ok) return;
    for (int k = 0; k < 7 + kNU; k++) s->qpos[k] = pdat->qpos[k];
    for (int k = 0; k < 6 + kNU; k++) s->qvel[k] = pdat->qvel[k];
  };
  if (lat_comp)
    std::fprintf(stderr, "[node] latency-comp ON: predict-forward %s%s (extra %.1fms, cap %.0fms); "
                 "plan + policy read at the landing time\n",
                 lat_fixed > 0.0 ? "FIXED " : "AUTO measured", lat_fixed > 0.0 ? "" : "+extra",
                 lat_extra * 1e3, lat_max * 1e3);
  else
    std::fprintf(stderr, "[node] latency-comp OFF (pass --latency_comp to enable)\n");

  std::vector<double> action(nu, 0.0);

  // ---- Title-5 baseline (B0) accumulators: tracking error, applied torque, balance ----
  // (policy phase only; printed as a summary on exit. The SAME node runs on the real
  //  robot -> per-row sim2real delta.)
  long m_ticks = 0;
  double m_err_sum[kNU] = {0}, m_err_sq[kNU] = {0}, m_err_max[kNU] = {0};
  double m_tau_sum[kNU] = {0}, m_tau_max[kNU] = {0};
  double m_z_sum = 0, m_z_sq = 0, m_z_min = 1e9, m_tilt_sum = 0, m_tilt_max = 0;
  long m_sat_count[kNU] = {0};   // ticks where |tau| exceeds the real H1-2 limit
  bool m_sat_warned = false;

  // ---- control loop @ ctrl_hz (mirrors app.cc physics thread) ----
  const bool use_twin_time = absl::GetFlag(FLAGS_use_twin_time);
  const double twin_dt = absl::GetFlag(FLAGS_twin_dt);
  uint32_t tick0 = 0; bool tick0_set = false;   // zero the twin sim-clock at the first state
  // single-stream base-velocity finite-diff + LPF state (see vel_lpf_ms flag).
  const double vel_lpf_tau = absl::GetFlag(FLAGS_vel_lpf_ms) * 1e-3;
  double fd_prev_p[3] = {0}, fd_vel[3] = {0}, fd_prev_t = 0.0; bool fd_have = false;
  std::fprintf(stderr, vel_lpf_tau > 0.0
                 ? "[node] base linvel: SINGLE-STREAM finite-diff + %.0fms LPF (drops the two-stream phantom)\n"
                 : "[node] base linvel: OLD cross-stream site_v - gyro x r (vel_lpf_ms<=0)\n",
               absl::GetFlag(FLAGS_vel_lpf_ms));
  auto t0 = std::chrono::steady_clock::now();
  auto next = t0;
  long ticks = 0;
  while (!g_exit.load()) {
    StateData cur;
    {
      std::lock_guard<std::mutex> lk(rs.mu);
      cur = rs.d;
    }
    auto t_tick = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t_tick - t0).count();
    bool warming = wall < warmup_sec;
    if (!tick0_set && cur.have_ls) { tick0 = cur.tick; tick0_set = true; }
    double twin_time = static_cast<double>(static_cast<int64_t>(cur.tick) -
                                           static_cast<int64_t>(tick0)) * twin_dt;

    fill_state(cur);
    if (!arm_init_set && cur.have_ls) {                 // latch the measured power-on arm pose for the ramp
      for (int i = 0; i < kNU; i++) arm_q_init[i] = cur.q[i];
      arm_init_set = true;
      if (start_ramp_sec > 0.0) {
        // scale the ramp (and its policy-holdout) by how far off-home we latched: the
        // holdout is only survivable while a harness supports the robot -- from a HOME
        // start a full 5 s without active balance topples it (bench: fell at 2.7 s).
        // >=0.5 rad off-home -> full ramp; at home -> ~0 -> policy right after warmup
        // (the proven home-start behavior).
        double d0 = 0.0;
        for (int i = 0; i < kNU; i++) d0 = std::fmax(d0, std::fabs(arm_q_init[i] - home_q[i]));
        ramp_eff = start_ramp_sec * std::fmin(1.0, d0 / 0.5);
        std::fprintf(stderr, "[node] bring-up ramp: latched pose is %.2f rad from home -> "
                             "effective ramp %.1fs\n", d0, ramp_eff);
      }
    }
    // SINGLE-STREAM base linvel: replace the cross-stream (site_v - gyro x r) value with a finite
    // diff of the reconstructed pelvis position over the sim-clock + LPF. Updates only when the twin
    // tick advances (a genuinely new state); holds between. Set before the latency rollout so the
    // prediction starts from the de-noised velocity. vel_lpf_ms<=0 keeps the old behavior.
    if (vel_lpf_tau > 0.0 && cur.have_ls && cur.have_ss) {
      if (fd_have && twin_time > fd_prev_t + 1e-9) {       // new state -> differentiate over real dt
        double dt = twin_time - fd_prev_t;
        double a = dt / (vel_lpf_tau + dt);                // 1st-order LPF, step-size independent
        for (int k = 0; k < 3; k++) {
          double raw = (sd->qpos[k] - fd_prev_p[k]) / dt;
          fd_vel[k] += a * (raw - fd_vel[k]);
          fd_prev_p[k] = sd->qpos[k];
        }
        fd_prev_t = twin_time;
      } else if (!fd_have) {                                // seed at rest (v=0) on the first state
        for (int k = 0; k < 3; k++) { fd_prev_p[k] = sd->qpos[k]; fd_vel[k] = 0.0; }
        fd_prev_t = twin_time; fd_have = true;
      }
      for (int k = 0; k < 3; k++) sd->qvel[k] = fd_vel[k];
    }
    // snapshot the MEASURED base pose NOW -- latency prediction overwrites sd->qpos below, but B0
    // and the status line must report the real (measured) height/tilt/knees, not the predicted ones.
    double meas_base_z = sd->qpos[2];
    double meas_R8; { double Rm[9]; mju_quat2Mat(Rm, sd->qpos + 3); meas_R8 = Rm[8]; }
    double meas_kneeL = sd->qpos[10], meas_kneeR = sd->qpos[16];

    // LATENCY COMPENSATION: predict the state forward by Δ and plan from THERE, so the action we
    // emit is in-phase when it lands on the plant. dlt=0 (default, or warmup) -> identical to before.
    double dlt = 0.0;
    if (lat_comp && !warming) {
      dlt = (lat_fixed > 0.0) ? lat_fixed : (ewma_comp + lat_extra);
      if (dlt > lat_max) dlt = lat_max;
      predict_forward(sd, dlt, last_cmd_q);
    }
    // disc (which HOLDS) times the planner by SIM-time; the node used wall, which races ahead of a
    // slower-than-realtime twin -> mis-timed policy. --use_twin_time clocks it by the twin's tick.
    // + dlt reads the policy at the LANDING time, consistent with the predicted start state above.
    sd->time = (use_twin_time ? twin_time : wall) + dlt;
                       // FIX 2026-06-02: advance the planner clock. It was NEVER set (stuck at 0),
                       // so ActionFromPolicy(state.time) read the START of the plan every tick ->
                       // the node executed only the first instant of each trajectory. Static holds
                       // (stand) looked fine; DYNAMIC motions (crouch, arms_overhead) never played
                       // out. app.cc advances d->time the same way so the policy is read forward.
    mj_forward(g_model, sd);
    // THE MISSING CALL (found 2026-06-12 via residual forensics): the GUI's physics loop
    // runs Task::Transition every step -- it is what LOADS the strategy JSON (posture
    // keyframe + per-phase weights) and advances multi-phase strategies. This node NEVER
    // called it, so every deployment planned toward keyframe 0 (straight-knee 'home',
    // z=1.028): the hand-off knee snap, the 1.028 base parking, and the inert keyframe
    // edits were all this one absence.
    g_agent.ActiveTask()->Transition(g_model, sd);
    g_agent.SetState(sd);
    // synchronous planning on THIS (fresh) state -- the discriminator pattern.
    if (plan_rate_hz > 0.0) {   // FRACTIONAL: ~plan_rate_hz iters/sec, most ticks 0 -> ctrl-rate stays high
      static double plan_accum = 0.0;
      plan_accum += plan_rate_hz * ctrl_dt;
      int niter = static_cast<int>(plan_accum);
      plan_accum -= niter;
      for (int i = 0; i < niter; i++) { g_agent.PlanIteration(&plan_pool); plan_count.fetch_add(1); }
    } else if (sync_plan > 0) {  // INTEGER iters/tick (collapses ctrl-rate; comparison only)
      for (int i = 0; i < sync_plan; i++) { g_agent.PlanIteration(&plan_pool); plan_count.fetch_add(1); }
    }

    // policy: 27 target joint positions (rad). Held at measured q during warmup.
    if (!warming) {
      g_agent.ActivePlanner().ActionFromPolicy(action.data(), g_agent.state.state().data(),
                                             g_agent.state.time());
    }

    // gravity feedforward: tau = gff * qfrc_bias evaluated at qvel = 0.
    double tau[kNU] = {0};
    if (gff != 0.0) {
      mju_copy(gd->qpos, sd->qpos, nq);
      mju_zero(gd->qvel, nv);
      mj_forward(g_model, gd);
      for (int i = 0; i < kNU; i++) tau[i] = gff * gd->qfrc_bias[6 + i];
    }

    // ---- live strategy-switch blend: on a pending switch, snapshot the CURRENT measured pose
    //      as the blend start + stamp the switch time, so the target eases from where the robot
    //      IS into the new policy target (no snap). Serves stdin switches (strategy 18's internal
    //      stand<->crouch cycling is handled by the task phase machine, not here).
    if (g_switch_pending.exchange(false)) {
      for (int i = 0; i < kNU; i++) g_switch_from[i] = cur.q[i];
      g_switch_wall.store(wall);
    }

    // per-joint commanded position target with the BRING-UP ramp: blend from the measured
    // power-on pose to the home/policy target. start_ramp_sec>0 -> ALL joints (kills the
    // warmup->policy SNAP from off-home starts: legs included); 0 -> legacy arm-only ramp.
    double tgt_q[kNU];
    for (int i = 0; i < kNU; i++) {
      double ramp_dur = (start_ramp_sec > 0.0) ? ramp_eff
                                               : (i >= 13 ? arm_ramp_sec : 0.0);
      double base;
      if (ramp_dur > 0.0 && arm_init_set) {
        double aa = std::fmin(1.0, std::fmax(0.0, wall / ramp_dur));
        // SCRIPTED rise: target the stance for the whole ramp (policy steering a half-
        // risen, moving robot topples it -- bench-proven), then HOLD the stance scripted
        // for ramp_hold_sec more so CEM converges around the STATIC operating pose
        // before getting authority (kills the hand-off "first tug" toward the old basin).
        const double t_ho = ramp_dur + absl::GetFlag(FLAGS_ramp_hold_sec);
        const double pblend = absl::GetFlag(FLAGS_policy_blend_sec);
        double tgt;
        if (aa < 1.0 || warming || i >= nact || wall < t_ho) {
          tgt = home_q[i];                       // rising / warmup / non-policy joint / scripted hold
        } else if (pblend > 0.0 && wall < t_ho + pblend) {
          // POLICY-BLEND: ease the scripted stance -> the LIVE policy target over pblend so a
          // cold-started non-stand task descends smoothly instead of snapping at the handoff.
          double bb = (wall - t_ho) / pblend;    // 0..1
          tgt = (1.0 - bb) * home_q[i] + bb * action[i];
        } else {
          // full policy authority -- UNLESS a LIVE switch just armed a blend. Mirror the proven
          // cold-start path (rise -> HOLD while CEM converges -> blend): first hold the pre-switch
          // pose for switch_settle_sec so the planner re-converges on the NEW strategy, THEN ease
          // into its target over g_switch_blend. Without the settle the robot descends mid-replan
          // and collapses past the target (the Squatter "snap": cmd blended to 0.63 but q hit 0.99).
          double sw_dt = wall - g_switch_wall.load();
          double settle = absl::GetFlag(FLAGS_switch_settle_sec);
          double sbl = g_switch_blend.load();
          if (sw_dt >= 0.0 && sw_dt < settle) {
            tgt = g_switch_from[i];              // SETTLE: hold pose, let the planner converge
          } else if (sbl > 0.0 && sw_dt >= settle && sw_dt < settle + sbl) {
            double sb = (sw_dt - settle) / sbl;  // 0..1
            tgt = (1.0 - sb) * g_switch_from[i] + sb * action[i];
          } else {
            tgt = action[i];                     // full policy authority
          }
        }
        base = (1.0 - aa) * arm_q_init[i] + aa * tgt;
      } else {
        base = (warming || i >= nact) ? cur.q[i] : action[i];
      }
      tgt_q[i] = base;
    }

    // torque-aware target clamp (--target_clamp): see TAU_ESTOP. Applied to the FINAL
    // target (ramp + policy uniformly), and last_cmd_q below stores the clamped value so
    // the latency predictor models what was actually sent.
    if (absl::GetFlag(FLAGS_target_clamp)) {
      for (int i = 0; i < kNU; i++) {
        double dmax = 0.9 * TAU_ESTOP[i] / KP[i];
        double lo = cur.q[i] - dmax, hi = cur.q[i] + dmax;
        if (tgt_q[i] < lo) tgt_q[i] = lo;
        else if (tgt_q[i] > hi) tgt_q[i] = hi;
      }
    }

    // ankle-roll zero-offset: shift the COMMAND by the calibration so the physical joint reaches
    // the planner's intended (corrected) angle. Pairs with the belief correction in fill_state;
    // applied to tgt_q so mc.q() AND last_cmd_q (latency model) stay consistent. 0 = no-op.
    tgt_q[5]  += absl::GetFlag(FLAGS_ankle_roll_offset_l_deg) * M_PI / 180.0;
    tgt_q[11] += absl::GetFlag(FLAGS_ankle_roll_offset_r_deg) * M_PI / 180.0;
    // SAFETY: the offset is applied after the torque clamp, so hard-clamp the ankle_roll command
    // to the joint's ctrl range (+-0.26 rad == +-14.9 deg) -> the offset can NEVER drive past limit.
    tgt_q[5]  = std::fmin(0.26, std::fmax(-0.26, tgt_q[5]));
    tgt_q[11] = std::fmin(0.26, std::fmax(-0.26, tgt_q[11]));

    // build unitree_hg LowCmd_
    LowCmd cmd{};
    cmd.mode_pr() = 0;                       // PR mode
    cmd.mode_machine() = cur.mode_machine;   // echo (required by the real robot)
    for (int i = 0; i < kNU; i++) {
      auto& mc = cmd.motor_cmd().at(i);
      mc.mode() = 1;  // 1 = enable
      mc.q() = static_cast<float>(tgt_q[i]);
      mc.dq() = 0.0f;
      mc.tau() = static_cast<float>(tau[i]);
      mc.kp() = static_cast<float>(KP[i]);
      mc.kd() = static_cast<float>(KV[i]);
    }
    cmd.crc() = Crc32Core(reinterpret_cast<uint32_t*>(&cmd), (sizeof(LowCmd) >> 2) - 1);
    cmd_pub->Write(cmd);

    // latency-comp bookkeeping: this command is the in-flight target during the NEXT prediction
    // window; and (AUTO mode) fold this tick's compute time tick-start->post-write into the EWMA Δ.
    for (int i = 0; i < kNU; i++)
      last_cmd_q[i] = tgt_q[i];
    if (lat_comp && lat_fixed <= 0.0) {
      double comp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_tick).count();
      ewma_comp = 0.9 * ewma_comp + 0.1 * comp;
    }

    // ---- per-tick metrics (B0 baseline; balance every tick, accumulate in policy) ----
    // use the MEASURED snapshot taken before prediction (sd may now hold the predicted state).
    double base_z = meas_base_z;
    double tilt = std::acos(std::fmax(-1.0, std::fmin(1.0, meas_R8))) * 57.29577951308232;
    if (!warming) {
      m_ticks++;
      m_z_sum += base_z;
      m_z_sq += base_z * base_z;
      if (base_z < m_z_min) m_z_min = base_z;
      m_tilt_sum += tilt;
      if (tilt > m_tilt_max) m_tilt_max = tilt;
      for (int i = 0; i < kNU; i++) {
        double cmd_q = tgt_q[i];    // the actual ramped command the twin/estop sees
        double err = cmd_q - cur.q[i];                              // commanded - measured
        double ae = std::fabs(err);
        m_err_sum[i] += ae;
        m_err_sq[i] += err * err;
        if (ae > m_err_max[i]) m_err_max[i] = ae;
        // torque the onboard/twin PD actually applies (matches SimInterface law)
        double total_tau = tau[i] + KP[i] * err + KV[i] * (0.0 - cur.dq[i]);
        double at = std::fabs(total_tau);
        m_tau_sum[i] += at;
        if (at > m_tau_max[i]) m_tau_max[i] = at;
        if (at > TAU_LIMIT[i]) {
          m_sat_count[i]++;
          if (!m_sat_warned) {
            std::fprintf(stderr,
                         "[node] WARNING: torque on %s = %.0f Nm > operational H1-2 limit %.0f Nm "
                         "(the safety-layer estop trips BELOW this -> address before deploy)\n",
                         JOINT_NAMES[i], at, TAU_LIMIT[i]);
            m_sat_warned = true;
          }
        }
      }
    }
    if (++ticks % static_cast<long>(ctrl_hz) == 0) {
      static long last_pc = 0; static double last_w = 0.0;
      long pc = plan_count.load();
      double prate = (wall > last_w) ? (pc - last_pc) / (wall - last_w) : 0.0;
      last_pc = pc; last_w = wall;
      std::fprintf(stderr,
                   "[node] t=%5.1fs(twin=%5.1f) %s base_z=%.3f tilt=%4.1fdeg knee(L/R)=%+.2f/%+.2f  "
                   "plan=%.0f/s lat=%.0fms\n",
                   wall, twin_time, warming ? "WARMUP-hold" : "policy", base_z, tilt,
                   meas_kneeL, meas_kneeR, prate, dlt * 1e3);
    }

    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(ctrl_dt));
    std::this_thread::sleep_until(next);
  }

  std::fprintf(stderr, "[node] shutting down ...\n");
  // ---- Title-5 baseline (B0) summary ----
  if (m_ticks > 0) {
    double inv = 1.0 / m_ticks;
    double zmean = m_z_sum * inv;
    double zsd = std::sqrt(std::fmax(0.0, m_z_sq * inv - zmean * zmean));
    std::fprintf(stderr,
                 "\n[B0] ===== baseline strategy=%d over %ld policy ticks (%.1fs @ %.0fHz) =====\n",
                 absl::GetFlag(FLAGS_strategy), m_ticks, m_ticks * ctrl_dt, ctrl_hz);
    std::fprintf(stderr,
                 "[B0] base_z mean=%.3f sd=%.3f min=%.3f m | tilt mean=%.1f max=%.1f deg\n",
                 zmean, zsd, m_z_min, m_tilt_sum * inv, m_tilt_max);
    std::fprintf(stderr,
                 "[B0] joint   trackRMSE trackMax |tau|mean |tau|peak  limit  peak%%  sat%%\n");
    std::fprintf(stderr,
                 "[B0]          (deg)    (deg)     (Nm)     (Nm)      (Nm)\n");
    int n_over = 0;
    for (int i = 0; i < kNU; i++) {
      double pk = m_tau_max[i], lim = TAU_LIMIT[i];
      double pkpct = 100.0 * pk / lim, satpct = 100.0 * m_sat_count[i] * inv;
      if (pk > lim) n_over++;
      std::fprintf(stderr, "[B0] %-7s %8.2f %8.2f %8.1f %9.1f %6.0f %6.0f %5.1f%s\n",
                   JOINT_NAMES[i], std::sqrt(m_err_sq[i] * inv) * 57.29577951308232,
                   m_err_max[i] * 57.29577951308232, m_tau_sum[i] * inv, pk, lim, pkpct, satpct,
                   pk > lim ? "  <<OVER-LIMIT" : "");
    }
    std::fprintf(stderr,
                 "[B0] torque headroom: %d/%d joints exceed the real H1-2 limit%s\n",
                 n_over, kNU,
                 n_over == 0 ? " -> torque-safe for hardware" : " <-- ADDRESS before deploy");
    std::fprintf(stderr, "[B0] (run the SAME node on the real robot -> per-row sim2real delta)\n");
  }
#ifdef H12_NODE_GRPC
  if (grpc_server) grpc_server->Shutdown();
#endif
  plan_exit.store(true);
  if (planner.joinable()) planner.join();
  mj_deleteData(gd);
  mj_deleteData(sd);
  mj_deleteData(pdat);
  mj_deleteData(data);
  mj_deleteModel(g_model);
  return 0;
}
