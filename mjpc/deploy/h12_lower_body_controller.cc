// MJPC embedded DDS control node for the Unitree H1-2 -- LOWER-BODY variant:
// nu=12 (legs only), task "Stabilize H12 Magpie", publishes the safety layer's
// split-mode LOWER channel rt/safety/lowcmd_lower_in. Torso + arms are owned by
// the upper-body FrameTask IK (rt/safety/lowcmd_upper_in); this node balances
// UNDER whatever the arms do -- with --arm_aware (default) the planner's
// CoM/dynamics track the MEASURED upper joints via retargeted equality locks,
// so the legs PRE-compensate for a reach instead of only reacting to base tilt.
//
// The shared implementation lives in deploy_common.cc (this file is a thin main:
// per-node gain tables + the surviving CLI flags -> NodeConfig -> RunDeployNode).
// Settled values (ctrl_hz 200, twin-time clocking, latency comp, plan threads 12,
// ramp/hold/blend timings, torque-budget clamp, ...) are compiled-in constants --
// see deploy_common.h for the full flag-diet rationale.

#include <cstdlib>
#include <string>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "mjpc/deploy/deploy_common.h"

namespace {
// default DDS domain: follow the ROS 2 convention used by HAMS ($ROS_DOMAIN_ID
// must match across the rclpy/DDS halves or they silently diverge).
int DefaultDomainId() {
  const char* e = std::getenv("ROS_DOMAIN_ID");
  return e ? std::atoi(e) : 0;
}
}  // namespace

ABSL_FLAG(std::string, task, "Stabilize H12 Magpie",
          "MJPC task id (lower-body nu=12 stabilize task)");
ABSL_FLAG(int, strategy, 6,
          "Stabilize Strategy parameter (6=stand; 0-5=placeholders). The stabilize "
          "task is lower-body-only, so the lean reach/lean/crouch slots are absent.");
ABSL_FLAG(double, gravity_ff, 0.85,
          "joint gravity feedforward scale (tau = scale * qfrc_bias); 0 disables. "
          "REAL robot: 0.85. TWIN bench: 0 (the twin's gravcomp over-lightens the legs "
          "at 0.85 -> knee-strut collapse).");
ABSL_FLAG(double, twin_dt, 0.005,
          "plant tick period (s) -- MUST equal 1/lowstate_hz of the plant. The planner is "
          "clocked by lowstate tick * twin_dt (twin-time, the settled deploy mode) and the "
          "single-stream velocity finite-diff uses the same clock. 500 Hz twin -> 0.002; "
          "200 Hz twin -> 0.005; REAL robot (1 kHz tick) -> 0.001.");
ABSL_FLAG(std::string, sportstate_topic, "rt/sportmodestate",
          "DDS SportModeState input topic (IMU-site world pose). Pass rt/sportmodestate_est "
          "to run on the base estimator's output instead of plant truth (A2 rung).");
ABSL_FLAG(double, imu_pitch_offset_deg, 1.6,
          "IMU pitch zero-offset CALIBRATION (deg) -- DEFAULT 1.6 = the MEASURED offset for THIS "
          "H1-2 (body verified truly vertical reads -1.6 deg on the IMU; +1.6 zeroes it). The "
          "perceived base orientation is rotated by this about the body pitch axis before planning. "
          "Validated on real (most-upright stand to date). Pass --imu_pitch_offset_deg 0 for the "
          "TWIN (no IMU mounting offset there) or to A/B against the raw IMU.");
ABSL_FLAG(double, bad_orient_rad, 0.9,
          "R6 bad-orientation damp fallback (rad). Base tilt beyond this latches a permanent "
          "kp=0/kd=3/tau=0 damp command until restart (Unitree deploy parity: bad_orientation(1.0) "
          "-> Passive) so the planner never thrashes an unrecoverable fall. 0 disables. "
          "Default 0.9 rad (~52 deg) = engaged well past the 35 deg fall verdict.");
ABSL_FLAG(double, imu_roll_offset_deg, 0.0,
          "IMU roll zero-offset calibration (deg), same idea as --imu_pitch_offset_deg (body roll axis).");
ABSL_FLAG(double, ankle_roll_offset_l_deg, 0.0,
          "LEFT ankle-roll zero-offset calibration (deg): SUBTRACTED from the perceived roll AND "
          "ADDED to the command, so a foot the encoder reports rolled is both reasoned about and "
          "driven to its true angle. 0 = off. Dial to flatten an edge-rolled foot (free-hang L/R asym).");
ABSL_FLAG(double, ankle_roll_offset_r_deg, 0.0,
          "RIGHT ankle-roll zero-offset calibration (deg); see --ankle_roll_offset_l_deg. The "
          "free-hang showed the RIGHT foot resting ~6 deg more rolled (edge) than the left.");
ABSL_FLAG(std::string, network_interface, "",
          "DDS network interface (empty = auto-pin the 192.168.123.x robot NIC when present, "
          "else autodetermine/loopback for the twin)");
ABSL_FLAG(int, domain_id, DefaultDomainId(),
          "DDS domain id (default: $ROS_DOMAIN_ID if set, else 0)");
ABSL_FLAG(int, grpc_port, 10000,
          "if >0, host an MJPC gRPC server on this port so the monitor can attach "
          "(view state + switch Strategy live); 0 disables");
ABSL_FLAG(int, plan_trajectories, 0,
          "override sampling_trajectories (rollouts per plan) AFTER the per-strategy "
          "PlannerNumericOverrides. 0 = task default. Real-hardware plan-rate sweep lever "
          "(samples-per-plan vs replan-rate); see h12_control_node --help for the full story.");
ABSL_FLAG(int, plan_threads, 0,
          "override the planner ThreadPool size. 0 = compiled kPlanThreads (12). One-wave "
          "rule: plan_trajectories <= plan_threads maximizes replan rate.");
ABSL_FLAG(double, stale_sec, 0.05,
          "H1 stale-input watchdog threshold (s): either state stream older than this -> "
          "damping safe-hold. DEFAULT 0.05 = the REAL-robot value (1 kHz lowstate). Loosen "
          "ONLY for heavyweight sims whose lowstate publisher stalls on a shared sim lock "
          "(RoboCasa's sensor renders hold it 50-60 ms -> permanent safe-hold at 0.05; "
          "0.15 rides the stalls out while still catching a genuinely dead stream).");
ABSL_FLAG(bool, arm_aware, true,
          "ARM-AWARE balance (loco-manip): read the 15 upper-body joints (torso+arms, motor 12..26) "
          "from rt/lowstate and feed them to the planner so its CoM/dynamics model tracks where the "
          "arms ACTUALLY are -- the legs then PRE-compensate for a reach instead of only reacting to "
          "the resulting base tilt. The planner STILL actuates only the 12 legs (nu=12); the upper "
          "joints are held at the MEASURED pose during each rollout via the equality locks (retargeted "
          "live), so they don't snap to home. This node NEVER commands the arms (the FrameTask IK owns "
          "rt/safety/lowcmd_upper_in). No-op when the arms are at home (measured==0). DEFAULT TRUE "
          "(2026-07-02: strictly better on the F6 bench -- arm45 ankle margin 2x; also feeds the "
          "task-side arm_aware_plan retarget). --noarm_aware = legacy home-locked isolation mode.");

namespace {
constexpr int kNU = 12;  // LEGS-ONLY lower-body controller: the 12 actuated
                         // joints below the pelvis (L/R hip yaw/pitch/roll,
                         // knee, ankle pitch/roll). Torso + arms are owned by
                         // the upper-body IK; the stabilize planner model
                         // equality-locks them at home (nu=12).
// Per-joint gains == h1_2_modified actuator classes == real LowCmd kp/kd.
// (Must match the safety-layer / twin PD.) Patched into the planner model.
const double KP[kNU] = {150, 200, 200, 200, 80, 80,  150, 200, 200, 200, 80, 80};
const double KV[kNU] = {5, 5, 5, 5, 4, 4,  5, 5, 5, 5, 4, 4};
// SAFETY-LAYER TAU-ESTOP thresholds (estop torque_ratio x URDF torque limit).
// Basis of the H2 torque-budget clamp: |tau_ff + KP*(tgt-q) + KV*dq| <= 0.9x these.
const double TAU_ESTOP[kNU] = {60, 130, 200, 300, 54, 36,  60, 130, 200, 300, 54, 36};
// LEG forceranges are LEFT at the model default: tightening them to the estop
// bound clamped hip/ankle balance authority and regressed the hold. Arms are
// not actuated by this lower-body controller (nu=12) -> no forcerange patch.
const char* const JOINT_NAMES[kNU] = {
    "LhipY", "LhipP", "LhipR", "Lknee", "LankP", "LankR",
    "RhipY", "RhipP", "RhipR", "Rknee", "RankP", "RankR"};
// OPERATIONAL H1-2 joint torque limits (Nm) = Unitree URDF actuatorfrcrange.
// The B0 report grades against THIS basis; the estop trips at TAU_ESTOP (below it).
const double TAU_LIMIT[kNU] = {200, 200, 200, 300, 60, 40,
                               200, 200, 200, 300, 60, 40};
// FABEL (2026-07-07, env H12_FRC_PARITY=1): planner-model forceranges clamped
// to the torque the deploy chain can ACTUALLY deliver = kClampRatio (0.9) x
// TAU_ESTOP. The stock model believes ankle +-75 / hipY +-200 / hip +-300 Nm
// while the node's H2 torque-budget clamp caps commands at 48.6 / 54 / 117 --
// so every planned balance catch over-promises (ankle by 54%) and
// under-delivers at execution; the in-process probe (no clamp) stands where
// the chain tips. The old "forcerange tightening regressed the hold" verdict
// predates the ABI fix (PatchActuators wrote through skewed offsets then).
// Unset = stock forceranges (unchanged).
const double FRC_PARITY[kNU] = {54, 117, 180, 270, 48.6, 32.4,
                                54, 117, 180, 270, 48.6, 32.4};
}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  h12deploy::NodeConfig cfg;
  cfg.nu = kNU;
  cfg.kp = KP;
  cfg.kv = KV;
  cfg.tau_estop = TAU_ESTOP;
  cfg.tau_limit = TAU_LIMIT;
  cfg.frc_limit = nullptr;    // no forcerange patch (legs stay at model default)
  cfg.frc_limit_begin = 0;
  if (const char* e = std::getenv("H12_FRC_PARITY"); e && e[0] != '\0') {
    // '1'/estop = 0.9 x TAU_ESTOP (the H2 clamp's default budget);
    // 'urdf'    = TAU_LIMIT (pair with H12_CLAMP_URDF=1 so the clamp budget
    //             matches -- promise == deliverable == URDF).
    cfg.frc_limit = (e[0] == 'u') ? TAU_LIMIT : FRC_PARITY;
    cfg.frc_limit_begin = 0;
    std::fprintf(stderr, "[node] FABEL H12_FRC_PARITY=%s: planner leg "
                         "forceranges = %s\n", e,
                 e[0] == 'u' ? "URDF TAU_LIMIT" : "0.9 x TAU_ESTOP");
  }
  cfg.joint_names = JOINT_NAMES;
  cfg.telemetry = h12deploy::Telemetry::kLowerBody;
  cfg.upper_count = 15;       // read torso+arms (motor 12..26) for arm-aware balance

  cfg.task_id = absl::GetFlag(FLAGS_task);
  cfg.strategy = absl::GetFlag(FLAGS_strategy);
  cfg.gravity_ff = absl::GetFlag(FLAGS_gravity_ff);
  cfg.twin_dt = absl::GetFlag(FLAGS_twin_dt);
  cfg.lowcmd_topic = "rt/safety/lowcmd_lower_in";   // safety layer split-mode LOWER channel
  cfg.sportstate_topic = absl::GetFlag(FLAGS_sportstate_topic);
  cfg.imu_pitch_offset_deg = absl::GetFlag(FLAGS_imu_pitch_offset_deg);
  cfg.bad_orient_rad = absl::GetFlag(FLAGS_bad_orient_rad);
  cfg.imu_roll_offset_deg = absl::GetFlag(FLAGS_imu_roll_offset_deg);
  cfg.ankle_roll_offset_l_deg = absl::GetFlag(FLAGS_ankle_roll_offset_l_deg);
  cfg.ankle_roll_offset_r_deg = absl::GetFlag(FLAGS_ankle_roll_offset_r_deg);
  cfg.network_interface = absl::GetFlag(FLAGS_network_interface);
  cfg.domain_id = absl::GetFlag(FLAGS_domain_id);
  cfg.grpc_port = absl::GetFlag(FLAGS_grpc_port);
  cfg.arm_aware = absl::GetFlag(FLAGS_arm_aware);
  cfg.plan_trajectories = absl::GetFlag(FLAGS_plan_trajectories);
  cfg.plan_threads = absl::GetFlag(FLAGS_plan_threads);
  cfg.stale_sec = absl::GetFlag(FLAGS_stale_sec);
  return h12deploy::RunDeployNode(cfg);
}
