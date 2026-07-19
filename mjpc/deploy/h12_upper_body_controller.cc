// MJPC embedded DDS control node for the Unitree H1-2 -- UPPER-BODY variant:
// nu=15 (torso + 2x7 arm joints, motor rows 12..26), task "Upper H12 Magpie",
// publishes the safety layer's split-mode UPPER channel rt/safety/lowcmd_upper_in.
// This is the sampling-MPC replacement for the FrameTask IK: joint-space goals
// arrive over the node's gRPC monitor port (SetTaskParameters "Goal Active"/
// "Goal Sec"/"Goal J0..J14" -- the same seam the twin precision gate used), the
// task transports along a min-jerk segment, and a sticky quiescence hold-latch
// freezes the executed command at the goal pose for mm-class held precision
// (P6.2 twin gate: EE bias 13 mm, held wobble 0.22 mm std). The legs are OWNED
// by the lower-body controller (rt/safety/lowcmd_lower_in); with --leg_aware
// (default) this node's planner model tracks the MEASURED legs + base via
// retargeted equality locks and the pelvis world-weld, so the arm plan is
// computed against where the body actually is. This node NEVER commands the
// legs (rows 0..11 of its LowCmd are left default; the safety layer's split
// merge reads only rows 12..26 from this channel).
//
// The shared implementation lives in deploy_common.cc (this file is a thin main:
// per-node gain tables + the surviving CLI flags -> NodeConfig -> RunDeployNode).
// Gains: the h1_2_modified actuator-class values (torso 200/5, arms kp 40,
// kd 10x4 + 2x3 per arm) == the generated planner model == the twin PD, so
// PatchActuators is numerically a no-op and the deployed planner model is
// byte-identical to the P6.2 gate-tested one. The generated model's arm
// forceranges are ALREADY estop-bounded (32/14.4/9.5 Nm) -> no frc_limit patch.

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

ABSL_FLAG(std::string, task, "Upper H12 Magpie",
          "MJPC task id (upper-body nu=15 joint-goal task)");
ABSL_FLAG(double, gravity_ff, 0.85,
          "joint gravity feedforward scale (tau = scale * qfrc_bias); 0 disables. "
          "REAL robot: 0.85. TWIN bench: 0 (the twin's gravcomp over-lightens at 0.85).");
ABSL_FLAG(double, twin_dt, 0.005,
          "plant tick period (s) -- MUST equal 1/lowstate_hz of the plant. 500 Hz twin -> "
          "0.002; 200 Hz twin -> 0.005; REAL robot (1 kHz tick) -> 0.001.");
ABSL_FLAG(std::string, sportstate_topic, "rt/sportmodestate",
          "DDS SportModeState input topic (IMU-site world pose; feeds the pelvis "
          "world-weld retarget). Pass rt/sportmodestate_est on the base estimator.");
ABSL_FLAG(double, imu_pitch_offset_deg, 1.6,
          "IMU pitch zero-offset CALIBRATION (deg) -- 1.6 = the measured offset for THIS H1-2 "
          "(same as the lower node; both nodes must agree on the believed base pose). "
          "Pass 0 for the TWIN.");
ABSL_FLAG(double, imu_roll_offset_deg, 0.0,
          "IMU roll zero-offset calibration (deg), same idea as --imu_pitch_offset_deg.");
ABSL_FLAG(double, bad_orient_rad, 0.9,
          "bad-orientation damp fallback (rad): base tilt beyond this latches a permanent "
          "kp=0/kd=3/tau=0 damp command on the ARM rows until restart (the robot is falling; "
          "stop driving the arms). 0 disables. Default 0.9 rad (~52 deg), matching the lower node.");
ABSL_FLAG(std::string, network_interface, "",
          "DDS network interface (empty = auto-pin the 192.168.123.x robot NIC when present, "
          "else autodetermine/loopback for the twin)");
ABSL_FLAG(int, domain_id, DefaultDomainId(),
          "DDS domain id (default: $ROS_DOMAIN_ID if set, else 0)");
ABSL_FLAG(int, grpc_port, 10001,
          "if >0, host an MJPC gRPC server on this port: the monitor can attach, AND this is "
          "the GOAL-INGEST seam (SetTaskParameters Goal Active/Sec/J0..J14). DEFAULT 10001 so "
          "a co-run with the lower node (10000) never collides; 0 disables (no goals!).");
ABSL_FLAG(int, plan_trajectories, 0,
          "override sampling_trajectories (rollouts per plan) AFTER load. 0 = task default.");
ABSL_FLAG(int, plan_threads, 0,
          "override the planner ThreadPool size. 0 = AUTO = hw_threads - 6. For a co-run "
          "with the lower node on one PC, split the cores (e.g. 6/6) -- see the P6.4 gate.");
ABSL_FLAG(bool, straighten_start, false,
          "hold the measured (slumped) pose, wait for ENTER, then hand authority to the planner "
          "from the slump (SETTLE->BLEND, no drag). OFF by default.");
ABSL_FLAG(bool, cost, false,
          "dump the per-term cost breakdown to stderr once/sec (debug). OFF by default -- the "
          "concise [node] status line is unaffected.");
ABSL_FLAG(double, stale_sec, 0.05,
          "H1 stale-input watchdog threshold (s): either state stream older than this -> "
          "damping safe-hold. 0.05 = the REAL-robot value; loosen only for heavyweight sims.");
ABSL_FLAG(double, latency_rtf, 1.0,
          "latency-comp sim-time scale = measured real-time factor (1.0 = real/twin identity).");
ABSL_FLAG(bool, leg_aware, true,
          "LEG-AWARE planning: read the 12 leg joints (motor 0..11) from rt/lowstate and "
          "retarget the planner model's leg equality locks + pelvis world-weld to the MEASURED "
          "pose each tick, so arm plans are computed against where the body ACTUALLY is (the "
          "lower-body MPC owns balance; this node never commands the legs). --noleg_aware = "
          "plan against the frozen stand keyframe (bench isolation only).");

namespace {
constexpr int kNU = 15;  // UPPER-BODY controller: torso + left arm (7) + right
                         // arm (7) = motor rows 12..26 == the safety layer's
                         // upper split [12:27]. Legs are owned by the lower
                         // node; the planner model equality-locks them at the
                         // measured pose (leg_aware) and welds the pelvis.
// Per-joint gains == h1_2_modified actuator classes == the generated planner
// model == the twin PD (H12_KP/KV rows 12..26). PatchActuators re-writes these
// into the planner model -> numerically a NO-OP, keeping the deployed planner
// byte-identical to the P6.2 precision-gate model. (The FrameTask's stiffer
// debug.yaml gains -- shoulder 200-240 -- are a real-robot A/B lever, NOT the
// default: changing kp changes the gate-tested plant model.)
const double KP[kNU] = {200,  40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 40};
const double KV[kNU] = {5,    10, 10, 10, 10, 2, 2, 2,
                              10, 10, 10, 10, 2, 2, 2};
// SAFETY-LAYER TAU-ESTOP thresholds (estop torque_ratio x URDF torque limit),
// rows 12..26 of the full-body node's table -- MUST equal the safety layer's
// estop table. Feeds the over-budget telemetry (ticks where |tau_ff + KP*(tgt-q)
// + KV*dq| > 0.9x these); emitted torque is NOT clamped -- the estop is the
// backstop.
const double TAU_ESTOP[kNU] = {40,  32, 32, 14.4, 14.4, 9.5, 9.5, 9.5,
                                    32, 32, 14.4, 14.4, 9.5, 9.5, 9.5};
// OPERATIONAL H1-2 joint torque limits (Nm) = Unitree URDF actuatorfrcrange
// (B0 report basis; the estop trips at TAU_ESTOP, below it).
const double TAU_LIMIT[kNU] = {200,  40, 40, 18, 18, 19, 19, 19,
                                     40, 40, 18, 18, 19, 19, 19};
// NO forcerange patch: the generated upper model's arm forceranges are already
// estop-bounded (32/14.4/9.5) -- exactly what the precision gate ran with.
const char* const JOINT_NAMES[kNU] = {
    "torso",
    "LshP", "LshR", "LshY", "Lelb", "LwrR", "LwrP", "LwrY",
    "RshP", "RshR", "RshY", "Relb", "RwrR", "RwrP", "RwrY"};
}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  h12deploy::NodeConfig cfg;
  cfg.nu = kNU;
  cfg.kp = KP;
  cfg.kv = KV;
  cfg.tau_estop = TAU_ESTOP;
  cfg.tau_limit = TAU_LIMIT;
  cfg.frc_limit = nullptr;    // generated model already estop-bounded (see above)
  cfg.frc_limit_begin = 0;
  cfg.joint_names = JOINT_NAMES;
  cfg.telemetry = h12deploy::Telemetry::kUpperBody;
  cfg.motor_offset = 12;      // actuator i <-> motor 12+i (safety upper split rows)
  cfg.upper_count = 12;       // complement = the 12 LEGS ...
  cfg.comp_motor_offset = 0;  // ... at motor rows 0..11 (leg_aware)

  cfg.task_id = absl::GetFlag(FLAGS_task);
  cfg.strategy = 0;           // the upper task has no Strategy parameter (inert)
  cfg.gravity_ff = absl::GetFlag(FLAGS_gravity_ff);
  cfg.twin_dt = absl::GetFlag(FLAGS_twin_dt);
  cfg.lowcmd_topic = "rt/safety/lowcmd_upper_in";   // safety layer split-mode UPPER channel
  cfg.sportstate_topic = absl::GetFlag(FLAGS_sportstate_topic);
  cfg.imu_pitch_offset_deg = absl::GetFlag(FLAGS_imu_pitch_offset_deg);
  cfg.bad_orient_rad = absl::GetFlag(FLAGS_bad_orient_rad);
  cfg.imu_roll_offset_deg = absl::GetFlag(FLAGS_imu_roll_offset_deg);
  cfg.network_interface = absl::GetFlag(FLAGS_network_interface);
  cfg.domain_id = absl::GetFlag(FLAGS_domain_id);
  cfg.grpc_port = absl::GetFlag(FLAGS_grpc_port);
  cfg.arm_aware = absl::GetFlag(FLAGS_leg_aware);   // X-aware machinery, LEG flavor here
  cfg.plan_trajectories = absl::GetFlag(FLAGS_plan_trajectories);
  cfg.plan_threads = absl::GetFlag(FLAGS_plan_threads);
  cfg.cost_log = absl::GetFlag(FLAGS_cost);
  cfg.straighten_start = absl::GetFlag(FLAGS_straighten_start);
  cfg.stale_sec = absl::GetFlag(FLAGS_stale_sec);
  cfg.latency_rtf = absl::GetFlag(FLAGS_latency_rtf);
  return h12deploy::RunDeployNode(cfg);
}
