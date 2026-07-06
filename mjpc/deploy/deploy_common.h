// Shared core of the H1-2 MJPC deploy nodes (h12_control_node = full-body nu=27,
// h12_lower_body_controller = legs-only nu=12). Everything that used to be
// duplicated line-for-line across the two .cc files lives here once: DDS I/O
// (unitree_sdk2), the embedded mjpc::Agent + async planner thread, state
// reconstruction (IMU-site -> pelvis), latency compensation, bring-up ramp /
// policy blend / live strategy switch, the safety clamps, telemetry, and the
// optional gRPC monitor service. The two remaining .cc files are thin mains:
// per-node gain tables + the surviving CLI flags -> NodeConfig -> RunDeployNode.
//
// FLAG DIET (2026-07-02, HAMS_integration): the ~20 flags that were passed
// IDENTICALLY in every documented invocation (Command_Sheet_h12.html A1/A2/B1/
// B2/B-Stabilize, run_realchain.sh, tab launchers) are now compiled-in
// constants below; the flags that provably differ between plants/missions
// survive in the thin mains (task, strategy, gravity_ff, twin_dt,
// sportstate_topic, IMU/ankle calibration, network_interface, domain_id,
// grpc_port, arm_aware). Deleted dead paths: --sync_plan / --plan_rate_hz (all
// runs used the async planner thread), --arm_ramp_sec (only live when
// start_ramp_sec==0, which never happened), --require_sportstate=false debug
// mode (the base estimator / OptiTrack always publishes sportmodestate now),
// --execute_best (never used).
// NO TUNED VALUE CHANGED: only how it is supplied (compiled default vs CLI).
// RE-ADDED 2026-07-03: --plan_trajectories + --plan_threads. Cutting them was
// a diet mistake -- they are the R2 plan-rate sweep levers on REAL hardware
// (samples-per-plan vs replan-rate: trot's 36 traj @ 12 threads plans only
// ~28 Hz = the 06-29 starvation diagnosis; one-wave rule: traj <= threads).
// Both default 0 = the compiled/task value, so a bare invocation is unchanged.
#ifndef MJPC_DEPLOY_DEPLOY_COMMON_H_
#define MJPC_DEPLOY_DEPLOY_COMMON_H_

#include <string>

namespace h12deploy {

// hard upper bound on actuated joints across all node variants (full-body 27)
inline constexpr int kMaxNU = 27;

// ---- SETTLED VALUES (identical in every documented run; see flag diet above) ----
inline constexpr double kCtrlHz = 200.0;          // control / publish rate
inline constexpr double kWarmupSec = 1.0;         // hold measured pose while planner converges
inline constexpr double kStartRampSec = 5.0;      // all-joint measured->stance bring-up ramp
inline constexpr double kRampHoldSec = 3.0;       // scripted hold after ramp (CEM converges)
inline constexpr double kPolicyBlendSec = 4.5;    // ease scripted stance -> live policy target
inline constexpr double kSwitchSettleSec = 2.0;   // hold pose after a live strategy switch
inline constexpr bool   kUseTwinTime = true;      // planner clock = lowstate tick * twin_dt
inline constexpr int    kPlanThreads = 12;        // planner ThreadPool (leaves cores for twin/safety)
inline constexpr bool   kLatencyComp = true;      // predict-forward by the measured loop delay
inline constexpr double kLatencyFixedMs = 0.0;    // 0 = AUTO (EWMA of compute time + extra)
inline constexpr double kLatencyExtraMs = 4.0;    // transport + plant zero-order-hold (AUTO mode)
inline constexpr double kLatencyMaxMs = 40.0;     // hard cap on the predicted-forward horizon
inline constexpr double kVelLpfMs = 30.0;         // single-stream base-linvel finite-diff LPF
inline constexpr double kClampRatio = 0.9;        // torque budget clamp: 0.9 x TAU_ESTOP (audit H2)
inline constexpr double kStaleSec = 0.05;         // H1 watchdog: state older than this -> safe-hold
inline constexpr float  kSafeHoldKd = 2.0f;       // damping-stop kd on safe-hold (kp=0, tau=0)
// IMU site position in the pelvis (free-joint) frame, from h1_2_handless.xml.
// Must match the base estimator's IMU_OFFSET exactly (both ends of the
// site->pelvis reconstruction agree or the base pose is wrong).
inline constexpr double kImuOffset[3] = {-0.04452, -0.01891, 0.27756};

// Which per-second telemetry block the status line prints (the ONLY cosmetic
// divergence between the two nodes: shoulders for the full-body node, ankle
// pitch + knees for the legs-only node).
enum class Telemetry { kFullBody, kLowerBody };

struct NodeConfig {
  // ---- per-node compile-time tables (point at static arrays in the main) ----
  int nu = 0;                          // 27 full-body, 12 legs-only
  const double* kp = nullptr;          // [nu] onboard PD kp == planner actuator gains
  const double* kv = nullptr;          // [nu]
  const double* tau_estop = nullptr;   // [nu] safety-layer estop thresholds (H2 clamp basis)
  const double* tau_limit = nullptr;   // [nu] operational URDF limits (B0 report basis)
  const double* frc_limit = nullptr;   // [nu] planner forcerange patch, or nullptr = none
  int frc_limit_begin = 0;             // first index to patch (13 = arms only, full-body node)
  const char* const* joint_names = nullptr;  // [nu]
  Telemetry telemetry = Telemetry::kFullBody;
  int upper_count = 0;                 // 15 = read torso+arms from lowstate (legs-only node)

  // ---- surviving CLI flags (differ between plants / missions) ----
  std::string task_id;                 // MJPC task name
  int strategy = 6;
  double gravity_ff = 0.85;            // 0 on the twin (gravcomp over-lightens), 0.85 real
  double twin_dt = 0.005;              // 1/lowstate_hz of the plant (real 1 kHz tick -> 0.001)
  std::string lowcmd_topic;            // per-node default (full vs split safety channel)
  std::string lowstate_topic = "rt/lowstate";
  std::string sportstate_topic;        // truth vs rt/sportmodestate_est (estimator-in-loop)
  double imu_pitch_offset_deg = 0.0;   // real H1-2 mount calib 1.6; twin 0
  double imu_roll_offset_deg = 0.0;
  // R6 (2026-07-04): bad-orientation damp fallback (Unitree deploy parity:
  // bad_orientation(1.0 rad) -> Passive FSM). base tilt > this [rad] latches
  // a permanent kp=0 / kd-damp / tau=0 command until restart, so the node
  // never thrashes an unrecoverable fall. 0 = disabled (default; lean
  // full-body deployments byte-unchanged).
  double bad_orient_rad = 0.0;
  double ankle_roll_offset_l_deg = 0.0;
  double ankle_roll_offset_r_deg = 0.0;
  std::string network_interface;       // "" = auto-pin 192.168.123.x, else autodetermine
  int domain_id = 0;                   // default read from $ROS_DOMAIN_ID in the mains
  int grpc_port = 10000;               // monitor server; 0 disables
  bool arm_aware = false;              // legs-only node: retarget eq locks to measured arms
  int plan_trajectories = 0;           // >0 overrides sampling_trajectories AFTER the
                                       // per-strategy PlannerNumericOverrides; 0 = task default
  int plan_threads = 0;                // >0 overrides kPlanThreads(12); 0 = compiled default
  double stale_sec = kStaleSec;        // H1 watchdog threshold; default = the REAL-robot 50ms.
                                       // Loosen ONLY for sims whose lowstate publisher stalls
                                       // on a shared sim lock (RoboCasa sensor renders: 50-60ms
                                       // gaps -> permanent safe-hold at the default).
};

// Runs the full deploy node (blocks until SIGINT/SIGTERM/q). Returns exit code.
int RunDeployNode(const NodeConfig& cfg);

}  // namespace h12deploy

#endif  // MJPC_DEPLOY_DEPLOY_COMMON_H_
