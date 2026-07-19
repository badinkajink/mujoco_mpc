// Shared core of the H1-2 MJPC deploy nodes. Everything that would otherwise
// be duplicated across the node mains lives here once: DDS I/O (unitree_sdk2),
// the embedded mjpc::Agent + async planner thread, state reconstruction
// (IMU-site -> pelvis), latency compensation, bring-up ramp / policy blend /
// live strategy switch, the safety machinery, telemetry, and the optional gRPC
// monitor service. The FOUR .cc files beside it are thin mains -- per-node gain
// tables + the surviving CLI flags -> NodeConfig -> RunDeployNode:
//   h12_control_node.cc          full-body nu=27  -> colcon mjpc_fullbody_core
//   h12_lower_body_controller.cc legs-only nu=12  -> mjpc_lowerbody_core
//                                (the legacy chain)
//   h12_upper_body_controller.cc upper nu=15 (motors 12..26; fork build only,
//                                no colcon core)
//   h12_split_controller.cc      whole-body nu=27, split lower/upper output
//                                channels -> mjpc_split_core (the core the
//                                chain bringup runs)
//
// FLAG DIET: flags that were passed identically in every documented invocation
// are compiled-in constants below; only the flags that provably differ between
// plants/missions survive in the thin mains. NO TUNED VALUE CHANGED by the
// diet -- only how a value is supplied (compiled default vs CLI).
// --plan_trajectories / --plan_threads stay as CLI flags: they are the
// plan-rate sweep levers on real hardware (one-wave rule: traj + 1 <= threads);
// both default 0 = the compiled/task value, so a bare invocation is unchanged.
// (history: see mjpc/deploy/HISTORY.md)
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
// Planner ThreadPool size. 0 = AUTO = hw_threads - kPlanThreadsReserve (floor
// kPlanThreadsMin). Rationale for AUTO: CEM waits on num_trajectory + 1 rollout
// jobs per iteration, so an undersized pool divides the replan rate into
// thread-WAVES and starves the sampler-owned stepping strategies -- real legged
// sampling-MPC needs 50-100 plans/s. (history: see mjpc/deploy/HISTORY.md)
// Reserve covers: control/publish thread, DDS rx/tx, safety layer, OS.
// BENCH NOTE: when co-running the PYTHON twin on the same box, pass --plan_threads 12 --
// 18 planner threads starve the twin to ~0.5x real-time (measured 2026-07-03).
inline constexpr int    kPlanThreads = 0;         // 0 = AUTO (see kPlanThreadsReserve)
inline constexpr int    kPlanThreadsReserve = 6;  // hw threads held back from the planner
inline constexpr int    kPlanThreadsMin = 4;      // never auto-size below this
inline constexpr bool   kLatencyComp = true;      // predict-forward by the measured loop delay
inline constexpr double kLatencyFixedMs = 0.0;    // 0 = AUTO (EWMA of compute time + extra)
inline constexpr double kLatencyExtraMs = 4.0;    // transport + plant zero-order-hold (AUTO mode)
inline constexpr double kLatencyMaxMs = 40.0;     // hard cap on the predicted-forward horizon
inline constexpr double kVelLpfMs = 30.0;         // single-stream base-linvel finite-diff LPF
inline constexpr double kClampRatio = 0.9;        // 0.9 x TAU_ESTOP: frc_parity forcerange
                                                  // budget + over-budget telemetry (no emit clamp)
inline constexpr double kStaleSec = 0.05;         // H1 watchdog: state older than this -> safe-hold
inline constexpr float  kSafeHoldKd = 2.0f;       // damping-stop kd on safe-hold (kp=0, tau=0)
// IMU site position in the pelvis (free-joint) frame, from h1_2_handless.xml.
// Must match the base estimator's IMU_OFFSET exactly (both ends of the
// site->pelvis reconstruction agree or the base pose is wrong).
inline constexpr double kImuOffset[3] = {-0.04452, -0.01891, 0.27756};

// Which per-second telemetry block the status line prints (the ONLY cosmetic
// divergence between the nodes: shoulders for the full-body node, ankle
// pitch + knees for the legs-only node, torso+shoulders/elbows for the
// upper-body node).
enum class Telemetry { kFullBody, kLowerBody, kUpperBody };

struct NodeConfig {
  // ---- per-node compile-time tables (point at static arrays in the main) ----
  int nu = 0;                          // 27 full-body/split, 12 legs-only, 15 upper-body
  const double* kp = nullptr;          // [nu] onboard PD kp == planner actuator gains
  const double* kv = nullptr;          // [nu]
  const double* tau_estop = nullptr;   // [nu] safety-layer estop thresholds (frc_parity basis)
  const double* tau_limit = nullptr;   // [nu] operational URDF limits (B0 report basis)
  const double* frc_limit = nullptr;   // [nu] planner forcerange patch, or nullptr = none
  int frc_limit_begin = 0;             // first index to patch (13 = arms only, full-body node)
  // ---- ACTUATOR-AUTHORITY PARITY: the planner must not plan with torque the
  // deployment cannot afford. There is NO emit clamp -- the node publishes the
  // raw planner command -- so a plan that demands more than the safety-layer
  // estop budget (kClampRatio * tau_estop) is emitted as-is and can TRIP the
  // estop. frc_parity is the intended mitigation: when ON, every planner
  // forcerange is tightened to that budget (never loosened; see
  // PatchActuators), so the sampler only plans motions the safety layer will
  // tolerate. (history: see mjpc/deploy/HISTORY.md)
  // -1 = task default (the model's `deploy_frc_parity` numeric; no task ships it ON —
  //      the Lean XMLs set it 0 and PlannerNumericOverrides never sets it), 0 = force OFF
  //      (legacy model, byte-identical), 1 = force ON (what split_body_controller.py
  //      passes). Kill switch: --frc_parity=0.
  int frc_parity = -1;
  const char* const* joint_names = nullptr;  // [nu]
  Telemetry telemetry = Telemetry::kFullBody;
  // ---- actuated-block placement (UPPER-BODY node, 2026-07-10; additive) ----
  // Actuated joint i lives at lowstate/lowcmd motor (motor_offset + i) and at
  // planner qpos[7 + motor_offset + i] / dof[6 + motor_offset + i]. 0 = the
  // existing full-body / legs-only nodes (byte-identical); 12 = the upper-body
  // node (torso+arms = motor rows 12..26, the safety layer's upper split).
  int motor_offset = 0;
  // ---- COMPLEMENT joints ("X-aware" balance/eq retarget) ----
  // upper_count joints read from lowstate starting at motor comp_motor_offset
  // and injected into the planner state + equality-lock retargets each tick.
  //   legs-only node:  upper_count=15, comp_motor_offset=12 (ARM-aware: torso+arms)
  //   upper-body node: upper_count=12, comp_motor_offset=0  (LEG-aware: the 12 legs)
  int upper_count = 0;                 // count of complement joints to read (0 = off)
  int comp_motor_offset = 12;          // first complement motor idx (12 = historical arm-aware)

  // ---- surviving CLI flags (differ between plants / missions) ----
  std::string task_id;                 // MJPC task name
  int strategy = 6;
  double gravity_ff = 0.85;            // 0 on the twin (gravcomp over-lightens), 0.85 real
  double twin_dt = 0.005;              // 1/lowstate_hz of the plant (real 1 kHz tick -> 0.001)
  std::string lowcmd_topic;            // per-node default (full vs split safety channel)
  // ---- SPLIT-BODY OUTPUT + UPPER-BODY PAUSE TOGGLE (whole-body split core) ----
  // When set, the node ALSO writes its command to this second channel. The
  // whole-body split core sends legs on lowcmd_topic (the safety split LOWER
  // channel) and arms on upper_lowcmd_topic (the UPPER channel), so the safety
  // split-merge takes rows 0..11 from one and 12..26 from the other -- the same
  // 27-row cmd goes to both, the untouched rows on each are ignored by the merge.
  // "" = single channel (every existing core is byte-unchanged).
  std::string upper_lowcmd_topic;
  // DDS std_msgs/String topic carrying the upper-body pause toggle ("1" = paused;
  // unitree_sdk2 ships no ros2 Bool IDL). Paused => the node STOPS writing
  // upper_lowcmd_topic, so the frame_task IK owns the arms; the legs channel keeps
  // publishing. "" = no subscriber (off; existing cores byte-unchanged).
  std::string pause_upper_topic;
  // INITIAL pause state of the upper channel (only meaningful when
  // pause_upper_topic is set). true = come up PAUSED: the node does not write
  // the upper channel (and, when the model carries the upper eq locks, holds
  // the arms eq-locked to measured in the planner) until a "0" arrives on
  // pause_upper_topic. The dynamic startup handshake needs this: frame_task
  // initializes OWNING the arms, and the split launcher hands them to MJPC by
  // unpausing only once frame_task reports ready -- so the core must not race
  // ahead of the first toggle sample. false = legacy (active at boot).
  bool pause_upper_init = false;
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
  double ankle_pitch_offset_l_deg = 0.0;  // per-ankle PITCH zero calib; same belief/command
  double ankle_pitch_offset_r_deg = 0.0;  // pairing as the roll offsets (H1-2 stores no zero)
  // ---- PHASE-A START-POSE ALIGN (2026-07-13) ----
  // Power-on leg geometry is arbitrary (twisted / fore-aft / one knee folded), and the operator
  // was hand-straightening the legs before every run. With align_start the node does it itself:
  // BEFORE the planner is ever consulted it drags the legs from the measured power-on pose to
  // align_pose on a min-jerk profile, holds until they arrive, then RE-LATCHES the bring-up so
  // MJPC always takes over from the same stance (d0~0 -> the scripted rise collapses to nothing
  // and only the policy blend runs -- exactly the near-home start the full-body node always had).
  // WALL-clocked on purpose: the bring-up choreography runs on plant time (tick*twin_dt), so with
  // --twin_dt 0.001 on this 200 Hz plant it dilates 5x; the align must not inherit that.
  // false = byte-identical to the pre-2026-07-13 node.
  bool align_start = false;
  double align_pose[kMaxNU] = {0};     // target stance (rad, motor order); filled by the node
  bool align_pose_set = false;         // false -> fall back to the model's `stand` keyframe
  double align_sec = 4.0;              // min-jerk drive duration [s, WALL]
  double align_tol = 0.08;             // converged when max|q - align_pose| < this [rad] (~4.6 deg)
  double align_timeout = 8.0;          // hard ceiling [s, WALL]; the robot is load-bearing and
                                       // sags, so convergence is best-effort -- never block on it
  // ANTI-STICTION PUSH. A pure position PD stalls wherever breakaway friction exceeds kp*err:
  // at kp=200 a 3 deg residual is only 10 Nm, which a harmonic drive will happily sit on, so the
  // legs stop short of the stance. align_ki winds an integrator into the COMMANDED target once
  // the min-jerk reference has arrived, so kp*(tgt - q) keeps GROWING until the joint breaks
  // loose. align_i_max (a POSITION bound) is the ONLY limit on the push: there is no torque
  // clamp on the emitted command, so the motor PD torque from the wound-up target CAN reach
  // the safety-layer estop (kp*align_i_max bounds the extra torque any one joint can add).
  // 0 = off (pure PD, may stall short).
  double align_ki = 1.0;               // integral gain [rad/s per rad of error]
  double align_i_max = 0.25;           // windup limit [rad]: the most extra command any one joint
                                       // may accumulate (kp*this = the extra torque it can add;
                                       // e.g. knee kp=200 -> up to +50 Nm of breakaway push)
  bool align_wait = true;              // after the drag, PARK on the stance and hold it until the
                                       // operator presses Enter (stdin). The legs stay stiff and
                                       // the node keeps publishing at 200 Hz while it waits, so
                                       // the planner never inherits a robot that is still being
                                       // handled. Only consulted when align_start is on.
  std::string network_interface;       // "" = auto-pin 192.168.123.x, else autodetermine
  int domain_id = 0;                   // default read from $ROS_DOMAIN_ID in the mains
  int grpc_port = 10000;               // monitor server; 0 disables
  bool arm_aware = false;              // legs-only node: retarget eq locks to measured arms
  int plan_trajectories = 0;           // >0 overrides sampling_trajectories AFTER the
                                       // per-strategy PlannerNumericOverrides; 0 = task default
  int plan_threads = 0;                // >0 overrides the pool size; 0 = compiled kPlanThreads,
                                       // itself 0 = AUTO (hw_threads - kPlanThreadsReserve)
  double stale_sec = kStaleSec;        // H1 watchdog threshold; default = the REAL-robot 50ms.
                                       // Loosen ONLY for sims whose lowstate publisher stalls
                                       // on a shared sim lock (RoboCasa sensor renders: 50-60ms
                                       // gaps -> permanent safe-hold at the default).
  double latency_rtf = 1.0;            // latency-comp sim-time scale = measured real-time factor.
                                       // The predict-forward horizon dlt is a WALL-clock duration
                                       // but predict_forward rolls it as SIM steps; on a below-
                                       // realtime plant (RoboCasa RTF<1) that over-leads by 1/RTF.
                                       // Scaling dlt by RTF converts wall->sim. 1.0 = IDENTITY
                                       // (real/twin run RTF~1 -> byte-unchanged); set to the
                                       // measured RTF on a slow sim (RoboCasa ~0.45).
  // ---- DEBUG PLAN PUBLISH (2026-07-15; additive, off by default) ----
  // Serialize the ACTIVE planner's best trajectory (qpos rows only) to JSON and
  // publish it as std_msgs::msg::dds_::String_ -- the same wire type ROS 2 uses for
  // std_msgs/msg/String, so DDS 'rt/mjpc/plan' surfaces as ROS topic '/mjpc/plan'
  // with no bridge (the mechanism that already makes rt/lowstate visible as
  // /lowstate). Feeds the mjpc_debug_visualizer's blue ghost.
  //
  // WHERE: the PLANNER thread, right after PlanIteration() returns -- the only
  // race-free point. CrossEntropyPlanner::BestTrajectory() hands back
  // &nominal_trajectory WITHOUT taking mtx_, and that buffer is written by a pool
  // worker; but PlanIteration internally WaitCount()s every rollout job including
  // the nominal one, so on return no worker is touching it and the only thread that
  // rewrites it is the one serializing. The 200 Hz control loop is never involved.
  //
  // "" = OFF (default): no publisher constructed, no serialization, ZERO added work
  // -> real-robot deployments byte-unchanged. Wired ON for the sim only.
  std::string plan_pub_topic;          // "" = OFF (default); e.g. "rt/mjpc/plan"
  double plan_pub_hz = 20.0;           // publish rate cap; the planner free-runs at
                                       // 45-52/s and a plan is ~90 KB of JSON, so we
                                       // SKIP iterations to hit this (never sleep the
                                       // planner thread -- that would cost plan rate).
  bool cost_log = false;               // --cost: dump the per-term cost breakdown to stderr
                                       // once/sec. OFF by default; the concise [node] status
                                       // line is unaffected. Debug aid only -- changes no weights.
  bool straighten_start = false;       // --straighten_start: hold the measured (slumped) pose,
                                       // wait for operator ENTER, then hand authority to the
                                       // planner (straighten strat) from the slump via the SAME
                                       // SETTLE->BLEND as align_start. NO drag. Default off ->
                                       // the cold-start + align paths are byte-identical.
};

// Runs the full deploy node (blocks until SIGINT/SIGTERM/q). Returns exit code.
int RunDeployNode(const NodeConfig& cfg);

}  // namespace h12deploy

#endif  // MJPC_DEPLOY_DEPLOY_COMMON_H_
