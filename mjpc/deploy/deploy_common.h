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
// STRAIGHTEN boot (strategy 25) ONLY: the scripted stand-pose rise above is a joint-space
// lerp with NO balance term -- from a slumped/leaning power-on it extends the knees on a
// fixed schedule while the torso pitches wherever physics takes it, and hands the planner a
// LEANED robot ~9s in (REAL 2026-07-11: "knees extended, torso kept leaning forward"; no
// phase-0 weight can fight it because action[] is not even read during the ramp). Straighten's
// whole purpose is to COMPUTE that bring-up trajectory under balance costs, so there the
// planner is the destination from the end of warmup: hold measured -> blend measured->policy
// over this window -> full authority. The phase-0 C3 residual ramp (target_ramp_sec) starts its
// reference AT the measured pose, so the action is near-identity at handover = no snap.
// TWIN-TUNED 2026-07-12 (straighten_basin --deploy-ramp straighten, realboot slump + harness):
// the scripted HOLD is itself the enemy. A slumped pose is NOT statically holdable under the
// deploy PD (~15% gravity droop, proven 07-11), so every extra second the node pins the robot
// on a fixed target it creeps and gains momentum -- the planner then inherits a worse, MOVING
// robot. Measured settle vs hold length (3 reps each, same slump/rope):
//     warmup 1.0s -> 20-36 deg (fails)   0.5s -> 9-18 deg   0.3s -> 1/2 fail
//     warmup 0.2s + 0.2s blend -> 3/3 RECOVERED, 2.1-4.3 deg, hip -0.23..-0.25 (clean handoff)
//     warmup 0.1s + no blend   -> 5/5 RECOVERED, 2.4-4.0 deg
// 0.2 keeps ~20 CEM iterations of convergence (planner runs at 90-100/s on real, and the C3
// reference is seeded AT the measured pose so the initial optimal action is ~"hold" = safe even
// under-converged), while staying inside the survivable hold window. BENCH CAVEAT: the twin
// pre-converges the planner before every run, so it can prove the hold must be SHORT but says
// nothing about how many CEM iterations the real planner needs -- hence 0.2 not 0.1.
inline constexpr double kStraightenWarmupSec = 0.2;  // hold the latched pose (vs kWarmupSec 1.0)
inline constexpr double kStraightenRampSec = 0.2;    // measured -> PLANNER blend on a straighten boot
inline constexpr double kSwitchSettleSec = 2.0;   // hold pose after a live strategy switch
inline constexpr bool   kUseTwinTime = true;      // planner clock = lowstate tick * twin_dt
// Planner ThreadPool size. 0 = AUTO = hw_threads - kPlanThreadsReserve.
//
// WAS a hard 12 ("leaves cores for twin/safety"), which silently starved every STEPPING
// strategy on the real robot: CEM schedules num_trajectory + 1 jobs (the nominal rollout
// rides along, cross_entropy/planner.cc:469) and WAITS for all of them, so a plan iteration
// costs ceil((N+1)/threads) thread-WAVES. Trot's 36 trajectories on 12 threads = 4 waves =
// 27-30 plans/s measured on real -- below the 50-100 Hz band every real deployment needs
// (CMU Go1 MPPI: 30 samples @ 100 Hz, CPU-only, "limited computing is much better spent on
// achieving a ~100 Hz policy than additional sample evaluations"; Unitree's own H1-2 RL
// deploys decide at 50 Hz). The open-loop swing (lean::ModifyControl) is rate-INDEPENDENT
// so the legs still alternated -- but the stance-leg weight shift is SAMPLER-owned, so it
// starved: feet never unloaded. AUTO-sizing gives 18 on the dev laptop (24 hw threads),
// which measured 45-52 plans/s on the real robot -- in band.
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
inline constexpr double kClampRatio = 0.9;        // torque budget clamp: 0.9 x TAU_ESTOP (audit H2)
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
  int nu = 0;                          // 27 full-body, 12 legs-only
  const double* kp = nullptr;          // [nu] onboard PD kp == planner actuator gains
  const double* kv = nullptr;          // [nu]
  const double* tau_estop = nullptr;   // [nu] safety-layer estop thresholds (H2 clamp basis)
  const double* tau_limit = nullptr;   // [nu] operational URDF limits (B0 report basis)
  const double* frc_limit = nullptr;   // [nu] planner forcerange patch, or nullptr = none
  int frc_limit_begin = 0;             // first index to patch (13 = arms only, full-body node)
  // ---- ACTUATOR-AUTHORITY PARITY (2026-07-11): the planner must not plan with torque
  // the node will never emit. The H2 clamp bounds the EMITTED command to
  // kClampRatio * tau_estop, but the PLANNER model's forceranges were left at the MJCF
  // defaults for legs/torso -- so the sampler plans single-support balance believing it
  // has authority it does not have:
  //     joint    planner sees   node actually emits   over-estimate
  //     ankleP     +/-75 Nm          48.6 Nm             1.54x
  //     ankleR     +/-75 Nm          32.4 Nm             2.31x
  //     torso     +/-200 Nm          36.0 Nm             5.56x
  //     hipY      +/-200 Nm          54.0 Nm             3.70x
  // REAL 2026-07-11 (strat 23, plan rate healthy at 45-52/s): the stance ankle pitch railed
  // at EXACTLY 48.6 Nm and the torso at EXACTLY 36.0 Nm for 6 s while the stance knee locked
  // to -0.05 rad (a passive prop) AGAINST a weight-200 anti-strut cost -- the classic
  // weak-ankle crutch. The plan was valid in the planner's model and unexecutable in the
  // node's. Same bug CLASS as the phantom-table parity bug that faked the "forward walk needs
  // RL" verdict: the planner was solving the wrong physics.
  // -1 = task default (Task::PlannerNumericOverrides may set the `deploy_frc_parity` numeric;
  //      the lean task turns it ON for the stepping strategies only), 0 = force OFF (legacy
  //      model, byte-identical), 1 = force ON. Kill switch: --frc_parity=0.
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
  // STRAIGHTEN boot: bypass the scripted stand-pose rise and hand the planner authority
  // right after warmup (see kStraightenRampSec). Set by the node when --strategy 25 boots
  // (and --straighten_planner_bringup is on). false everywhere else -> every other
  // strategy's bring-up choreography is byte-unchanged.
  bool straighten_boot = false;
  // <0 = use the compiled kStraightenWarmupSec / kStraightenRampSec. Exposed as CLI flags so the
  // hold-vs-convergence tradeoff can be A/B'd on the REAL robot (the twin pre-converges its
  // planner, so it can prove the hold must be short but not how many CEM iterations real needs).
  double straighten_warmup_sec = -1.0;
  double straighten_ramp_sec = -1.0;
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
  double latency_rtf = 1.0;            // latency-comp sim-time scale = measured real-time factor.
                                       // The predict-forward horizon dlt is a WALL-clock duration
                                       // but predict_forward rolls it as SIM steps; on a below-
                                       // realtime plant (RoboCasa RTF<1) that over-leads by 1/RTF.
                                       // Scaling dlt by RTF converts wall->sim. 1.0 = IDENTITY
                                       // (real/twin run RTF~1 -> byte-unchanged); set to the
                                       // measured RTF on a slow sim (RoboCasa ~0.45).
};

// Runs the full deploy node (blocks until SIGINT/SIGTERM/q). Returns exit code.
int RunDeployNode(const NodeConfig& cfg);

}  // namespace h12deploy

#endif  // MJPC_DEPLOY_DEPLOY_COMMON_H_
