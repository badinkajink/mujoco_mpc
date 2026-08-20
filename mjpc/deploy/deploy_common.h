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
  double imu_yaw_offset_deg = 0.0;     // WORLD-frame heading correction (deg).
  bool yaw_fusion = true;              // slew the yaw offset live from the tag
                                       // bridge's aux_odom.position[2] (2026-08-18)
                                       // The IMU yaw is a gyro integration with
                                       // no absolute reference: it random-walks
                                       // ~0.1 deg/s and rotates the whole
                                       // planning frame off the real table
                                       // (08-14: all 4 evening runs stood at a
                                       // believed yaw of 9-26 deg -> corkscrew
                                       // dives). Measure with the tag bridge
                                       // (yaw_off print, <2 deg accurate) and
                                       // pass the NEGATIVE of the believed
                                       // error. Applied as a world z pre-
                                       // multiply in fill_state; 0 = off.
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
  // --ankle_autocalib: measure the per-power-on ankle zeros DURING the bring-up ramp hold
  // (still, loaded, planner muzzled) with the floor-referenced flat-sole solve, and REPLACE
  // the manual offsets above before policy handover. Fail-safe: any gate failure applies
  // nothing and keeps the manual flags. OFF by default = byte-identical.
  bool ankle_autocalib = false;
  bool ankle_autocalib_selftest = false;  // run the planted-offset solver selftest and exit
  // GRAVITY ANCHOR (2026-07-18): base tilt for the autocalib solve from the
  // time-averaged RAW accelerometer in a ZUPT-verified still window instead of
  // the fused quat (which carried 0.5-0.7 deg of run-to-run support-handling
  // error -- the same power-on zeros solved differently across two runs).
  // Adds gates: | |a|-g | < 0.5, std|a| < 0.35, mean|gyro| < 0.15. Empty accel
  // stream (twin bridges) auto-falls back to the fused quat. --ac_gravity=0
  // reverts to the exact pre-2026-07-18 behavior without a rebuild.
  bool ankle_autocalib_gravity = true;
  // SESSION IMU ALIGNMENT (2026-07-19): when the autocalib APPLIES, also add the
  // measured gravity-vs-fused pitch/roll delta (mean of witness+confirm windows,
  // capped +-4 deg) to the planning-path IMU correction -- real 07-19 evidence:
  // the fused frame is a CONSTANT ~+2.3 deg pitch off the plumb-line (7 windows,
  // 3 runs), i.e. the planner balanced around a vertical ~3 cm off-centre.
  // Requires ankle_autocalib_gravity. --ac_imu_align=0 reverts.
  bool ankle_autocalib_imu_align = true;
  // EXTENDED CALIB HOLD (2026-07-19): with the stock 3.0 s hold only ONE calib
  // window fits before policy handover -- the confirm window then measures a
  // robot the policy is already fighting (with ~6 deg zeros it never re-quiets:
  // real runs 2 & 4 of 07-19 burned all 10 windows on MOVING rejects and ran
  // uncalibrated). Extending the scripted hold by this many seconds (ONLY when
  // ankle_autocalib is on; 0 = stock timing) fits witness + confirm + APPLY
  // before the planner gets a vote -- the original "solve BEFORE handover"
  // design intent. 3.5 => hold 6.5 s: witness ~3.3-5.5, confirm ~6.5-8.7,
  // apply ~8.7, handover ~9.3.
  double ac_hold_extra_sec = 3.5;
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
  // BRING-UP RAMP DESTINATION. The scripted rise+hold (kStartRampSec + kRampHoldSec) drives ALL
  // joints here open-loop before the planner gets a vote. It was hard-wired to `stand`, which is
  // perfectly SYMMETRIC (hipP -0.15/-0.15) -- so booting a staggered strategy WITHOUT --align_start
  // spent 8 s scripted to drag the operator's hand-placed stagger back to a square stance. With
  // both feet planted that cannot move the feet, it only shears the soles and shoves the base:
  // strat 27 lat-lean grew monotonically -0.3 -> -7.0 deg across exactly the ramp window on two
  // real runs. Empty -> `stand` (every existing strategy: byte-identical).
  const char* ramp_dest_key = nullptr;
  double align_sec = 4.0;              // min-jerk drive duration [s, WALL]
  double align_tol = 0.08;             // converged when max|q - align_pose| < this [rad] (~4.6 deg)
  double align_timeout = 8.0;          // hard ceiling [s, WALL]; the robot is load-bearing and
                                       // sags, so convergence is best-effort -- never block on it
  // ANTI-STICTION PUSH. A pure position PD stalls wherever breakaway friction exceeds kp*err:
  // at kp=200 a 3 deg residual is only 10 Nm, which a harmonic drive will happily sit on, so the
  // legs stop short of the stance. align_ki winds an integrator into the COMMANDED target once
  // the min-jerk reference has arrived, so kp*(tgt - q) keeps GROWING until the joint breaks
  // loose. The head-room is real -- the H2 clamp permits |tgt - q| up to
  // (0.9*tau_estop - |tau_ff| - kv*|dq|)/kp = 0.36..1.35 rad on these joints -- and that SAME
  // clamp is the backstop: the emitted torque still cannot exceed 0.9 x the safety estop, so
  // this pushes hard but never leaves the safety envelope. 0 = off (pure PD, may stall short).
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
  int plan_threads = 0;                // >0 overrides kPlanThreads(12); 0 = compiled default
  bool cem_best_action = false;        // execute lowest-cost elite instead of elite mean
                                       // (CEM anti-hedging; --cem_best_action)
  double stale_sec = kStaleSec;        // H1 watchdog threshold; default = the REAL-robot 50ms.
                                       // Loosen ONLY for sims whose lowstate publisher stalls
                                       // on a shared sim lock (RoboCasa sensor renders: 50-60ms
                                       // gaps -> permanent safe-hold at the default).
  double latency_extra_ms = 4.0;       // latency-comp ADDITIVE ms on top of measured compute time
                                       // (transport + plant zero-order-hold). 2026-07-19 A/B lever:
                                       // the twin cannot test this axis (clean timing), so it is
                                       // probed on real: 4 (default) vs 8 vs 12.
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
