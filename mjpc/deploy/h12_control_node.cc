// MJPC embedded DDS control node for the Unitree H1-2 (digital twin + real robot).
// FULL-BODY variant: nu=27 (legs + torso + arms), task "Lean H12 Magpie",
// publishes the safety layer's full-body channel rt/safety/lowcmd_in.
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
//   planner thread : g_agent.PlanIteration(pool)  -- replans forever on the latest state
//   control thread : @200 Hz  read DDS state -> SetState -> ActionFromPolicy
//                    -> q* + gravity-FF tau -> unitree_hg LowCmd_ -> rt/safety/lowcmd_in
//
// STATE  pelvis (free-joint, qpos[0:7]) is backed out of the reported IMU-site
//   pose:  base_p = site_p - R(quat)*IMU_OFFSET ;  base_v = site_v - (R*gyro) x roff.
//   Identical math to mjpc_dds_bridge.py:pelvis_from_site (unit-tested vs ground truth).
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

ABSL_FLAG(std::string, task, "Lean H12 Magpie", "MJPC task id");
ABSL_FLAG(int, strategy, 6,
          "Lean Strategy parameter (6=stand 8=crouch 11=arms_overhead 13=lean_left "
          "16=counterbalance 18=squatter 20=stumble 21=reach 23=trot "
          "25=straighten/bring-up 31-35=lean pipeline ...). For slumped/leaning "
          "power-on boot with --strategy 25: phase 0 drives to upright+centered "
          "from the measured pose, then a basin gate (tilt<=3deg, z>=1.00, "
          "knees<=0.50rad i.e. legs actually extended, quiescent, 1.5s "
          "sustained) AUTO-ADVANCES into a stand phase running "
          "strat-6 weights/keyframe -- no manual switch needed; live-switch to "
          "other strategies as usual afterward.");
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
ABSL_FLAG(double, bad_orient_rad, 0.0,
          "R6 bad-orientation damp fallback (rad); 0 = OFF (default for the full-body/lean node -- "
          "validated deployments unchanged). See h12_lower_body_controller for the rationale.");
ABSL_FLAG(double, imu_roll_offset_deg, 0.0,
          "IMU roll zero-offset calibration (deg), same idea as --imu_pitch_offset_deg (body roll axis).");
ABSL_FLAG(double, ankle_roll_offset_l_deg, 0.0,
          "LEFT ankle-roll zero-offset calibration (deg): SUBTRACTED from the perceived roll AND "
          "ADDED to the command, so a foot the encoder reports rolled is both reasoned about and "
          "driven to its true angle. 0 = off. Dial to flatten an edge-rolled foot (free-hang L/R asym).");
ABSL_FLAG(double, ankle_roll_offset_r_deg, 0.0,
          "RIGHT ankle-roll zero-offset calibration (deg); see --ankle_roll_offset_l_deg. The "
          "free-hang showed the RIGHT foot resting ~6 deg more rolled (edge) than the left.");
ABSL_FLAG(double, ankle_pitch_offset_l_deg, 0.0,
          "LEFT ankle-pitch zero-offset calibration (deg), off = encoder - true; same "
          "belief/command pairing as the roll offsets. Measure with ankle_zero_snap.py.");
ABSL_FLAG(double, ankle_pitch_offset_r_deg, 0.0,
          "RIGHT ankle-pitch zero-offset calibration (deg); see --ankle_pitch_offset_l_deg.");
ABSL_FLAG(bool, ankle_autocalib, false,
          "SELF-CALIBRATE the per-power-on ankle zeros at bring-up: during the scripted ramp "
          "HOLD (robot still, feet LOADED, planner muzzled) sample the raw encoders + IMU, "
          "solve the flat-sole ankle angles (the floor is the reference -- same validated math "
          "as ankle_zero_snap.py), and REPLACE the four --ankle_*_offset flags with the result "
          "before policy handover. The H1-2 stores no ankle calibration and the zero re-rolls "
          "EVERY power-on (and can move on a violent event), so a per-bring-up self-check "
          "catches every case. FAIL-SAFE: if the window is not LOADED (>15 Nm legs) + SETTLED "
          "(|dq|<0.03) + STABLE (base std<0.5 deg) + self-consistent (halves agree <0.5 deg, "
          "|off|<8 deg cap), it applies NOTHING and keeps the manual flags. Requires the feet "
          "on the ground during bring-up (suspended -> clean REJECT). Skipped under "
          "--straighten_start. OFF by default = byte-identical. (Ported from "
          "h12_lower_body_controller 2026-07-20; the machinery is shared deploy_common.)");
ABSL_FLAG(bool, ankle_autocalib_selftest, false,
          "Validate the auto-calib solver against planted +-6 deg offsets (no robot, no DDS) "
          "and exit 0/1. Run after any edit to the anklecalib:: code.");
ABSL_FLAG(double, ac_hold_extra, 3.5,
          "EXTENDED CALIB HOLD (2026-07-19): extra seconds of scripted post-ramp hold when "
          "--ankle_autocalib is on, so BOTH calib windows (witness + confirm) and the APPLY land "
          "BEFORE policy handover. The stock 3.0s hold fits only one window; the confirm then "
          "measures a robot the policy is already fighting (6-deg zeros -> never quiet -> ran "
          "uncalibrated, real 07-19 runs 2+4). 0 = stock timing.");
ABSL_FLAG(bool, ac_imu_align, true,
          "SESSION IMU ALIGNMENT (2026-07-19): when --ankle_autocalib APPLIES, also correct the "
          "PLANNER's perceived orientation by the measured gravity-vs-fused delta (mean of the "
          "witness+confirm windows, cap 4 deg, blended). Real 07-19: the fused frame is a CONSTANT "
          "~+2.3 deg pitch off the plumb-line across runs -- the planner balanced around a vertical "
          "~3 cm off-centre (40% of the heel margin). Requires --ac_gravity. 0 = planner keeps the "
          "static --imu_pitch_offset_deg frame only.");
ABSL_FLAG(bool, ac_gravity, true,
          "GRAVITY ANCHOR for --ankle_autocalib (2026-07-18): take the solve's base tilt from "
          "the time-averaged RAW accelerometer in a ZUPT-verified still window (| |a|-g |<0.5, "
          "std|a|<0.35, mean|gyro|<0.15) instead of the fused IMU quat. The fused quat carried "
          "0.5-0.7 deg of run-to-run support-handling error (same power-on zeros solved "
          "pitch L +4.19 vs +3.50 across two runs); gravity is an absolute plumb-line no "
          "static support force can bias. Empty accel stream (twin) -> auto-fallback to the "
          "fused quat. 0 = exact pre-2026-07-18 fused-quat behavior.");
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
          "PlannerNumericOverrides. 0 = use the task default (e.g. trot slot 23 = 36). On the "
          "LOCKSTEP twin trajectories are free (planner waits for physics) so more = strictly "
          "better; on the REAL robot they are NOT -- 36 traj plans ~28Hz on 12 threads, 16 at "
          "~80Hz. Samples-per-plan vs replan-rate is a real tradeoff the lockstep twin cannot "
          "see. Sweep {16,18,24,36} on hardware / the real-chain bench without a rebuild. "
          "(Re-added 2026-07-03; the 07-02 flag diet cut it by mistake.)");
ABSL_FLAG(int, plan_threads, 0,
          "override the planner ThreadPool size. 0 = AUTO = hw_threads - 6 (18 on the 24-thread "
          "dev laptop; measured 45-52 plans/s on real). ONE-WAVE RULE: CEM schedules "
          "trajectories+1 jobs and waits for ALL, so plan rate divides by "
          "ceil((traj+1)/threads) -- keep plan_trajectories < plan_threads. The node now WARNS "
          "when you are multi-wave. BENCH: pass 12 when co-running the Python twin (18 planner "
          "threads starve it to ~0.5x real-time).");
ABSL_FLAG(bool, straighten_start, false,
          "hold the measured (slumped) pose, wait for ENTER, then hand authority to the planner "
          "from the slump (SETTLE->BLEND, no drag). Pair with --strategy 25 (straighten). OFF by default.");
ABSL_FLAG(bool, cost, false,
          "dump the per-term cost breakdown to stderr once/sec (debug). OFF by default -- the "
          "concise [node] status line is unaffected.");
ABSL_FLAG(int, frc_parity, -1,
          "ACTUATOR-AUTHORITY PARITY: tighten the PLANNER model's actuator forceranges to the "
          "torque the node can actually emit (0.9 x tau_estop = the H2 clamp budget), so the "
          "sampler stops planning balance it cannot execute. The planner model shipped ankle "
          "+/-75 Nm and torso +/-200 Nm while the node emits at most 48.6 / 36.0 -- on the real "
          "trot BOTH railed at exactly their budget for 6 s and the stance knee locked to a "
          "passive prop (the weak-ankle crutch). -1 = task default (ON for the stepping "
          "strategies via PlannerNumericOverrides), 0 = OFF (legacy model, byte-identical), "
          "1 = force ON. Kill switch for a real-robot A/B: --frc_parity=0.");
ABSL_FLAG(double, stale_sec, 0.05,
          "H1 stale-input watchdog threshold (s): either state stream older than this -> "
          "damping safe-hold. DEFAULT 0.05 = the REAL-robot value (1 kHz lowstate). Loosen "
          "ONLY for heavyweight sims whose lowstate publisher stalls on a shared sim lock "
          "(RoboCasa's sensor renders hold it 50-60 ms -> permanent safe-hold at 0.05; "
          "0.15 rides the stalls out while still catching a genuinely dead stream).");
ABSL_FLAG(double, latency_rtf, 1.0,
          "latency-comp sim-time scale = measured real-time factor. The predict-forward "
          "horizon is a WALL-clock duration rolled forward as SIM steps; on a below-realtime "
          "plant (RoboCasa RTF<1) that over-leads by 1/RTF. Scaling by RTF converts wall->sim. "
          "1.0 = IDENTITY (real/twin run RTF~1 -> UNCHANGED); set to the measured RTF (e.g. 0.45) "
          "only on a slow sim.");

namespace {
constexpr int kNU = 27;  // actuated joints on the handless H1-2
// Per-joint gains == h1_2_modified actuator classes == real LowCmd kp/kd.
// (Must match mjpc_dds_bridge.py / _lockstep_capability.py.)
// Safety-layer arm-gain fix (2026-06-09): ARMS ONLY -- arm kp 40 -> 30/20/15 (shoulder_p/r,
// shoulder_yaw+elbow, wrist) so the onboard arm PD torque stays under the arm estop bound. LEG/TORSO/
// hip_yaw kp LEFT AT ORIGINAL: lowering torso/hipY kp + tightening leg forceranges REGRESSED the hold
// (fell ~15s vs ~30s); the strut/lean is a SEPARATE balance problem, not a gain issue. Patched into the
// planner+latency model (PatchActuators) so node KP[]/KV[] == planner kp/kv == twin PD.
// ANKLE-PITCH kp 80 -> 200 (2026-07-20 DIAGNOSTIC): the deploy gravity feedforward is
// FREE-BODY qfrc_bias (deploy_common.cc:1929) -> ankle_pitch tau_ff = -0.2 Nm (only the 0.85 kg
// foot is outboard). Standing needs 17-44 Nm there, so at kp=80 the ankle can only make that
// torque by drooping 18 deg (kp*err), and that droop IS the forward lean (verified: required
// droop == observed droop to 0.5 deg across back3/back5). Raising kp to 200 (== hip/knee) cuts
// the droop to ~7 deg for the same torque. This is a MITIGATION test for the real fix
// (support-aware feedforward); ankle ROLL (idx 5,11) left at 80. Clamp still bounds
// |tau| <= 0.9*54 = 48.6 Nm, so at kp=200 the clamp bites past 13.9 deg error -> safe.
const double KP[kNU] = {150, 200, 200, 200, 200, 80,  150, 200, 200, 200, 200, 80,  200,
                        30, 30, 20, 20, 15, 15, 15,   30, 30, 20, 20, 15, 15, 15};
const double KV[kNU] = {5, 5, 5, 5, 4, 4,  5, 5, 5, 5, 4, 4,  5,
                        10, 10, 10, 10, 2, 2, 2,  10, 10, 10, 10, 2, 2, 2};
// SAFETY-LAYER TAU-ESTOP thresholds (estop torque_ratio x URDF torque limit, from
// default_safety_full.yaml + h12_safety_layer/core/joint_limits.py). Basis of the
// H2 torque-budget clamp: |tau_ff + KP*(tgt-q) + KV*dq| is capped at 0.9x these,
// so a commanded position can never demand estop-level torque.
const double TAU_ESTOP[kNU] = {60, 130, 200, 300, 54, 36,  60, 130, 200, 300, 54, 36,  40,
                               32, 32, 14.4, 14.4, 9.5, 9.5, 9.5,
                               32, 32, 14.4, 14.4, 9.5, 9.5, 9.5};
// ARM actuator force limit = OPERATIONAL URDF (= twin actuatorfrcrange). Patched into the planner+latency
// model for the ARMS ONLY (idx 13..26) so the planner caps arm torque at the real 18/40/19 Nm instead of
// the motor-peak 120 it otherwise assumes. LEG/TORSO forceranges are LEFT at the model default -- tightening
// them to the estop bound clamped the planner's hip/ankle balance authority and regressed the hold.
const double FRC_LIMIT[kNU] = {200, 200, 200, 300, 60, 40,  200, 200, 200, 300, 60, 40,  200,
                               40, 40, 18, 18, 19, 19, 19,   40, 40, 18, 18, 19, 19, 19};
// short joint names (qpos[7..33] order) for the Title-5 baseline (B0) report.
const char* const JOINT_NAMES[kNU] = {
    "LhipY", "LhipP", "LhipR", "Lknee", "LankP", "LankR",
    "RhipY", "RhipP", "RhipR", "Rknee", "RankP", "RankR", "torso",
    "LshP", "LshR", "LshY", "Lelb", "LwrR", "LwrP", "LwrY",
    "RshP", "RshR", "RshY", "Relb", "RwrR", "RwrP", "RwrY"};
// OPERATIONAL H1-2 joint torque limits (Nm) = Unitree URDF actuatorfrcrange == the twin's
// actuatorfrcrange == the safety-layer URDF_TORQUE_LIMITS. NOT the motor-PEAK from specs.md (which is
// ~6.7x higher on the arms: that table lists motor stall torque, e.g. elbow MOTOR 120 Nm, but the
// operational/control limit is 18). The B0 report grades against THIS basis.
//   legs: hipY/hipP/hipR 200, knee 300, ankP 60, ankR 40; torso 200;
//   arms: shoulderP/R 40, shoulderY 18, elbow 18, wrist 19.
const double TAU_LIMIT[kNU] = {200, 200, 200, 300, 60, 40,
                               200, 200, 200, 300, 60, 40, 200,
                               40, 40, 18, 18, 19, 19, 19,
                               40, 40, 18, 18, 19, 19, 19};
}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  h12deploy::NodeConfig cfg;
  cfg.nu = kNU;
  cfg.kp = KP;
  cfg.kv = KV;
  cfg.tau_estop = TAU_ESTOP;
  cfg.tau_limit = TAU_LIMIT;
  cfg.frc_limit = FRC_LIMIT;
  cfg.frc_limit_begin = 13;   // arms only (see FRC_LIMIT comment)
  cfg.joint_names = JOINT_NAMES;
  cfg.telemetry = h12deploy::Telemetry::kFullBody;
  cfg.upper_count = 0;        // full-body node actuates the arms itself

  cfg.task_id = absl::GetFlag(FLAGS_task);
  cfg.strategy = absl::GetFlag(FLAGS_strategy);
  cfg.gravity_ff = absl::GetFlag(FLAGS_gravity_ff);
  cfg.twin_dt = absl::GetFlag(FLAGS_twin_dt);
  cfg.lowcmd_topic = "rt/safety/lowcmd_in";   // safety layer full-body channel
  cfg.sportstate_topic = absl::GetFlag(FLAGS_sportstate_topic);
  cfg.imu_pitch_offset_deg = absl::GetFlag(FLAGS_imu_pitch_offset_deg);
  cfg.bad_orient_rad = absl::GetFlag(FLAGS_bad_orient_rad);
  cfg.imu_roll_offset_deg = absl::GetFlag(FLAGS_imu_roll_offset_deg);
  cfg.ankle_roll_offset_l_deg = absl::GetFlag(FLAGS_ankle_roll_offset_l_deg);
  cfg.ankle_roll_offset_r_deg = absl::GetFlag(FLAGS_ankle_roll_offset_r_deg);
  cfg.ankle_pitch_offset_l_deg = absl::GetFlag(FLAGS_ankle_pitch_offset_l_deg);
  cfg.ankle_pitch_offset_r_deg = absl::GetFlag(FLAGS_ankle_pitch_offset_r_deg);
  cfg.ankle_autocalib = absl::GetFlag(FLAGS_ankle_autocalib);
  cfg.ankle_autocalib_selftest = absl::GetFlag(FLAGS_ankle_autocalib_selftest);
  cfg.ankle_autocalib_gravity = absl::GetFlag(FLAGS_ac_gravity);
  cfg.ankle_autocalib_imu_align = absl::GetFlag(FLAGS_ac_imu_align);
  cfg.ac_hold_extra_sec = absl::GetFlag(FLAGS_ac_hold_extra);
  cfg.network_interface = absl::GetFlag(FLAGS_network_interface);
  cfg.domain_id = absl::GetFlag(FLAGS_domain_id);
  cfg.grpc_port = absl::GetFlag(FLAGS_grpc_port);
  cfg.arm_aware = false;
  // STRAIGHTEN boot (strategy 25): let the planner, not the scripted stand-pose lerp, drive
  // the rise. Only on a strat-25 boot -> every other strategy keeps its proven choreography.
  cfg.plan_trajectories = absl::GetFlag(FLAGS_plan_trajectories);
  cfg.plan_threads = absl::GetFlag(FLAGS_plan_threads);
  cfg.cost_log = absl::GetFlag(FLAGS_cost);
  cfg.straighten_start = absl::GetFlag(FLAGS_straighten_start);
  cfg.frc_parity = absl::GetFlag(FLAGS_frc_parity);
  cfg.stale_sec = absl::GetFlag(FLAGS_stale_sec);
  cfg.latency_rtf = absl::GetFlag(FLAGS_latency_rtf);
  return h12deploy::RunDeployNode(cfg);
}
