// MJPC embedded DDS control node for the Unitree H1-2 -- SPLIT-BODY variant.
//
// This is the C++ core started by the h12_deploy_mjpc "split body" launcher
// (mjpc_deploy_splitbody_controller / split_body_controller.py). It is a WHOLE-BODY
// controller (nu=27: legs + torso + two 7-DoF arms) that publishes its output on
// BOTH safety split channels: legs (rows 0..11) on rt/safety/lowcmd_lower_in and
// arms (rows 12..26) on rt/safety/lowcmd_upper_in. The UPPER channel is GATED by
// the upper-body pause toggle (deploy_common's cfg.pause_upper_topic): when paused
// the core stops writing the arms so the frame_task IK owns them (the legs channel
// keeps publishing); when active MJPC drives the arms as part of whole-body balance.
// This lets control of the arms hand back and forth between MJPC and the frame_task
// server. Kept a SEPARATE compilation unit / binary (mjpc_split_core) so this
// whole-body-split behavior never touches the plain lower-body / full-body cores.
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

// ---------------------------------------------------------------------------
// START POSE (--align_start): the stance the node drags the legs into BEFORE
// handing the robot to MJPC. R^12, radians, in DDS motor order (rows 0..11 of
// lowcmd -- the same rows this node publishes to rt/safety/lowcmd_lower_in).
//
// PROVENANCE (2026-07-13). Derived from the only long good stand on record,
// logs/stand_cost_3_20260711_175521: averaging the MEASURED pose over its 164 s
// of stable standing (70 s -> 234 s after hand-off) gives
//     { 0.011,-0.276, 0.173, 0.370,-0.154,-0.133,
//      -0.006,-0.269,-0.108, 0.371,-0.233, 0.221 }
// i.e. feet 0.549 m apart, knees bent 21.2/21.3 deg, pelvis 1.017 m. Every joint
// of that lands within 5.8 deg of the model's `stand` keyframe, so the keyframe
// was already right; the run just confirms it on hardware.
//
// SHIPPED below == the 'stand_up'/'stand' keyframe legs EXACTLY (2026-07-14, user
// request): the align target now equals what strategy-6 Posture pulls toward, so
// there is NO pose discontinuity at handover. (Previously knee 0.37 / hip_pitch
// -0.27 / ankle_pitch -0.21 were taken from a measured good-stand HOLD -- droop-
// aware -- which differed from the keyframe by up to 7 deg and caused a small jump
// at handover.) Under load the robot droops FROM this commanded pose; the align
// exits on SETTLED (not q==target), so it rests slightly off and reports a residual
// -- expected. Edit freely; this is the knob.
//
//                          feet ~0.516 m apart, knees ~20 deg bent (== keyframe)
constexpr double kLowerStartPose[12] = {
     0.00,   // [ 0] left_hip_yaw_joint        0 deg
    -0.15,   // [ 1] left_hip_pitch_joint    -8.6 deg -- == 'stand_up' keyframe
     0.12,   // [ 2] left_hip_roll_joint     +6.9 deg -- opens the stance
     0.35,   // [ 3] left_knee_joint        +20.1 deg -- bent (== keyframe, was 0.37)
    -0.28,   // [ 4] left_ankle_pitch_joint -16.0 deg -- sole flat (== keyframe, was -0.21)
    -0.12,   // [ 5] left_ankle_roll_joint   -6.9 deg -- sole flat laterally
     0.00,   // [ 6] right_hip_yaw_joint      0 deg
    -0.15,   // [ 7] right_hip_pitch_joint   -8.6 deg
    -0.12,   // [ 8] right_hip_roll_joint    -6.9 deg  (mirror of left)
     0.35,   // [ 9] right_knee_joint       +20.1 deg
    -0.28,   // [10] right_ankle_pitch_joint-16.0 deg
     0.12,   // [11] right_ankle_roll_joint  +6.9 deg  (mirror of left)
};

// LOCKSTAND (strategy 26) align target = the 'lockstand' keyframe legs: LOCKED knee
// + WIDE stance. Used ONLY when --strategy 26, so the bring-up places the feet apart
// and the knees straight BEFORE handover (a balance hold cannot widen PLANTED feet --
// they must start wide). Matches the own-sim-validated pose (held 3/3), so the robot
// lands at lockstand's target instead of straightening under load after handover.
//                          feet ~0.635 m apart, knees ~4.6 deg (locked strut)
constexpr double kLockstandStartPose[12] = {
     0.00,   // [ 0] left_hip_yaw_joint       0 deg
    -0.03,   // [ 1] left_hip_pitch_joint    -1.7 deg -- thigh near-vertical (straight leg)
     0.19,   // [ 2] left_hip_roll_joint    +10.9 deg -- WIDE splay
     0.08,   // [ 3] left_knee_joint         +4.6 deg -- LOCKED strut
    -0.14,   // [ 4] left_ankle_pitch_joint  -8.0 deg -- sole flat under straight shin
    -0.19,   // [ 5] left_ankle_roll_joint  -10.9 deg -- sole flat laterally at the wide stance
     0.00,   // [ 6] right_hip_yaw_joint      0 deg
    -0.03,   // [ 7] right_hip_pitch_joint   -1.7 deg
    -0.19,   // [ 8] right_hip_roll_joint   -10.9 deg  (mirror of left)
     0.08,   // [ 9] right_knee_joint        +4.6 deg
    -0.14,   // [10] right_ankle_pitch_joint -8.0 deg
     0.19,   // [11] right_ankle_roll_joint +10.9 deg  (mirror of left)
};
}  // namespace

ABSL_FLAG(std::string, task, "Lean H12 Magpie Split",
          "MJPC task id (WHOLE-BODY nu=27 task; the launcher/bringup normally override it. "
          "Must be a 27-DoF task -- this core actuates legs + torso + arms). The default "
          "'Lean H12 Magpie Split' = the Magpie model + 15 INACTIVE upper-joint equality "
          "locks, which the pause toggle engages at the measured pose while the frame_task "
          "IK owns the arms (plain 'Lean H12 Magpie' lacks the locks -> pausing would only "
          "gate the wire and the node warns).");
ABSL_FLAG(int, strategy, 6,
          "Stabilize Strategy parameter. The stabilize task is lower-body-only "
          "(nu=12), so the lean reach/lean/crouch slots are absent. Slots:\n"
          "    6  = stand   (free-standing balance hold -- the validated default)\n"
          "   20  = stumble (balance-gated march + catch-march push recovery)\n"
          "   22  = walk    (trot + a baked forward v_des; walk_des_vel_x)\n"
          "   23  = trot    (capture-point in-place trot; lifts ~5-8 cm)\n"
          "   24  = drive   (WSS teleop: stand<->trot FSM on live cmd_vel; idle "
          "plants the feet and stands, a command engages the gait, release settles "
          "upright. Set Cmd Active/Vx/Vy/Wz/Seq over gRPC -- Seq is a heartbeat and "
          "MUST keep changing or the watchdog stops the robot after 1 s.)\n"
          "  20/22/23/24 are the STEPPING family: they share the gait clock and the "
          "ModifyControl swing forcer, and get spline 5 / 17 trajectories via "
          "PlannerNumericOverrides (17+1 = one thread wave -- see the plan-rate note "
          "in stabilize.cc; 36 traj starved the planner to 27-30 plans/s on real).");
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
ABSL_FLAG(double, ankle_pitch_offset_l_deg, 0.0,
          "LEFT ankle-pitch zero-offset calibration (deg), off = encoder - true: SUBTRACTED from "
          "the perceived pitch AND ADDED to the wire command (same pairing as the roll offsets). "
          "The H1-2 stores NO ankle zero -- zero = wherever the A/B linkage sat at power-on -- so "
          "a common-mode pitch error tilts the whole body -> steady lean / slow fore-aft hunt. "
          "Measure with ankle_zero_snap.py over a LOADED quiet-stand recording. 0 = off.");
ABSL_FLAG(double, ankle_pitch_offset_r_deg, 0.0,
          "RIGHT ankle-pitch zero-offset calibration (deg); see --ankle_pitch_offset_l_deg.");
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
ABSL_FLAG(bool, straighten_start, false,
          "hold the measured (slumped) pose, wait for ENTER, then hand authority to the planner "
          "from the slump (SETTLE->BLEND, no drag). Pair with --strategy 25 (straighten). OFF by default.");
ABSL_FLAG(bool, cost, false,
          "dump the per-term cost breakdown to stderr once/sec (debug). OFF by default -- the "
          "concise [node] status line is unaffected.");
ABSL_FLAG(int, frc_parity, -1,
          "ACTUATOR-AUTHORITY PARITY: tighten the PLANNER model's actuator forceranges to the "
          "torque the node can actually emit (0.9 x tau_estop = the H2 clamp budget), so the "
          "sampler stops planning balance it cannot execute. The stabilize planner model ships "
          "the ankle at +/-75 Nm while the node emits at most 48.6 -- a 1.54x overestimate on "
          "the joint that OWNS fore-aft balance, so the sampler buys sway correction with ankle "
          "torque that the clamp then eats (real stand: LankP railed at 48.2/48.6, clamp 6.9%). "
          "-1 = task default (the `deploy_frc_parity` numeric; absent on stabilize -> OFF), "
          "0 = OFF (legacy model, byte-identical), 1 = force ON. Real-robot A/B: --frc_parity=0.");
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

// ---- PHASE-A START-POSE ALIGN (2026-07-13) ----
ABSL_FLAG(bool, align_start, false,
          "GO-TO-START-POSE before the planner runs. The H1-2 powers on with arbitrary leg "
          "geometry (twisted, fore-aft, a knee folded) and the operator was hand-straightening "
          "the legs before every launch; whether the stand survived then depended on where the "
          "hands left it. With this on, the node drags the legs from the measured power-on pose "
          "to kLowerStartPose (top of this file -- feet ~0.52 m apart, knees ~21 deg bent, taken "
          "from the 164 s stable hold in logs/stand_cost_3_20260711_175521) on a min-jerk profile, "
          "holds until they arrive, and only THEN hands over -- so MJPC always inherits the same "
          "stance instead of a lottery. Because the robot is then AT the stance, the scripted "
          "bring-up rise that follows collapses to ~0 (d0~0) and only the policy blend runs, which "
          "is exactly the near-home hand-off the full-body node always had. WALL-clocked, so "
          "--twin_dt does not dilate it. Default false = byte-identical to the old node.");
ABSL_FLAG(double, align_sec, 4.0,
          "--align_start: min-jerk drive duration [s, WALL]. Zero velocity AND acceleration at "
          "both ends, and the target starts AT the measured pose, so nothing is yanked. RAISE "
          "this to drag more gently (it is the gentleness knob); the gains are untouched.");
ABSL_FLAG(double, align_tol, 0.08,
          "--align_start: QUALITY WARNING threshold [rad] (0.08 = 4.6 deg), NOT the exit gate. "
          "The align exits when the legs SETTLE (|dq| < 0.05 rad/s), because the robot is "
          "load-bearing and a knee commanded to 0.37 comes to rest nearer 0.55 under the deploy "
          "PD's gravity droop (the good run: cmd 0.35 vs measured 0.568 -- and it stood fine). "
          "Gating on |q - target| would never fire. If the settled residual exceeds this, the "
          "node prints a NOTE -- harmless if the droop is spread across the legs, worth a look "
          "if ONE joint is far off (a foot jammed on the floor).");
ABSL_FLAG(double, align_ki, 1.0,
          "--align_start: ANTI-STICTION PUSH gain [rad/s per rad of error]. A pure position PD "
          "stalls wherever breakaway friction beats kp*err -- at kp=200 a 3 deg residual is only "
          "10 Nm, which a harmonic drive will just sit on, so the legs stop short of the stance. "
          "This integrates the residual into the COMMANDED target once the drag profile has "
          "arrived, so kp*(tgt-q) keeps GROWING until the joint breaks loose. The head-room is "
          "real: the H2 clamp allows |tgt-q| up to 0.36-1.35 rad on these joints. And that same "
          "clamp is the backstop -- emitted torque still cannot exceed 0.9 x the safety estop, so "
          "this pushes HARD but never outside the safety envelope. Raise it if the legs still "
          "stall; 0 = off (pure PD).");
ABSL_FLAG(double, align_i_max, 0.25,
          "--align_start: windup limit [rad] on the anti-stiction push -- the most extra command "
          "any ONE joint may accumulate. kp * this = the extra breakaway torque available (knee "
          "kp=200 -> up to +50 Nm). Bounds how hard a genuinely JAMMED joint gets shoved: it "
          "parks at this and the node reports it, rather than winding up forever. The torque "
          "clamp still caps the real output regardless.");
ABSL_FLAG(bool, align_wait, true,
          "--align_start: after the legs reach the start pose, PARK there and hold it until you "
          "press ENTER (bare Enter, or 'g'/'go', on the node's stdin). Nothing else changes: the "
          "legs stay stiff and the node keeps publishing at 200 Hz the whole time it waits, so "
          "the safety layer never sees a stale command and the stance does not drift. This is the "
          "point of the whole feature -- MJPC must not inherit the robot while you are still "
          "handling it. Press Enter only once you have let go and it is settled. 'q'+Enter quits "
          "without ever engaging the planner. --noalign_wait = hand over the instant the legs "
          "settle (no operator in the loop).");
ABSL_FLAG(double, align_timeout, 15.0,
          "--align_start: hard ceiling [s, WALL] on the drag+push. Reached => stop pushing and "
          "move on (to the operator ENTER gate, if --align_wait). Generous by default so the "
          "anti-stiction integral has time to break a sticky joint loose; the align normally "
          "exits earlier via REACHED or STALLED. This ceiling is NOT optional -- a joint that "
          "cannot arrive must never block the node forever.");

// ---- DEBUG PLAN PUBLISH (2026-07-15) ----
ABSL_FLAG(std::string, plan_topic, "",
          "DEBUG: publish the active planner's best trajectory (qpos rows, JSON) to this DDS "
          "topic for the mjpc_debug_visualizer's blue ghost. The payload is a "
          "std_msgs::msg::dds_::String_, which is byte-identical on the wire to ROS 2's "
          "std_msgs/msg/String -- so 'rt/mjpc/plan' here surfaces as ROS topic '/mjpc/plan' with "
          "no bridge node (the same DDS<->ROS name mangling that already makes rt/lowstate visible "
          "as /lowstate). Serialized on the PLANNER thread right after PlanIteration returns (the "
          "one race-free point -- BestTrajectory() does NOT lock), so the 200 Hz control loop is "
          "untouched. EMPTY = OFF (default): no publisher, no serialization, zero added work. "
          "Debug only -- nothing in the control path reads this.");
ABSL_FLAG(double, plan_hz, 20.0,
          "--plan_topic publish rate cap [Hz]. The planner free-runs at 45-52 iters/s and a plan "
          "is ~90 KB of JSON; iterations are SKIPPED to hit this rate (the planner thread is never "
          "slept -- that would cost real plan rate). Ignored when --plan_topic is empty.");

// ---- UPPER-BODY PAUSE TOGGLE (whole-body split core) ----
ABSL_FLAG(std::string, pause_upper_topic, "rt/mjpc/pause_upperbody",
          "DDS std_msgs/String topic ('1'=paused) toggling whether this core PUBLISHES the "
          "UPPER channel (arms) and whether the planner treats the arms as free (whole-body) "
          "or eq-locked to the measured pose (legs-only planning). Paused -> the frame_task "
          "IK owns the arms; the legs channel keeps publishing. Empty = off (single-channel). "
          "The launcher's toggle_pause_upperbody service publishes this.");
ABSL_FLAG(bool, pause_upper_init, true,
          "INITIAL upper-channel pause state. DEFAULT TRUE = come up PAUSED (frame_task owns "
          "the arms; the dynamic startup handshake: the launcher unpauses this core only "
          "after frame_task reports ready and hands the arms over). --nopause_upper_init = "
          "legacy active-at-boot. Ignored when --pause_upper_topic is empty.");

namespace {
constexpr int kNU = 27;  // WHOLE-BODY split controller: all 27 actuated joints
                         // (12 legs + torso + two 7-DoF arms). The legs go to the
                         // safety split LOWER channel and the arms to the UPPER
                         // channel (gated by the pause toggle); see main().
// Per-joint gains == h1_2_modified actuator classes == real LowCmd kp/kd (== the
// full-body node). ARMS kp 40 -> 30/20/15 (shoulder_p/r, shoulder_yaw+elbow, wrist)
// so the onboard arm PD torque stays under the arm estop bound; legs/torso/hipY kp
// left at original (lowering them regressed the hold). Legs keep kp 80/kv 4 on the
// ankles (softening was A/B-tested and REJECTED 2026-07-10; stiffness is load-
// bearing). Patched into the planner+latency model so node KP[]/KV[] == planner
// kp/kv == twin PD.
const double KP[kNU] = {150, 200, 200, 200, 80, 80,  150, 200, 200, 200, 80, 80,  200,
                        30, 30, 20, 20, 15, 15, 15,   30, 30, 20, 20, 15, 15, 15};
const double KV[kNU] = {5, 5, 5, 5, 4, 4,  5, 5, 5, 5, 4, 4,  5,
                        10, 10, 10, 10, 2, 2, 2,  10, 10, 10, 10, 2, 2, 2};
// SAFETY-LAYER TAU-ESTOP thresholds (estop torque_ratio x URDF torque limit).
// Basis of the H2 torque-budget clamp: |tau_ff + KP*(tgt-q) + KV*dq| <= 0.9x these.
const double TAU_ESTOP[kNU] = {60, 130, 200, 300, 54, 36,  60, 130, 200, 300, 54, 36,  40,
                               32, 32, 14.4, 14.4, 9.5, 9.5, 9.5,
                               32, 32, 14.4, 14.4, 9.5, 9.5, 9.5};
// ARM actuator force limit = OPERATIONAL URDF, patched into the planner+latency
// model for the ARMS ONLY (idx 13..26) so the planner caps arm torque at the real
// 18/40/19 Nm instead of the motor-peak it otherwise assumes. LEG/TORSO forceranges
// are LEFT at the model default (tightening them regressed the leg hold).
const double FRC_LIMIT[kNU] = {200, 200, 200, 300, 60, 40,  200, 200, 200, 300, 60, 40,  200,
                               40, 40, 18, 18, 19, 19, 19,   40, 40, 18, 18, 19, 19, 19};
const char* const JOINT_NAMES[kNU] = {
    "LhipY", "LhipP", "LhipR", "Lknee", "LankP", "LankR",
    "RhipY", "RhipP", "RhipR", "Rknee", "RankP", "RankR", "torso",
    "LshP", "LshR", "LshY", "Lelb", "LwrR", "LwrP", "LwrY",
    "RshP", "RshR", "RshY", "Relb", "RwrR", "RwrP", "RwrY"};
// OPERATIONAL H1-2 joint torque limits (Nm) = Unitree URDF actuatorfrcrange.
// The B0 report grades against THIS basis; the estop trips at TAU_ESTOP (below it).
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
  cfg.frc_limit = FRC_LIMIT;  // arms-only forcerange patch (idx 13..26)
  cfg.frc_limit_begin = 13;
  cfg.joint_names = JOINT_NAMES;
  cfg.telemetry = h12deploy::Telemetry::kFullBody;
  // ARM-AWARE complement (2026-07-17): read torso+arms (motor 12..26) from
  // rt/lowstate and retarget the model's upper eq locks to the MEASURED pose
  // each tick -- exactly the legs-only core's arm_aware config. The rows
  // OVERLAP this core's own actuated nu=27 set on purpose: while the upper
  // channel is ACTIVE the locks stay inactive (Split model ships them
  // active=false) and this is just a redundant state write; while PAUSED the
  // deploy core engages the locks so the planner holds the arms where the
  // frame_task IK really has them and plans the legs around that. Gated on
  // --arm_aware (on by default; --noarm_aware = wire-gating only + warning).
  cfg.upper_count = 15;
  cfg.comp_motor_offset = 12;

  cfg.task_id = absl::GetFlag(FLAGS_task);
  cfg.strategy = absl::GetFlag(FLAGS_strategy);
  cfg.gravity_ff = absl::GetFlag(FLAGS_gravity_ff);
  cfg.twin_dt = absl::GetFlag(FLAGS_twin_dt);
  cfg.lowcmd_topic = "rt/safety/lowcmd_lower_in";        // safety split LOWER channel (legs)
  cfg.upper_lowcmd_topic = "rt/safety/lowcmd_upper_in";  // safety split UPPER channel (arms)
  cfg.pause_upper_topic = absl::GetFlag(FLAGS_pause_upper_topic);
  cfg.pause_upper_init = absl::GetFlag(FLAGS_pause_upper_init);
  cfg.sportstate_topic = absl::GetFlag(FLAGS_sportstate_topic);
  cfg.imu_pitch_offset_deg = absl::GetFlag(FLAGS_imu_pitch_offset_deg);
  cfg.bad_orient_rad = absl::GetFlag(FLAGS_bad_orient_rad);
  cfg.imu_roll_offset_deg = absl::GetFlag(FLAGS_imu_roll_offset_deg);
  cfg.ankle_roll_offset_l_deg = absl::GetFlag(FLAGS_ankle_roll_offset_l_deg);
  cfg.ankle_roll_offset_r_deg = absl::GetFlag(FLAGS_ankle_roll_offset_r_deg);
  cfg.ankle_pitch_offset_l_deg = absl::GetFlag(FLAGS_ankle_pitch_offset_l_deg);
  cfg.ankle_pitch_offset_r_deg = absl::GetFlag(FLAGS_ankle_pitch_offset_r_deg);
  cfg.network_interface = absl::GetFlag(FLAGS_network_interface);
  cfg.domain_id = absl::GetFlag(FLAGS_domain_id);
  cfg.grpc_port = absl::GetFlag(FLAGS_grpc_port);
  cfg.arm_aware = absl::GetFlag(FLAGS_arm_aware);
  cfg.plan_trajectories = absl::GetFlag(FLAGS_plan_trajectories);
  cfg.plan_threads = absl::GetFlag(FLAGS_plan_threads);
  cfg.cost_log = absl::GetFlag(FLAGS_cost);
  cfg.straighten_start = absl::GetFlag(FLAGS_straighten_start);
  cfg.frc_parity = absl::GetFlag(FLAGS_frc_parity);
  cfg.stale_sec = absl::GetFlag(FLAGS_stale_sec);
  cfg.latency_rtf = absl::GetFlag(FLAGS_latency_rtf);
  cfg.plan_pub_topic = absl::GetFlag(FLAGS_plan_topic);   // "" = OFF (debug only)
  cfg.plan_pub_hz = absl::GetFlag(FLAGS_plan_hz);
  // PHASE-A start-pose align: kLowerStartPose is the R^12 stance at the top of this file.
  cfg.align_start = absl::GetFlag(FLAGS_align_start);
  cfg.align_sec = absl::GetFlag(FLAGS_align_sec);
  cfg.align_tol = absl::GetFlag(FLAGS_align_tol);
  cfg.align_timeout = absl::GetFlag(FLAGS_align_timeout);
  cfg.align_wait = absl::GetFlag(FLAGS_align_wait);
  cfg.align_ki = absl::GetFlag(FLAGS_align_ki);
  cfg.align_i_max = absl::GetFlag(FLAGS_align_i_max);
  // Strategy 26 (lockstand) aligns to the wide+locked pose; every other strategy
  // keeps the tested bent-knee stand bring-up. Feet must START wide (a hold can't
  // slide planted feet outward), so the align pose owns the stance width.
  const bool lockstand = absl::GetFlag(FLAGS_strategy) == 26;
  const double* start_pose = lockstand ? kLockstandStartPose : kLowerStartPose;
  // legs only (the start-pose arrays are R^12); arms stay home (align_pose 0).
  for (int i = 0; i < 12; i++) cfg.align_pose[i] = start_pose[i];
  cfg.align_pose_set = true;
  return h12deploy::RunDeployNode(cfg);
}
