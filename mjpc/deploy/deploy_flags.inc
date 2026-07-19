// H1-2 deploy shared CLI-flag manifest (stage 2 of the 2026-07-18 reorg).
//
// Included by each thin main AFTER <absl/flags/flag.h> and
// "mjpc/deploy/deploy_common.h" (DefaultDomainId lives there). Defines the
// flag set every core shares, exactly once, with per-node overrides via
// macros the main #defines BEFORE including this file:
//
//   H12_DEFAULT_BAD_ORIENT   (default 0.9; the full-body main sets 0.0)
//   H12_DEFAULT_GRPC_PORT    (default 10000; the upper main sets 10001)
//   H12_HELP_<NAME>          full help-string override for flags whose help is
//                            genuinely node-specific (--arm_aware, --grpc_port,
//                            --bad_orient_rad, --frc_parity, --straighten_start)
//   H12_ANKLE_HELP_NOTE      string literal appended to the four ankle-calibration
//                            helps (the upper main marks them INERT: calibration
//                            applies at motor rows 4/5/10/11 only)
//
// NOT here (defined per-main, help is the load-bearing per-node contract):
//   --task, --strategy (absent on upper), --pause_upper_topic/--pause_upper_init
//   (split only).
//
// deploy_common.cc reads every manifest flag via FillCommonConfig()
// (deploy_flags.h declares them); each binary links exactly one main, so the
// one-definition rule holds per binary.

#ifndef H12_DEFAULT_BAD_ORIENT
#define H12_DEFAULT_BAD_ORIENT 0.9
#endif
#ifndef H12_DEFAULT_GRPC_PORT
#define H12_DEFAULT_GRPC_PORT 10000
#endif
#ifndef H12_ANKLE_HELP_NOTE
#define H12_ANKLE_HELP_NOTE ""
#endif

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
#ifdef H12_HELP_BAD_ORIENT
ABSL_FLAG(double, bad_orient_rad, H12_DEFAULT_BAD_ORIENT, H12_HELP_BAD_ORIENT);
#else
ABSL_FLAG(double, bad_orient_rad, H12_DEFAULT_BAD_ORIENT,
          "R6 bad-orientation damp fallback (rad). Base tilt beyond this latches a permanent "
          "kp=0/kd=3/tau=0 damp command until restart (Unitree deploy parity: bad_orientation(1.0) "
          "-> Passive) so the planner never thrashes an unrecoverable fall. 0 disables. "
          "0.9 rad (~52 deg) = engaged well past the 35 deg fall verdict.");
#endif
ABSL_FLAG(double, imu_roll_offset_deg, 0.0,
          "IMU roll zero-offset calibration (deg), same idea as --imu_pitch_offset_deg (body roll axis).");
ABSL_FLAG(double, ankle_roll_offset_l_deg, 0.0,
          "LEFT ankle-roll zero-offset calibration (deg): SUBTRACTED from the perceived roll AND "
          "ADDED to the command, so a foot the encoder reports rolled is both reasoned about and "
          "driven to its true angle. 0 = off. Dial to flatten an edge-rolled foot (free-hang L/R "
          "asym)." H12_ANKLE_HELP_NOTE);
ABSL_FLAG(double, ankle_roll_offset_r_deg, 0.0,
          "RIGHT ankle-roll zero-offset calibration (deg); see --ankle_roll_offset_l_deg. The "
          "free-hang showed the RIGHT foot resting ~6 deg more rolled (edge) than the left."
          H12_ANKLE_HELP_NOTE);
ABSL_FLAG(double, ankle_pitch_offset_l_deg, 0.0,
          "LEFT ankle-pitch zero-offset calibration (deg), off = encoder - true: SUBTRACTED from "
          "the perceived pitch AND ADDED to the wire command (same pairing as the roll offsets). "
          "The H1-2 stores NO ankle zero -- zero = wherever the A/B linkage sat at power-on -- so "
          "a common-mode pitch error tilts the whole body -> steady lean / slow fore-aft hunt. "
          "Measure with ankle_zero_snap.py over a LOADED quiet-stand recording. 0 = off."
          H12_ANKLE_HELP_NOTE);
ABSL_FLAG(double, ankle_pitch_offset_r_deg, 0.0,
          "RIGHT ankle-pitch zero-offset calibration (deg); see --ankle_pitch_offset_l_deg."
          H12_ANKLE_HELP_NOTE);
ABSL_FLAG(std::string, network_interface, "",
          "DDS network interface (empty = auto-pin the 192.168.123.x robot NIC when present, "
          "else autodetermine/loopback for the twin)");
ABSL_FLAG(int, domain_id, h12deploy::DefaultDomainId(),
          "DDS domain id (default: $ROS_DOMAIN_ID if set, else 0)");
#ifdef H12_HELP_GRPC_PORT
ABSL_FLAG(int, grpc_port, H12_DEFAULT_GRPC_PORT, H12_HELP_GRPC_PORT);
#else
ABSL_FLAG(int, grpc_port, H12_DEFAULT_GRPC_PORT,
          "if >0, host an MJPC gRPC server on this port so the monitor can attach "
          "(view state + switch Strategy live); 0 disables. NOTE: compiled in only on "
          "fork builds (MJPC_BUILD_GRPC_SERVICE); the colcon cores parse this but have "
          "no gRPC server.");
#endif
ABSL_FLAG(int, plan_trajectories, 0,
          "override sampling_trajectories (rollouts per plan) AFTER the per-strategy "
          "PlannerNumericOverrides. 0 = task default. Real-hardware plan-rate sweep lever "
          "(samples-per-plan vs replan-rate; one-wave rule: traj+1 <= threads). "
          "(history: see mjpc/deploy/HISTORY.md)");
ABSL_FLAG(int, plan_threads, 0,
          "override the planner ThreadPool size. 0 = AUTO = hw_threads - 6. One-wave "
          "rule: keep plan_trajectories < plan_threads (CEM schedules traj+1 jobs).");
#ifdef H12_HELP_STRAIGHTEN_START
ABSL_FLAG(bool, straighten_start, false, H12_HELP_STRAIGHTEN_START);
#else
ABSL_FLAG(bool, straighten_start, false,
          "hold the measured (slumped) pose, wait for ENTER, then hand authority to the planner "
          "from the slump (SETTLE->BLEND, no drag). Pair with --strategy 25 (straighten). OFF by default.");
#endif
ABSL_FLAG(bool, cost, false,
          "dump the per-term cost breakdown to stderr once/sec (debug). OFF by default -- the "
          "concise [node] status line is unaffected.");
#ifdef H12_HELP_FRC_PARITY
ABSL_FLAG(int, frc_parity, -1, H12_HELP_FRC_PARITY);
#else
ABSL_FLAG(int, frc_parity, -1,
          "ACTUATOR-AUTHORITY PARITY: tighten the PLANNER model's actuator forceranges to "
          "0.9 x tau_estop (the safety-layer torque budget), so the sampler stops planning "
          "balance the safety envelope cannot deliver. The torque-budget clamp is REMOVED: "
          "emitted torque is unclamped and CAN trip the safety estop, so parity is the "
          "intended mitigation. -1 = task default (the model's `deploy_frc_parity` numeric; "
          "no shipped task sets it ON). 0 = force OFF (legacy model, byte-identical), "
          "1 = force ON.");
#endif
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
#ifdef H12_HELP_ARM_AWARE
ABSL_FLAG(bool, arm_aware, true, H12_HELP_ARM_AWARE);
#else
ABSL_FLAG(bool, arm_aware, true,
          "ARM-AWARE balance (loco-manip): read the 15 upper-body joints (torso+arms, motor 12..26) "
          "from rt/lowstate and feed them to the planner so its CoM/dynamics model tracks where the "
          "arms ACTUALLY are -- the legs then PRE-compensate for a reach instead of only reacting to "
          "the resulting base tilt. The planner STILL actuates only the 12 legs (nu=12); the upper "
          "joints are held at the MEASURED pose during each rollout via the equality locks (retargeted "
          "live), so they don't snap to home. This node NEVER commands the arms (the FrameTask IK owns "
          "rt/safety/lowcmd_upper_in). No-op when the arms are at home (measured==0). DEFAULT TRUE "
          "(strictly better on the F6 bench -- arm45 ankle margin 2x; also feeds the "
          "task-side arm_aware_plan retarget). --noarm_aware = legacy home-locked isolation mode.");
#endif

// ---- PHASE-A START-POSE ALIGN ----
ABSL_FLAG(bool, align_start, false,
          "GO-TO-START-POSE before the planner runs. The H1-2 powers on with arbitrary leg "
          "geometry (twisted, fore-aft, a knee folded); with this on, the node drags the legs "
          "from the measured power-on pose to the compiled start pose (h12_start_poses.h on the "
          "leg-owning mains; the model's 'stand' keyframe otherwise) on a min-jerk profile, holds "
          "until they arrive, and only THEN hands over -- so MJPC always inherits the same stance "
          "instead of a lottery. Because the robot is then AT the stance, the scripted bring-up "
          "rise collapses to ~0 and only the policy blend runs. WALL-clocked, so --twin_dt does "
          "not dilate it. Default false = byte-identical to the old node.");
ABSL_FLAG(double, align_sec, 4.0,
          "--align_start: min-jerk drive duration [s, WALL]. Zero velocity AND acceleration at "
          "both ends, and the target starts AT the measured pose, so nothing is yanked. RAISE "
          "this to drag more gently (it is the gentleness knob); the gains are untouched.");
ABSL_FLAG(double, align_tol, 0.08,
          "--align_start: QUALITY WARNING threshold [rad] (0.08 = 4.6 deg), NOT the exit gate. "
          "The align exits when the legs SETTLE (|dq| < 0.05 rad/s), because the robot is "
          "load-bearing and a knee commanded to 0.37 comes to rest nearer 0.55 under the deploy "
          "PD's gravity droop. Gating on |q - target| would never fire. If the settled residual "
          "exceeds this, the node prints a NOTE -- harmless if the droop is spread across the "
          "legs, worth a look if ONE joint is far off (a foot jammed on the floor).");
ABSL_FLAG(double, align_ki, 1.0,
          "--align_start: ANTI-STICTION PUSH gain [rad/s per rad of error]. A pure position PD "
          "stalls wherever breakaway friction beats kp*err -- at kp=200 a 3 deg residual is only "
          "10 Nm, which a harmonic drive will just sit on, so the legs stop short of the stance. "
          "This integrates the residual into the COMMANDED target once the drag profile has "
          "arrived, so kp*(tgt-q) keeps GROWING until the joint breaks loose. The wound-up "
          "target is bounded ONLY by --align_i_max (the torque clamp is removed): the PD "
          "torque CAN reach the safety-layer estop, which is the backstop. Raise it if the "
          "legs still stall; 0 = off (pure PD).");
ABSL_FLAG(double, align_i_max, 0.25,
          "--align_start: windup limit [rad] on the anti-stiction push -- the most extra command "
          "any ONE joint may accumulate. kp * this = the extra breakaway torque available (knee "
          "kp=200 -> up to +50 Nm). Bounds how hard a genuinely JAMMED joint gets shoved: it "
          "parks at this and the node reports it, rather than winding up forever. This is the "
          "ONLY bound (the torque clamp is removed): the resulting PD torque CAN reach the "
          "safety-layer estop.");
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

// ---- DEBUG PLAN PUBLISH ----
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
