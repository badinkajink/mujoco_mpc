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
// Thin main: shared flags come from deploy_flags.inc, gain tables from
// h12_gain_tables.h, start poses from h12_start_poses.h; everything else is
// deploy_common.cc (FillCommonConfig -> NodeConfig -> RunDeployNode). Settled
// values (ctrl_hz 200, twin-time clocking, latency comp, AUTO-sized plan
// threads, ramp/hold/blend timings, ...) are compiled-in constants -- see
// deploy_common.h.

#include <string>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "mjpc/deploy/deploy_common.h"
#include "mjpc/deploy/h12_gain_tables.h"
#include "mjpc/deploy/h12_start_poses.h"

// Node-specific help override: the launcher context matters on this core.
#define H12_HELP_FRC_PARITY \
          "ACTUATOR-AUTHORITY PARITY: tighten the PLANNER model's actuator forceranges to " \
          "0.9 x tau_estop (the safety-layer torque budget), so the sampler stops planning " \
          "balance the safety envelope cannot deliver. The torque-budget clamp is REMOVED: " \
          "emitted torque is unclamped and CAN trip the safety estop, so parity is the " \
          "intended mitigation. -1 = task default (the `deploy_frc_parity` numeric; the " \
          "Lean XMLs ship 0 and lean.cc never sets it -> OFF). NOTE the launcher " \
          "(split_body_controller.py) passes --frc_parity 1, so bringup runs are parity-ON. " \
          "0 = force OFF (legacy model, byte-identical), 1 = force ON."
#include "mjpc/deploy/deploy_flags.inc"

ABSL_FLAG(std::string, task, "Lean H12 Magpie Split",
          "MJPC task id (WHOLE-BODY nu=27 task; the launcher/bringup normally override it. "
          "Must be a 27-DoF task -- this core actuates legs + torso + arms). The default "
          "'Lean H12 Magpie Split' = the Magpie model + 15 INACTIVE upper-joint equality "
          "locks, which the pause toggle engages at the measured pose while the frame_task "
          "IK owns the arms (plain 'Lean H12 Magpie' lacks the locks -> pausing would only "
          "gate the wire and the node warns).");
ABSL_FLAG(int, strategy, 6,
          "Lean Strategy parameter (the default task is the whole-body 'Lean H12 Magpie "
          "Split', nu=27 -- NOT the nu=12 stabilize task). Slots (lean.h "
          "GetStrategyNames; 0-5 = the lean pipeline, 7-19/21 = single-skill poses):\n"
          "    6  = stand         (free-standing balance hold -- the validated default)\n"
          "   20  = stumble       (gait-clock step-in-place; walks on a commanded CoM vel)\n"
          "   22  = forearm_brace (pre-lean forearm brace)\n"
          "   23  = trot          (capture-point in-place trot)\n"
          "   24  = drive         (WSS teleop: stand<->trot FSM on live cmd_vel; the Seq "
          "heartbeat MUST keep changing or the watchdog stops the robot after 1 s)\n"
          "   25  = straighten    (pre-stand bring-up; pair with --straighten_start)\n"
          "   26  = jump          (h12_simple_jump, one-shot in-place hop). COLLISION: "
          "this main ALSO maps --strategy 26 to the LOCKSTAND wide+locked align pose "
          "(h12_start_poses.h), so an --align_start boot at 26 aligns for lockstand while "
          "the planner runs the jump task\n"
          "   27-30 = reserved (stand), 31-35 = the lean ladder "
          "(stand/reach/counterbalance/brace/full).");

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

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  h12deploy::NodeConfig cfg;
  cfg.nu = h12deploy::kNU27;
  cfg.kp = h12deploy::kKp27;
  cfg.kv = h12deploy::kKv27;
  cfg.tau_estop = h12deploy::kTauEstop27;
  cfg.tau_limit = h12deploy::kTauLimit27;
  cfg.frc_limit = h12deploy::kFrcLimit27;  // arms-only forcerange patch
  cfg.frc_limit_begin = h12deploy::kArmsBegin;
  cfg.joint_names = h12deploy::kJointNames27;
  cfg.telemetry = h12deploy::Telemetry::kFullBody;
  // ARM-AWARE complement: read torso+arms (motor 12..26) from rt/lowstate and
  // retarget the model's upper eq locks to the MEASURED pose each tick --
  // exactly the legs-only core's arm_aware config. The rows OVERLAP this
  // core's own actuated nu=27 set on purpose: while the upper channel is
  // ACTIVE the locks stay inactive (Split model ships them active=false) and
  // this is just a redundant state write; while PAUSED the deploy core engages
  // the locks so the planner holds the arms where the frame_task IK really has
  // them and plans the legs around that. Gated on --arm_aware (on by default;
  // --noarm_aware = wire-gating only + warning).
  cfg.upper_count = h12deploy::kUpperCount;
  cfg.comp_motor_offset = h12deploy::kUpperOffset;

  h12deploy::FillCommonConfig(&cfg);
  cfg.task_id = absl::GetFlag(FLAGS_task);
  cfg.strategy = absl::GetFlag(FLAGS_strategy);
  cfg.lowcmd_topic = "rt/safety/lowcmd_lower_in";        // safety split LOWER channel (legs)
  cfg.upper_lowcmd_topic = "rt/safety/lowcmd_upper_in";  // safety split UPPER channel (arms)
  cfg.pause_upper_topic = absl::GetFlag(FLAGS_pause_upper_topic);
  cfg.pause_upper_init = absl::GetFlag(FLAGS_pause_upper_init);
  // PHASE-A start-pose align target: legs only (R^12); arms stay home.
  const double* start_pose =
      h12deploy::AlignPoseForStrategy(absl::GetFlag(FLAGS_strategy));
  for (int i = 0; i < 12; i++) cfg.align_pose[i] = start_pose[i];
  cfg.align_pose_set = true;
  return h12deploy::RunDeployNode(cfg);
}
