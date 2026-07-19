// MJPC embedded DDS control node for the Unitree H1-2 -- UPPER-BODY variant:
// nu=15 (torso + 2x7 arm joints, motor rows 12..26), task "Upper H12 Magpie",
// publishes the safety layer's split-mode UPPER channel rt/safety/lowcmd_upper_in.
// This is the sampling-MPC replacement for the FrameTask IK: joint-space goals
// arrive over the node's gRPC monitor port (SetTaskParameters "Goal Active"/
// "Goal Sec"/"Goal J0..J14" -- the same seam the twin precision gate used), the
// task transports along a min-jerk segment, and a sticky quiescence hold-latch
// freezes the executed command at the goal pose for mm-class held precision
// (P6.2 twin gate: EE bias 13 mm, held wobble 0.22 mm std). The legs are OWNED
// by the lower-body controller (rt/safety/lowcmd_lower_in); with --arm_aware
// (default; the complement here is the LEGS) this node's planner model tracks
// the MEASURED legs + base via retargeted equality locks and the pelvis
// world-weld, so the arm plan is computed against where the body actually is.
// This node NEVER commands the legs (rows 0..11 of its LowCmd are left default;
// the safety layer's split merge reads only rows 12..26 from this channel).
// FORK-BUILD ONLY today (no colcon target); superseded in bringup by the split
// core's pause-gated arm handover.
//
// Thin main: shared flags come from deploy_flags.inc (CLI CHANGE 2026-07-18:
// --leg_aware was renamed to the canonical --arm_aware -- same complement
// machinery, leg flavor); gain tables from h12_gain_tables.h (the kp-40 gate
// tables + TAU rows 12..26 of the canonical 27). Everything else is
// deploy_common.cc (FillCommonConfig -> NodeConfig -> RunDeployNode).

#include <string>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "mjpc/deploy/deploy_common.h"
#include "mjpc/deploy/h12_gain_tables.h"

// Node-specific overrides: reversed complement (legs), goal-ingest gRPC port,
// arm-row damp latch, no strategy pairing, inert ankle calibration.
#define H12_DEFAULT_GRPC_PORT 10001
#define H12_HELP_GRPC_PORT \
          "if >0, host an MJPC gRPC server on this port: the monitor can attach, AND this is " \
          "the GOAL-INGEST seam (SetTaskParameters Goal Active/Sec/J0..J14). DEFAULT 10001 so " \
          "a co-run with the lower node (10000) never collides; 0 disables (no goals!)."
#define H12_HELP_ARM_AWARE \
          "LEG-AWARE planning (this node's complement is the LEGS; the flag keeps the canonical " \
          "--arm_aware name, renamed from --leg_aware 2026-07-18): read the 12 leg joints " \
          "(motor 0..11) from rt/lowstate and retarget the planner model's leg equality locks + " \
          "pelvis world-weld to the MEASURED pose each tick, so arm plans are computed against " \
          "where the body ACTUALLY is (the lower-body MPC owns balance; this node never commands " \
          "the legs). --noarm_aware = plan against the frozen stand keyframe (bench isolation only)."
#define H12_HELP_BAD_ORIENT \
          "bad-orientation damp fallback (rad): base tilt beyond this latches a permanent " \
          "kp=0/kd=3/tau=0 damp command on the ARM rows until restart (the robot is falling; " \
          "stop driving the arms). 0 disables. Default 0.9 rad (~52 deg), matching the lower node."
#define H12_HELP_STRAIGHTEN_START \
          "hold the measured (slumped) pose, wait for ENTER, then hand authority to the planner " \
          "from the slump (SETTLE->BLEND, no drag). OFF by default. (The upper task has no " \
          "Strategy parameter.)"
#define H12_ANKLE_HELP_NOTE \
          " INERT ON THIS NODE: ankle calibration applies at motor rows 4/5/10/11, which this " \
          "upper-body node (motor_offset 12) does not touch; accepted for launcher-arg parity."
#include "mjpc/deploy/deploy_flags.inc"

ABSL_FLAG(std::string, task, "Upper H12 Magpie",
          "MJPC task id (upper-body nu=15 joint-goal task)");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  h12deploy::NodeConfig cfg;
  // kp-40 gate tables (h12_gain_tables.h): == the h1_2_modified actuator
  // classes == the generated upper planner model == the twin PD, so
  // PatchActuators is numerically a no-op and the deployed planner model stays
  // byte-identical to the P6.2 precision-gate one. TAU tables = rows 12..26 of
  // the canonical 27. NO forcerange patch: the generated model's arm
  // forceranges are already estop-bounded (32/14.4/9.5).
  cfg.nu = h12deploy::kUpperCount;
  cfg.kp = h12deploy::kUpperKpGate15;
  cfg.kv = h12deploy::kUpperKvGate15;
  cfg.tau_estop = h12deploy::kTauEstop27 + h12deploy::kUpperOffset;
  cfg.tau_limit = h12deploy::kTauLimit27 + h12deploy::kUpperOffset;
  cfg.frc_limit = nullptr;
  cfg.frc_limit_begin = 0;
  cfg.joint_names = h12deploy::kJointNames27 + h12deploy::kUpperOffset;
  cfg.telemetry = h12deploy::Telemetry::kUpperBody;
  cfg.motor_offset = h12deploy::kUpperOffset;  // actuator i <-> motor 12+i
  cfg.upper_count = h12deploy::kLegsCount;     // complement = the 12 LEGS ...
  cfg.comp_motor_offset = 0;                   // ... at motor rows 0..11

  h12deploy::FillCommonConfig(&cfg);
  cfg.task_id = absl::GetFlag(FLAGS_task);
  cfg.strategy = 0;           // the upper task has no Strategy parameter (inert)
  cfg.lowcmd_topic = "rt/safety/lowcmd_upper_in";   // safety layer split-mode UPPER channel
  return h12deploy::RunDeployNode(cfg);
}
