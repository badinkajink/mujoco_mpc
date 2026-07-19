// MJPC embedded DDS control node for the Unitree H1-2 -- LOWER-BODY variant:
// nu=12 (legs only), task "Stabilize H12 Magpie", publishes the safety layer's
// split-mode LOWER channel rt/safety/lowcmd_lower_in. Torso + arms are owned by
// the upper-body FrameTask IK (rt/safety/lowcmd_upper_in); this node balances
// UNDER whatever the arms do -- with --arm_aware (default) the planner's
// CoM/dynamics track the MEASURED upper joints via retargeted equality locks,
// so the legs PRE-compensate for a reach instead of only reacting to base tilt.
// LEGACY chain: bringup launches the split core (mjpc_split_core) instead;
// this binary is kept for the legs-only Stabilize workflow.
//
// Thin main: shared flags come from deploy_flags.inc, gain tables (rows 0..11
// of the canonical 27) from h12_gain_tables.h, start poses from
// h12_start_poses.h; everything else is deploy_common.cc (FillCommonConfig ->
// NodeConfig -> RunDeployNode).

#include <string>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "mjpc/deploy/deploy_common.h"
#include "mjpc/deploy/h12_gain_tables.h"
#include "mjpc/deploy/h12_start_poses.h"

#include "mjpc/deploy/deploy_flags.inc"

ABSL_FLAG(std::string, task, "Stabilize H12 Magpie",
          "MJPC task id (lower-body nu=12 stabilize task)");
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
          "   25  = straighten (pre-stand slump recovery; pair with --straighten_start)\n"
          "   26  = lockstand (locked-knee wide-stance strut hold; this main also "
          "swaps the --align_start target to the wide+locked lockstand pose)\n"
          "  20/22/23/24 are the STEPPING family: they share the gait clock and the "
          "ModifyControl swing forcer, and get spline 5 / 17 trajectories via "
          "PlannerNumericOverrides (17+1 = one thread wave -- see the plan-rate note "
          "in stabilize.cc; 36 traj starved the planner to 27-30 plans/s on real).");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  h12deploy::NodeConfig cfg;
  // Legs = rows 0..11 of the canonical tables (h12_gain_tables.h). Ankle kp/kv
  // stay 80/4 -- softening was A/B-tested and REJECTED (stiffness is
  // load-bearing; history: see mjpc/deploy/HISTORY.md).
  cfg.nu = h12deploy::kLegsCount;
  cfg.kp = h12deploy::kKp27;
  cfg.kv = h12deploy::kKv27;
  cfg.tau_estop = h12deploy::kTauEstop27;
  cfg.tau_limit = h12deploy::kTauLimit27;
  cfg.frc_limit = nullptr;    // no forcerange patch (legs stay at model default --
  cfg.frc_limit_begin = 0;    // tightening them regressed the hold)
  cfg.joint_names = h12deploy::kJointNames27;
  cfg.telemetry = h12deploy::Telemetry::kLowerBody;
  cfg.upper_count = h12deploy::kUpperCount;  // read torso+arms (motor 12..26)
                                             // for arm-aware balance

  h12deploy::FillCommonConfig(&cfg);
  cfg.task_id = absl::GetFlag(FLAGS_task);
  cfg.strategy = absl::GetFlag(FLAGS_strategy);
  cfg.lowcmd_topic = "rt/safety/lowcmd_lower_in";   // safety layer split-mode LOWER channel
  // PHASE-A start-pose align target: legs only (R^12), h12_start_poses.h.
  const double* start_pose =
      h12deploy::AlignPoseForStrategy(absl::GetFlag(FLAGS_strategy));
  for (int i = 0; i < 12; i++) cfg.align_pose[i] = start_pose[i];
  cfg.align_pose_set = true;
  return h12deploy::RunDeployNode(cfg);
}
