// MJPC embedded DDS control node for the Unitree H1-2 (digital twin + real robot).
// FULL-BODY variant: nu=27 (legs + torso + arms), task "Lean H12 Magpie",
// publishes the safety layer's full-body channel rt/safety/lowcmd_in.
// See mjpc/deploy/README.md.  Build: cmake .. -DMJPC_BUILD_DEPLOY=ON  (needs unitree_sdk2).
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
// Thin main: shared flags come from deploy_flags.inc, gain tables from
// h12_gain_tables.h; everything else is deploy_common.cc (FillCommonConfig ->
// NodeConfig -> RunDeployNode). --align_start here aligns to the model's
// 'stand' keyframe (no compiled leg pose -- that is the leg-owning mains').

#include <string>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "mjpc/deploy/deploy_common.h"
#include "mjpc/deploy/h12_gain_tables.h"

// Node-specific overrides: this node actuates the arms itself.
#define H12_DEFAULT_BAD_ORIENT 0.0
#define H12_HELP_BAD_ORIENT \
          "R6 bad-orientation damp fallback (rad); 0 = OFF (default for the full-body/lean node -- " \
          "validated deployments unchanged). See h12_lower_body_controller for the rationale."
#define H12_HELP_ARM_AWARE \
          "ACCEPTED FOR LAUNCHER-ARG PARITY ONLY: the full-body node actuates the arms itself " \
          "(upper_count=0) and hardcodes the complement machinery OFF -- this flag has NO effect " \
          "here. It exists so the shared launchers can pass --arm_aware to every core."
#include "mjpc/deploy/deploy_flags.inc"

ABSL_FLAG(std::string, task, "Lean H12 Magpie", "MJPC task id");
ABSL_FLAG(int, strategy, 6,
          "Lean Strategy parameter (6=stand 8=crouch 11=arms_overhead 13=lean_left "
          "16=counterbalance 18=squatter 20=stumble 21=reach 23=trot "
          "25=straighten/bring-up 31-35=lean ladder ...). For slumped/leaning "
          "power-on boot with --strategy 25: phase 0 drives to upright+centered "
          "from the measured pose, then a basin gate (tilt<=3deg, z>=1.00, "
          "knees<=0.50rad i.e. legs actually extended, quiescent, 1.5s "
          "sustained) AUTO-ADVANCES into a stand phase running "
          "strat-6 weights/keyframe -- no manual switch needed; live-switch to "
          "other strategies as usual afterward.");

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
  cfg.upper_count = 0;        // full-body node actuates the arms itself

  h12deploy::FillCommonConfig(&cfg);
  cfg.task_id = absl::GetFlag(FLAGS_task);
  cfg.strategy = absl::GetFlag(FLAGS_strategy);
  cfg.lowcmd_topic = "rt/safety/lowcmd_in";   // safety layer full-body channel
  cfg.arm_aware = false;      // no complement on this node (--arm_aware is inert here)
  return h12deploy::RunDeployNode(cfg);
}
