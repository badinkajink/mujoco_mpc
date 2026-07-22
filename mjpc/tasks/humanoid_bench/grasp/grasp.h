// Full-body MPC GRASP controller task (2026-07-20). Plan:
// h12/docs/grasp_controller_plan_2026-07-20.md.
//
// The grasp task is a whole-body (nu=27, arms ACTUATED) reach-to-a-6-DOF-pose
// controller that stands AND brings the gripper to a perception-supplied grasp
// pose while balancing, then holds for the Magpie gripper to close. It is a thin
// subclass of the validated full-body Lean task (Lean_H12_Magpie -- the SAME
// task h12_control_node runs by default), inheriting all of lean's stand / reach
// / balance / transition machinery with ZERO logic duplication. Only the task
// name and (from P1) the model path + grasp-specific reach residual diverge.
//
// P0 (skeleton that STANDS): reuse lean's proven whole-body model verbatim so the
// grasp task stands + reaches exactly like Lean H12 Magpie under strategy 6/21.
// P1 swaps XmlPath() to a grasp wrapper model that adds the quaternion target
// sensor for the 6-DOF (mju_subQuat log-map) orientation residual.
#ifndef MJPC_TASKS_HUMANOID_BENCH_GRASP_GRASP_H_
#define MJPC_TASKS_HUMANOID_BENCH_GRASP_GRASP_H_

#include <string>
#include <vector>

#include "mjpc/tasks/humanoid_bench/lean/lean.h"

namespace mjpc {

// Whole-body grasp task. Full-body base = Lean_H12_Magpie (nu=27, actuated arms,
// the h12_control_node default), NOT Stabilize (nu=12 legs-only, arms eq-locked).
class Grasp_H12_Magpie : public Lean_H12_Magpie {
 public:
  std::string Name() const override { return "Grasp H12 Magpie"; }

  // Grasp OWNS its model: a copy of lean's validated whole-body magpie model
  // (same robot include, so mesh resolution is identical to lean's) with the
  // `lean.xml` include repointed to ../lean/ and a `right_hand_quat` framequat
  // sensor appended (6-DOF orientation telemetry / future use). The RESIDUAL
  // ENGINE stays lean's (inherited) -- a grasp.cc copying lean.cc's 300 KB would
  // instantly drift from the tuned original, so it is intentionally absent.
  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/grasp/Grasp_H12_Magpie.xml");
  }

  // Grasp reuses lean's whole strategy roster (stand=6, drive=24, ...) but swaps
  // the REACH slot (21) to a grasp-OWNED strategy whose "Reaching Hand Dist"
  // weight DOMINATES (1500 vs lean's 80). Twin measurement (2026-07-20, pelvis-
  // fixed): dominant weight drops the isolated reach bias 9.9->4.1 cm AND creates
  // the sustained dwell (convergence NULL->2.6 s) that the deploy-side hold-latch
  // / IK-seed PD-hold needs. Strategy lives in grasp/strategies/ (grasp OWNS it);
  // the inherited loader reads from lean/strategies/ but MotionStrategy::Load just
  // does path+name+".json", so a "../../grasp/strategies/..." NAME redirects the read
  // into grasp/ with ZERO lean edit. 6-DOF ORIENTATION is delivered by the mink IK
  // q* + terminal PD-hold in the orchestrator (P2/P5), NOT an MPC quaternion
  // residual -- so lean.cc's reach (position + torso-lean) is reused unchanged.
  std::vector<std::string> GetStrategyNames() const override {
    std::vector<std::string> names = Lean_H12_Magpie::GetStrategyNames();
    if (names.size() > 21) names[21] = "../../grasp/strategies/h12_grasp_reach";
    return names;
  }
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_GRASP_GRASP_H_
