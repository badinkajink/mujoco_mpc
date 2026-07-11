// Upper-body MPC controller task (nu=15: torso + 14 arms) -- the sampling-MPC
// FrameTask replacement. Plan doc: h12/upper_body_mpc_controller_plan_2026-07-09.md.
//
// Legs are unactuated + eq-locked; TransitionLocked retargets the leg locks
// and the pelvis world-weld to the MEASURED pose every plan ("leg_aware",
// mirror of stabilize's arm_aware F1-A). Goals are joint-space (hybrid
// default after the P0 precision probe): "Goal Active" rising edge latches a
// min-jerk segment from the measured upper pose to "Goal J0..J14" over
// "Goal Sec" -- the residual tracks the moving target, so goal duration is
// REAL trajectory time (unlike FrameTask, where duration was a timeout).
// ModifyControl provides the mm-class HOLD: once the segment is complete and
// the plant is quiescent at the goal, the executed ctrl is hard-written to
// the goal pose (sampler wobble removed); any disturbance breaks quiescence
// and returns authority to the planner.

#ifndef MJPC_TASKS_HUMANOID_BENCH_UPPER_UPPER_H_
#define MJPC_TASKS_HUMANOID_BENCH_UPPER_UPPER_H_

#include <memory>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {

// parameter indices (XML residual_ numeric order in Upper_H12_Magpie.xml)
constexpr int kUpperGoalActiveParameterIndex = 0;
constexpr int kUpperGoalSecParameterIndex = 1;
constexpr int kUpperGoalJ0ParameterIndex = 2;   // J0..J14 -> 2..16

class Upper : public Task {
 public:
  std::string Name() const override = 0;
  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Upper *task)
        : mjpc::BaseResidualFn(task), upper_(task) {}
    // terms (order == sensor list): Joint Goal (15), Joint Vel (15)
    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

   private:
    const Upper *upper_;
  };

  Upper() : residual_(this) {}

  // leg_aware + pelvis-weld retarget, goal-segment latch. Once per plan,
  // under the transition lock, before rollout workers fan out.
  void TransitionLocked(mjModel *model, mjData *data) override;

  // hold-latch: quiescent at a completed goal -> ctrl hard-written to the
  // goal pose (executed action AND rollouts -- both paths call this).
  void ModifyControl(const mjModel *model, const double *qpos,
                     const double *qvel, double time,
                     double *ctrl) const override;

  // goal target at time t: min-jerk between latched q0 and goal (per dof of
  // the 15 upper actuators, actuator order). Used by residual + hold check.
  void GoalAt(double t, const mjModel *model, double *q_ref) const;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;

  // goal-segment state; written ONLY in TransitionLocked (same fan-out
  // guarantee as stabilize's F1-A shared-eq_data write).
  bool goal_active_ = false;      // segment latched (may be complete)
  double goal_t0_ = 0.0;
  double goal_T_ = 1.0;
  double goal_q0_[15] = {0};
  double goal_qg_[15] = {0};
  // sticky hold-latch (decided in TransitionLocked on the REAL plant state,
  // once per plan under the lock -- catch-march pattern): enter after 0.3 s
  // of quiescence at a completed goal, exit on excursion. ModifyControl just
  // reads the flag, so the executed ctrl doesn't flicker between the sampler
  // and the hold on borderline ticks.
  bool hold_latched_ = false;
  double hold_ok_since_ = -1.0;
  friend class ResidualFn;
};

class Upper_H12_Magpie : public Upper {
 public:
  std::string Name() const override { return "Upper H12 Magpie"; }
  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/upper/Upper_H12_Magpie.xml");
  }
};

}  // namespace mjpc
#endif  // MJPC_TASKS_HUMANOID_BENCH_UPPER_UPPER_H_
