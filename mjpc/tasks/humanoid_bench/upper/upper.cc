// Upper-body MPC controller task -- see upper.h and the plan doc
// h12/upper_body_mpc_controller_plan_2026-07-09.md.

#include "mjpc/tasks/humanoid_bench/upper/upper.h"

#include <algorithm>
#include <cstring>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {

namespace {
// min-jerk position/. s in [0,1]
inline double MinJerk(double tau) {
  if (tau <= 0.0) return 0.0;
  if (tau >= 1.0) return 1.0;
  return tau * tau * tau * (10.0 + tau * (-15.0 + 6.0 * tau));
}
}  // namespace

void Upper::GoalAt(double t, const mjModel *model, double *q_ref) const {
  double s = 1.0;
  if (goal_active_) s = MinJerk((t - goal_t0_) / goal_T_);
  for (int i = 0; i < model->nu && i < 15; i++) {
    q_ref[i] = goal_active_
                   ? goal_q0_[i] + (goal_qg_[i] - goal_q0_[i]) * s
                   : goal_qg_[i];   // never armed: hold home/goal default
  }
}

// terms: [Joint Goal 15][Joint Vel 15] -- must match the XML sensor list.
void Upper::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                 double *residual) const {
  int counter = 0;
  double q_ref[15] = {0};
  upper_->GoalAt(data->time, model, q_ref);
  for (int i = 0; i < model->nu; i++) {
    int jid = model->actuator_trnid[2 * i];
    residual[counter++] = data->qpos[model->jnt_qposadr[jid]] - q_ref[i];
  }
  for (int i = 0; i < model->nu; i++) {
    int jid = model->actuator_trnid[2 * i];
    residual[counter++] = data->qvel[model->jnt_dofadr[jid]];
  }
  // sanity vs sensor dims (house pattern)
  CheckSensorDim(model, counter);
}

void Upper::TransitionLocked(mjModel *model, mjData *data) {
  // ---- leg_aware (mirror of stabilize arm_aware F1-A): retarget the leg
  // eq-locks and the pelvis world-weld to the MEASURED pose, once per plan
  // under the transition lock (rollout workers fan out after this, so the
  // shared eq_data write is thread-safe). Every rollout then holds the lower
  // body where it ACTUALLY is, instead of frozen at home.
  for (int e = 0; e < model->neq; e++) {
    if (model->eq_type[e] == mjEQ_JOINT) {
      int jid = model->eq_obj1id[e];
      model->eq_data[e * mjNEQDATA + 0] = data->qpos[model->jnt_qposadr[jid]];
    } else if (model->eq_type[e] == mjEQ_WELD) {
      // body1-to-world weld: relpose (data[3..9]) = measured base pos+quat.
      for (int k = 0; k < 3; k++)
        model->eq_data[e * mjNEQDATA + 3 + k] = data->qpos[k];
      for (int k = 0; k < 4; k++)
        model->eq_data[e * mjNEQDATA + 6 + k] = data->qpos[3 + k];
    }
  }

  // ---- goal-segment latch ("Goal Active" rising edge) ------------------- //
  if ((int)parameters.size() > kUpperGoalJ0ParameterIndex + 14) {
    bool want = parameters[kUpperGoalActiveParameterIndex] > 0.5;
    if (want && !goal_active_) {
      goal_t0_ = data->time;
      goal_T_ = std::max(0.2, (double)parameters[kUpperGoalSecParameterIndex]);
      for (int i = 0; i < model->nu && i < 15; i++) {
        int jid = model->actuator_trnid[2 * i];
        goal_q0_[i] = data->qpos[model->jnt_qposadr[jid]];
        goal_qg_[i] = parameters[kUpperGoalJ0ParameterIndex + i];
      }
      goal_active_ = true;
    } else if (!want && goal_active_) {
      // drop back to plain hold-at-goal (segment forgotten, target = qg)
      goal_active_ = false;
    }
  }

  // ---- sticky hold-latch state machine (real plant state, once per plan) --
  bool complete = !goal_active_ || data->time >= goal_t0_ + goal_T_;
  double max_err = 0.0, max_vel = 0.0;
  for (int i = 0; i < model->nu && i < 15; i++) {
    int jid = model->actuator_trnid[2 * i];
    max_err = std::max(max_err,
                       std::abs(data->qpos[model->jnt_qposadr[jid]] - goal_qg_[i]));
    max_vel = std::max(max_vel, std::abs(data->qvel[model->jnt_dofadr[jid]]));
  }
  if (hold_latched_) {
    if (max_err > 0.10 || !complete) {   // excursion / new goal -> release
      hold_latched_ = false;
      hold_ok_since_ = -1.0;
    }
  } else if (complete && max_err < 0.06 && max_vel < 0.5) {
    if (hold_ok_since_ < 0.0) hold_ok_since_ = data->time;
    else if (data->time - hold_ok_since_ >= 0.3) hold_latched_ = true;
  } else {
    hold_ok_since_ = -1.0;
  }
}

// HOLD-LATCH: while latched (sticky flag from TransitionLocked's quiescence
// state machine), hard-write ctrl to the goal pose -- the sampler's
// steady-state hunting (P0 probe: 14-18 mm std at the hand) is removed from
// the EXECUTED action; the planner keeps planning underneath and any
// excursion releases the latch on the next plan cycle.
void Upper::ModifyControl(const mjModel *model, const double *qpos,
                          const double *qvel, double time,
                          double *ctrl) const {
  if (!hold_latched_) return;
  for (int i = 0; i < model->nu && i < 15; i++) ctrl[i] = goal_qg_[i];
}

}  // namespace mjpc
