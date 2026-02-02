// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/planners/idto/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include <mujoco/mujoco.h>

#include "mjpc/planners/idto/policy.h"
#include "mjpc/planners/idto/settings.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

// Initialize data and settings
void IDTOPlanner::Initialize(mjModel* model, const Task& task) {
  // Store references
  model_ = model;
  task_ = &task;

  // Get dimensions
  dim_state_ = model->nq;
  dim_state_derivative_ = model->nv;
  dim_action_ = model->nu;
  dim_sensor_ = model->nsensordata;
  dim_max_ = std::max(std::max(dim_state_, dim_state_derivative_),
                      std::max(dim_action_, dim_sensor_));

  // Enable discrete inverse dynamics in model
  model_->opt.enableflags |= mjENBL_INVDISCRETE;

  // Initialize settings with defaults
  settings_ = IDTOSettings();

  // Initialize trust region
  trust_region_radius_ = settings_.trust_region_radius_initial;
  regularization_ = settings_.regularization_initial;

  // Initialize state
  state_.resize(dim_state_);
  mocap_.resize(7 * model->nmocap);
  userdata_.resize(model->nuserdata);

  // Initialize counters
  iterations_ = 0;
  cost_ = 0.0;
  cost_previous_ = 0.0;
  gradient_norm_ = 0.0;
  cost_improvement_ = 0.0;

  // Initialize timers
  trajectory_compute_time_ = 0.0;
  cost_compute_time_ = 0.0;
  gradient_compute_time_ = 0.0;
  hessian_compute_time_ = 0.0;
  solve_compute_time_ = 0.0;

  // GUI state
  max_iterations_gui_ = settings_.max_iterations;
}

// Allocate memory
void IDTOPlanner::Allocate() {
  // Get horizon (will be set properly in Reset)
  int max_horizon = kMaxTrajectory;

  // Allocate configuration trajectory
  configuration_.resize(dim_state_ * (max_horizon + 1));
  std::fill(configuration_.begin(), configuration_.end(), 0.0);

  // Allocate derived quantities
  velocity_.resize(dim_state_derivative_ * (max_horizon + 1));
  acceleration_.resize(dim_state_derivative_ * max_horizon);
  force_.resize(dim_state_derivative_ * max_horizon);

  // Allocate cost derivatives
  gradient_.resize(dim_state_ * (max_horizon + 1));
  // TODO(vincekurtz): proper banded Hessian allocation
  hessian_.resize(dim_state_ * (max_horizon + 1) * 5);  // Penta-diagonal

  // Allocate search direction
  search_direction_.resize(dim_state_ * (max_horizon + 1));

  // Allocate trajectory
  trajectory_.Initialize(dim_state_ + dim_state_derivative_, dim_action_,
                        dim_sensor_, 3, max_horizon);
  trajectory_.Allocate(max_horizon);

  // Allocate policy
  policy_.trajectory.Initialize(dim_state_ + dim_state_derivative_, dim_action_,
                               dim_sensor_, 3, max_horizon);
  policy_.trajectory.Allocate(max_horizon);
  policy_.configuration.resize(dim_state_ * (max_horizon + 1));
  policy_.velocity.resize(dim_state_derivative_ * (max_horizon + 1));
  policy_.force.resize(dim_state_derivative_ * max_horizon);
  policy_.times.resize(max_horizon + 1);
}

// Reset memory to zeros
void IDTOPlanner::Reset(int horizon,
                        const double* initial_repeated_action) {
  // Set horizon
  horizon_ = std::min(horizon, kMaxTrajectory);

  // Reset configuration to zeros (or initial state if available)
  std::fill(configuration_.begin(),
            configuration_.begin() + dim_state_ * (horizon_ + 1), 0.0);

  // Reset derived quantities
  std::fill(velocity_.begin(),
            velocity_.begin() + dim_state_derivative_ * (horizon_ + 1), 0.0);
  std::fill(acceleration_.begin(),
            acceleration_.begin() + dim_state_derivative_ * horizon_, 0.0);
  std::fill(force_.begin(),
            force_.begin() + dim_state_derivative_ * horizon_, 0.0);

  // Reset cost derivatives
  std::fill(gradient_.begin(),
            gradient_.begin() + dim_state_ * (horizon_ + 1), 0.0);

  // Reset trust region
  trust_region_radius_ = settings_.trust_region_radius_initial;
  regularization_ = settings_.regularization_initial;

  // Reset counters
  iterations_ = 0;
  cost_ = 0.0;
  cost_previous_ = 0.0;
  gradient_norm_ = 0.0;

  // Reset trajectory
  trajectory_.Reset(horizon_);

  // Reset policy
  policy_.trajectory.Reset(horizon_);
  policy_.timestep = model_->opt.timestep;
}

// Set current state
void IDTOPlanner::SetState(const State& state) {
  std::unique_lock<std::shared_mutex> lock(mtx_);

  // Copy state
  state_.assign(state.state().begin(), state.state().end());
  time_ = state.time();

  // Copy mocap
  if (!state.mocap().empty()) {
    mocap_.assign(state.mocap().begin(), state.mocap().end());
  }

  // Copy userdata
  if (!state.userdata().empty()) {
    userdata_.assign(state.userdata().begin(), state.userdata().end());
  }

  // Set initial configuration
  mju_copy(configuration_.data(), state_.data(), dim_state_);
}

// Optimize nominal policy using IDTO
void IDTOPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // Lock for thread safety
  std::unique_lock<std::shared_mutex> lock(mtx_);

  // Warm start from previous solution
  WarmStart();

  // Get max iterations (real-time mode may limit this)
  int max_iter = settings_.real_time_mode ?
                 settings_.real_time_max_iterations :
                 settings_.max_iterations;

  // Main optimization loop
  for (iterations_ = 0; iterations_ < max_iter; iterations_++) {
    Iteration(horizon, pool);

    // Check convergence
    if (gradient_norm_ < settings_.gradient_tolerance ||
        cost_improvement_ < settings_.cost_tolerance) {
      break;
    }
  }

  // Update policy from optimized trajectory
  UpdatePolicy();
}

// Update policy from current optimized trajectory
void IDTOPlanner::UpdatePolicy() {
  // Copy configuration trajectory
  policy_.configuration = configuration_;
  policy_.velocity = velocity_;
  policy_.force = force_;

  // Set timestep
  policy_.timestep = model_->opt.timestep;

  // Set times
  policy_.times.resize(horizon_ + 1);
  for (int t = 0; t <= horizon_; t++) {
    policy_.times[t] = time_ + t * policy_.timestep;
  }

  // Copy to trajectory object for visualization
  for (int t = 0; t <= horizon_; t++) {
    // State: [qpos, qvel]
    mju_copy(trajectory_.states.data() + t * (dim_state_ + dim_state_derivative_),
             configuration_.data() + t * dim_state_, dim_state_);
    mju_copy(trajectory_.states.data() + t * (dim_state_ + dim_state_derivative_) + dim_state_,
             velocity_.data() + t * dim_state_derivative_, dim_state_derivative_);

    // Time
    trajectory_.times[t] = policy_.times[t];

    // Action (force)
    if (t < horizon_) {
      mju_copy(trajectory_.actions.data() + t * dim_action_,
               force_.data() + t * dim_state_derivative_,
               std::min(dim_action_, dim_state_derivative_));
    }
  }

  trajectory_.total_return = cost_;
}

// Compute trajectory using nominal policy
void IDTOPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  // The IDTO trajectory IS the nominal trajectory
  // Just evaluate it to ensure velocity/acceleration/force are up to date
  EvaluateTrajectory();

  // Copy to trajectory object for visualization
  // TODO(vincekurtz): implement proper trajectory copying
}

// Set action from policy
void IDTOPlanner::ActionFromPolicy(double* action, const double* state,
                                  double time, bool use_previous) {
  // Lock for reading
  std::shared_lock<std::shared_mutex> lock(mtx_);

  const IDTOPolicy& policy = use_previous ? previous_policy_ : policy_;

  // Find the timestep corresponding to this time
  double t_policy = time - time_;
  int t_idx = static_cast<int>(t_policy / policy.timestep);

  // Clamp to valid range
  t_idx = std::max(0, std::min(t_idx, horizon_ - 1));

  // For now, just return the nominal force
  // TODO(vincekurtz): add interpolation and optional feedback
  if (!policy.force.empty() && t_idx < horizon_) {
    mju_copy(action, policy.force.data() + t_idx * dim_state_derivative_,
             dim_action_);
  } else {
    // No valid action, return zeros
    mju_zero(action, dim_action_);
  }
}

// Return trajectory with best total return
const Trajectory* IDTOPlanner::BestTrajectory() {
  return &trajectory_;
}

// Visualize planner-specific traces in GUI
void IDTOPlanner::Traces(mjvScene* scn) {
  // TODO(vincekurtz): implement trajectory visualization
  // Similar to ilqg/planner.cc Traces
}

// Planner-specific GUI elements
void IDTOPlanner::GUI(mjUI& ui) {
  // TODO(vincekurtz): add GUI controls for IDTO settings
  mjuiDef defIDTO[] = {
      {mjITEM_SECTION, "IDTO", 1, nullptr, "AP"},
      {mjITEM_SLIDERINT, "Max iterations", 2, &max_iterations_gui_, "1 200"},
      {mjITEM_END}};

  // Add UI elements
  mjui_add(&ui, defIDTO);

  // Update settings from GUI
  settings_.max_iterations = max_iterations_gui_;
}

// Planner-specific plots
void IDTOPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                       int planner_shift, int timer_shift, int planning,
                       int* shift) {
  // TODO(vincekurtz): implement plotting
  // Show cost, gradient norm, trust region radius, etc.
}

// ----- IDTO-specific methods ----- //

// Single IDTO optimization iteration
void IDTOPlanner::Iteration(int horizon, ThreadPool& pool) {
  auto start = std::chrono::steady_clock::now();

  // 1. Evaluate trajectory (compute v, a, tau from q)
  EvaluateTrajectory();

  auto eval_end = std::chrono::steady_clock::now();
  trajectory_compute_time_ =
      std::chrono::duration<double>(eval_end - start).count();

  // 2. Compute cost
  cost_previous_ = cost_;
  cost_ = ComputeCost();

  auto cost_end = std::chrono::steady_clock::now();
  cost_compute_time_ =
      std::chrono::duration<double>(cost_end - eval_end).count();

  // 3. Compute gradient
  ComputeGradient();

  auto grad_end = std::chrono::steady_clock::now();
  gradient_compute_time_ =
      std::chrono::duration<double>(grad_end - cost_end).count();

  // 4. Compute Hessian
  ComputeHessian();

  auto hess_end = std::chrono::steady_clock::now();
  hessian_compute_time_ =
      std::chrono::duration<double>(hess_end - grad_end).count();

  // 5. Solve trust-region subproblem
  SolveTrustRegionSubproblem(search_direction_);

  auto solve_end = std::chrono::steady_clock::now();
  solve_compute_time_ =
      std::chrono::duration<double>(solve_end - hess_end).count();

  // 6. Evaluate cost at new point
  std::vector<double> configuration_new = configuration_;
  for (int i = 0; i < dim_state_ * (horizon_ + 1); i++) {
    configuration_new[i] += search_direction_[i];
  }

  // Temporarily update configuration
  std::swap(configuration_, configuration_new);
  EvaluateTrajectory();
  double cost_new = ComputeCost();
  std::swap(configuration_, configuration_new);  // Restore

  // 7. Compute reduction ratio and update trust region
  double actual_reduction = cost_ - cost_new;

  // Predicted reduction from linear model: -g' * dq
  double predicted_reduction = 0.0;
  for (int i = 0; i < dim_state_ * (horizon_ + 1); i++) {
    predicted_reduction -= gradient_[i] * search_direction_[i];
  }

  // Compute reduction ratio
  double reduction_ratio = 0.0;
  if (std::abs(predicted_reduction) > 1e-12) {
    reduction_ratio = actual_reduction / predicted_reduction;
  }

  if (actual_reduction > 0.0 &&
      reduction_ratio > settings_.trust_region_shrink_threshold) {
    // Accept step
    configuration_ = configuration_new;
    cost_improvement_ = actual_reduction;
  } else {
    // Reject step
    cost_improvement_ = 0.0;
  }

  UpdateTrustRegion(actual_reduction, predicted_reduction);

  // Update gradient norm
  gradient_norm_ = 0.0;
  for (double g : gradient_) {
    gradient_norm_ += g * g;
  }
  gradient_norm_ = std::sqrt(gradient_norm_);
}

// Evaluate trajectory (compute v, a, tau from q)
void IDTOPlanner::EvaluateTrajectory() {
  // Core IDTO algorithm:
  // Given configuration trajectory q[0:T], compute:
  // - v[t] via finite differences
  // - a[t] via finite differences
  // - tau[t] via inverse dynamics

  double dt = model_->opt.timestep;

  // Compute velocities via finite differences: v[t] = (q[t] - q[t-1]) / dt
  for (int t = 0; t <= horizon_; t++) {
    if (t == 0) {
      // Initial velocity from current state
      mju_copy(velocity_.data(), state_.data() + dim_state_,
               dim_state_derivative_);
    } else {
      // Finite difference velocity
      for (int i = 0; i < dim_state_derivative_; i++) {
        velocity_[t * dim_state_derivative_ + i] =
            (configuration_[t * dim_state_ + i] -
             configuration_[(t - 1) * dim_state_ + i]) /
            dt;
      }
    }
  }

  // Compute accelerations via finite differences: a[t] = (v[t+1] - v[t]) / dt
  for (int t = 0; t < horizon_; t++) {
    for (int i = 0; i < dim_state_derivative_; i++) {
      acceleration_[t * dim_state_derivative_ + i] =
          (velocity_[(t + 1) * dim_state_derivative_ + i] -
           velocity_[t * dim_state_derivative_ + i]) /
          dt;
    }
  }

  // Compute forces via inverse dynamics: tau[t] = ID(q[t+1], v[t+1], a[t])
  // Use thread 0's data for inverse dynamics computation
  mjData* d = data_[0].get();

  for (int t = 0; t < horizon_; t++) {
    // Set state at t+1
    mju_copy(d->qpos, configuration_.data() + (t + 1) * dim_state_, dim_state_);
    mju_copy(d->qvel, velocity_.data() + (t + 1) * dim_state_derivative_,
             dim_state_derivative_);
    mju_copy(d->qacc, acceleration_.data() + t * dim_state_derivative_,
             dim_state_derivative_);

    // Compute inverse dynamics
    // mj_inverse requires forward kinematics to be up to date
    mj_kinematics(model_, d);
    mj_comPos(model_, d);
    mj_camlight(model_, d);
    mj_crb(model_, d);  // Composite rigid body inertia
    mj_inverse(model_, d);

    // Extract generalized forces
    // For actuated joints, extract from qfrc_inverse
    // Note: qfrc_inverse contains all generalized forces needed to produce qacc
    mju_copy(force_.data() + t * dim_state_derivative_, d->qfrc_inverse,
             dim_state_derivative_);
  }
}

// Compute cost
double IDTOPlanner::ComputeCost() {
  // IDTO cost: sum of stage costs along the trajectory
  // For each timestep, evaluate task residual and compute cost

  double total_cost = 0.0;
  mjData* d = data_[0].get();

  // Allocate residual storage
  std::vector<double> residual(task_->num_residual);

  for (int t = 0; t <= horizon_; t++) {
    // Set configuration and velocity
    mju_copy(d->qpos, configuration_.data() + t * dim_state_, dim_state_);
    mju_copy(d->qvel, velocity_.data() + t * dim_state_derivative_,
             dim_state_derivative_);

    // Set mocap and userdata
    if (!mocap_.empty()) {
      for (int i = 0; i < model_->nmocap; i++) {
        mju_copy(d->mocap_pos + 3 * i, mocap_.data() + 7 * i, 3);
        mju_copy(d->mocap_quat + 4 * i, mocap_.data() + 7 * i + 3, 4);
      }
    }
    if (!userdata_.empty()) {
      mju_copy(d->userdata, userdata_.data(), model_->nuserdata);
    }

    // Compute forward kinematics and sensors
    mj_forward(model_, d);

    // Compute task residual
    task_->Residual(model_, d, residual.data());

    // Compute stage cost
    double stage_cost = task_->CostValue(residual.data());
    total_cost += stage_cost;

    // Optional: add control cost (R * ||tau||^2)
    if (t < horizon_ && settings_.weight_control > 0.0) {
      double control_cost = 0.0;
      for (int i = 0; i < dim_state_derivative_; i++) {
        double tau = force_[t * dim_state_derivative_ + i];
        control_cost += tau * tau;
      }
      total_cost += settings_.weight_control * control_cost;
    }
  }

  // Normalize by horizon (following mjpc convention)
  if (horizon_ > 0) {
    total_cost /= (horizon_ + 1);
  }

  return total_cost;
}

// Compute gradient
void IDTOPlanner::ComputeGradient() {
  // Compute ∂L/∂q via finite differences
  // gradient_[t * dim_state + i] = ∂L/∂q[t][i]

  std::fill(gradient_.begin(),
            gradient_.begin() + dim_state_ * (horizon_ + 1), 0.0);

  const double eps = settings_.finite_difference_epsilon;
  const double baseline_cost = cost_;

  // Finite difference for each configuration variable
  for (int t = 0; t <= horizon_; t++) {
    for (int i = 0; i < dim_state_; i++) {
      int idx = t * dim_state_ + i;

      // Perturb configuration
      configuration_[idx] += eps;

      // Recompute trajectory and cost
      EvaluateTrajectory();
      double perturbed_cost = ComputeCost();

      // Finite difference gradient
      gradient_[idx] = (perturbed_cost - baseline_cost) / eps;

      // Restore configuration
      configuration_[idx] -= eps;
    }
  }

  // Restore trajectory
  EvaluateTrajectory();
}

// Compute Hessian
void IDTOPlanner::ComputeHessian() {
  // TODO(vincekurtz): implement Hessian computation (Gauss-Newton)
}

// Solve trust-region subproblem
void IDTOPlanner::SolveTrustRegionSubproblem(std::vector<double>& dq) {
  // Solve: min_dq  g'*dq + 0.5*dq'*H*dq
  //        s.t.    ||dq|| <= trust_region_radius
  //
  // For now: simple gradient descent with trust region constraint
  // Future: implement full Newton step with Hessian

  // Compute gradient norm
  double grad_norm = 0.0;
  for (double g : gradient_) {
    grad_norm += g * g;
  }
  grad_norm = std::sqrt(grad_norm);

  if (grad_norm < settings_.gradient_tolerance) {
    // Already at optimum, no step needed
    std::fill(dq.begin(), dq.end(), 0.0);
    return;
  }

  // Gradient descent direction
  double step_size = trust_region_radius_ / grad_norm;

  // Scale to stay within trust region
  for (int i = 0; i < dim_state_ * (horizon_ + 1); i++) {
    dq[i] = -step_size * gradient_[i];
  }
}

// Update trust region radius
void IDTOPlanner::UpdateTrustRegion(double actual_reduction,
                                   double predicted_reduction) {
  if (predicted_reduction <= 0) return;

  double ratio = actual_reduction / predicted_reduction;

  if (ratio > settings_.trust_region_expand_threshold) {
    // Good step, expand trust region
    trust_region_radius_ *= settings_.trust_region_expand_factor;
    trust_region_radius_ =
        std::min(trust_region_radius_, settings_.trust_region_radius_max);
  } else if (ratio < settings_.trust_region_shrink_threshold) {
    // Poor step, shrink trust region
    trust_region_radius_ *= settings_.trust_region_shrink_factor;
    trust_region_radius_ =
        std::max(trust_region_radius_, settings_.trust_region_radius_min);
  }
}

// Warm start from previous solution
void IDTOPlanner::WarmStart() {
  // Shift trajectory forward by one timestep
  for (int t = 0; t < horizon_; t++) {
    mju_copy(configuration_.data() + t * dim_state_,
             configuration_.data() + (t + 1) * dim_state_, dim_state_);
  }

  // Last timestep: duplicate
  mju_copy(configuration_.data() + horizon_ * dim_state_,
           configuration_.data() + (horizon_ - 1) * dim_state_, dim_state_);

  // Set initial configuration from current state
  mju_copy(configuration_.data(), state_.data(), dim_state_);
}

}  // namespace mjpc
