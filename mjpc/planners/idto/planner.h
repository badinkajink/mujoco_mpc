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

#ifndef MJPC_PLANNERS_IDTO_PLANNER_H_
#define MJPC_PLANNERS_IDTO_PLANNER_H_

#include <shared_mutex>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/planners/idto/policy.h"
#include "mjpc/planners/idto/settings.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/trajectory.h"

namespace mjpc {

// IDTO (Inverse Dynamics Trajectory Optimization) planner
// Based on: Kurtz et al., "Inverse Dynamics Trajectory Optimization for
// Contact-Implicit Model Predictive Control" (2023)
// https://arxiv.org/abs/2309.01813
class IDTOPlanner : public Planner {
 public:
  // constructor
  IDTOPlanner() = default;

  // destructor
  ~IDTOPlanner() override = default;

  // ----- Planner interface implementation ----- //

  // Initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // Allocate memory
  void Allocate() override;

  // Reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // Set current state
  void SetState(const State& state) override;

  // Optimize nominal policy using IDTO
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // Compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // Set action from policy
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;

  // Return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // Visualize planner-specific traces in GUI
  void Traces(mjvScene* scn) override;

  // Planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // Planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // Return number of parameters optimized by planner
  int NumParameters() override {
    return dim_state_ * (horizon_ + 1);  // Optimize all configurations
  }

  // ----- IDTO-specific methods ----- //

  // Single IDTO optimization iteration
  void Iteration(int horizon, ThreadPool& pool);

  // Evaluate trajectory (compute v, a, tau from q)
  void EvaluateTrajectory();

  // Compute cost and derivatives
  double ComputeCost();
  void ComputeGradient();
  void ComputeHessian();

  // Solve trust-region subproblem
  void SolveTrustRegionSubproblem(std::vector<double>& dq);

  // Update trust region radius based on reduction ratio
  void UpdateTrustRegion(double actual_reduction, double predicted_reduction);

  // Warm start from previous solution
  void WarmStart();

  // Update policy from optimized trajectory
  void UpdatePolicy();

  // ----- Members ----- //

  // Model and task
  mjModel* model_ = nullptr;
  const Task* task_ = nullptr;

  // State
  std::vector<double> state_;
  double time_;
  std::vector<double> mocap_;
  std::vector<double> userdata_;

  // Policy
  IDTOPolicy policy_;
  IDTOPolicy previous_policy_;

  // Dimensions
  int dim_state_;             // State dimension (nq)
  int dim_state_derivative_;  // State derivative dimension (nv)
  int dim_action_;            // Action dimension (nu)
  int dim_sensor_;            // Sensor dimension
  int dim_max_;               // Maximum dimension

  // Horizon
  int horizon_;

  // Decision variables: configuration trajectory q[0:T]
  std::vector<double> configuration_;  // [nq * (T+1)]

  // Cached derived quantities
  std::vector<double> velocity_;       // [nv * (T+1)]
  std::vector<double> acceleration_;   // [nv * T]
  std::vector<double> force_;          // [nv * T]

  // Cost and derivatives
  double cost_;
  double cost_previous_;
  std::vector<double> gradient_;  // [nq * (T+1)]
  std::vector<double> hessian_;   // Stored as band matrix

  // Search direction
  std::vector<double> search_direction_;  // [nq * (T+1)]

  // Trust region
  double trust_region_radius_;

  // Regularization
  double regularization_;

  // Settings
  IDTOSettings settings_;

  // Status
  int iterations_;
  double gradient_norm_;
  double cost_improvement_;

  // Compute times
  double trajectory_compute_time_;
  double cost_compute_time_;
  double gradient_compute_time_;
  double hessian_compute_time_;
  double solve_compute_time_;

  // Trajectory for visualization
  Trajectory trajectory_;

  // Mutex for thread safety
  mutable std::shared_mutex mtx_;

 private:
  // GUI state
  int max_iterations_gui_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_IDTO_PLANNER_H_
