// Copyright 2022 DeepMind Technologies Limited
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
//
// ============================================================================
// CoVO-MPC (Covariance-Optimal MPC), after Yi et al., L4DC 2024
// (arXiv:2401.07369).
//
// PAPER IDEA. CoVO-MPC replaces the isotropic sampling covariance of vanilla
// sampling MPC (MPPI) with a covariance DESIGNED from the local cost landscape:
// samples are drawn from N(nominal, Sigma) with Sigma related to the inverse
// curvature (Hessian) of the cost-to-go around the nominal, so that samples are
// stretched along low-curvature (flat) directions and compressed along
// high-curvature directions. The nominal is then updated with the usual
// MPPI-style exponentially-weighted average of the samples.
//
// WHAT THIS IMPLEMENTATION DOES (documented approximation). MJPC is a
// derivative-free sampling controller, so we do NOT form the full dense Hessian
// of the paper. Instead we implement the paper's tractable DIAGONAL variant:
//
//   1. Each planning step draws `num_trajectory` samples around the resampled
//      nominal using a PER-DIMENSION (diagonal) covariance `variance[d]`
//      persisted across steps (initialized isotropically).
//   2. From this same batch (no extra rollouts) we estimate a diagonal
//      curvature h_d for each control parameter d by regressing the sample
//      cost J_i onto the squared perturbation (delta_{i,d})^2. Because the
//      perturbations are zero-mean and symmetric, the linear term averages out
//      and
//              h_d = 2 * cov_i( J_i , delta_{i,d}^2 ) / var_i( delta_{i,d}^2 ).
//      This is a finite-sample, diagonal, second-difference estimate of the
//      cost curvature along each nominal-perturbation direction.
//   3. The next step's per-dimension sampling std is set to the inverse-sqrt
//      curvature law from the task spec,
//              std_d = covo_scale / sqrt(max(h_d, eps)),
//      clamped to [std_min, std_max]. This realizes Sigma ~ (diag Hessian)^-1
//      up to the `covo_scale` factor (the paper's temperature/step scaling).
//   4. The nominal is updated by an MPPI-style softmax-weighted average of all
//      samples, w_i = exp(-(J_i - J_min)/lambda), lambda = `covo_temperature`.
//
// DIFFERENCES vs the paper: diagonal (not full) covariance; curvature is
// estimated empirically from the sample batch rather than via autodiff of the
// dynamics/cost; no explicit optimal step-size / horizon-dependent scaling
// (folded into covo_scale). The interface, sampling, and MPPI fold are
// faithful; the covariance design is the documented diagonal approximation.
//
// XML knobs: `covo_scale` (default 1.0), `covo_temperature` (default 1.0),
// plus the shared `sampling_exploration` (initial std), `std_min`,
// `sampling_trajectories`, `sampling_spline_points`.
// ============================================================================

#ifndef MJPC_PLANNERS_COVO_PLANNER_H_
#define MJPC_PLANNERS_COVO_PLANNER_H_

#include <atomic>
#include <shared_mutex>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"

namespace mjpc {

class CoVOPlanner : public Planner {
 public:
  // constructor
  CoVOPlanner() = default;

  // destructor
  ~CoVOPlanner() override = default;

  // ----- methods ----- //

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // set state
  void SetState(const State& state) override;

  // optimize nominal policy using CoVO sampling + MPPI fold
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;
  void NominalTrajectory(int horizon);

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;

  // resample nominal policy
  void ResamplePolicy(int horizon);

  // add noise to nominal policy
  void AddNoiseToPolicy(int i, double std_min);

  // compute candidate trajectories
  void Rollouts(int num_trajectory, int horizon, ThreadPool& pool);

  // estimate diagonal cost curvature and update the sampling covariance
  void UpdateCovariance(int num_trajectory);

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // return number of parameters optimized by planner
  int NumParameters() override {
    return policy.num_spline_points * policy.model->nu;
  };

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  SamplingPolicy policy;  // (Guarded by mtx_)
  SamplingPolicy candidate_policy[kMaxTrajectory];
  SamplingPolicy resampled_policy;
  SamplingPolicy previous_policy;

  // scratch
  std::vector<double> parameters_scratch;
  std::vector<double> times_scratch;

  // trajectories
  Trajectory trajectory[kMaxTrajectory];
  Trajectory nominal_trajectory;

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;

  // ----- noise ----- //
  double std_initial_;  // initial standard deviation N(0, std)
  double std_min_;      // the minimum allowable std
  double std_max_;      // the maximum allowable std (clamp for stability)
  std::vector<double> noise;
  std::vector<double> variance;  // per-parameter sampling variance (diagonal)

  // ----- CoVO ----- //
  double covo_scale_ = 1.0;        // scale on inverse-sqrt-curvature std
  double covo_temperature_ = 1.0;  // MPPI lambda for the fold

  // improvement
  double improvement;

  // timing
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  mjpc::spline::SplineInterpolation interpolation_ =
      mjpc::spline::SplineInterpolation::kZeroSpline;
  int num_trajectory_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_COVO_PLANNER_H_
