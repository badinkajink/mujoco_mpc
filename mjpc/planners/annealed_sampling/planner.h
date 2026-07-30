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

#ifndef MJPC_PLANNERS_ANNEALED_SAMPLING_PLANNER_H_
#define MJPC_PLANNERS_ANNEALED_SAMPLING_PLANNER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include <vector>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"

namespace mjpc {

class AnnealedSamplingPlanner : public RankedPlanner {
 public:
  AnnealedSamplingPlanner() = default;
  ~AnnealedSamplingPlanner() override = default;

  // ----- methods ----- //

  void Initialize(mjModel* model, const Task& task) override;
  void Allocate() override;
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;
  void SetState(const State& state) override;
  void OptimizePolicy(int horizon, ThreadPool& pool) override;
  void NominalTrajectory(int horizon, ThreadPool& pool) override;
  void ActionFromPolicy(double* action, const double* state,
                        double time, bool use_previous = false) override;

  // resample nominal policy
  void UpdateNominalPolicy(int horizon);

  // add noise to nominal policy (with annealing schedule)
  void AddNoiseToPolicy(double start_time, int i);

  // compute candidate trajectories
  void Rollouts(int num_trajectory, int horizon, ThreadPool& pool);

  // MPPI cost-weighted policy update
  void MPPIUpdate(int num_trajectory);

  const Trajectory* BestTrajectory() override;
  void Traces(mjvScene* scn) override;
  void GUI(mjUI& ui) override;
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  int NumParameters() override {
    return policy.num_spline_points * model->nu;
  };

  int OptimizePolicyCandidates(int ncandidates, int horizon,
                               ThreadPool& pool) override;
  double CandidateScore(int candidate) const override;
  void ActionFromCandidatePolicy(double* action, int candidate,
                                 const double* state, double time) override;
  void CopyCandidateToPolicy(int candidate) override;

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
  SamplingPolicy previous_policy;

  // scratch
  mjpc::spline::TimeSpline plan_scratch;

  // trajectories
  Trajectory trajectory[kMaxTrajectory];

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;

  // ----- noise ----- //
  double noise_exploration[2] = {0};
  std::vector<double> noise;
  mjpc::spline::SplineInterpolation interpolation_ =
      mjpc::spline::SplineInterpolation::kZeroSpline;

  // best trajectory
  int winner;

  // improvement
  double improvement;

  // flags
  int processed_noise_status;

  // timing
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  // If true, use sliding plans (no resampling)
  std::uint8_t sliding_plan_ = false;

  int num_trajectory_;
  mutable std::shared_mutex mtx_;

  // ----- DIAL-MPC parameters ----- //

  // MPPI temperature (lambda): lower = sharper selection, higher = softer
  double temperature_ = 1.0;

  // dual-loop annealing
  int num_annealing_iters_ = 4;    // N: inner loop iterations
  double beta_trajectory_ = 0.5;   // trajectory-level annealing temperature
  double beta_action_ = 0.5;       // action-level annealing temperature

  // internal state for current annealing iteration
  int current_annealing_iter_ = 0;
  int current_annealing_total_ = 1;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_ANNEALED_SAMPLING_PLANNER_H_
