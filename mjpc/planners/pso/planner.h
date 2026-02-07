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

#ifndef MJPC_PLANNERS_PSO_PLANNER_H_
#define MJPC_PLANNERS_PSO_PLANNER_H_

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

// Particle Swarm Optimization planner
// A population-based metaheuristic that optimizes by iteratively improving
// candidate solutions (particles) with regard to a given cost function.
class PSOPlanner : public RankedPlanner {
 public:
  // constructor
  PSOPlanner() = default;

  // destructor
  ~PSOPlanner() override = default;

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

  // optimize nominal policy using PSO
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;

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
    return policy.num_spline_points * model->nu;
  }

  // ----- RankedPlanner interface ----- //

  // optimizes policies and returns number of candidates
  int OptimizePolicyCandidates(int ncandidates, int horizon,
                               ThreadPool& pool) override;

  // returns the total return for the nth candidate
  double CandidateScore(int candidate) const override;

  // set action from candidate policy
  void ActionFromCandidatePolicy(double* action, int candidate,
                                 const double* state, double time) override;

  // copies the nth candidate to the active policy
  void CopyCandidateToPolicy(int candidate) override;

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // output policy
  SamplingPolicy policy;  // (Guarded by mtx_)
  SamplingPolicy previous_policy;

  // particles - positions are stored in candidate_policy
  SamplingPolicy candidate_policy[kMaxTrajectory];  // current positions
  std::vector<std::vector<double>> velocities;      // particle velocities
  SamplingPolicy personal_best[kMaxTrajectory];     // personal best positions
  std::vector<double> personal_best_costs;          // personal best costs

  // global best
  int global_best_index;
  double global_best_cost;

  // trajectories
  Trajectory trajectory[kMaxTrajectory];
  Trajectory nominal_trajectory;

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;

  // PSO hyperparameters
  double inertia_weight;    // w: momentum term (0.4-0.9)
  double cognitive_coeff;   // c1: personal best attraction (~1.5-2.0)
  double social_coeff;      // c2: global best attraction (~1.5-2.0)
  double velocity_scale;    // velocity scaling relative to control range

  // improvement
  double improvement;

  // timing
  std::atomic<double> velocity_update_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  // spline settings
  mjpc::spline::SplineInterpolation interpolation_ =
      mjpc::spline::SplineInterpolation::kZeroSpline;

  int num_particles_;
  mutable std::shared_mutex mtx_;

 private:
  // ----- PSO-specific methods ----- //

  // resample policy to current time
  void ResamplePolicy(int horizon);

  // initialize particle positions randomly around nominal
  void InitializeParticles(int horizon);

  // evaluate all particles via parallel rollouts
  void Rollouts(int num_particles, int horizon, ThreadPool& pool);

  // update velocities using PSO equations
  void UpdateVelocities();

  // update positions by adding velocities
  void UpdatePositions();

  // update personal bests based on current costs
  void UpdatePersonalBests(int num_particles);

  // update global best
  void UpdateGlobalBest(int num_particles);

  // scratch space for resampling
  std::vector<double> parameters_scratch;
  std::vector<double> times_scratch;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_PSO_PLANNER_H_
