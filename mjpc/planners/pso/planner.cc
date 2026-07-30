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

#include "mjpc/planners/pso/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <shared_mutex>

#include <absl/random/random.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::TimeSpline;

// initialize data and settings
void PSOPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();

  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // PSO hyperparameters from model or defaults
  num_particles_ = GetNumberOrDefault(20, model, "pso_num_particles");
  inertia_weight = GetNumberOrDefault(0.7, model, "pso_inertia");
  cognitive_coeff = GetNumberOrDefault(1.5, model, "pso_cognitive");
  social_coeff = GetNumberOrDefault(1.5, model, "pso_social");
  velocity_scale = GetNumberOrDefault(0.1, model, "pso_velocity_scale");
  publish_evaluated_ =
      GetNumberOrDefault(1.0, model, "pso_publish_evaluated") != 0.0;

  interpolation_ = GetNumberOrDefault(
      mjpc::spline::SplineInterpolation::kCubicSpline, model,
      "sampling_representation");

  if (num_particles_ > kMaxTrajectory) {
    mju_error_i("Too many particles, %d is the maximum allowed.",
                kMaxTrajectory);
  }

  global_best_index = 0;
}

// allocate memory
void PSOPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na;

  // state
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // policy
  int num_max_parameter = model->nu * kMaxTrajectoryHorizon;
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);

  // scratch
  parameters_scratch.resize(num_max_parameter);
  times_scratch.resize(kMaxTrajectoryHorizon);

  // particles
  velocities.resize(kMaxTrajectory);
  personal_best_costs.resize(kMaxTrajectory,
                              std::numeric_limits<double>::max());

  trajectory_order.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory_order[i] = i;
  }

  // trajectories and particle policies
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
    personal_best[i].Allocate(model, *task, kMaxTrajectoryHorizon);
    evaluated_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
    velocities[i].resize(num_max_parameter, 0.0);
  }

  nominal_trajectory.Initialize(num_state, model->nu, task->num_residual,
                                task->num_trace, kMaxTrajectoryHorizon);
  nominal_trajectory.Allocate(kMaxTrajectoryHorizon);

  global_best_cost = std::numeric_limits<double>::max();
}

// reset memory to zeros
void PSOPlanner::Reset(int horizon, const double* initial_repeated_action) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  // policy parameters
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    policy.Reset(horizon, initial_repeated_action);
    previous_policy.Reset(horizon, initial_repeated_action);
  }

  // scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // reset all particles
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon, initial_repeated_action);
    personal_best[i].Reset(horizon, initial_repeated_action);
    personal_best_costs[i] = std::numeric_limits<double>::max();
    std::fill(velocities[i].begin(), velocities[i].end(), 0.0);
  }
  nominal_trajectory.Reset(kMaxTrajectoryHorizon);

  for (const auto& d : data_) {
    mju_zero(d->ctrl, model->nu);
  }

  // reset global best
  global_best_cost = std::numeric_limits<double>::max();
  global_best_index = 0;

  // improvement
  improvement = 0.0;
}

// set state
void PSOPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

// resample policy to current time
void PSOPlanner::ResamplePolicy(int horizon) {
  // dimensions
  int num_spline_points = policy.num_spline_points;

  // time
  double nominal_time = time;
  double time_shift = mju_max(
      (horizon - 1) * model->opt.timestep / (num_spline_points - 1), 1.0e-5);

  // get spline points
  for (int t = 0; t < num_spline_points; t++) {
    times_scratch[t] = nominal_time;
    policy.Action(DataAt(parameters_scratch, t * model->nu), nullptr,
                  nominal_time);
    nominal_time += time_shift;
  }
}

// initialize particles with random perturbations around nominal
void PSOPlanner::InitializeParticles(int horizon) {
  int num_spline_points = policy.num_spline_points;
  absl::BitGen gen;

  for (int i = 0; i < num_particles_; i++) {
    // copy nominal policy to particle
    candidate_policy[i].plan.Clear();
    candidate_policy[i].plan.SetInterpolation(interpolation_);

    for (int t = 0; t < num_spline_points; t++) {
      // start with nominal parameters
      std::vector<double> values(model->nu);
      for (int k = 0; k < model->nu; k++) {
        values[k] = parameters_scratch[t * model->nu + k];
      }

      // add random perturbation (except for first particle which stays nominal)
      if (i > 0) {
        for (int k = 0; k < model->nu; k++) {
          double range = model->actuator_ctrlrange[2 * k + 1] -
                         model->actuator_ctrlrange[2 * k];
          double noise = absl::Gaussian<double>(gen, 0.0, 0.1 * range);
          values[k] += noise;
        }
      }

      candidate_policy[i].plan.AddNode(
          times_scratch[t], absl::MakeConstSpan(values.data(), model->nu));
    }
    candidate_policy[i].num_spline_points = num_spline_points;

    // initialize velocity to small random values
    for (int t = 0; t < num_spline_points; t++) {
      for (int k = 0; k < model->nu; k++) {
        double range = model->actuator_ctrlrange[2 * k + 1] -
                       model->actuator_ctrlrange[2 * k];
        velocities[i][t * model->nu + k] =
            absl::Gaussian<double>(gen, 0.0, 0.01 * range);
      }
    }
  }
}

int PSOPlanner::OptimizePolicyCandidates(int ncandidates, int horizon,
                                          ThreadPool& pool) {
  // if num_particles_ has changed, use it in this new iteration
  int num_particles = num_particles_;
  ncandidates = std::min(ncandidates, num_particles);

  // resize number of mjData
  ResizeMjData(model, pool.NumThreads());

  // resample nominal policy to current time
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    ResamplePolicy(horizon);
  }

  // initialize particles if this is first iteration or they need resampling
  InitializeParticles(horizon);

  // ----- rollout all particles ----- //
  auto rollouts_start = std::chrono::steady_clock::now();
  Rollouts(num_particles, horizon, pool);
  rollouts_compute_time = GetDuration(rollouts_start);

  // update personal and global bests
  UpdatePersonalBests(num_particles);
  UpdateGlobalBest(num_particles);

  // sort trajectories by cost for ranking. This reads only total_return, which
  // the velocity/position update below does not touch, so ranking is the same
  // whichever side of the update it runs on.
  for (int i = 0; i < num_particles; i++) {
    trajectory_order[i] = i;
  }
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + ncandidates,
      trajectory_order.begin() + num_particles,
      [&trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });

  // Snapshot the ranked candidates *before* the swarm update mutates them.
  // trajectory[i].total_return is the cost of candidate_policy[i] as it was
  // rolled out; UpdatePositions overwrites candidate_policy[i] in place with
  // the next iterate. Publishing after that update ships parameters whose cost
  // was never evaluated -- the ranking says one thing and the executed policy
  // is another. Keeping a pre-update copy is what makes the published policy
  // the one that actually earned its score.
  if (publish_evaluated_) {
    for (int c = 0; c < ncandidates; c++) {
      int i = trajectory_order[c];
      evaluated_policy[i].CopyFrom(candidate_policy[i],
                                   candidate_policy[i].num_spline_points);
    }
  }

  // ----- update velocities and positions ----- //
  auto velocity_start = std::chrono::steady_clock::now();
  UpdateVelocities();
  UpdatePositions();
  velocity_update_time = GetDuration(velocity_start);

  return ncandidates;
}

// optimize nominal policy using PSO
void PSOPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  OptimizePolicyCandidates(1, horizon, pool);

  // ----- update policy ----- //
  auto policy_update_start = std::chrono::steady_clock::now();

  CopyCandidateToPolicy(0);

  // improvement: compare previous best to current best
  double best_return = trajectory[trajectory_order[0]].total_return;
  improvement = mju_max(global_best_cost - best_return, 0.0);

  policy_update_compute_time = GetDuration(policy_update_start);
}

// compute trajectory using nominal policy
void PSOPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  // set policy
  auto nominal_policy_fn = [&policy = this->policy](double* action,
                                                     const double* state,
                                                     double time) {
    policy.Action(action, state, time);
  };

  // rollout nominal policy
  nominal_trajectory.Rollout(nominal_policy_fn, task, model,
                             data_[ThreadPool::WorkerId()].get(), state.data(),
                             time, mocap.data(), userdata.data(), horizon);
}

// set action from policy
void PSOPlanner::ActionFromPolicy(double* action, const double* state,
                                   double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
}

// compute candidate trajectories via parallel rollouts
void PSOPlanner::Rollouts(int num_particles, int horizon, ThreadPool& pool) {
  int count_before = pool.GetCount();

  for (int i = 0; i < num_particles; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   i]() {
      // policy
      auto particle_policy = [&candidate_policy = s.candidate_policy, &i](
                                 double* action, const double* state,
                                 double time) {
        candidate_policy[i].Action(action, state, time);
      };

      // rollout
      s.trajectory[i].Rollout(particle_policy, task, model,
                              s.data_[ThreadPool::WorkerId()].get(),
                              state.data(), time, mocap.data(), userdata.data(),
                              horizon);
    });
  }

  pool.WaitCount(count_before + num_particles);
  pool.ResetCount();
}

// update velocities using PSO equations
void PSOPlanner::UpdateVelocities() {
  absl::BitGen gen;
  int num_spline_points = policy.num_spline_points;

  for (int i = 0; i < num_particles_; i++) {
    // skip if personal best not yet set
    if (personal_best_costs[i] >= std::numeric_limits<double>::max()) {
      continue;
    }

    for (int t = 0; t < num_spline_points; t++) {
      // get current position from candidate_policy
      auto pos_node = candidate_policy[i].plan.NodeAt(t);
      auto pbest_node = personal_best[i].plan.NodeAt(t);
      auto gbest_node = candidate_policy[global_best_index].plan.NodeAt(t);

      for (int k = 0; k < model->nu; k++) {
        double r1 = absl::Uniform(gen, 0.0, 1.0);
        double r2 = absl::Uniform(gen, 0.0, 1.0);

        double pos = pos_node.values()[k];
        double pbest = pbest_node.values()[k];
        double gbest = gbest_node.values()[k];

        // PSO velocity update
        double cognitive = cognitive_coeff * r1 * (pbest - pos);
        double social = social_coeff * r2 * (gbest - pos);
        double new_vel =
            inertia_weight * velocities[i][t * model->nu + k] +
            cognitive + social;

        // clamp velocity
        double range = model->actuator_ctrlrange[2 * k + 1] -
                       model->actuator_ctrlrange[2 * k];
        double max_vel = velocity_scale * range;
        new_vel = mju_clip(new_vel, -max_vel, max_vel);

        velocities[i][t * model->nu + k] = new_vel;
      }
    }
  }
}

// update positions by adding velocities
void PSOPlanner::UpdatePositions() {
  int num_spline_points = policy.num_spline_points;

  for (int i = 0; i < num_particles_; i++) {
    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::Node pos_node = candidate_policy[i].plan.NodeAt(t);

      for (int k = 0; k < model->nu; k++) {
        pos_node.values()[k] += velocities[i][t * model->nu + k];
      }

      // clamp to control limits
      Clamp(pos_node.values().data(), model->actuator_ctrlrange, model->nu);
    }
  }
}

// update personal bests based on current costs
void PSOPlanner::UpdatePersonalBests(int num_particles) {
  for (int i = 0; i < num_particles; i++) {
    if (trajectory[i].total_return < personal_best_costs[i]) {
      personal_best_costs[i] = trajectory[i].total_return;
      personal_best[i].CopyFrom(candidate_policy[i],
                                 candidate_policy[i].num_spline_points);
    }
  }
}

// update global best
void PSOPlanner::UpdateGlobalBest(int num_particles) {
  for (int i = 0; i < num_particles; i++) {
    if (personal_best_costs[i] < global_best_cost) {
      global_best_cost = personal_best_costs[i];
      global_best_index = i;
    }
  }
}

// return trajectory with best total return
const Trajectory* PSOPlanner::BestTrajectory() {
  return global_best_index >= 0 ? &trajectory[global_best_index] : nullptr;
}

// visualize planner-specific traces
void PSOPlanner::Traces(mjvScene* scn) {
  // particle color
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  // width of a sample trace, in pixels
  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // best trajectory
  auto best = this->BestTrajectory();
  if (!best) return;

  // particle traces
  for (int p = 0; p < num_particles_; p++) {
    // skip global best (will be drawn separately)
    if (p == global_best_index) continue;

    // plot particle trajectory
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     color);

        // make geometry
        mjv_connector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[p].trace.data() + 3 * task->num_trace * i + 3 * j,
            trajectory[p].trace.data() + 3 * task->num_trace * (i + 1) + 3 * j);

        // increment number of geometries
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void PSOPlanner::GUI(mjUI& ui) {
  mjuiDef defPSO[] = {
      {mjITEM_SLIDERINT, "Particles", 2, &num_particles_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_, "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Inertia", 2, &inertia_weight, "0.1 0.99"},
      {mjITEM_SLIDERNUM, "Cognitive", 2, &cognitive_coeff, "0.0 4.0"},
      {mjITEM_SLIDERNUM, "Social", 2, &social_coeff, "0.0 4.0"},
      {mjITEM_SLIDERNUM, "Vel Scale", 2, &velocity_scale, "0.01 0.5"},
      {mjITEM_END}};

  // set number of particles slider limits
  mju::sprintf_arr(defPSO[0].other, "%i %i", 2, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defPSO[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);

  // add PSO planner
  mjui_add(&ui, defPSO);
}

// planner-specific plots
void PSOPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                        int planner_shift, int timer_shift, int planning,
                        int* shift) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improvement");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // bounds
  double timer_bounds[2] = {0.0, 1.0};

  // ----- timer ----- //
  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[0 + timer_shift][0] + 1,
                 1.0e-3 * velocity_update_time * planning, 100,
                 0 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[1 + timer_shift][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100,
                 1 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[2 + timer_shift][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100,
                 2 + timer_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Velocity");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

  // planner shift
  shift[0] += 1;

  // timer shift
  shift[1] += 3;
}

double PSOPlanner::CandidateScore(int candidate) const {
  return trajectory[trajectory_order[candidate]].total_return;
}

// set action from candidate policy
void PSOPlanner::ActionFromCandidatePolicy(double* action, int candidate,
                                            const double* state, double time) {
  candidate_policy[trajectory_order[candidate]].Action(action, state, time);
}

void PSOPlanner::CopyCandidateToPolicy(int candidate) {
  int best_idx = trajectory_order[candidate];

  // Publish the parameters that were rolled out, not the post-swarm-update
  // iterate that happens to sit in the same slot. See the snapshot in
  // OptimizePolicyCandidates. Set the model numeric "pso_publish_evaluated" to
  // 0 to restore the original behaviour for comparison.
  const SamplingPolicy& source =
      publish_evaluated_ ? evaluated_policy[best_idx]
                         : candidate_policy[best_idx];

  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    policy.CopyFrom(source, source.num_spline_points);
  }
}

}  // namespace mjpc
