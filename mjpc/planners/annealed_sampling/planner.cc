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

#include "mjpc/planners/annealed_sampling/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <shared_mutex>

#include <absl/random/random.h>
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
using mjpc::spline::SplineInterpolation;
using mjpc::spline::TimeSpline;

// initialize data and settings
void AnnealedSamplingPlanner::Initialize(mjModel* model, const Task& task) {
  data_.clear();
  ResizeMjData(model, 1);

  this->model = model;
  this->task = &task;

  // sampling noise std
  noise_exploration[0] =
      GetNumberOrDefault(0.1, model, "sampling_exploration");

  // optional second std (defaults to 0)
  int se_id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_exploration");
  if (se_id >= 0 && model->numeric_size[se_id] > 1) {
    int se_adr = model->numeric_adr[se_id];
    noise_exploration[1] = model->numeric_data[se_adr + 1];
  }

  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  interpolation_ = GetNumberOrDefault(SplineInterpolation::kCubicSpline, model,
                                      "sampling_representation");
  sliding_plan_ = GetNumberOrDefault(0, model, "sampling_sliding_plan");

  // DIAL-MPC parameters
  temperature_ = GetNumberOrDefault(1.0, model, "sampling_temperature");
  num_annealing_iters_ =
      GetNumberOrDefault(4, model, "annealing_iterations");
  beta_trajectory_ =
      GetNumberOrDefault(0.5, model, "annealing_beta_trajectory");
  beta_action_ = GetNumberOrDefault(0.5, model, "annealing_beta_action");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }

  winner = 0;
}

// allocate memory
void AnnealedSamplingPlanner::Allocate() {
  int num_state = model->nq + model->nv + model->na;

  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  plan_scratch = TimeSpline(/*dim=*/model->nu);

  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

  winner = -1;
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
}

// reset memory to zeros
void AnnealedSamplingPlanner::Reset(int horizon,
                                    const double* initial_repeated_action) {
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    policy.Reset(horizon, initial_repeated_action);
    previous_policy.Reset(horizon, initial_repeated_action);
  }

  plan_scratch.Clear();
  std::fill(noise.begin(), noise.end(), 0.0);

  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon, initial_repeated_action);
  }

  for (const auto& d : data_) {
    if (initial_repeated_action) {
      mju_copy(d->ctrl, initial_repeated_action, model->nu);
    } else {
      mju_zero(d->ctrl, model->nu);
    }
  }

  improvement = 0.0;
  winner = 0;
  current_annealing_iter_ = 0;
  current_annealing_total_ = 1;
}

// set state
void AnnealedSamplingPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

int AnnealedSamplingPlanner::OptimizePolicyCandidates(int ncandidates,
                                                      int horizon,
                                                      ThreadPool& pool) {
  // resample nominal policy to current time
  this->UpdateNominalPolicy(horizon);

  int num_trajectory = num_trajectory_;
  ncandidates = std::min(ncandidates, num_trajectory);
  ResizeMjData(model, pool.NumThreads());

  auto rollouts_start = std::chrono::steady_clock::now();

  policy.plan.SetInterpolation(interpolation_);
  this->Rollouts(num_trajectory, horizon, pool);

  // sort candidate policies and trajectories by score
  trajectory_order.clear();
  trajectory_order.reserve(num_trajectory);
  for (int i = 0; i < num_trajectory; i++) {
    trajectory_order.push_back(i);
  }

  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + ncandidates,
      trajectory_order.end(), [trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });

  rollouts_compute_time = GetDuration(rollouts_start);

  return ncandidates;
}

// optimize nominal policy using DIAL-MPC annealed sampling
void AnnealedSamplingPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  int N = std::max(num_annealing_iters_, 1);

  // Timers are per-part, not cumulative over the whole call. Reported as one
  // total they would say "the iteration is 100% policy update", which is both
  // useless in the GUI plot and wrong: nearly all of it is the N x
  // num_trajectory rollouts below.
  rollouts_compute_time = 0.0;
  double update_time = 0.0;

  for (int iter = 0; iter < N; iter++) {
    // store annealing state for AddNoiseToPolicy
    current_annealing_iter_ = iter;
    current_annealing_total_ = N;

    // resample, rollout, sort
    this->UpdateNominalPolicy(horizon);

    int num_trajectory = num_trajectory_;
    ResizeMjData(model, pool.NumThreads());

    policy.plan.SetInterpolation(interpolation_);
    auto rollouts_start = std::chrono::steady_clock::now();
    this->Rollouts(num_trajectory, horizon, pool);
    rollouts_compute_time += GetDuration(rollouts_start);

    // sort all trajectories by cost
    trajectory_order.clear();
    trajectory_order.reserve(num_trajectory);
    for (int i = 0; i < num_trajectory; i++) {
      trajectory_order.push_back(i);
    }
    std::sort(trajectory_order.begin(), trajectory_order.end(),
              [trajectory = trajectory](int a, int b) {
                return trajectory[a].total_return < trajectory[b].total_return;
              });

    // MPPI-weighted update
    auto update_start = std::chrono::steady_clock::now();
    MPPIUpdate(num_trajectory);
    update_time += GetDuration(update_start);
  }

  // compute improvement: compare nominal (trajectory[0]) to best
  winner = trajectory_order[0];
  double best_return = trajectory[0].total_return;
  improvement = mju_max(best_return - trajectory[winner].total_return, 0.0);

  policy_update_compute_time = update_time;
}

// MPPI cost-weighted policy update
void AnnealedSamplingPlanner::MPPIUpdate(int num_trajectory) {
  if (num_trajectory <= 0) return;

  // 1. Compute weights using log-sum-exp for numerical stability
  double min_cost = trajectory[trajectory_order[0]].total_return;

  std::vector<double> weights(num_trajectory);
  double weight_sum = 0.0;
  for (int i = 0; i < num_trajectory; i++) {
    double cost = trajectory[i].total_return;
    weights[i] = std::exp(-(cost - min_cost) / temperature_);
    weight_sum += weights[i];
  }

  // 2. Normalize weights
  if (weight_sum > 0.0) {
    for (int i = 0; i < num_trajectory; i++) {
      weights[i] /= weight_sum;
    }
  } else {
    // fallback: uniform
    for (int i = 0; i < num_trajectory; i++) {
      weights[i] = 1.0 / num_trajectory;
    }
  }

  // 3. Weighted average of candidate policy parameters
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;

    // Zero out policy parameters, then accumulate weighted sum
    for (const TimeSpline::Node& node : policy.plan) {
      for (int k = 0; k < model->nu; k++) {
        node.values()[k] = 0.0;
      }
    }

    for (int i = 0; i < num_trajectory; i++) {
      if (weights[i] < 1.0e-15) continue;  // skip negligible weights

      int t = 0;
      auto policy_it = policy.plan.begin();
      for (const TimeSpline::Node& cand_node : candidate_policy[i].plan) {
        if (policy_it == policy.plan.end()) break;
        for (int k = 0; k < model->nu; k++) {
          policy_it->values()[k] += weights[i] * cand_node.values()[k];
        }
        ++policy_it;
        t++;
      }
    }

    // Clamp to control limits
    for (const TimeSpline::Node& node : policy.plan) {
      Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
    }
  }

  // set winner to the best trajectory for BestTrajectory()
  winner = trajectory_order[0];
}

// compute trajectory using nominal policy
void AnnealedSamplingPlanner::NominalTrajectory(int horizon,
                                                ThreadPool& pool) {
  auto nominal_policy = [&cp = candidate_policy[0]](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  trajectory[0].Rollout(nominal_policy, task, model, data_[0].get(),
                        state.data(), time, mocap.data(), userdata.data(),
                        horizon);
}

// set action from policy
void AnnealedSamplingPlanner::ActionFromPolicy(double* action,
                                               const double* state,
                                               double time,
                                               bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
}

// update policy via resampling
void AnnealedSamplingPlanner::UpdateNominalPolicy(int horizon) {
  int num_spline_points = candidate_policy[winner].num_spline_points;
  double nominal_time = time;
  double time_horizon = (horizon - 1) * model->opt.timestep;

  if (sliding_plan_) {
    int extra_points;
    switch (interpolation_) {
      case spline::SplineInterpolation::kZeroSpline:
        extra_points = 1;
        break;
      case spline::SplineInterpolation::kLinearSpline:
        extra_points = 2;
        break;
      case spline::SplineInterpolation::kCubicSpline:
        extra_points = 4;
        break;
    }

    double time_shift;
    if (num_spline_points > extra_points) {
      time_shift = mju_max(
          time_horizon / (num_spline_points - extra_points), 1.0e-5);
    } else {
      time_shift = time_horizon;
    }

    const std::unique_lock<std::shared_mutex> lock(mtx_);

    if (policy.plan.Size() && policy.plan.begin()->time() > nominal_time) {
      policy.plan.ShiftTime(nominal_time);
      previous_policy.plan.ShiftTime(nominal_time);
    }

    policy.plan.DiscardBefore(nominal_time);
    if (policy.plan.Size() == 0) {
      policy.plan.AddNode(nominal_time);
    }
    while (policy.plan.Size() < num_spline_points) {
      double new_node_time = (policy.plan.end() - 1)->time() + time_shift;
      TimeSpline::Node new_node = policy.plan.AddNode(new_node_time);
      std::copy((policy.plan.end() - 2)->values().begin(),
                (policy.plan.end() - 2)->values().end(),
                new_node.values().begin());
    }
  } else {
    double time_shift;
    if (interpolation_ == spline::SplineInterpolation::kZeroSpline) {
      time_shift = mju_max(time_horizon / num_spline_points, 1.0e-5);
    } else {
      time_shift = mju_max(time_horizon / (num_spline_points - 1), 1.0e-5);
    }

    plan_scratch.Clear();
    plan_scratch.SetInterpolation(interpolation_);
    plan_scratch.Reserve(num_spline_points);

    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::Node node = plan_scratch.AddNode(nominal_time);
      candidate_policy[winner].Action(node.values().data(), /*state=*/nullptr,
                                      nominal_time);
      nominal_time += time_shift;
    }

    {
      const std::unique_lock<std::shared_mutex> lock(mtx_);
      policy.plan = plan_scratch;
    }
  }
}

// add random noise to nominal policy with DIAL-MPC annealing schedule
void AnnealedSamplingPlanner::AddNoiseToPolicy(double start_time, int i) {
  auto noise_start = std::chrono::steady_clock::now();

  absl::BitGen gen_;

  // base noise std
  double base_std = noise_exploration[0];
  constexpr double kStd2Proportion = 0.2;
  if (noise_exploration[1] > 0 && absl::Bernoulli(gen_, kStd2Proportion)) {
    base_std = noise_exploration[1];
  }

  int N = current_annealing_total_;
  int iter = current_annealing_iter_;

  // count spline points for horizon-dependent scaling
  int H = 0;
  for (const TimeSpline::Node& node : candidate_policy[i].plan) {
    (void)node;
    H++;
  }
  if (H == 0) H = 1;  // avoid division by zero

  int h = 0;
  for (const TimeSpline::Node& node : candidate_policy[i].plan) {
    // dual-loop annealing scale (DIAL-MPC eq. 7)
    // trajectory annealing: decreases noise over iterations
    //   iter=0 (first) -> traj_anneal = 1.0 (full noise)
    //   iter=N-1 (last) -> reduced noise
    double traj_anneal = (N > 1)
        ? std::exp(-(double)iter / (beta_trajectory_ * N))
        : 1.0;

    // action annealing: more noise for farther horizon actions
    //   h=0 (near-term) -> small noise (well-refined)
    //   h=H-1 (far-horizon) -> large noise (needs exploration)
    double action_anneal = (H > 1)
        ? std::exp(-(double)(H - 1 - h) / (beta_action_ * H))
        : 1.0;

    double effective_std = base_std * traj_anneal * action_anneal;

    for (int k = 0; k < model->nu; k++) {
      double scale = 0.5 * (model->actuator_ctrlrange[2 * k + 1] -
                            model->actuator_ctrlrange[2 * k]);
      node.values()[k] += absl::Gaussian<double>(gen_, 0.0,
                                                 scale * effective_std);
    }
    Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
    h++;
  }

  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void AnnealedSamplingPlanner::Rollouts(int num_trajectory, int horizon,
                                       ThreadPool& pool) {
  noise_compute_time = 0.0;

  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   i]() {
      // copy nominal policy
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);
        s.candidate_policy[i].CopyFrom(s.policy, s.policy.num_spline_points);
      }

      // sample noise policy (skip i=0 to keep nominal)
      if (i != 0) s.AddNoiseToPolicy(time, i);

      // policy
      auto sample_policy_i = [&candidate_policy = s.candidate_policy, &i](
                                 double* action, const double* state,
                                 double time) {
        candidate_policy[i].Action(action, state, time);
      };

      // policy rollout
      s.trajectory[i].Rollout(
          sample_policy_i, task, model, s.data_[ThreadPool::WorkerId()].get(),
          state.data(), time, mocap.data(), userdata.data(), horizon);
    });
  }
  pool.WaitCount(count_before + num_trajectory);
  pool.ResetCount();
}

// return trajectory with best total return
const Trajectory* AnnealedSamplingPlanner::BestTrajectory() {
  return winner >= 0 ? &trajectory[winner] : nullptr;
}

// visualize planner-specific traces
void AnnealedSamplingPlanner::Traces(mjvScene* scn) {
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  double zero3[3] = {0};
  double zero9[9] = {0};

  auto best = this->BestTrajectory();

  for (int k = 0; k < num_trajectory_; k++) {
    if (k == winner) continue;

    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3,
                     zero9, color);

        mjv_connector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[k].trace.data() + 3 * task->num_trace * i + 3 * j,
            trajectory[k].trace.data() + 3 * task->num_trace * (i + 1) +
                3 * j);

        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void AnnealedSamplingPlanner::GUI(mjUI& ui) {
  mjuiDef defAnnealed[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_,
       "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std", 2, noise_exploration, "0 1"},
      {mjITEM_SLIDERINT, "Anneal Iters", 2, &num_annealing_iters_, "0 1"},
      {mjITEM_SLIDERNUM, "Temperature", 2, &temperature_, "0 1"},
      {mjITEM_SLIDERNUM, "Beta Traj", 2, &beta_trajectory_, "0 1"},
      {mjITEM_SLIDERNUM, "Beta Action", 2, &beta_action_, "0 1"},
      {mjITEM_CHECKBYTE, "Sliding plan", 2, &sliding_plan_, ""},
      {mjITEM_END}};

  // set slider limits
  mju::sprintf_arr(defAnnealed[0].other, "%i %i", 1, kMaxTrajectory);
  mju::sprintf_arr(defAnnealed[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);
  mju::sprintf_arr(defAnnealed[3].other, "%f %f", MinNoiseStdDev,
                   MaxNoiseStdDev);
  mju::sprintf_arr(defAnnealed[4].other, "%i %i", 1, 8);
  mju::sprintf_arr(defAnnealed[5].other, "%f %f", 0.001, 10.0);
  mju::sprintf_arr(defAnnealed[6].other, "%f %f", 0.01, 2.0);
  mju::sprintf_arr(defAnnealed[7].other, "%f %f", 0.01, 2.0);

  mjui_add(&ui, defAnnealed);
}

// planner-specific plots
void AnnealedSamplingPlanner::Plots(mjvFigure* fig_planner,
                                    mjvFigure* fig_timer, int planner_shift,
                                    int timer_shift, int planning,
                                    int* shift) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improvement");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // ----- timer ----- //
  double timer_bounds[2] = {0.0, 1.0};

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[0 + timer_shift][0] + 1,
                 1.0e-3 * noise_compute_time * planning, 100,
                 0 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[1 + timer_shift][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100,
                 1 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[2 + timer_shift][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100,
                 2 + timer_shift, 0, 1, -100);

  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

  shift[0] += 1;
  shift[1] += 3;
}

double AnnealedSamplingPlanner::CandidateScore(int candidate) const {
  return trajectory[trajectory_order[candidate]].total_return;
}

void AnnealedSamplingPlanner::ActionFromCandidatePolicy(double* action,
                                                        int candidate,
                                                        const double* state,
                                                        double time) {
  candidate_policy[trajectory_order[candidate]].Action(action, state, time);
}

void AnnealedSamplingPlanner::CopyCandidateToPolicy(int candidate) {
  winner = trajectory_order[candidate];

  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    policy = candidate_policy[winner];
  }
}

}  // namespace mjpc
