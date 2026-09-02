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
// See planner.h for a full description of the CoVO-MPC approximation
// implemented here.

#include "mjpc/planners/covo/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
void CoVOPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();

  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // sampling noise
  std_initial_ =
      GetNumberOrDefault(0.1, model, "sampling_exploration");  // initial std
  std_min_ = GetNumberOrDefault(0.01, model, "std_min");       // minimum std
  std_max_ = GetNumberOrDefault(1.0, model, "std_max");        // maximum std

  // CoVO knobs
  covo_scale_ = GetNumberOrDefault(1.0, model, "covo_scale");
  covo_temperature_ = GetNumberOrDefault(1.0, model, "covo_temperature");

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }
}

// allocate memory
void CoVOPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na;

  // state
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // policy
  int num_max_parameter = model->nu * kMaxTrajectoryHorizon;
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  resampled_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);

  // scratch
  parameters_scratch.resize(num_max_parameter);
  times_scratch.resize(kMaxTrajectoryHorizon);

  // noise
  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

  // variance
  variance.resize(model->nu * kMaxTrajectoryHorizon);  // (nu * horizon)

  // need to initialize an arbitrary order of the trajectories
  trajectory_order.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory_order[i] = i;
  }

  // trajectories and parameters
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
  nominal_trajectory.Initialize(num_state, model->nu, task->num_residual,
                                task->num_trace, kMaxTrajectoryHorizon);
  nominal_trajectory.Allocate(kMaxTrajectoryHorizon);
}

// reset memory to zeros
void CoVOPlanner::Reset(int horizon, const double* initial_repeated_action) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  // policy parameters
  policy.Reset(horizon, initial_repeated_action);
  resampled_policy.Reset(horizon, initial_repeated_action);
  previous_policy.Reset(horizon, initial_repeated_action);

  // scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // variance (isotropic initialization)
  double var = std_initial_ * std_initial_;
  std::fill(variance.begin(), variance.end(), var);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon);
  }
  nominal_trajectory.Reset(kMaxTrajectoryHorizon);

  for (const auto& d : data_) {
    mju_zero(d->ctrl, model->nu);
  }

  // improvement
  improvement = 0.0;
}

// set state
void CoVOPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

// optimize nominal policy using CoVO sampling + MPPI fold
void CoVOPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  resampled_policy.plan.SetInterpolation(interpolation_);

  // if num_trajectory_ has changed, use it in this new iteration.
  int num_trajectory = std::min(num_trajectory_, kMaxTrajectory);

  // resize number of mjData
  ResizeMjData(model, pool.NumThreads());

  // copy nominal policy
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    resampled_policy.CopyFrom(policy, policy.num_spline_points);
  }

  // resample nominal policy to current time (fills parameters_scratch /
  // times_scratch with the nominal at the spline points)
  this->ResamplePolicy(horizon);

  // ----- rollout samples from the current (diagonal) covariance ----- //
  auto rollouts_start = std::chrono::steady_clock::now();
  this->Rollouts(num_trajectory, horizon, pool);

  // sort trajectories by score
  for (int i = 0; i < num_trajectory; i++) {
    trajectory_order[i] = i;
  }
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + num_trajectory,
      trajectory_order.begin() + num_trajectory,
      [&trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });

  rollouts_compute_time = GetDuration(rollouts_start);

  // ----- MPPI-style weighted-average fold ----- //
  auto policy_update_start = std::chrono::steady_clock::now();

  int num_spline_points = resampled_policy.num_spline_points;
  int num_parameters = num_spline_points * model->nu;

  // minimum cost for overflow-safe exponent
  double j_min = trajectory[0].total_return;
  for (int i = 1; i < num_trajectory; i++) {
    j_min = std::min(j_min, trajectory[i].total_return);
  }

  double lambda = std::max(covo_temperature_, 1.0e-10);
  double weight_sum = 0.0;
  std::vector<double> weights(num_trajectory);
  for (int i = 0; i < num_trajectory; i++) {
    double w = std::exp(-(trajectory[i].total_return - j_min) / lambda);
    if (!std::isfinite(w)) w = 0.0;
    weights[i] = w;
    weight_sum += w;
  }

  std::fill(parameters_scratch.begin(),
            parameters_scratch.begin() + num_parameters, 0.0);

  bool degenerate = !(weight_sum > 0.0) || !std::isfinite(weight_sum);
  if (degenerate) {
    // fall back to argmin sample
    int best = trajectory_order[0];
    const TimeSpline& best_plan = candidate_policy[best].plan;
    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::ConstNode n = best_plan.NodeAt(t);
      for (int j = 0; j < model->nu; j++) {
        parameters_scratch[t * model->nu + j] = n.values()[j];
      }
    }
  } else {
    for (int i = 0; i < num_trajectory; i++) {
      double wi = weights[i] / weight_sum;
      const TimeSpline& plan_i = candidate_policy[i].plan;
      for (int t = 0; t < num_spline_points; t++) {
        TimeSpline::ConstNode n = plan_i.NodeAt(t);
        for (int j = 0; j < model->nu; j++) {
          parameters_scratch[t * model->nu + j] += wi * n.values()[j];
        }
      }
    }
  }

  // write folded nominal into the policy
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    policy.plan.Clear();
    policy.plan.SetInterpolation(interpolation_);
    for (int t = 0; t < num_spline_points; t++) {
      absl::Span<const double> values =
          absl::MakeConstSpan(parameters_scratch.data() + t * model->nu,
                              parameters_scratch.data() + (t + 1) * model->nu);
      policy.plan.AddNode(times_scratch[t], values);
    }
  }

  // ----- design the covariance for the NEXT step from cost curvature ----- //
  UpdateCovariance(num_trajectory);

  // improvement: best sample vs mean
  double avg_return = 0.0;
  for (int i = 0; i < num_trajectory; i++) avg_return += trajectory[i].total_return;
  avg_return /= num_trajectory;
  improvement =
      mju_max(avg_return - trajectory[trajectory_order[0]].total_return, 0.0);

  policy_update_compute_time = GetDuration(policy_update_start);
}

// estimate diagonal cost curvature from the sample batch and set the
// per-parameter sampling variance to (covo_scale / sqrt(max(h, eps)))^2.
void CoVOPlanner::UpdateCovariance(int num_trajectory) {
  if (num_trajectory < 2) return;

  int num_spline_points = resampled_policy.num_spline_points;
  int num_parameters = num_spline_points * model->nu;
  int stride = model->nu * kMaxTrajectoryHorizon;  // per-sample noise stride

  constexpr double kEps = 1.0e-9;
  const double var_min = std_min_ * std_min_;
  const double var_max = std_max_ * std_max_;

  // mean cost
  double j_mean = 0.0;
  for (int i = 0; i < num_trajectory; i++) j_mean += trajectory[i].total_return;
  j_mean /= num_trajectory;

  for (int d = 0; d < num_parameters; d++) {
    // per-dimension statistics over samples:
    //   x2_i = (delta_{i,d})^2  ;  regress J on x2 to isolate 0.5*h_d.
    double mean_x2 = 0.0;
    for (int i = 0; i < num_trajectory; i++) {
      double x = noise[i * stride + d];
      mean_x2 += x * x;
    }
    mean_x2 /= num_trajectory;

    double cov_jx2 = 0.0;  // cov(J, x2)
    double var_x2 = 0.0;   // var(x2)
    for (int i = 0; i < num_trajectory; i++) {
      double x = noise[i * stride + d];
      double x2c = x * x - mean_x2;
      cov_jx2 += (trajectory[i].total_return - j_mean) * x2c;
      var_x2 += x2c * x2c;
    }

    double h_d;
    if (var_x2 > kEps) {
      // y ~ a + b x + 0.5 h x^2  =>  h = 2 cov(J, x2) / var(x2)
      h_d = 2.0 * cov_jx2 / var_x2;
    } else {
      h_d = 0.0;  // no signal -> treated as flat (explore)
    }

    // inverse-sqrt-curvature std law; non-positive curvature -> max std
    double std_d;
    if (h_d > kEps) {
      std_d = covo_scale_ / std::sqrt(h_d);
    } else {
      std_d = std_max_;
    }

    double var_d = std_d * std_d;
    if (!std::isfinite(var_d)) var_d = var_max;
    var_d = std::min(std::max(var_d, var_min), var_max);
    variance[d] = var_d;
  }
}

// compute trajectory using nominal policy
void CoVOPlanner::NominalTrajectory(int horizon) {
  auto nominal_policy = [&cp = resampled_policy](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };
  nominal_trajectory.Rollout(nominal_policy, task, model,
                             data_[ThreadPool::WorkerId()].get(), state.data(),
                             time, mocap.data(), userdata.data(), horizon);
}
void CoVOPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  NominalTrajectory(horizon);
}

// set action from policy
void CoVOPlanner::ActionFromPolicy(double* action, const double* state,
                                   double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
}

// update policy via resampling
void CoVOPlanner::ResamplePolicy(int horizon) {
  int num_spline_points = resampled_policy.num_spline_points;

  double nominal_time = time;
  double time_shift = mju_max(
      (horizon - 1) * model->opt.timestep / (num_spline_points - 1), 1.0e-5);

  for (int t = 0; t < num_spline_points; t++) {
    times_scratch[t] = nominal_time;
    resampled_policy.Action(DataAt(parameters_scratch, t * model->nu), nullptr,
                            nominal_time);
    nominal_time += time_shift;
  }

  resampled_policy.plan.Clear();
  for (int t = 0; t < num_spline_points; t++) {
    absl::Span<const double> values =
        absl::MakeConstSpan(parameters_scratch.data() + t * model->nu,
                            parameters_scratch.data() + (t + 1) * model->nu);
    resampled_policy.plan.AddNode(times_scratch[t], values);
  }
  resampled_policy.plan.SetInterpolation(policy.plan.Interpolation());
}

// add random noise to nominal policy using the per-dimension variance
void CoVOPlanner::AddNoiseToPolicy(int i, double std_min) {
  auto noise_start = std::chrono::steady_clock::now();

  int num_spline_points = candidate_policy[i].num_spline_points;
  int num_parameters = num_spline_points * model->nu;

  absl::BitGen gen_;

  int shift = i * (model->nu * kMaxTrajectoryHorizon);

  for (int k = 0; k < num_parameters; k++) {
    noise[k + shift] = absl::Gaussian<double>(
        gen_, 0.0, std::max(std::sqrt(variance[k]), std_min));
  }

  for (int k = 0; k < candidate_policy[i].plan.Size(); k++) {
    TimeSpline::Node n = candidate_policy[i].plan.NodeAt(k);
    mju_addTo(n.values().data(), DataAt(noise, shift + k * model->nu),
              model->nu);
    Clamp(n.values().data(), model->actuator_ctrlrange, model->nu);
  }

  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void CoVOPlanner::Rollouts(int num_trajectory, int horizon, ThreadPool& pool) {
  noise_compute_time = 0.0;

  double std_min = std_min_;

  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   std_min, i]() {
      // copy nominal policy and sample noise
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);
        s.candidate_policy[i].CopyFrom(s.resampled_policy,
                                       s.resampled_policy.num_spline_points);
        s.candidate_policy[i].plan.SetInterpolation(
            s.resampled_policy.plan.Interpolation());

        // sample noise from the diagonal covariance
        s.AddNoiseToPolicy(i, std_min);
      }

      // ----- rollout sample policy ----- //
      auto sample_policy_i = [&candidate_policy = s.candidate_policy, &i](
                                 double* action, const double* state,
                                 double time) {
        candidate_policy[i].Action(action, state, time);
      };

      s.trajectory[i].Rollout(
          sample_policy_i, task, model, s.data_[ThreadPool::WorkerId()].get(),
          state.data(), time, mocap.data(), userdata.data(), horizon);
    });
  }
  // nominal
  pool.Schedule([&s = *this, horizon]() { s.NominalTrajectory(horizon); });

  pool.WaitCount(count_before + num_trajectory + 1);
  pool.ResetCount();
}

// returns the nominal trajectory
const Trajectory* CoVOPlanner::BestTrajectory() { return &nominal_trajectory; }

// visualize planner-specific traces
void CoVOPlanner::Traces(mjvScene* scn) {
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  double zero3[3] = {0};
  double zero9[9] = {0};

  auto best = this->BestTrajectory();

  int num_trajectory = num_trajectory_;
  for (int k = 0; k < num_trajectory; k++) {
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     color);
        int idx = trajectory_order[k];
        mjv_connector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[idx].trace.data() + 3 * task->num_trace * i + 3 * j,
            trajectory[idx].trace.data() + 3 * task->num_trace * (i + 1) + 3 * j);
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void CoVOPlanner::GUI(mjUI& ui) {
  mjuiDef defCoVO[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_, "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Init. Std", 2, &std_initial_, "0 1"},
      {mjITEM_SLIDERNUM, "Min. Std", 2, &std_min_, "0.001 0.5"},
      {mjITEM_SLIDERNUM, "Max. Std", 2, &std_max_, "0.1 2"},
      {mjITEM_SLIDERNUM, "CoVO Scale", 2, &covo_scale_, "0.01 10"},
      {mjITEM_SLIDERNUM, "Temperature", 2, &covo_temperature_, "0.001 10"},
      {mjITEM_END}};

  mju::sprintf_arr(defCoVO[0].other, "%i %i", 1, kMaxTrajectory);
  mju::sprintf_arr(defCoVO[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);
  mju::sprintf_arr(defCoVO[3].other, "%f %f", MinNoiseStdDev, MaxNoiseStdDev);

  mjui_add(&ui, defCoVO);
}

// planner-specific plots
void CoVOPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                        int planner_shift, int timer_shift, int planning,
                        int* shift) {
  double planner_bounds[2] = {-6.0, 6.0};

  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Avg - Best");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  double timer_bounds[2] = {0.0, 1.0};

  PlotUpdateData(
      fig_timer, timer_bounds, fig_timer->linedata[0 + timer_shift][0] + 1,
      1.0e-3 * noise_compute_time * planning, 100, 0 + timer_shift, 0, 1, -100);

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

}  // namespace mjpc
