// Copyright 2026 — iCEM planner port for MJPC.
//
// Derived from mjpc/planners/cross_entropy/planner.cc. Diff to CEM is
// localized: AddNoiseToPolicy() applies an AR(1) filter along the spline
// time axis for temporally-correlated noise, and OptimizePolicy() carries
// over a small set of elite samples from the previous Plan() call.

#include "mjpc/planners/icem/planner.h"

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
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::TimeSpline;

void iCEMPlanner::Initialize(mjModel* model, const Task& task) {
  data_.clear();
  ResizeMjData(model, 1);

  this->model = model;
  this->task = &task;

  std_initial_ = GetNumberOrDefault(0.1, model, "sampling_exploration");
  std_min_ = GetNumberOrDefault(0.01, model, "std_min");
  explore_fraction_ = GetNumberOrDefault(0.0, model, "explore_fraction");

  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  // CEM elite-set size; same lookup as CrossEntropyPlanner for parity.
  n_elite_ =
      GetNumberOrDefault(std::max(num_trajectory_ / 10, 2), model, "n_elite");

  // iCEM additions: colored-noise AR(1) coefficient and elite-memory size.
  // Defaults are mild — α=0.7 gives roughly pink-noise behaviour, which is
  // a good general-purpose setting for humanoid tasks; the user can tune
  // via <numeric name="icem_alpha" data="..."/> in the task XML.
  colored_alpha_ = GetNumberOrDefault(0.7, model, "icem_alpha");
  if (colored_alpha_ < 0.0) colored_alpha_ = 0.0;
  if (colored_alpha_ > 0.99) colored_alpha_ = 0.99;
  // Default keeps top 2 elites across Plan() calls. Set to 0 to disable.
  n_elite_keep_ =
      GetNumberOrDefault(2, model, "icem_elite_keep");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }
}

void iCEMPlanner::Allocate() {
  int num_state = model->nq + model->nv + model->na;

  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  int num_max_parameter = model->nu * kMaxTrajectoryHorizon;
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  resampled_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);

  parameters_scratch.resize(num_max_parameter);
  times_scratch.resize(kMaxTrajectoryHorizon);

  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));
  variance.resize(model->nu * kMaxTrajectoryHorizon);

  trajectory_order.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory_order[i] = i;
  }

  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
  nominal_trajectory.Initialize(num_state, model->nu, task->num_residual,
                                task->num_trace, kMaxTrajectoryHorizon);
  nominal_trajectory.Allocate(kMaxTrajectoryHorizon);

  // Pre-allocate the elite memory buffer. Sized to the max we'd ever keep.
  elite_memory.resize(std::max(1, n_elite_keep_));
  for (auto& p : elite_memory) {
    p.Allocate(model, *task, kMaxTrajectoryHorizon);
  }
  elite_memory_valid_ = false;
}

void iCEMPlanner::Reset(int horizon, const double* initial_repeated_action) {
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  policy.Reset(horizon, initial_repeated_action);
  resampled_policy.Reset(horizon, initial_repeated_action);
  previous_policy.Reset(horizon, initial_repeated_action);

  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  std::fill(noise.begin(), noise.end(), 0.0);

  double var = std_initial_ * std_initial_;
  std::fill(variance.begin(), variance.end(), var);

  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon);
  }
  nominal_trajectory.Reset(kMaxTrajectoryHorizon);

  for (auto& p : elite_memory) p.Reset(horizon);
  elite_memory_valid_ = false;

  for (const auto& d : data_) {
    mju_zero(d->ctrl, model->nu);
  }

  improvement = 0.0;
}

void iCEMPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

void iCEMPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  resampled_policy.plan.SetInterpolation(interpolation_);

  int num_trajectory = num_trajectory_;
  n_elite_ = std::min(n_elite_, num_trajectory);
  int n_elite = std::min(n_elite_, num_trajectory);
  int n_keep = std::min(n_elite_keep_, n_elite);

  ResizeMjData(model, pool.NumThreads());

  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    resampled_policy.CopyFrom(policy, policy.num_spline_points);
  }
  this->ResamplePolicy(horizon);

  auto rollouts_start = std::chrono::steady_clock::now();
  this->Rollouts(num_trajectory, horizon, pool);

  for (int i = 0; i < num_trajectory; i++) trajectory_order[i] = i;
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + num_trajectory,
      trajectory_order.begin() + num_trajectory,
      [&trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });

  rollouts_compute_time = GetDuration(rollouts_start);

  auto policy_update_start = std::chrono::steady_clock::now();

  int num_spline_points = resampled_policy.num_spline_points;
  int num_parameters = num_spline_points * model->nu;

  double avg_return = 0.0;
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);

  for (int i = 0; i < n_elite; i++) {
    int idx = trajectory_order[i];
    const TimeSpline& elite_plan = candidate_policy[idx].plan;
    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::ConstNode n = elite_plan.NodeAt(t);
      for (int j = 0; j < model->nu; j++) {
        parameters_scratch[t * model->nu + j] += n.values()[j];
      }
    }
    avg_return += trajectory[idx].total_return;
  }
  mju_scl(parameters_scratch.data(), parameters_scratch.data(), 1.0 / n_elite,
          num_parameters);
  avg_return /= n_elite;

  std::fill(variance.begin(), variance.end(), 0.0);
  for (int i = 0; i < n_elite; i++) {
    int idx = trajectory_order[i];
    const TimeSpline& elite_plan = candidate_policy[idx].plan;
    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::ConstNode n = elite_plan.NodeAt(t);
      for (int j = 0; j < model->nu; j++) {
        double p_avg = parameters_scratch[t * model->nu + j];
        double pi = n.values()[j];
        double diff = pi - p_avg;
        variance[t * model->nu + j] += diff * diff / (n_elite - 1);
      }
    }
  }

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

  // iCEM elite memory: snapshot the top-k elite candidate policies for use
  // as seeds in the next Plan() call. This is the "shifted-elites" trick
  // from arXiv 2008.06389 §3.3.
  if (n_keep > 0) {
    if ((int)elite_memory.size() < n_keep) {
      int old = elite_memory.size();
      elite_memory.resize(n_keep);
      for (int k = old; k < n_keep; k++) {
        elite_memory[k].Allocate(model, *task, kMaxTrajectoryHorizon);
      }
    }
    for (int k = 0; k < n_keep; k++) {
      int idx = trajectory_order[k];
      elite_memory[k].CopyFrom(candidate_policy[idx], num_spline_points);
    }
    elite_memory_valid_ = true;
  } else {
    elite_memory_valid_ = false;
  }

  improvement =
      mju_max(avg_return - trajectory[trajectory_order[0]].total_return, 0.0);

  policy_update_compute_time = GetDuration(policy_update_start);
}

void iCEMPlanner::NominalTrajectory(int horizon) {
  auto nominal_policy = [&cp = resampled_policy](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };
  nominal_trajectory.Rollout(nominal_policy, task, model,
                             data_[ThreadPool::WorkerId()].get(), state.data(),
                             time, mocap.data(), userdata.data(), horizon);
}
void iCEMPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  NominalTrajectory(horizon);
}

void iCEMPlanner::ActionFromPolicy(double* action, const double* state,
                                   double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
}

void iCEMPlanner::ResamplePolicy(int horizon) {
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

// iCEM colored noise: AR(1) along the time (spline-point) axis, independent
// per control dimension. The recursion is
//     n[t] = α · n[t-1] + sqrt(1 - α²) · w[t]
// where w[t] ~ N(0, σ²). This preserves the marginal variance σ² while
// inducing temporal correlation cor(n[t], n[t+1]) = α. Approximates the
// paper's PSD ∝ 1/f^β trick at a fraction of the implementation cost; for
// β≈1 use α≈0.5, for β≈2 use α≈0.9.
void iCEMPlanner::AddNoiseToPolicy(int i, double std_min) {
  auto noise_start = std::chrono::steady_clock::now();

  int num_spline_points = candidate_policy[i].num_spline_points;
  int shift = i * (model->nu * kMaxTrajectoryHorizon);

  absl::BitGen gen_;
  const double a = colored_alpha_;
  const double sqrt_one_minus_a2 = std::sqrt(std::max(0.0, 1.0 - a * a));
  // Previous-step noise per control dim; starts at zero (cold-start) which
  // is fine since the next step is dominated by sqrt(1-α²)·w.
  std::vector<double> prev_n(model->nu, 0.0);

  for (int t = 0; t < num_spline_points; t++) {
    for (int j = 0; j < model->nu; j++) {
      int k = t * model->nu + j;
      double std = std::max(std::sqrt(variance[k]), std_min);
      double w = absl::Gaussian<double>(gen_, 0.0, std);
      double n = a * prev_n[j] + sqrt_one_minus_a2 * w;
      noise[k + shift] = n;
      prev_n[j] = n;
    }
  }

  for (int k = 0; k < candidate_policy[i].plan.Size(); k++) {
    TimeSpline::Node n = candidate_policy[i].plan.NodeAt(k);
    mju_addTo(n.values().data(), DataAt(noise, shift + k * model->nu),
              model->nu);
    Clamp(n.values().data(), model->actuator_ctrlrange, model->nu);
  }

  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

void iCEMPlanner::Rollouts(int num_trajectory, int horizon, ThreadPool& pool) {
  noise_compute_time = 0.0;

  double std_min = std_min_;
  double std_initial = std_initial_;
  int n_keep = elite_memory_valid_
                   ? std::min(n_elite_keep_, num_trajectory)
                   : 0;

  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    double std;
    if (i < num_trajectory * explore_fraction_) {
      std = std_initial;
    } else {
      std = std_min;
    }
    // For the first n_keep rollouts (if any), seed from saved elite memory
    // instead of adding fresh noise — these are the "kept" elites.
    bool use_elite_memory = (i < n_keep);
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   std, i, use_elite_memory]() {
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);
        if (use_elite_memory) {
          s.candidate_policy[i].CopyFrom(
              s.elite_memory[i], s.elite_memory[i].num_spline_points);
          s.candidate_policy[i].plan.SetInterpolation(
              s.resampled_policy.plan.Interpolation());
          // Re-evaluate verbatim, no extra noise — we want to see if the
          // remembered policy is still good given the new state.
        } else {
          s.candidate_policy[i].CopyFrom(s.resampled_policy,
                                         s.resampled_policy.num_spline_points);
          s.candidate_policy[i].plan.SetInterpolation(
              s.resampled_policy.plan.Interpolation());
          s.AddNoiseToPolicy(i, std);
        }
      }

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
  pool.Schedule([&s = *this, horizon]() { s.NominalTrajectory(horizon); });

  pool.WaitCount(count_before + num_trajectory + 1);
  pool.ResetCount();
}

const Trajectory* iCEMPlanner::BestTrajectory() {
  return &nominal_trajectory;
}

void iCEMPlanner::Traces(mjvScene* scn) {
  float color[4] = {1.0, 1.0, 1.0, 1.0};
  double width = GetNumberOrDefault(3, model, "agent_sample_width");
  double zero3[3] = {0};
  double zero9[9] = {0};

  auto best = this->BestTrajectory();
  int n_elite = n_elite_;
  for (int k = 0; k < n_elite; k++) {
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     color);
        int idx = trajectory_order[k];
        mjv_connector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[idx].trace.data() + 3 * task->num_trace * i + 3 * j,
            trajectory[idx].trace.data() + 3 * task->num_trace * (i + 1) +
                3 * j);
        scn->ngeom += 1;
      }
    }
  }
}

void iCEMPlanner::GUI(mjUI& ui) {
  mjuiDef defICEM[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_, "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Init. Std", 2, &std_initial_, "0 1"},
      {mjITEM_SLIDERNUM, "Min. Std", 2, &std_min_, "0.01 0.5"},
      {mjITEM_SLIDERNUM, "Explore", 2, &explore_fraction_, "0.0 1.0"},
      {mjITEM_SLIDERINT, "Elite", 2, &n_elite_, "2 128"},
      {mjITEM_SLIDERNUM, "Colored α", 2, &colored_alpha_, "0.0 0.99"},
      {mjITEM_SLIDERINT, "Elite Keep", 2, &n_elite_keep_, "0 16"},
      {mjITEM_END}};

  mju::sprintf_arr(defICEM[0].other, "%i %i", 1, kMaxTrajectory);
  mju::sprintf_arr(defICEM[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);
  mju::sprintf_arr(defICEM[3].other, "%f %f", MinNoiseStdDev, MaxNoiseStdDev);

  mjui_add(&ui, defICEM);
}

void iCEMPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
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

}  // namespace mjpc
