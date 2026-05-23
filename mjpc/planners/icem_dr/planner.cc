// Copyright 2026 — ensemble-DR iCEM planner for MJPC.
//
// Derived from mjpc/planners/icem/planner.cc. The ONLY algorithmic change vs
// iCEM is the scoring of each candidate: instead of a single rollout on the
// nominal model, each candidate is rolled out on R domain-randomized model
// copies and scored by the MIN total return across the ensemble (risk-seeking
// aggregation). The AR(1) colored noise, elite fraction, and cross-Plan()
// elite memory are inherited verbatim.

#include "mjpc/planners/icem_dr/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <random>
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

// ---------------------------------------------------------------------------
// Domain-randomization ranges (applied to each of the R model copies).
// Each is a uniform multiplicative/explicit range; tweak here to retune DR.
// ---------------------------------------------------------------------------
// Tangential (slide) friction scale: geom_friction[g][0] *= U[lo, hi].
inline constexpr double kFrictionScaleLo = 0.5;
inline constexpr double kFrictionScaleHi = 1.5;
// Body mass scale: body_mass[b] *= U[lo, hi]; body_inertia[b][*] scaled by the
// SAME factor so the inertia stays physically consistent with the mass.
inline constexpr double kMassScaleLo = 0.8;
inline constexpr double kMassScaleHi = 1.2;
// Contact stiffness via solref timeconst: geom_solref[g][0] SET to U[lo, hi] s.
// (Smaller timeconst => stiffer contact.) Only applied to positive timeconsts
// so direct (negative-encoded stiffness/damping) solref entries are left alone.
inline constexpr double kSolrefTimeConstLo = 0.01;
inline constexpr double kSolrefTimeConstHi = 0.03;

void iCEMDRPlanner::Initialize(mjModel* model, const Task& task) {
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

  // iCEM additions (inherited): colored-noise AR(1) coefficient + elite memory.
  colored_alpha_ = GetNumberOrDefault(0.7, model, "icem_alpha");
  if (colored_alpha_ < 0.0) colored_alpha_ = 0.0;
  if (colored_alpha_ > 0.99) colored_alpha_ = 0.99;
  n_elite_keep_ = GetNumberOrDefault(2, model, "icem_elite_keep");

  // ensemble-DR: number of randomized model copies. Default 16; tunable via
  // <custom><numeric name="icem_dr_ensemble" data="..."/>. Clamped to bounds.
  n_ensemble_ = GetNumberOrDefault(kDefaultEnsembleSize, model,
                                   "icem_dr_ensemble");
  if (n_ensemble_ < 1) n_ensemble_ = 1;
  if (n_ensemble_ > kMaxEnsembleSize) n_ensemble_ = kMaxEnsembleSize;

  // Fixed, seedable RNG base so runs are reproducible. Tunable via
  // <numeric name="icem_dr_seed" data="..."/>.
  dr_seed_ = static_cast<uint64_t>(
      GetNumberOrDefault(12345, model, "icem_dr_seed"));
  plan_counter_ = 0;

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }

  // Build the ensemble once at init for a single thread; OptimizePolicy()
  // resizes/rebuilds it to the pool's thread count and refreshes the DR.
  ensemble_models_.clear();
  ensemble_data_.clear();
  ensemble_data_threads_ = 0;
  BuildEnsemble(1);
}

void iCEMDRPlanner::Allocate() {
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

  // iCEM elite memory buffer.
  elite_memory.resize(std::max(1, n_elite_keep_));
  for (auto& p : elite_memory) {
    p.Allocate(model, *task, kMaxTrajectoryHorizon);
  }
  elite_memory_valid_ = false;
}

void iCEMDRPlanner::Reset(int horizon, const double* initial_repeated_action) {
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
  for (const auto& d : ensemble_data_) {
    if (d) mju_zero(d->ctrl, model->nu);
  }

  plan_counter_ = 0;
  improvement = 0.0;
}

void iCEMDRPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

// ---------------------------------------------------------------------------
// ensemble-DR machinery
// ---------------------------------------------------------------------------

// Perturb a single freshly-copied model in place. NEVER call on the planner's
// nominal model — only on owned copies in ensemble_models_.
void iCEMDRPlanner::RandomizeModel(mjModel* m, uint64_t seed) const {
  // Deterministic per-model RNG seeded from `seed` for reproducibility.
  // absl::BitGen takes a [rand.req.seed_seq]-conforming seed sequence, so split
  // the 64-bit seed into two 32-bit words and feed a std::seed_seq.
  std::seed_seq seq{static_cast<uint32_t>(seed & 0xFFFFFFFFull),
                    static_cast<uint32_t>(seed >> 32)};
  absl::BitGen gen(seq);

  // Skip the world body (index 0). World geoms are those attached to body 0.
  for (int b = 1; b < m->nbody; b++) {
    double mass_scale = absl::Uniform<double>(gen, kMassScaleLo, kMassScaleHi);
    m->body_mass[b] *= mass_scale;
    // Scale the 3 diagonal inertia entries by the SAME factor to stay physical.
    m->body_inertia[3 * b + 0] *= mass_scale;
    m->body_inertia[3 * b + 1] *= mass_scale;
    m->body_inertia[3 * b + 2] *= mass_scale;
  }

  for (int g = 0; g < m->ngeom; g++) {
    // Skip geoms attached to the world body (index 0).
    if (m->geom_bodyid[g] == 0) continue;

    // Tangential (slide) friction scale.
    double fric_scale =
        absl::Uniform<double>(gen, kFrictionScaleLo, kFrictionScaleHi);
    m->geom_friction[3 * g + 0] *= fric_scale;

    // Contact stiffness via solref timeconst. solref[0] > 0 is a (timeconst,
    // dampratio) standard encoding; solref[0] <= 0 is the direct (negative)
    // stiffness/damping encoding — leave the latter untouched.
    double tc =
        absl::Uniform<double>(gen, kSolrefTimeConstLo, kSolrefTimeConstHi);
    if (m->geom_solref[mjNREF * g + 0] > 0.0) {
      m->geom_solref[mjNREF * g + 0] = tc;
    }
  }
}

// (Re)build the R randomized model copies and the per-(thread x member) mjData
// pool. `num_threads` is the rollout pool size. DR is freshly resampled here,
// so calling this each OptimizePolicy() refreshes the randomization (Hydrax-
// style); the per-model seed mixes dr_seed_, plan_counter_, and member index
// so the sequence is reproducible across runs with the same seed.
void iCEMDRPlanner::BuildEnsemble(int num_threads) {
  int R = std::max(1, std::min(n_ensemble_, kMaxEnsembleSize));
  int T = std::max(1, num_threads);

  // (Re)create the R model copies if their count changed.
  if ((int)ensemble_models_.size() != R) {
    ensemble_models_.clear();
    ensemble_models_.reserve(R);
    for (int r = 0; r < R; r++) {
      ensemble_models_.push_back(MakeUniqueMjModel(mj_copyModel(nullptr,
                                                                model)));
    }
    // Model set changed => existing mjData are stale; force a rebuild below.
    ensemble_data_.clear();
    ensemble_data_threads_ = 0;
  } else {
    // Refresh each copy from the nominal model before re-randomizing, so DR
    // perturbations don't compound across plan iterations.
    for (int r = 0; r < R; r++) {
      mj_copyModel(ensemble_models_[r].get(), model);
    }
  }

  // Apply fresh domain randomization to every copy.
  for (int r = 0; r < R; r++) {
    uint64_t seed = dr_seed_ ^ (plan_counter_ * 0x9E3779B97F4A7C15ull) ^
                    (static_cast<uint64_t>(r) * 0x100000001B3ull);
    RandomizeModel(ensemble_models_[r].get(), seed);
  }

  // (Re)allocate the per-(thread x member) mjData pool if its shape changed.
  // Each mjData MUST be made from the model it will be stepped against.
  if (ensemble_data_threads_ != T ||
      (int)ensemble_data_.size() != T * R) {
    ensemble_data_.clear();
    ensemble_data_.reserve(static_cast<size_t>(T) * R);
    for (int t = 0; t < T; t++) {
      for (int r = 0; r < R; r++) {
        ensemble_data_.push_back(
            MakeUniqueMjData(mj_makeData(ensemble_models_[r].get())));
      }
    }
    ensemble_data_threads_ = T;
  }

  // (Re)size per-worker scratch trajectories.
  if ((int)ensemble_scratch_.size() != T) {
    ensemble_scratch_.resize(T);
  }
  int num_state = model->nq + model->nv + model->na;
  for (int t = 0; t < T; t++) {
    ensemble_scratch_[t].Initialize(num_state, model->nu, task->num_residual,
                                    task->num_trace, kMaxTrajectoryHorizon);
    ensemble_scratch_[t].Allocate(kMaxTrajectoryHorizon);
  }
}

// Roll candidate i's policy across the whole ensemble; write the MIN return
// into trajectory[i]. Member 0 records into trajectory[i] (so traces/best
// plumbing is preserved); members 1..R-1 use the calling worker's scratch
// trajectory and only contribute their scalar total_return to the min.
void iCEMDRPlanner::EnsembleRolloutCandidate(int i, int horizon) {
  int wid = ThreadPool::WorkerId();
  if (wid < 0) wid = 0;  // defensive: should always be on a worker.
  int R = (int)ensemble_models_.size();

  auto policy_i = [&candidate_policy = this->candidate_policy, i](
                      double* action, const double* state, double time) {
    candidate_policy[i].Action(action, state, time);
  };

  double min_return = 0.0;
  for (int r = 0; r < R; r++) {
    mjData* d = ensemble_data_[static_cast<size_t>(wid) * R + r].get();
    mjModel* m = ensemble_models_[r].get();
    Trajectory* traj =
        (r == 0) ? &trajectory[i] : &ensemble_scratch_[wid];
    traj->Rollout(policy_i, task, m, d, state.data(), time, mocap.data(),
                  userdata.data(), horizon);
    if (r == 0) {
      min_return = traj->total_return;
    } else {
      min_return = std::min(min_return, traj->total_return);
    }
  }

  // Risk-seeking aggregation: score the candidate by its best-case (min) cost
  // across the randomized ensemble. trajectory[i] keeps member 0's full record
  // (states/trace) but its scalar return is overwritten with the aggregate so
  // the CEM elite sort uses the ensemble-min.
  trajectory[i].total_return = min_return;
}

// ---------------------------------------------------------------------------
// OptimizePolicy — identical to iCEM except Rollouts() scores via the ensemble.
// ---------------------------------------------------------------------------
void iCEMDRPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  resampled_policy.plan.SetInterpolation(interpolation_);

  int num_trajectory = num_trajectory_;
  n_elite_ = std::min(n_elite_, num_trajectory);
  int n_elite = std::min(n_elite_, num_trajectory);
  int n_keep = std::min(n_elite_keep_, n_elite);

  ResizeMjData(model, pool.NumThreads());

  // Rebuild/refresh the DR ensemble + its mjData pool for this plan iteration.
  // Resampling DR every call (rather than fixing it at Init) keeps the
  // randomization fresh, matching Hydrax-style domain-randomized sampling MPC.
  plan_counter_++;
  BuildEnsemble(pool.NumThreads());

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

  // iCEM elite memory: snapshot the top-k elite candidate policies for use as
  // seeds in the next Plan() call (the "shifted-elites" trick).
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

void iCEMDRPlanner::NominalTrajectory(int horizon) {
  auto nominal_policy = [&cp = resampled_policy](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };
  nominal_trajectory.Rollout(nominal_policy, task, model,
                             data_[ThreadPool::WorkerId()].get(), state.data(),
                             time, mocap.data(), userdata.data(), horizon);
}
void iCEMDRPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  NominalTrajectory(horizon);
}

void iCEMDRPlanner::ActionFromPolicy(double* action, const double* state,
                                     double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
}

void iCEMDRPlanner::ResamplePolicy(int horizon) {
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
// per control dimension. n[t] = α·n[t-1] + sqrt(1-α²)·w[t], w ~ N(0, σ²).
void iCEMDRPlanner::AddNoiseToPolicy(int i, double std_min) {
  auto noise_start = std::chrono::steady_clock::now();

  int num_spline_points = candidate_policy[i].num_spline_points;
  int shift = i * (model->nu * kMaxTrajectoryHorizon);

  absl::BitGen gen_;
  const double a = colored_alpha_;
  const double sqrt_one_minus_a2 = std::sqrt(std::max(0.0, 1.0 - a * a));
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

// Identical scheduling to iCEM, but each candidate is SCORED via the ensemble
// (EnsembleRolloutCandidate) instead of a single nominal-model rollout.
void iCEMDRPlanner::Rollouts(int num_trajectory, int horizon,
                             ThreadPool& pool) {
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
    bool use_elite_memory = (i < n_keep);
    pool.Schedule([&s = *this, horizon, std, i, use_elite_memory]() {
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);
        if (use_elite_memory) {
          s.candidate_policy[i].CopyFrom(
              s.elite_memory[i], s.elite_memory[i].num_spline_points);
          s.candidate_policy[i].plan.SetInterpolation(
              s.resampled_policy.plan.Interpolation());
        } else {
          s.candidate_policy[i].CopyFrom(s.resampled_policy,
                                         s.resampled_policy.num_spline_points);
          s.candidate_policy[i].plan.SetInterpolation(
              s.resampled_policy.plan.Interpolation());
          s.AddNoiseToPolicy(i, std);
        }
      }

      // Score this candidate by rolling it out across the DR ensemble and
      // taking the min total return.
      s.EnsembleRolloutCandidate(i, horizon);
    });
  }
  pool.Schedule([&s = *this, horizon]() { s.NominalTrajectory(horizon); });

  pool.WaitCount(count_before + num_trajectory + 1);
  pool.ResetCount();
}

const Trajectory* iCEMDRPlanner::BestTrajectory() {
  return &nominal_trajectory;
}

void iCEMDRPlanner::Traces(mjvScene* scn) {
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

void iCEMDRPlanner::GUI(mjUI& ui) {
  mjuiDef defICEMDR[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_, "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Init. Std", 2, &std_initial_, "0 1"},
      {mjITEM_SLIDERNUM, "Min. Std", 2, &std_min_, "0.01 0.5"},
      {mjITEM_SLIDERNUM, "Explore", 2, &explore_fraction_, "0.0 1.0"},
      {mjITEM_SLIDERINT, "Elite", 2, &n_elite_, "2 128"},
      {mjITEM_SLIDERNUM, "Colored α", 2, &colored_alpha_, "0.0 0.99"},
      {mjITEM_SLIDERINT, "Elite Keep", 2, &n_elite_keep_, "0 16"},
      {mjITEM_SLIDERINT, "DR Ensemble", 2, &n_ensemble_, "1 64"},
      {mjITEM_END}};

  mju::sprintf_arr(defICEMDR[0].other, "%i %i", 1, kMaxTrajectory);
  mju::sprintf_arr(defICEMDR[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);
  mju::sprintf_arr(defICEMDR[3].other, "%f %f", MinNoiseStdDev, MaxNoiseStdDev);
  mju::sprintf_arr(defICEMDR[9].other, "%i %i", 1, kMaxEnsembleSize);

  mjui_add(&ui, defICEMDR);
}

void iCEMDRPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
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
