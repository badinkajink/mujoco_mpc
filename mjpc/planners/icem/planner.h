// Copyright 2026 — iCEM planner port for MJPC.
//
// Based on Pinneri et al., "Sample-efficient Cross-Entropy Method for
// Real-time Planning" (CoRL 2020, arXiv 2008.06389) and the reference
// implementation at github.com/martius-lab/iCEM.
//
// Two algorithmic additions on top of MJPC's CrossEntropyPlanner:
//   1. Temporally correlated ("colored") action noise via an AR(1) filter
//      along the spline-point time axis. Approximates the paper's
//      PSD ∝ 1/f^β noise without the FFT machinery; one parameter
//      controls the correlation strength.
//   2. Optional elite-memory: a small number of top elite samples from the
//      previous Plan() call are re-evaluated in the next call, biasing
//      exploration toward known-good trajectories. (Implicit warm-start
//      via previous_policy is preserved on top of this.)
//
// All other plumbing (rollout pool, spline policy, GUI elements, plots) is
// inherited unchanged from the CEM planner.

#ifndef MJPC_PLANNERS_ICEM_PLANNER_H_
#define MJPC_PLANNERS_ICEM_PLANNER_H_

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

class iCEMPlanner : public Planner {
 public:
  iCEMPlanner() = default;
  ~iCEMPlanner() override = default;

  // ----- Planner interface ----- //
  void Initialize(mjModel* model, const Task& task) override;
  void Allocate() override;
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;
  void SetState(const State& state) override;
  void OptimizePolicy(int horizon, ThreadPool& pool) override;
  void NominalTrajectory(int horizon, ThreadPool& pool) override;
  void NominalTrajectory(int horizon);
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;

  // ----- iCEM-specific helpers ----- //
  void ResamplePolicy(int horizon);
  // AR(1) colored-noise variant of CEM's per-spline Gaussian sampling.
  void AddNoiseToPolicy(int i, double std_min);
  void Rollouts(int num_trajectory, int horizon, ThreadPool& pool);

  const Trajectory* BestTrajectory() override;
  void Traces(mjvScene* scn) override;
  void GUI(mjUI& ui) override;
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

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
  SamplingPolicy policy;
  SamplingPolicy candidate_policy[kMaxTrajectory];
  SamplingPolicy resampled_policy;
  SamplingPolicy previous_policy;

  // scratch
  std::vector<double> parameters_scratch;
  std::vector<double> times_scratch;

  // trajectories
  Trajectory trajectory[kMaxTrajectory];
  Trajectory nominal_trajectory;
  std::vector<int> trajectory_order;

  // ----- noise / iCEM tuning ----- //
  double std_initial_;
  double std_min_;
  double explore_fraction_ = 0;
  std::vector<double> noise;
  std::vector<double> variance;

  // AR(1) temporal correlation coefficient α in [0, 0.99].
  //   α = 0    → white noise (vanilla CEM)
  //   α = 0.7  → moderate correlation, ≈ pink noise (β≈1)
  //   α = 0.95 → strong correlation, ≈ brownian (β≈2)
  // Tuned per task via the model's <custom><numeric name="icem_alpha"/> field.
  double colored_alpha_;

  // Number of elite samples to KEEP across Plan() calls (re-evaluated next
  // iteration instead of being resampled).
  int n_elite_keep_;
  // Saved elite policies from the previous Plan() call (size up to
  // n_elite_keep_). At the start of the next call, the first n_elite_keep_
  // candidate_policies are seeded from these.
  std::vector<SamplingPolicy> elite_memory;
  bool elite_memory_valid_ = false;

  int n_elite_;
  double improvement;
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  mjpc::spline::SplineInterpolation interpolation_ =
      mjpc::spline::SplineInterpolation::kZeroSpline;
  int num_trajectory_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_ICEM_PLANNER_H_
