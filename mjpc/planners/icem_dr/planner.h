// Copyright 2026 — ensemble-DR iCEM planner for MJPC.
//
// "Risk-aware domain-randomized iCEM" — the Route-2 online-MPC brain for
// sim2real on the Unitree H1-2. Algorithmically identical to the iCEM planner
// (mjpc/planners/icem/) EXCEPT for how each sampled action-sequence is scored:
//
//   - The planner maintains R domain-randomized COPIES of the mjModel
//     (default R = 16; see n_ensemble_). Each copy perturbs geom friction,
//     body mass+inertia, and contact stiffness within fixed ranges.
//   - Every candidate spline is rolled out on ALL R models from the current
//     state, and its score is the MIN total cost across the ensemble
//     (optimistic / risk-seeking aggregation — per the risk-aware-DR result
//     that min beats mean/max on contact-rich tasks).
//   - The CEM elite selection + AR(1) refit then proceed on these aggregated
//     costs exactly as iCEM does.
//
// Everything else — AR(1) colored noise, elite fraction, cross-plan elite
// memory, spline representation, GUI/plots — is inherited from iCEM. This is
// an ADDITIVE planner (a new index/name); the original iCEM (#7) is untouched.
//
// Conceptually mirrors Hydrax's vmap'd domain-randomized sampling MPC and the
// "risk-aware MPC via DR" min-aggregation result, re-expressed in MJPC's C++
// threading idiom (no JAX/vmap; an explicit per-thread ensemble loop instead).

#ifndef MJPC_PLANNERS_ICEM_DR_PLANNER_H_
#define MJPC_PLANNERS_ICEM_DR_PLANNER_H_

#include <atomic>
#include <memory>
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

// UniqueMjModel / MakeUniqueMjModel and UniqueMjData / MakeUniqueMjData are
// provided by mjpc/utilities.h (pulled in transitively via planner.h).

// Default number of domain-randomized model copies in the ensemble.
inline constexpr int kDefaultEnsembleSize = 16;
// Hard cap so the GUI slider and allocations stay bounded.
inline constexpr int kMaxEnsembleSize = 64;

class iCEMDRPlanner : public Planner {
 public:
  iCEMDRPlanner() = default;
  ~iCEMDRPlanner() override = default;

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

  // ----- iCEM-specific helpers (inherited behavior) ----- //
  void ResamplePolicy(int horizon);
  // AR(1) colored-noise variant of CEM's per-spline Gaussian sampling.
  void AddNoiseToPolicy(int i, double std_min);
  void Rollouts(int num_trajectory, int horizon, ThreadPool& pool);

  // ----- ensemble-DR-specific helpers ----- //
  // (Re)build the R randomized model copies from the nominal model, and
  // (re)allocate the per-(thread x member) mjData pool to match.
  void BuildEnsemble(int num_threads);
  // Perturb a single freshly-copied model in place using the DR ranges.
  void RandomizeModel(mjModel* m, uint64_t seed) const;
  // Roll out candidate i's policy across the whole ensemble and write the
  // min-aggregated return into trajectory[i]. Runs on a worker thread.
  void EnsembleRolloutCandidate(int i, int horizon);

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

  // AR(1) temporal correlation coefficient α in [0, 0.99]. See icem planner.
  double colored_alpha_;

  // Cross-Plan() elite memory (inherited from iCEM).
  int n_elite_keep_;
  std::vector<SamplingPolicy> elite_memory;
  bool elite_memory_valid_ = false;

  int n_elite_;
  double improvement;
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  // ----- ensemble-DR members ----- //
  // Number of domain-randomized model copies. GUI-tunable, clamped to
  // [1, kMaxEnsembleSize]. Tuned per task via <numeric name="icem_dr_ensemble"/>.
  int n_ensemble_;
  // Owned randomized model copies (size n_ensemble_).
  std::vector<UniqueMjModel> ensemble_models_;
  // Per-(thread x member) mjData, flat-indexed [thread * n_ensemble_ + member].
  // Each entry is allocated against ensemble_models_[member] and must only be
  // touched by its owning worker thread. Rebuilt when thread count or ensemble
  // size changes.
  std::vector<UniqueMjData> ensemble_data_;
  // Number of threads ensemble_data_ was last sized for.
  int ensemble_data_threads_ = 0;
  // Per-worker scratch trajectories for ensemble members 1..R-1 (member 0
  // reuses trajectory[i] so traces/BestTrajectory plumbing is preserved).
  // Flat-indexed [thread]; one scratch per worker is enough because a worker
  // evaluates the ensemble members sequentially.
  std::vector<Trajectory> ensemble_scratch_;
  // Base RNG seed; the per-model seed mixes this with the member index and a
  // monotonically increasing plan counter so runs are reproducible yet DR is
  // refreshed each OptimizePolicy() call.
  uint64_t dr_seed_ = 0;
  uint64_t plan_counter_ = 0;

  mjpc::spline::SplineInterpolation interpolation_ =
      mjpc::spline::SplineInterpolation::kZeroSpline;
  int num_trajectory_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_ICEM_DR_PLANNER_H_
