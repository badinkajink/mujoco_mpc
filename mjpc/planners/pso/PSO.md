# Particle Swarm Optimization (PSO) Planner Implementation Plan

## Overview

This document outlines the implementation plan for adding a Particle Swarm Optimization (PSO) planner to MuJoCo MPC. PSO is a population-based metaheuristic optimization algorithm that, like the existing sampling and cross-entropy planners, does not require gradient information and only needs access to the cost/objective function.

---

## 1. Analysis of Existing Sampling-Based Planners

### 1.1 Common Architecture

All sampling-based planners inherit from `Planner` (or `RankedPlanner`) and share a common structure:

```
mjpc/planners/
├── planner.h              # Base virtual interface
├── sampling/
│   ├── planner.h/cc       # Random sampling planner
│   └── policy.h/cc        # SamplingPolicy (spline-based)
├── cross_entropy/
│   └── planner.h/cc       # CEM planner (uses SamplingPolicy)
└── robust/
    └── robust_planner.h/cc # Wrapper for perturbation robustness
```

### 1.2 Base Planner Interface (`planner.h:32-80`)

```cpp
class Planner {
  virtual void Initialize(mjModel* model, const Task& task) = 0;
  virtual void Allocate() = 0;
  virtual void Reset(int horizon, const double* initial_repeated_action = nullptr) = 0;
  virtual void SetState(const State& state) = 0;
  virtual void OptimizePolicy(int horizon, ThreadPool& pool) = 0;
  virtual void NominalTrajectory(int horizon, ThreadPool& pool) = 0;
  virtual void ActionFromPolicy(double* action, const double* state, double time, bool use_previous = false) = 0;
  virtual const Trajectory* BestTrajectory() = 0;
  virtual void Traces(mjvScene* scn) = 0;
  virtual void GUI(mjUI& ui) = 0;
  virtual void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, ...) = 0;
  virtual int NumParameters() = 0;
};
```

### 1.3 RankedPlanner Interface (Optional)

For planners that want to generate multiple ranked policy candidates:

```cpp
class RankedPlanner : public Planner {
  virtual int OptimizePolicyCandidates(int ncandidates, int horizon, ThreadPool& pool) = 0;
  virtual double CandidateScore(int candidate) const = 0;
  virtual void ActionFromCandidatePolicy(double* action, int candidate, const double* state, double time) = 0;
  virtual void CopyCandidateToPolicy(int candidate) = 0;
};
```

### 1.4 Sampling Planner Analysis (`sampling/planner.cc`)

**Key Components:**
- **Policy Representation**: Uses `SamplingPolicy` with `TimeSpline` for interpolation
- **Noise Generation**: Gaussian noise scaled by control range (`AddNoiseToPolicy`)
- **Parallel Rollouts**: Uses `ThreadPool` to evaluate trajectories in parallel
- **Selection**: Picks trajectory with lowest `total_return` (cost)

**Optimization Loop:**
1. `UpdateNominalPolicy(horizon)` - Resample policy to current time
2. Generate `num_trajectory` candidates by adding noise to nominal
3. `Rollouts()` - Parallel simulation of all candidate policies
4. `partial_sort` trajectories by `total_return`
5. Copy best candidate to nominal policy

**Key Parameters:**
- `num_trajectory_`: Number of samples (default 10)
- `noise_exploration[0,1]`: Noise standard deviation(s)
- `interpolation_`: Spline type (Zero/Linear/Cubic)
- `policy.num_spline_points`: Number of control points

### 1.5 Cross-Entropy Planner Analysis (`cross_entropy/planner.cc`)

**Key Differences from Sampling:**
- Maintains `variance` vector updated from elite samples
- Uses elite averaging to update mean policy
- Adapts variance based on elite distribution

**Optimization Loop:**
1. `ResamplePolicy(horizon)` - Resample policy to current time
2. Generate candidates with noise scaled by per-parameter variance
3. `Rollouts()` - Parallel simulation
4. `partial_sort` trajectories by `total_return`
5. **Elite Update**: Average parameters of top `n_elite_` trajectories
6. **Variance Update**: Compute variance from elite parameter distribution
7. Copy averaged parameters to policy

**Key Parameters:**
- `n_elite_`: Number of elite samples (default: 10% of trajectories)
- `std_initial_`: Initial standard deviation
- `std_min_`: Minimum standard deviation floor
- `explore_fraction_`: Fraction using full exploration noise

---

## 2. PSO Algorithm Overview

### 2.1 Standard PSO

Each particle maintains:
- **Position** `x_i`: Current solution (control parameters)
- **Velocity** `v_i`: Rate of change
- **Personal Best** `p_i`: Best position found by this particle
- **Personal Best Cost** `p_cost_i`: Cost at personal best

Global state:
- **Global Best** `g`: Best position found by any particle
- **Global Best Cost** `g_cost`: Cost at global best

**Update Equations:**
```
v_i = w * v_i + c1 * r1 * (p_i - x_i) + c2 * r2 * (g - x_i)
x_i = x_i + v_i
```

Where:
- `w`: Inertia weight (typically 0.4-0.9)
- `c1`: Cognitive coefficient (personal best attraction, ~2.0)
- `c2`: Social coefficient (global best attraction, ~2.0)
- `r1, r2`: Random values in [0, 1]

### 2.2 Adaptations for MPC

For receding-horizon MPC, PSO needs modifications:

1. **Warm-Starting**: When time advances, shift particle positions and bests
2. **Velocity Clamping**: Prevent velocity explosion
3. **Position Clamping**: Enforce actuator limits
4. **Time-Varying Inertia**: Potentially adapt `w` during optimization

---

## 3. Implementation Plan

### 3.1 File Structure

```
mjpc/planners/pso/
├── PSO.md          # This document
├── planner.h       # PSOPlanner class declaration
├── planner.cc      # PSOPlanner implementation
└── (optional: policy.h/cc if custom policy needed)
```

### 3.2 Class Design

```cpp
// mjpc/planners/pso/planner.h

class PSOPlanner : public RankedPlanner {
 public:
  PSOPlanner() = default;
  ~PSOPlanner() override = default;

  // ----- Planner Interface ----- //
  void Initialize(mjModel* model, const Task& task) override;
  void Allocate() override;
  void Reset(int horizon, const double* initial_repeated_action = nullptr) override;
  void SetState(const State& state) override;
  void OptimizePolicy(int horizon, ThreadPool& pool) override;
  void NominalTrajectory(int horizon, ThreadPool& pool) override;
  void ActionFromPolicy(double* action, const double* state, double time, bool use_previous) override;
  const Trajectory* BestTrajectory() override;
  void Traces(mjvScene* scn) override;
  void GUI(mjUI& ui) override;
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, ...) override;
  int NumParameters() override;

  // ----- RankedPlanner Interface ----- //
  int OptimizePolicyCandidates(int ncandidates, int horizon, ThreadPool& pool) override;
  double CandidateScore(int candidate) const override;
  void ActionFromCandidatePolicy(double* action, int candidate, const double* state, double time) override;
  void CopyCandidateToPolicy(int candidate) override;

 private:
  // ----- PSO-Specific Methods ----- //
  void InitializeSwarm();
  void UpdateVelocities();
  void UpdatePositions();
  void EvaluateParticles(int horizon, ThreadPool& pool);
  void UpdatePersonalBests();
  void UpdateGlobalBest();
  void ShiftSwarmForNewTime(double new_time);

  // ----- Members ----- //
  mjModel* model_;
  const Task* task_;

  // State
  std::vector<double> state_;
  double time_;
  std::vector<double> mocap_;
  std::vector<double> userdata_;

  // Particles (each particle = one policy candidate)
  int num_particles_;
  std::vector<SamplingPolicy> particle_positions_;    // Current positions (policies)
  std::vector<SamplingPolicy> particle_velocities_;   // Velocities
  std::vector<SamplingPolicy> personal_bests_;        // Personal best positions
  std::vector<double> personal_best_costs_;           // Personal best costs

  // Global best
  SamplingPolicy global_best_;
  double global_best_cost_;
  int global_best_index_;

  // Output policy
  SamplingPolicy policy_;          // (Guarded by mtx_)
  SamplingPolicy previous_policy_;

  // Trajectories for evaluation
  Trajectory trajectory_[kMaxTrajectory];
  std::vector<int> trajectory_order_;

  // PSO Hyperparameters
  double inertia_weight_;          // w: typically 0.4-0.9
  double cognitive_coeff_;         // c1: personal best attraction
  double social_coeff_;            // c2: global best attraction
  double velocity_clamp_;          // Maximum velocity magnitude

  // Spline settings
  mjpc::spline::SplineInterpolation interpolation_;

  // Timing
  std::atomic<double> velocity_update_time_;
  double rollouts_compute_time_;
  double policy_update_compute_time_;

  mutable std::shared_mutex mtx_;
};
```

### 3.3 Implementation Details

#### 3.3.1 `Initialize()`

```cpp
void PSOPlanner::Initialize(mjModel* model, const Task& task) {
  data_.clear();
  ResizeMjData(model, 1);

  model_ = model;
  task_ = &task;

  // Read hyperparameters from model or use defaults
  num_particles_ = GetNumberOrDefault(20, model, "pso_particles");
  inertia_weight_ = GetNumberOrDefault(0.7, model, "pso_inertia");
  cognitive_coeff_ = GetNumberOrDefault(1.5, model, "pso_cognitive");
  social_coeff_ = GetNumberOrDefault(1.5, model, "pso_social");
  velocity_clamp_ = GetNumberOrDefault(0.2, model, "pso_velocity_clamp");

  interpolation_ = GetNumberOrDefault(SplineInterpolation::kCubicSpline, model,
                                       "sampling_representation");

  if (num_particles_ > kMaxTrajectory) {
    mju_error_i("Too many particles, %d is the maximum allowed.", kMaxTrajectory);
  }

  global_best_index_ = 0;
}
```

#### 3.3.2 `Allocate()`

```cpp
void PSOPlanner::Allocate() {
  int num_state = model_->nq + model_->nv + model_->na;

  // State vectors
  state_.resize(num_state);
  mocap_.resize(7 * model_->nmocap);
  userdata_.resize(model_->nuserdata);

  // Particle arrays
  particle_positions_.resize(kMaxTrajectory);
  particle_velocities_.resize(kMaxTrajectory);
  personal_bests_.resize(kMaxTrajectory);
  personal_best_costs_.resize(kMaxTrajectory, std::numeric_limits<double>::max());

  for (int i = 0; i < kMaxTrajectory; i++) {
    particle_positions_[i].Allocate(model_, *task_, kMaxTrajectoryHorizon);
    particle_velocities_[i].Allocate(model_, *task_, kMaxTrajectoryHorizon);
    personal_bests_[i].Allocate(model_, *task_, kMaxTrajectoryHorizon);
    trajectory_[i].Initialize(num_state, model_->nu, task_->num_residual,
                               task_->num_trace, kMaxTrajectoryHorizon);
    trajectory_[i].Allocate(kMaxTrajectoryHorizon);
  }

  // Global best and output policy
  global_best_.Allocate(model_, *task_, kMaxTrajectoryHorizon);
  policy_.Allocate(model_, *task_, kMaxTrajectoryHorizon);
  previous_policy_.Allocate(model_, *task_, kMaxTrajectoryHorizon);

  trajectory_order_.resize(kMaxTrajectory);
  global_best_cost_ = std::numeric_limits<double>::max();
}
```

#### 3.3.3 `Reset()`

```cpp
void PSOPlanner::Reset(int horizon, const double* initial_repeated_action) {
  // Reset state
  std::fill(state_.begin(), state_.end(), 0.0);
  std::fill(mocap_.begin(), mocap_.end(), 0.0);
  std::fill(userdata_.begin(), userdata_.end(), 0.0);
  time_ = 0.0;

  // Reset all particles with random initialization
  absl::BitGen gen;
  for (int i = 0; i < num_particles_; i++) {
    particle_positions_[i].Reset(horizon, initial_repeated_action);
    particle_velocities_[i].Reset(horizon);  // Zero velocity
    personal_bests_[i].Reset(horizon, initial_repeated_action);
    personal_best_costs_[i] = std::numeric_limits<double>::max();
    trajectory_[i].Reset(kMaxTrajectoryHorizon);

    // Random initialization of positions (except first particle = nominal)
    if (i > 0) {
      for (auto& node : particle_positions_[i].plan) {
        for (int k = 0; k < model_->nu; k++) {
          double scale = 0.5 * (model_->actuator_ctrlrange[2*k + 1] -
                                 model_->actuator_ctrlrange[2*k]);
          node.values()[k] += absl::Gaussian<double>(gen, 0.0, scale * 0.3);
        }
        Clamp(node.values().data(), model_->actuator_ctrlrange, model_->nu);
      }
    }
  }

  // Reset global best
  global_best_.Reset(horizon, initial_repeated_action);
  global_best_cost_ = std::numeric_limits<double>::max();
  global_best_index_ = 0;

  // Reset output policy
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    policy_.Reset(horizon, initial_repeated_action);
    previous_policy_.Reset(horizon, initial_repeated_action);
  }
}
```

#### 3.3.4 `OptimizePolicy()` - Main PSO Loop

```cpp
void PSOPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  int num_particles = num_particles_;
  ResizeMjData(model_, pool.NumThreads());

  // ----- Evaluate Current Positions ----- //
  auto rollouts_start = std::chrono::steady_clock::now();
  EvaluateParticles(horizon, pool);
  rollouts_compute_time_ = GetDuration(rollouts_start);

  // ----- Update Personal and Global Bests ----- //
  UpdatePersonalBests();
  UpdateGlobalBest();

  // ----- Update Velocities and Positions ----- //
  auto velocity_start = std::chrono::steady_clock::now();
  UpdateVelocities();
  UpdatePositions();
  velocity_update_time_ = GetDuration(velocity_start);

  // ----- Copy Global Best to Output Policy ----- //
  auto policy_start = std::chrono::steady_clock::now();
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy_ = policy_;
    policy_.CopyFrom(global_best_, global_best_.num_spline_points);
  }
  policy_update_compute_time_ = GetDuration(policy_start);
}
```

#### 3.3.5 `UpdateVelocities()`

```cpp
void PSOPlanner::UpdateVelocities() {
  absl::BitGen gen;

  for (int i = 0; i < num_particles_; i++) {
    auto& pos = particle_positions_[i].plan;
    auto& vel = particle_velocities_[i].plan;
    auto& pbest = personal_bests_[i].plan;
    auto& gbest = global_best_.plan;

    for (int t = 0; t < pos.Size(); t++) {
      auto pos_node = pos.NodeAt(t);
      auto vel_node = vel.NodeAt(t);
      auto pbest_node = pbest.NodeAt(t);
      auto gbest_node = gbest.NodeAt(t);

      for (int k = 0; k < model_->nu; k++) {
        double r1 = absl::Uniform(gen, 0.0, 1.0);
        double r2 = absl::Uniform(gen, 0.0, 1.0);

        double cognitive = cognitive_coeff_ * r1 * (pbest_node.values()[k] - pos_node.values()[k]);
        double social = social_coeff_ * r2 * (gbest_node.values()[k] - pos_node.values()[k]);

        double new_vel = inertia_weight_ * vel_node.values()[k] + cognitive + social;

        // Clamp velocity
        double scale = 0.5 * (model_->actuator_ctrlrange[2*k + 1] -
                               model_->actuator_ctrlrange[2*k]);
        new_vel = mju_clip(new_vel, -velocity_clamp_ * scale, velocity_clamp_ * scale);

        vel_node.values()[k] = new_vel;
      }
    }
  }
}
```

#### 3.3.6 `UpdatePositions()`

```cpp
void PSOPlanner::UpdatePositions() {
  for (int i = 0; i < num_particles_; i++) {
    auto& pos = particle_positions_[i].plan;
    auto& vel = particle_velocities_[i].plan;

    for (int t = 0; t < pos.Size(); t++) {
      auto pos_node = pos.NodeAt(t);
      auto vel_node = vel.NodeAt(t);

      for (int k = 0; k < model_->nu; k++) {
        pos_node.values()[k] += vel_node.values()[k];
      }

      // Clamp to control limits
      Clamp(pos_node.values().data(), model_->actuator_ctrlrange, model_->nu);
    }
  }
}
```

#### 3.3.7 `EvaluateParticles()` - Parallel Rollouts

```cpp
void PSOPlanner::EvaluateParticles(int horizon, ThreadPool& pool) {
  int count_before = pool.GetCount();

  for (int i = 0; i < num_particles_; i++) {
    pool.Schedule([this, horizon, i]() {
      auto particle_policy = [&pos = particle_positions_[i]](
          double* action, const double* state, double time) {
        pos.Action(action, state, time);
      };

      trajectory_[i].Rollout(
          particle_policy, task_, model_,
          data_[ThreadPool::WorkerId()].get(),
          state_.data(), time_, mocap_.data(), userdata_.data(), horizon);
    });
  }

  pool.WaitCount(count_before + num_particles_);
  pool.ResetCount();
}
```

#### 3.3.8 `UpdatePersonalBests()`

```cpp
void PSOPlanner::UpdatePersonalBests() {
  for (int i = 0; i < num_particles_; i++) {
    if (trajectory_[i].total_return < personal_best_costs_[i]) {
      personal_best_costs_[i] = trajectory_[i].total_return;
      personal_bests_[i].CopyFrom(particle_positions_[i],
                                   particle_positions_[i].num_spline_points);
    }
  }
}
```

#### 3.3.9 `UpdateGlobalBest()`

```cpp
void PSOPlanner::UpdateGlobalBest() {
  for (int i = 0; i < num_particles_; i++) {
    if (personal_best_costs_[i] < global_best_cost_) {
      global_best_cost_ = personal_best_costs_[i];
      global_best_.CopyFrom(personal_bests_[i], personal_bests_[i].num_spline_points);
      global_best_index_ = i;
    }
  }
}
```

### 3.4 GUI Elements

```cpp
void PSOPlanner::GUI(mjUI& ui) {
  mjuiDef defPSO[] = {
      {mjITEM_SLIDERINT, "Particles", 2, &num_particles_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_, "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy_.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Inertia", 2, &inertia_weight_, "0.1 1.0"},
      {mjITEM_SLIDERNUM, "Cognitive", 2, &cognitive_coeff_, "0.0 4.0"},
      {mjITEM_SLIDERNUM, "Social", 2, &social_coeff_, "0.0 4.0"},
      {mjITEM_SLIDERNUM, "Vel Clamp", 2, &velocity_clamp_, "0.01 1.0"},
      {mjITEM_END}};

  mju::sprintf_arr(defPSO[0].other, "%i %i", 2, kMaxTrajectory);
  mju::sprintf_arr(defPSO[2].other, "%i %i", MinSamplingSplinePoints, MaxSamplingSplinePoints);

  mjui_add(&ui, defPSO);
}
```

---

## 4. Registration & Build Integration

### 4.1 Update `include.h`

Add to `PlannerType` enum:
```cpp
enum PlannerType : int {
  kSamplingPlanner = 0,
  kGradientPlanner,
  kILQGPlanner,
  kILQSPlanner,
  kRobustPlanner,
  kCrossEntropyPlanner,
  kSampleGradientPlanner,
  kPSOPlanner,  // <-- Add
};
```

### 4.2 Update `include.cc`

```cpp
#include "mjpc/planners/pso/planner.h"  // Add include

const char kPlannerNames[] =
    "Sampling\n"
    "Gradient\n"
    "iLQG\n"
    "iLQS\n"
    "Robust Sampling\n"
    "Cross Entropy\n"
    "Sample Gradient\n"
    "PSO";  // Add

std::vector<std::unique_ptr<mjpc::Planner>> LoadPlanners() {
  std::vector<std::unique_ptr<mjpc::Planner>> planners;

  planners.emplace_back(new mjpc::SamplingPlanner);
  planners.emplace_back(new mjpc::GradientPlanner);
  planners.emplace_back(new mjpc::iLQGPlanner);
  planners.emplace_back(new mjpc::iLQSPlanner);
  planners.emplace_back(new RobustPlanner(std::make_unique<mjpc::SamplingPlanner>()));
  planners.emplace_back(new mjpc::CrossEntropyPlanner);
  planners.emplace_back(new mjpc::SampleGradientPlanner);
  planners.emplace_back(new mjpc::PSOPlanner);  // Add

  return planners;
}
```

### 4.3 Update CMakeLists.txt

Add to `mjpc/CMakeLists.txt` in the planners source list:
```cmake
planners/pso/planner.h
planners/pso/planner.cc
```

---

## 5. PSO Variants to Consider

### 5.1 Basic Improvements

1. **Linearly Decreasing Inertia**: Start with high `w` for exploration, decrease over iterations
   ```cpp
   w = w_max - (w_max - w_min) * iteration / max_iterations;
   ```

2. **Constriction Factor**: Use Clerc's constriction coefficient
   ```cpp
   phi = c1 + c2;  // Should be > 4
   chi = 2.0 / fabs(2 - phi - sqrt(phi*phi - 4*phi));
   v = chi * (v + c1*r1*(pbest-x) + c2*r2*(gbest-x));
   ```

3. **Velocity Reinitialization**: Reset velocity when particle stagnates

### 5.2 Advanced Variants

1. **Ring Topology**: Each particle only influenced by neighbors, not global best
2. **Hybrid PSO-CEM**: Use CEM variance estimation to inform PSO exploration
3. **Adaptive PSO**: Auto-tune hyperparameters based on swarm diversity

### 5.3 MPC-Specific Adaptations

1. **Warm-Start Shifting**: When time advances, shift particle positions/bests
2. **Reset Strategy**: Periodically reinitialize worst particles
3. **Multi-Objective**: Handle multiple cost terms separately

---

## 6. Testing Plan

### 6.1 Unit Tests

Create `mjpc/test/pso_planner/` with:
- `pso_planner_test.cc`: Test initialization, reset, optimization

### 6.2 Integration Tests

- Test on existing tasks (Cartpole, Walker, etc.)
- Compare performance with Sampling and Cross-Entropy planners

### 6.3 Benchmark Comparisons

| Metric | Sampling | CEM | PSO (Expected) |
|--------|----------|-----|----------------|
| Convergence Speed | Slow | Medium | Fast |
| Sample Efficiency | Low | Medium | Medium-High |
| Exploration | High | Adaptive | High |
| Exploitation | Low | High | Medium-High |

---

## 7. Implementation Checklist

- [ ] Create `mjpc/planners/pso/` directory
- [ ] Implement `planner.h` with class declaration
- [ ] Implement `planner.cc` with full functionality
  - [ ] `Initialize()`, `Allocate()`, `Reset()`
  - [ ] `SetState()`, `OptimizePolicy()`
  - [ ] `UpdateVelocities()`, `UpdatePositions()`
  - [ ] `EvaluateParticles()`, `UpdatePersonalBests()`, `UpdateGlobalBest()`
  - [ ] `ActionFromPolicy()`, `BestTrajectory()`
  - [ ] `GUI()`, `Plots()`, `Traces()`
- [ ] Update `include.h` with `kPSOPlanner` enum
- [ ] Update `include.cc` to register planner
- [ ] Update `CMakeLists.txt`
- [ ] Write unit tests
- [ ] Test on standard tasks
- [ ] Tune default hyperparameters
- [ ] Document model-specific parameters (`pso_particles`, etc.)

---

## 8. Empirical Findings

PSO is implemented and functional. Key observations from testing:

- **Requires aggressive tuning**: default hyperparameters (c1=1.5, c2=1.5, w=0.7) do not work well. Best results came from low social/cognitive coefficients, high velocity scale, and maximum particles.
- **Does not match MPPI quality**: the Sampling planner (MPPI with cost-weighted averaging) consistently produces better plans than PSO with the same number of rollouts. PSO discards cost information from non-best particles, while MPPI uses soft exponential weighting across all samples.
- **Speed is comparable**: the bottleneck for both PSO and MPPI is the parallel MuJoCo rollout step, not the update rule. PSO's velocity update adds negligible overhead.

See [PSO_Upgrades.md](PSO_Upgrades.md) for a detailed analysis of why PSO underperforms MPPI, potential improvement paths (including DIAL-MPC dual-loop annealing), and GPU scaling considerations.

---

## 9. References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*.
2. Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability, and convergence. *IEEE TEC*.
3. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. *IEEE CEC*.
4. MuJoCo MPC: https://github.com/google-deepmind/mujoco_mpc
5. Xue, Pan, Yi, Qu, Shi. "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing." (DIAL-MPC)
