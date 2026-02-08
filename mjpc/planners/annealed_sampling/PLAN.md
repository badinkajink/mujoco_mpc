# DIAL-MPC (Annealed Sampling) Implementation Plan

## Goal

Implement DIAL-MPC's dual-loop diffusion-inspired annealing on top of the Sampling planner. This adds two key changes on top of vanilla predictive sampling:

1. **MPPI cost-weighted update** (replace "pick best" with exponentially-weighted average of all samples)
2. **Dual-loop annealing schedule** (decrease noise across inner iterations; increase noise for farther-horizon actions)

Reference: Xue, Pan, Yi, Qu, Shi — "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing" (`../pso/dial_mpc_paper.pdf`)

---

## Step 0: Scaffolding — Rename & Register

The files are currently exact copies of `planners/sampling/`. Before any algorithm changes:

### 0a. Rename classes

In `planner.h` / `planner.cc`:
- `SamplingPlanner` → `AnnealedSamplingPlanner`
- Update include guard: `MJPC_PLANNERS_ANNEALED_SAMPLING_PLANNER_H_`
- Update `#include` paths to `"mjpc/planners/annealed_sampling/policy.h"`

In `policy.h` / `policy.cc`:
- `SamplingPolicy` → keep as `SamplingPolicy` (reuse from `sampling/policy.h`), OR rename to `AnnealedSamplingPolicy`
- **Simpler approach**: Don't copy policy.h/cc at all. Just `#include "mjpc/planners/sampling/policy.h"` and reuse `SamplingPolicy` directly. The policy class has no planner-specific logic.

### 0b. Register in framework

**`mjpc/planners/include.h`** — add to enum:
```cpp
kAnnealedSamplingPlanner,  // after kPSOPlanner
```

**`mjpc/planners/include.cc`** — add:
```cpp
#include "mjpc/planners/annealed_sampling/planner.h"
// in kPlannerNames:
"Annealed Sampling"   // append after "PSO"
// in LoadPlanners():
planners.emplace_back(new mjpc::AnnealedSamplingPlanner);
```

**`mjpc/CMakeLists.txt`** — add to libmjpc sources:
```cmake
planners/annealed_sampling/planner.h
planners/annealed_sampling/planner.cc
```
(No policy files needed if reusing `SamplingPolicy`.)

### 0c. Verify

Build and confirm the new planner appears in the GUI dropdown and behaves identically to Sampling (since the algorithm hasn't changed yet).

---

## Step 1: MPPI Cost-Weighted Update

The current sampling planner picks the single best trajectory. DIAL-MPC requires the MPPI update (eq. 1 in the paper):

```
U+ = U + [ Σ_i exp(-J_i/λ) * W_i ] / [ Σ_j exp(-J_j/λ) ]
```

where `W_i = candidate_policy[i] - nominal_policy` (the perturbation), and `λ` is the temperature parameter.

### 1a. Add temperature parameter

New member in `AnnealedSamplingPlanner`:
```cpp
double temperature_;  // λ: MPPI temperature (default 1.0)
```

Read from model:
```cpp
temperature_ = GetNumberOrDefault(1.0, model, "sampling_temperature");
```

Add to GUI.

### 1b. Implement weighted update in `OptimizePolicy`

After rollouts and sorting, instead of `CopyCandidateToPolicy(0)`:

```cpp
void AnnealedSamplingPlanner::MPPIUpdate() {
    // 1. Compute weights: w_i = exp(-J_i / λ)
    //    Use log-sum-exp for numerical stability:
    //    Find min cost, subtract it, then exponentiate
    double min_cost = trajectory[trajectory_order[0]].total_return;

    std::vector<double> weights(num_trajectory);
    double weight_sum = 0.0;
    for (int i = 0; i < num_trajectory; i++) {
        double cost = trajectory[i].total_return;
        weights[i] = exp(-(cost - min_cost) / temperature_);
        weight_sum += weights[i];
    }

    // 2. Normalize weights
    for (int i = 0; i < num_trajectory; i++) {
        weights[i] /= weight_sum;
    }

    // 3. Weighted average of candidate policy parameters
    //    For each spline node and each actuator dimension:
    //    policy_new[t][k] = Σ_i weights[i] * candidate_policy[i][t][k]
    //
    //    This is equivalent to: U+ = U + Σ weights[i] * W_i
    //    since candidate_policy[i] = policy + W_i
    {
        const std::unique_lock<std::shared_mutex> lock(mtx_);
        previous_policy = policy;

        // Zero out policy parameters, then accumulate weighted sum
        for (auto& node : policy.plan) {
            for (int k = 0; k < model->nu; k++) {
                node.values()[k] = 0.0;
            }
        }

        for (int i = 0; i < num_trajectory; i++) {
            int t = 0;
            for (auto& node : policy.plan) {
                auto candidate_node = candidate_policy[i].plan.NodeAt(t);
                for (int k = 0; k < model->nu; k++) {
                    node.values()[k] += weights[i] * candidate_node.values()[k];
                }
                t++;
            }
        }

        // Clamp to control limits
        for (auto& node : policy.plan) {
            Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
        }
    }
}
```

**Important**: The weighted average operates on the full candidate parameters (not just perturbations), because `candidate_policy[i] = nominal + noise_i`, so `Σ w_i * candidate_policy[i] = nominal + Σ w_i * noise_i`.

### 1c. Verify

Test that the MPPI-weighted planner produces smoother, better-quality plans than pick-best on a simple task (e.g., cartpole, particle). The temperature parameter controls the softness — low λ approaches pick-best, high λ approaches uniform average.

---

## Step 2: Dual-Loop Annealing

### 2a. Add annealing parameters

New members:
```cpp
int num_annealing_iters_;    // N: inner iterations (default 1, range 1-8)
double beta_trajectory_;     // β₁: trajectory-level annealing temp (default 0.5)
double beta_action_;         // β₂: action-level annealing temp (default 0.5)
```

Read from model:
```cpp
num_annealing_iters_ = GetNumberOrDefault(1, model, "annealing_iterations");
beta_trajectory_ = GetNumberOrDefault(0.5, model, "annealing_beta_trajectory");
beta_action_ = GetNumberOrDefault(0.5, model, "annealing_beta_action");
```

Add to GUI as sliders.

### 2b. Modify `OptimizePolicy` to run N iterations

```cpp
void AnnealedSamplingPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
    int N = num_annealing_iters_;

    for (int iter = 0; iter < N; iter++) {
        // Store current iteration index for AddNoiseToPolicy to use
        current_annealing_iter_ = iter;
        current_annealing_total_ = N;

        // Resample, rollout, sort (same as before)
        UpdateNominalPolicy(horizon);
        Rollouts(num_trajectory_, horizon, pool);
        SortTrajectories();

        // MPPI-weighted update (from Step 1)
        MPPIUpdate();
    }
}
```

### 2c. Modify `AddNoiseToPolicy` for dual-loop noise schedule

The DIAL-MPC noise schedule (eq. 7 from the paper):
```
Σ^i_{t+h} = exp(-(N-i)/(β₁·N) - (H-h)/(β₂·H)) · I
```

Where:
- `i` = current annealing iteration (1 to N)
- `h` = horizon index (0 to H)
- `β₁` = trajectory-level temperature
- `β₂` = action-level temperature

Modified `AddNoiseToPolicy`:
```cpp
void AnnealedSamplingPlanner::AddNoiseToPolicy(double start_time, int i) {
    absl::BitGen gen_;

    int N = current_annealing_total_;
    int iter = current_annealing_iter_;  // 0-indexed, so iter=0 is first (largest noise)
    int H = policy.num_spline_points;

    // Base noise std from GUI/model
    double base_std = noise_exploration[0];

    int h = 0;  // horizon index
    for (const TimeSpline::Node& node : candidate_policy[i].plan) {
        // Dual-loop annealing scale (eq. 7)
        // iter goes 0..N-1, map to "i goes N..1" in paper notation:
        //   paper_i = N - iter, so (N - paper_i)/(β₁·N) = iter/(β₁·N)
        double traj_anneal = exp(-(double)iter / (beta_trajectory_ * N));
        double action_anneal = exp(-(double)(H - 1 - h) / (beta_action_ * H));
        double effective_std = base_std * traj_anneal * action_anneal;

        for (int k = 0; k < model->nu; k++) {
            double scale = 0.5 * (model->actuator_ctrlrange[2*k+1] -
                                  model->actuator_ctrlrange[2*k]);
            node.values()[k] += absl::Gaussian<double>(gen_, 0.0, scale * effective_std);
        }
        Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
        h++;
    }
}
```

**Behavior**:
- `iter=0` (first iteration): `traj_anneal ≈ 1.0` → full exploration noise
- `iter=N-1` (last iteration): `traj_anneal = exp(-(N-1)/(β₁·N))` → reduced noise
- `h=0` (near-term actions): `action_anneal = exp(-(H-1)/(β₂·H))` → small noise (well-refined)
- `h=H-1` (far-horizon actions): `action_anneal ≈ 1.0` → large noise (needs more exploration)

### 2d. Verify

Compare on a locomotion task (e.g., ur5) with:
- `num_annealing_iters = 1, temperature = ∞`: should match vanilla sampling (pick-best)
- `num_annealing_iters = 1, temperature = 1.0`: MPPI-weighted but no annealing
- `num_annealing_iters = 4, beta1 = 0.5, beta2 = 0.5`: full DIAL-MPC

---

## Step 3: GUI & Plots

### 3a. GUI sliders

```cpp
void AnnealedSamplingPlanner::GUI(mjUI& ui) {
    mjuiDef defAnnealed[] = {
        {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
        {mjITEM_SELECT, "Spline", 2, &interpolation_, "Zero\nLinear\nCubic"},
        {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
        {mjITEM_SLIDERNUM, "Noise Std", 2, noise_exploration, "0 1"},
        {mjITEM_SLIDERINT, "Anneal Iters", 2, &num_annealing_iters_, "1 8"},
        {mjITEM_SLIDERNUM, "Temperature", 2, &temperature_, "0.001 10.0"},
        {mjITEM_SLIDERNUM, "Beta Traj", 2, &beta_trajectory_, "0.01 2.0"},
        {mjITEM_SLIDERNUM, "Beta Action", 2, &beta_action_, "0.01 2.0"},
        {mjITEM_END}
    };
    // ... set slider limits, add to UI
}
```

### 3b. Plots

Add per-iteration cost tracking to see the annealing convergence within each MPC step. Plot:
- Best cost per annealing iteration (should decrease across iterations)
- Timer breakdown: rollout time × N iterations, noise time, policy update time

---

## New Members Summary

```cpp
class AnnealedSamplingPlanner : public RankedPlanner {
    // ... inherited from SamplingPlanner ...

    // MPPI
    double temperature_;              // λ: softmax temperature

    // Dual-loop annealing
    int num_annealing_iters_;         // N: inner loop count
    double beta_trajectory_;          // β₁: trajectory-level annealing
    double beta_action_;              // β₂: action-level annealing

    // Internal state for current iteration
    int current_annealing_iter_;      // which inner iteration we're on
    int current_annealing_total_;     // total inner iterations

    // Scratch for MPPI weights
    std::vector<double> mppi_weights_;
};
```

---

## File Change Summary

| File | Change |
|------|--------|
| `annealed_sampling/planner.h` | Rename class, add new members, add `MPPIUpdate()` method |
| `annealed_sampling/planner.cc` | Rename class, implement MPPI update, annealing loop, modified `AddNoiseToPolicy` |
| `annealed_sampling/policy.h` | Delete — reuse `sampling/policy.h` |
| `annealed_sampling/policy.cc` | Delete — reuse `sampling/policy.cc` |
| `planners/include.h` | Add `kAnnealedSamplingPlanner` to enum |
| `planners/include.cc` | Add include, name, and loader entry |
| `mjpc/CMakeLists.txt` | Add `annealed_sampling/planner.h` and `planner.cc` |

---

## Default Hyperparameters

Based on the DIAL-MPC paper's experimental setup:

| Parameter | Default | Paper Value | Notes |
|-----------|---------|-------------|-------|
| `num_trajectory_` | 10 | 2048 | Paper uses GPU; CPU budget is ~10-128 |
| `temperature_` | 1.0 | task-specific | Lower = sharper selection |
| `num_annealing_iters_` | 4 | 4 | Paper uses 4 steps for crate climbing |
| `beta_trajectory_` | 0.5 | task-specific | Controls outer-loop decay rate |
| `beta_action_` | 0.5 | task-specific | Controls inner-loop horizon scaling |
| `noise_exploration[0]` | 0.1 | 0.05-0.2 | Base noise std, scaled by annealing |

**Note on sample budget**: DIAL-MPC runs on GPU with 2048 parallel samples. With CPU and ~128 max particles, the annealing effect will be weaker because each iteration has fewer samples to estimate the score function. The temperature may need to be higher (softer weighting) to compensate for noisier score estimates. Empirical tuning per task will be necessary.
