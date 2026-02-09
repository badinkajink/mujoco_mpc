# Annealed Sampling Planner - Self Audit

## Executive Summary

The Annealed Sampling Planner implements **DIAL-MPC** (Diffusion-Inspired Annealing for Legged Model Predictive Control). It extends the base random sampling planner with two key innovations:

1. **MPPI-style weighted averaging** instead of simply picking the best trajectory
2. **Dual-loop annealing schedule** that modulates noise based on iteration and horizon position

---

## Algorithm Overview

### Base Sampling (what we're comparing against)
```
For each optimization step:
  1. Copy nominal policy to N candidate policies
  2. Add random Gaussian noise to candidates 1..N-1 (candidate 0 = nominal)
  3. Rollout all candidates, compute costs
  4. Sort by cost
  5. Pick best trajectory → becomes new nominal policy
```

### Annealed Sampling (DIAL-MPC)
```
For each optimization step:
  For iter = 0 to N-1 (dual-loop):                    ← NEW: multiple inner iterations
    1. Copy nominal policy to candidates
    2. Add ANNEALED noise (decreases over iter)       ← CHANGED: noise schedule
    3. Rollout all candidates, compute costs
    4. Sort by cost
    5. MPPI weighted average of ALL policies → nominal  ← CHANGED: soft selection
```

---

## New Member Variables (planner.h:130-142)

```cpp
// ----- DIAL-MPC parameters ----- //

// MPPI temperature (lambda): lower = sharper selection, higher = softer
double temperature_ = 1.0;                    // Line 133

// dual-loop annealing
int num_annealing_iters_ = 4;                 // Line 136: N inner iterations
double beta_trajectory_ = 0.5;                // Line 137: trajectory annealing rate
double beta_action_ = 0.5;                    // Line 138: action/horizon annealing rate

// internal state for current annealing iteration
int current_annealing_iter_ = 0;              // Line 141
int current_annealing_total_ = 1;             // Line 142
```

### Parameter Initialization (planner.cc:67-73)
```cpp
// DIAL-MPC parameters
temperature_ = GetNumberOrDefault(1.0, model, "sampling_temperature");
num_annealing_iters_ = GetNumberOrDefault(4, model, "annealing_iterations");
beta_trajectory_ = GetNumberOrDefault(0.5, model, "annealing_beta_trajectory");
beta_action_ = GetNumberOrDefault(0.5, model, "annealing_beta_action");
```

---

## Key Algorithmic Changes

### 1. The Dual-Loop Structure (planner.cc:182-222)

**Base Sampling** runs one optimization pass per call:
```cpp
void SamplingPlanner::OptimizePolicy(...) {
  OptimizePolicyCandidates(1, horizon, pool);  // One pass
  CopyCandidateToPolicy(0);                    // Pick best
}
```

**Annealed Sampling** runs multiple iterations with decreasing noise:
```cpp
void AnnealedSamplingPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  int N = std::max(num_annealing_iters_, 1);                   // Line 183

  for (int iter = 0; iter < N; iter++) {                       // Line 187
    // Store annealing state for AddNoiseToPolicy
    current_annealing_iter_ = iter;                            // Line 189
    current_annealing_total_ = N;                              // Line 190

    // Resample, rollout, sort
    this->UpdateNominalPolicy(horizon);                        // Line 193
    this->Rollouts(num_trajectory, horizon, pool);             // Line 199

    // Sort ALL trajectories (not partial_sort)
    std::sort(trajectory_order.begin(), ...);                  // Line 207

    // MPPI-weighted update (NOT just picking best)
    MPPIUpdate(num_trajectory);                                // Line 213
  }
}
```

**Why this matters:**
- Multiple iterations allow progressive refinement
- Each iteration: explore → evaluate → soft-update → repeat
- Noise decreases over iterations (annealing schedule)

---

### 2. MPPI Weighted Policy Update (planner.cc:225-286)

This is the **core difference** from base sampling. Instead of picking the single best trajectory, DIAL-MPC computes a **weighted average of ALL trajectories**.

**Base Sampling** (CopyCandidateToPolicy):
```cpp
policy = candidate_policy[winner];  // Just copy the best one
```

**Annealed Sampling** (MPPIUpdate):
```cpp
void AnnealedSamplingPlanner::MPPIUpdate(int num_trajectory) {
  // 1. Compute weights using softmax over negative costs
  double min_cost = trajectory[trajectory_order[0]].total_return;  // Line 229

  std::vector<double> weights(num_trajectory);
  double weight_sum = 0.0;
  for (int i = 0; i < num_trajectory; i++) {
    double cost = trajectory[i].total_return;
    weights[i] = std::exp(-(cost - min_cost) / temperature_);      // Line 235
    weight_sum += weights[i];
  }

  // 2. Normalize weights to sum to 1
  for (int i = 0; i < num_trajectory; i++) {
    weights[i] /= weight_sum;                                       // Line 242
  }

  // 3. Weighted average of ALL candidate policies
  // Zero out policy
  for (const TimeSpline::Node& node : policy.plan) {
    for (int k = 0; k < model->nu; k++) {
      node.values()[k] = 0.0;                                       // Line 259
    }
  }

  // Accumulate weighted sum from ALL trajectories
  for (int i = 0; i < num_trajectory; i++) {
    if (weights[i] < 1.0e-15) continue;
    for (const TimeSpline::Node& cand_node : candidate_policy[i].plan) {
      policy_it->values()[k] += weights[i] * cand_node.values()[k]; // Line 271
    }
  }
}
```

**MPPI Weight Equation:**
```
weight[i] = exp(-(cost[i] - min_cost) / temperature) / Z
```
where Z is the normalizing constant (sum of unnormalized weights).

**Temperature effects:**
- `temperature_ → 0`: Sharp selection (converges to argmin, like base sampling)
- `temperature_ → ∞`: Uniform weights (average all equally)
- `temperature_ = 1.0` (default): Balanced cost-weighted averaging

---

### 3. Dual-Loop Annealing Schedule (planner.cc:386-440)

The noise added to candidate policies follows an **annealing schedule** that varies by:
1. **Iteration number** (trajectory annealing): less noise in later iterations
2. **Horizon position** (action annealing): more noise for distant actions

**Base Sampling** (AddNoiseToPolicy):
```cpp
double std = noise_exploration[0];  // Fixed noise level
for (each spline node) {
  node.values()[k] += Gaussian(0, scale * std);  // Same std everywhere
}
```

**Annealed Sampling** (AddNoiseToPolicy):
```cpp
void AnnealedSamplingPlanner::AddNoiseToPolicy(double start_time, int i) {
  double base_std = noise_exploration[0];

  int N = current_annealing_total_;   // Total iterations           Line 399
  int iter = current_annealing_iter_; // Current iteration          Line 400

  // Count spline points for horizon H
  int H = 0;
  for (...) H++;                                                  // Lines 403-408

  int h = 0;
  for (const TimeSpline::Node& node : candidate_policy[i].plan) {
    // TRAJECTORY ANNEALING: decreases noise over iterations
    //   iter=0 (first) → traj_anneal = 1.0 (full noise)
    //   iter=N-1 (last) → traj_anneal → 0 (minimal noise)
    double traj_anneal = (N > 1)
        ? std::exp(-(double)iter / (beta_trajectory_ * N))        // Line 417
        : 1.0;

    // ACTION ANNEALING: more noise for farther horizon
    //   h=0 (near-term) → action_anneal small (refined)
    //   h=H-1 (far-horizon) → action_anneal large (explore)
    double action_anneal = (H > 1)
        ? std::exp(-(double)(H - 1 - h) / (beta_action_ * H))     // Line 424
        : 1.0;

    // Combined effective noise
    double effective_std = base_std * traj_anneal * action_anneal; // Line 427

    for (int k = 0; k < model->nu; k++) {
      node.values()[k] += Gaussian(0, scale * effective_std);     // Line 432-433
    }
    h++;
  }
}
```

**Annealing Equations (DIAL-MPC paper eq. 7):**
```
traj_anneal   = exp(-iter / (beta_trajectory * N))
action_anneal = exp(-(H-1-h) / (beta_action * H))
effective_std = base_std * traj_anneal * action_anneal
```

**Visual intuition:**
```
Iteration 0 (full exploration):
  h=0  h=1  h=2  h=3  h=4  (horizon position)
  ███  ███  ███  ████ █████  (noise level)

Iteration N-1 (refinement):
  h=0  h=1  h=2  h=3  h=4
  ▪    ▪    █    ██   ███
```

---

## GUI Parameters (planner.cc:521-547)

New GUI sliders for DIAL-MPC parameters:

```cpp
mjuiDef defAnnealed[] = {
    {mjITEM_SLIDERINT, "Rollouts", ...},
    {mjITEM_SELECT, "Spline", ...},
    {mjITEM_SLIDERINT, "Spline Pts", ...},
    {mjITEM_SLIDERNUM, "Noise Std", ...},
    {mjITEM_SLIDERINT, "Anneal Iters", 2, &num_annealing_iters_, ...},  // NEW
    {mjITEM_SLIDERNUM, "Temperature", 2, &temperature_, ...},           // NEW
    {mjITEM_SLIDERNUM, "Beta Traj", 2, &beta_trajectory_, ...},         // NEW
    {mjITEM_SLIDERNUM, "Beta Action", 2, &beta_action_, ...},           // NEW
    {mjITEM_CHECKBYTE, "Sliding plan", ...},
};
```

---

## Summary of Code Changes

| Aspect | Base Sampling | Annealed Sampling | Location |
|--------|---------------|-------------------|----------|
| Outer loop | 1 iteration | N iterations | planner.cc:187 |
| Selection | argmin (pick best) | softmax weighted avg | planner.cc:225-286 |
| Noise level | Fixed std | Annealed (iter + horizon) | planner.cc:416-427 |
| Temperature | N/A | Configurable | planner.h:133 |
| Sorting | partial_sort (top k) | full sort (all) | planner.cc:207 |

---

## References

- DIAL-MPC paper: See `dial_mpc_paper.pdf` in this directory
- MPPI (Model Predictive Path Integral): Williams et al., 2017
