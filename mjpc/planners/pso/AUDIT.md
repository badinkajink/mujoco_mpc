# PSO Planner - Self Audit

## Executive Summary

The PSO (Particle Swarm Optimization) Planner replaces random noise-based exploration with a **population-based swarm intelligence** approach. Instead of randomly perturbing a nominal policy, PSO maintains a swarm of "particles" that move through the policy space guided by:

1. **Personal best** - each particle remembers its own best-found solution
2. **Global best** - particles are attracted toward the swarm's overall best solution
3. **Inertia** - particles maintain momentum from previous velocities

---

## Algorithm Overview

### Base Sampling
```
For each optimization step:
  1. Copy nominal policy → candidates
  2. Add RANDOM noise to candidates
  3. Rollout all, compute costs
  4. Pick best → new nominal
```

### PSO
```
For each optimization step:
  1. Initialize particles around nominal
  2. Rollout all particles, compute costs
  3. Update personal bests (per particle)
  4. Update global best (across swarm)
  5. Update velocities using PSO equation
  6. Update positions by adding velocities
  7. Copy best particle → nominal policy
```

---

## New Member Variables (planner.h)

### Particle State (Lines 117-125)
```cpp
// particles - positions are stored in candidate_policy
SamplingPolicy candidate_policy[kMaxTrajectory];  // current positions    Line 118
std::vector<std::vector<double>> velocities;      // particle velocities  Line 119
SamplingPolicy personal_best[kMaxTrajectory];     // personal best pos    Line 120
std::vector<double> personal_best_costs;          // personal best costs  Line 121

// global best
int global_best_index;                            // which particle       Line 124
double global_best_cost;                          // best cost found      Line 125
```

### PSO Hyperparameters (Lines 134-138)
```cpp
double inertia_weight;    // w: momentum term (0.4-0.9)              Line 135
double cognitive_coeff;   // c1: personal best attraction (~1.5-2.0) Line 136
double social_coeff;      // c2: global best attraction (~1.5-2.0)   Line 137
double velocity_scale;    // max velocity as fraction of ctrl range  Line 138
```

### Initialization (planner.cc:57-62)
```cpp
num_particles_ = GetNumberOrDefault(20, model, "pso_num_particles");
inertia_weight = GetNumberOrDefault(0.7, model, "pso_inertia");
cognitive_coeff = GetNumberOrDefault(1.5, model, "pso_cognitive");
social_coeff = GetNumberOrDefault(1.5, model, "pso_social");
velocity_scale = GetNumberOrDefault(0.1, model, "pso_velocity_scale");
```

---

## Key Algorithmic Changes

### 1. Particle Initialization (planner.cc:189-230)

**Base Sampling:** Add random noise to copies of nominal policy.

**PSO:** Initialize particles with positions around nominal + small random velocities:

```cpp
void PSOPlanner::InitializeParticles(int horizon) {
  absl::BitGen gen;

  for (int i = 0; i < num_particles_; i++) {
    for (int t = 0; t < num_spline_points; t++) {
      // Start with nominal parameters
      std::vector<double> values(model->nu);
      for (int k = 0; k < model->nu; k++) {
        values[k] = parameters_scratch[t * model->nu + k];           // Line 202
      }

      // Add random perturbation (except particle 0 = nominal)
      if (i > 0) {
        for (int k = 0; k < model->nu; k++) {
          double range = model->actuator_ctrlrange[2*k+1] - ...;
          double noise = absl::Gaussian<double>(gen, 0.0, 0.1 * range); // Line 210
          values[k] += noise;
        }
      }

      candidate_policy[i].plan.AddNode(times_scratch[t], values);
    }

    // Initialize velocity to small random values
    for (int t = 0; t < num_spline_points; t++) {
      for (int k = 0; k < model->nu; k++) {
        velocities[i][t * model->nu + k] =
            absl::Gaussian<double>(gen, 0.0, 0.01 * range);           // Line 225-226
      }
    }
  }
}
```

---

### 2. PSO Velocity Update (planner.cc:350-391)

This is the **core PSO equation**. Each particle's velocity is updated based on three components:

```cpp
void PSOPlanner::UpdateVelocities() {
  absl::BitGen gen;

  for (int i = 0; i < num_particles_; i++) {
    for (int t = 0; t < num_spline_points; t++) {
      // Get positions
      auto pos_node = candidate_policy[i].plan.NodeAt(t);
      auto pbest_node = personal_best[i].plan.NodeAt(t);
      auto gbest_node = candidate_policy[global_best_index].plan.NodeAt(t);

      for (int k = 0; k < model->nu; k++) {
        // Random coefficients for stochasticity
        double r1 = absl::Uniform(gen, 0.0, 1.0);                    // Line 367
        double r2 = absl::Uniform(gen, 0.0, 1.0);                    // Line 368

        double pos = pos_node.values()[k];
        double pbest = pbest_node.values()[k];
        double gbest = gbest_node.values()[k];

        // PSO VELOCITY UPDATE EQUATION
        double cognitive = cognitive_coeff * r1 * (pbest - pos);     // Line 375
        double social = social_coeff * r2 * (gbest - pos);           // Line 376
        double new_vel =
            inertia_weight * velocities[i][t * model->nu + k] +      // Line 377-379
            cognitive + social;

        // Clamp velocity to prevent explosion
        double max_vel = velocity_scale * range;
        new_vel = mju_clip(new_vel, -max_vel, max_vel);              // Line 385

        velocities[i][t * model->nu + k] = new_vel;
      }
    }
  }
}
```

**The PSO Velocity Equation:**
```
v_new = w * v_old + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        ├────────┤   ├────────────────────┤   ├────────────────────┤
         Inertia      Cognitive component      Social component
         (momentum)   (personal memory)        (swarm knowledge)
```

**Components explained:**
- **Inertia (w * v_old):** Particle maintains momentum. `w=0.7` means 70% of previous velocity carries over.
- **Cognitive (c1 * r1 * (pbest - x)):** Attracted toward personal best. Like "memory" of good locations.
- **Social (c2 * r2 * (gbest - x)):** Attracted toward global best. Swarm intelligence.
- **r1, r2:** Random in [0,1] for stochastic exploration.

---

### 3. Position Update (planner.cc:394-409)

Simple Euler integration: add velocity to position.

```cpp
void PSOPlanner::UpdatePositions() {
  for (int i = 0; i < num_particles_; i++) {
    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::Node pos_node = candidate_policy[i].plan.NodeAt(t);

      for (int k = 0; k < model->nu; k++) {
        pos_node.values()[k] += velocities[i][t * model->nu + k];    // Line 402
      }

      // Clamp to control limits
      Clamp(pos_node.values().data(), model->actuator_ctrlrange, ...); // Line 406
    }
  }
}
```

**Position Update Equation:**
```
x_new = x_old + v_new
```

---

### 4. Personal Best Tracking (planner.cc:412-420)

Each particle remembers the best position it has personally visited.

```cpp
void PSOPlanner::UpdatePersonalBests(int num_particles) {
  for (int i = 0; i < num_particles; i++) {
    if (trajectory[i].total_return < personal_best_costs[i]) {       // Line 414
      personal_best_costs[i] = trajectory[i].total_return;
      personal_best[i].CopyFrom(candidate_policy[i], ...);           // Line 416-417
    }
  }
}
```

---

### 5. Global Best Tracking (planner.cc:423-430)

Track the best solution found by ANY particle.

```cpp
void PSOPlanner::UpdateGlobalBest(int num_particles) {
  for (int i = 0; i < num_particles; i++) {
    if (personal_best_costs[i] < global_best_cost) {                 // Line 425
      global_best_cost = personal_best_costs[i];
      global_best_index = i;
    }
  }
}
```

---

### 6. Optimization Flow (planner.cc:232-277)

Complete optimization step:

```cpp
int PSOPlanner::OptimizePolicyCandidates(...) {
  // 1. Resample nominal policy to current time
  ResamplePolicy(horizon);                                           // Line 244

  // 2. Initialize particles around nominal
  InitializeParticles(horizon);                                      // Line 248

  // 3. Rollout all particles
  Rollouts(num_particles, horizon, pool);                            // Line 252

  // 4. Update personal bests
  UpdatePersonalBests(num_particles);                                // Line 256

  // 5. Update global best
  UpdateGlobalBest(num_particles);                                   // Line 257

  // 6. Update velocities (PSO equation)
  UpdateVelocities();                                                // Line 261

  // 7. Update positions
  UpdatePositions();                                                 // Line 262

  // 8. Sort for ranking
  std::partial_sort(...);                                            // Line 269-274
}
```

---

### 7. Rollouts Without Noise (planner.cc:322-347)

**Base Sampling:** Adds noise during rollout (`if (i != 0) AddNoiseToPolicy(...)`).

**PSO:** No noise addition - particles already have PSO-determined positions:

```cpp
void PSOPlanner::Rollouts(int num_particles, int horizon, ThreadPool& pool) {
  for (int i = 0; i < num_particles; i++) {
    pool.Schedule([...]() {
      auto particle_policy = [&](double* action, ...) {
        candidate_policy[i].Action(action, state, time);
      };
      // Note: NO AddNoiseToPolicy call here
      s.trajectory[i].Rollout(particle_policy, ...);                 // Line 338
    });
  }
}
```

---

## GUI Parameters (planner.cc:484-504)

```cpp
mjuiDef defPSO[] = {
    {mjITEM_SLIDERINT, "Particles", 2, &num_particles_, "0 1"},
    {mjITEM_SELECT, "Spline", 2, &interpolation_, "Zero\nLinear\nCubic"},
    {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
    {mjITEM_SLIDERNUM, "Inertia", 2, &inertia_weight, "0.1 0.99"},     // PSO
    {mjITEM_SLIDERNUM, "Cognitive", 2, &cognitive_coeff, "0.0 4.0"},   // PSO
    {mjITEM_SLIDERNUM, "Social", 2, &social_coeff, "0.0 4.0"},         // PSO
    {mjITEM_SLIDERNUM, "Vel Scale", 2, &velocity_scale, "0.01 0.5"},   // PSO
};
```

---

## Summary of Code Changes

| Aspect | Base Sampling | PSO | Location |
|--------|---------------|-----|----------|
| Exploration | Random Gaussian noise | Velocity-based movement | planner.cc:350-409 |
| Memory | None | Personal best per particle | planner.cc:412-420 |
| Coordination | None | Global best attraction | planner.cc:423-430 |
| Noise | `AddNoiseToPolicy()` | Removed (uses velocities) | - |
| Parameters | `noise_exploration` | `inertia, cognitive, social` | planner.h:135-138 |
| Best tracking | `winner` index | `global_best_index/cost` | planner.h:124-125 |

---

## Visual Intuition

```
Iteration 1: Particles scattered, exploring
    P1 ──→        P3
         ↘      ↗
    P2 ──→  G*  ←── P4     (G* = global best)
              ↖
        P5 ──→

Iteration N: Particles converging toward global best
         P1  P3
           ↘↗
    P2 →  G* ← P4
           ↑
          P5

Velocity update each step:
  - Particles remember their personal best (cognitive)
  - Particles attracted to global best (social)
  - Particles maintain momentum (inertia)
```

---

## Parameter Tuning Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `inertia_weight` | 0.4-0.9 | Higher = more exploration, slower convergence |
| `cognitive_coeff` | 1.0-2.5 | Higher = stronger personal memory |
| `social_coeff` | 1.0-2.5 | Higher = faster convergence to global best |
| `velocity_scale` | 0.01-0.5 | Limits maximum step size |

**Recommended starting point:** `w=0.7, c1=1.5, c2=1.5`

---

## Comparison: PSO vs Base Sampling

| Property | Base Sampling | PSO |
|----------|---------------|-----|
| Search strategy | Random restart | Guided evolution |
| Information use | Discards all but best | Uses swarm history |
| Convergence | May be slow (random) | Faster (directed) |
| Exploration | Good (random) | Depends on parameters |
| Complexity | O(N) per step | O(N) per step + state |
| Memory | O(N) policies | O(N) policies + velocities + bests |

---

## References

- Kennedy, J.; Eberhart, R. (1995). "Particle Swarm Optimization"
- Shi, Y.; Eberhart, R. (1998). "A Modified Particle Swarm Optimizer"
