# PSO Upgrades: Performance Analysis & Improvement Paths

## Status Quo

PSO is implemented and functional as a sampling-based planner in MuJoCo MPC. However, empirical results show:

- **Requires aggressive tuning**: low social/cognitive coefficients, high velocity scale, maximum particles.
- **Slower than MPPI (Sampling planner)**: same rollout cost but no better solution quality to compensate.
- **Lower solution quality than MPPI**: standard random sampling with cost-weighted averaging consistently outperforms PSO's velocity-guided search.

This document analyzes *why* PSO underperforms, what can be done about speed, and evaluates the DIAL-MPC annealing framework as a potential upgrade path.

---

## 1. Why PSO Underperforms MPPI

### 1.1 Information Utilization

The core issue is how each method uses cost information from rollouts:

**MPPI** computes a cost-weighted average over *all* samples:
```
U+ = U + Σ * [ Σ_i exp(-J(U+W_i)/λ) * W_i ] / [ Σ_j exp(-J(U+W_j)/λ) ]
```
Every sample contributes to the update, weighted by its cost. This is equivalent to score ascent on the convolved target distribution (Proposition 1 in the DIAL-MPC paper). The update direction is an unbiased Monte Carlo estimate of the gradient of log p₁(U).

**PSO** uses only the *best* particle (global best) and each particle's *personal best*:
```
v_i = w * v_i + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
```
This discards cost information from all non-best particles. A particle that's second-best gets zero influence on the global update direction. In a high-dimensional control space with noisy cost landscapes, this is a massive waste of information.

**In short**: MPPI uses a soft ranking (exponential weighting) across all samples. PSO uses a hard ranking (only the best). The soft ranking is strictly more informative per rollout.

### 1.2 Exploration-Exploitation Balance

MPPI's exploration is controlled by a single parameter (noise std / temperature λ) with a clear theoretical interpretation: the noise level defines the convolution kernel, and the temperature controls how sharply costs are weighted.

PSO's exploration depends on the interaction of four coupled parameters (inertia, cognitive, social, velocity clamp) with no clear theoretical grounding for MPC. The velocity dynamics can either cause premature convergence (particles collapse to global best) or divergence (velocities explode). The sweet spot is narrow, task-dependent, and changes as the cost landscape evolves during receding-horizon execution.

### 1.3 Stale Memory in Receding Horizons

PSO's personal bests and global best are evaluated at *previous* states. In MPC, the state changes every timestep, so the cost landscape shifts. A particle's personal best from 5 timesteps ago may now be a poor control sequence. The warm-start shifting helps but doesn't fully account for landscape changes.

MPPI has no memory — it evaluates everything fresh from the current state. This is actually an advantage in fast-changing MPC settings.

---

## 2. Speed Improvements (Keeping PSO as-is)

### 2.1 Current Bottleneck: Rollouts

The timing profile shows that `Rollouts()` (parallel MuJoCo simulation of all particles) dominates wall-clock time. The velocity/position update is negligible by comparison. This is the same bottleneck as every other sampling-based planner in the framework.

### 2.2 Reduce Rollout Cost

**Shorter horizons with more iterations**: Use fewer spline points / shorter planning horizon to make each rollout cheaper, then iterate more within the same wall-clock budget. This trades planning depth for solution refinement.

**Cheaper dynamics**: Disable unnecessary contacts (as DIAL-MPC does — contacts only on feet), reduce solver iterations, use a coarser timestep for planning vs. execution.

**Early termination**: If a particle's partial cost already exceeds the current best, abort its rollout early. This requires restructuring the parallel rollout loop but can save significant time when many particles are poor.

### 2.3 Reduce Per-Iteration Overhead

The velocity/position update is already fast (O(particles × spline_points × actuators)), but:

- **Avoid per-element random draws**: Pre-generate all random numbers in a batch rather than calling `absl::Uniform` in a tight loop. Or use a faster RNG.
- **SIMD/vectorize the update**: The velocity update loop is embarrassingly parallelizable across particles and dimensions. Even on CPU, explicit vectorization could help.

These are minor optimizations — the rollout cost dominates by orders of magnitude.

### 2.4 Multi-Iteration PSO Between Policy Applications

Currently, `OptimizePolicy` runs one PSO iteration (evaluate → update velocities → update positions). Within the MPC planning budget, you could run multiple PSO iterations:

```
for iter in 1..K:
    EvaluateParticles()      // rollouts — expensive
    UpdatePersonalBests()
    UpdateGlobalBest()
    UpdateVelocities()       // cheap
    UpdatePositions()        // cheap
```

This is already possible by calling `OptimizePolicy` multiple times per MPC step (the agent does this). But each iteration requires a full set of rollouts. There's no way around the N-rollouts-per-iteration cost without changing the algorithm fundamentally.

---

## 3. GPU Scaling: PSO vs. MPPI

### 3.1 Would PSO Scale Better on GPU?

**No.** Both methods have the same dominant cost: N parallel forward simulations. The difference is only in the update rule applied to the results:

| Operation | MPPI | PSO |
|-----------|------|-----|
| N rollouts | O(N × H × sim_cost) | O(N × H × sim_cost) |
| Update rule | Cost-weighted mean (O(N × d)) | Velocity update (O(N × d)) |
| Memory | O(N × d) samples | O(3N × d) positions + velocities + pbests |

The rollout cost is identical and is where GPU parallelism helps. GPU physics engines (MJX, Isaac Gym, Brax) would benefit both methods equally because the bottleneck — stepping the physics simulator for each sample — is the same.

PSO has slightly *more* memory overhead (3x vs 1x per particle for positions, velocities, personal bests) which is mildly unfavorable for GPU memory bandwidth but negligible compared to the physics state.

### 3.2 What GPU Scaling Actually Enables

With GPU-based MuJoCo (MJX) or similar:
- **DIAL-MPC uses 2048 samples** on GPU at 50 Hz real-time
- **MuJoCo MPC sampling planner** uses 10-128 samples on CPU
- GPU enables ~10-100x more samples per wall-clock second

Both MPPI and PSO benefit equally from this. The question is whether PSO's update rule extracts more value from 2048+ particles than MPPI's — and the answer from the information-theoretic argument above is *no*. MPPI's cost-weighted averaging is strictly more sample-efficient.

**Conclusion**: GPU scaling is a rising tide that lifts all boats. PSO does not have a structural advantage here.

---

## 4. DIAL-MPC Annealing: Analysis & Adaptation

### 4.1 Core Insight

The DIAL-MPC paper's key contribution is connecting MPPI to single-step score-based diffusion and then applying the diffusion insight: **iterate at decreasing noise levels** to achieve both global coverage (large noise) and local convergence (small noise).

The dual-loop annealing design:

**Outer loop (trajectory-level)**: Across the N inner iterations at each MPC step, decrease the sampling variance from large (exploration) to small (exploitation):
```
det(Σ^i) ∝ exp(-(N-i)/(β₁·N) · H·d_u)    for i = N,...,1
```

**Inner loop (action-level)**: Within each iteration, use larger variance for actions farther in the horizon (less refined) and smaller variance for near-term actions (more refined):
```
det(Σ^i_{t+h}) ∝ exp(-(H-h)/(β₂·H) · d_u)    for h = 0,...,H
```

Combined:
```
Σ^i_{t+h} = exp(-(N-i)/(β₁·N) - (H-h)/(β₂·H)) · I
```

### 4.2 Can This Be Adapted to PSO?

In principle, you could apply annealing to PSO by varying the velocity clamp (or injecting additional noise) according to the DIAL-MPC schedule:

```cpp
// Annealing-inspired PSO velocity scaling
double traj_anneal = exp(-(N - iter) / (beta1 * N));
double action_anneal = exp(-(H - h) / (beta2 * H));
double effective_vel_scale = velocity_scale * traj_anneal * action_anneal;
```

This would make particles explore broadly in early iterations and for far-horizon actions, then tighten as iterations progress and for near-term actions.

**However, this misses the fundamental point.** The DIAL-MPC annealing is specifically designed around MPPI's score-ascent property (Proposition 1). MPPI's update *is* a gradient step on the log-probability of the convolved distribution. The noise level directly controls which convolved distribution you're ascending on. Decreasing noise = ascending on distributions closer to the true target.

PSO's velocity update has no such interpretation. Varying the velocity scale changes exploration range but doesn't correspond to ascending on a different smoothed objective. The theoretical guarantee of DIAL-MPC doesn't transfer to PSO.

### 4.3 The Honest Recommendation

**If the goal is to match or exceed MPPI performance with annealing, the right path is to implement DIAL-MPC directly on top of the existing Sampling planner**, not to bolt annealing onto PSO. The changes would be:

1. Add an outer loop of N iterations (currently the agent already calls `OptimizePolicy` multiple times)
2. At each iteration i, set the noise std according to the exponential schedule (eq. 7)
3. Per-action noise scaling: scale noise by horizon position h
4. Use MPPI cost-weighted averaging (already implemented in the Sampling planner)

This is a relatively contained change to `SamplingPlanner::OptimizePolicy()` — add two parameters (β₁, β₂), and modify `AddNoiseToPolicy` to accept a per-timestep noise scale.

---

## 5. Concrete Upgrade Paths (Ranked by Expected Impact)

### Path A: DIAL-MPC on Sampling Planner (High Impact, Moderate Effort)

Implement the dual-loop annealing schedule on the existing `SamplingPlanner`. This is the highest-impact change because:
- The Sampling planner already does MPPI-style cost-weighted updates
- Annealing addresses the exact coverage-vs-convergence tradeoff that limits MPPI
- DIAL-MPC showed 13.4x error reduction vs vanilla MPPI on locomotion tasks
- Requires minimal new code — mainly a noise schedule and an inner iteration loop

**Implementation sketch for `SamplingPlanner`**:

```cpp
void SamplingPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
    int N = num_annealing_steps_;  // e.g., 4
    for (int iter = 0; iter < N; iter++) {
        // Compute per-timestep noise scale with dual-loop annealing
        double traj_scale = exp(-(double)(N - 1 - iter) / (beta1_ * N));

        for (int h = 0; h < horizon; h++) {
            double action_scale = exp(-(double)(horizon - 1 - h) / (beta2_ * horizon));
            noise_per_timestep_[h] = noise_exploration_ * traj_scale * action_scale;
        }

        // Standard MPPI: sample, rollout, cost-weighted average
        AddNoiseToPolicies(noise_per_timestep_);
        Rollouts(num_trajectory_, horizon, pool);
        UpdateNominalWithMPPIWeights();  // cost-weighted mean
    }
}
```

New GUI parameters:
- `Anneal Steps` (1-8): Number of inner iterations N
- `β₁ Trajectory` (0.1-2.0): Trajectory-level annealing temperature
- `β₂ Action` (0.1-2.0): Action-level annealing temperature

### Path B: Hybrid PSO-MPPI (Medium Impact, Medium Effort)

Replace PSO's hard global-best selection with MPPI-style soft weighting:

```cpp
// Instead of: only use global best
// Do: cost-weighted average across all particles
for (int i = 0; i < num_particles; i++) {
    weights[i] = exp(-trajectory[i].total_return / temperature);
}
// Normalize weights
// Update each particle's target as weighted mean instead of global best
```

This gives PSO the information-utilization advantage of MPPI while retaining velocity-based exploration dynamics. It's a novel hybrid that might outperform either pure method, but has no theoretical backing.

### Path C: PSO with Annealing (Low Impact, Low Effort)

Add the velocity-scale annealing described in 4.2 above. Quick to implement, but unlikely to close the gap with MPPI because it doesn't fix the fundamental information-utilization problem.

### Path D: GPU-Accelerated Rollouts via MJX (High Impact, High Effort)

Port the rollout step to MuJoCo XLA (MJX) for GPU-parallel simulation. This would benefit *all* planners equally and enable 2048+ samples in real-time. This is the same approach DIAL-MPC uses (they run on Brax/JAX with GPU).

This is an infrastructure change, not PSO-specific. It would make both MPPI and PSO dramatically faster, and MPPI would still likely produce better plans per sample.

---

## 6. PSO as Meta-Optimizer for Planner Hyperparameters

PSO is a poor fit for the *control* problem (high-dimensional, non-smooth, fast-changing landscape), but it is a natural fit for *hyperparameter tuning* of any planner — including DIAL-MPC itself.

The meta-optimization problem looks like:
- **Search space**: ~4-8 scalar parameters (β₁, β₂, λ, N, noise_std, etc.)
- **Objective**: aggregate task performance (e.g., average cost over K rollouts of a full MPC episode)
- **Evaluation cost**: expensive (run a full episode per evaluation), but low-dimensional and relatively smooth
- **No gradients available**: hyperparameter → episode cost has no closed-form gradient

This is exactly where PSO excels: low-dimensional, smooth-ish, gradient-free, population-based search. Each PSO "particle" is a hyperparameter vector, and each "rollout" is a full MPC episode scored by total cost.

**Practical setup**:
```
Outer loop (PSO over hyperparameters):
    particle_i = (β₁, β₂, λ, N, noise_std, ...)

    Inner loop (MPC episode with those hyperparameters):
        Run DIAL-MPC / Sampling / any planner with particle_i's params
        Score = total episode cost (or tracking error, success rate, etc.)

    PSO update on particle positions using episode scores
```

This could run offline as a tuning sweep, or online with a warm-started population if the task distribution shifts. The key point: PSO's strengths (population diversity, no gradient needed, simple implementation) align well with the meta-problem even though they don't align well with the control problem.

This also means PSO's implementation in the codebase has lasting value beyond direct planning — it's a general-purpose optimizer that can tune any planner's parameters.

---

## 7. Summary

| Path | Expected Quality Gain | Speed Gain | Effort | Recommendation |
|------|----------------------|------------|--------|----------------|
| A: DIAL-MPC on Sampling | Large (13x per paper) | Neutral (more iters, same total rollouts) | Moderate | **Do this** |
| B: Hybrid PSO-MPPI | Medium (speculative) | Neutral | Medium | Interesting experiment |
| C: PSO + Annealing | Small | Neutral | Low | Quick test, don't expect much |
| D: GPU rollouts (MJX) | N/A (infra) | 10-100x samples | High | Long-term, benefits all planners |

**The fundamental issue is not PSO's speed — it's PSO's sample efficiency.** Both PSO and MPPI are bottlenecked by rollout cost. MPPI extracts more information per rollout via cost-weighted averaging. DIAL-MPC further improves MPPI by adding principled noise annealing with theoretical grounding in diffusion processes.

PSO's velocity-based dynamics are an interesting exploration mechanism, but they don't provide a sample-efficiency advantage over MPPI's cost-weighted updates, and there's no theoretical framework (analogous to the MPPI-diffusion connection) that would give PSO a principled annealing schedule.

The most impactful next step is **Path A**: implement DIAL-MPC's dual-loop annealing on the Sampling planner, which already has the right MPPI update rule. This would be a new planner variant (e.g., `DIALPlanner` or an annealing mode on `SamplingPlanner`).

---

## References

1. Xue, Pan, Yi, Qu, Shi. "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing." (DIAL-MPC paper, provided as `dial_mpc_paper.pdf`)
2. Pan, Yi, Shi, Qu. "Model-Based Diffusion for Trajectory Optimization." arXiv:2407.01573, 2024.
3. Williams, Drews, Goldfain, Rehg, Theodorou. "Aggressive driving with model predictive path integral control." ICRA 2016.
4. Howell, Gileadi, Tunyasuvunakool, Zakka, Erez, Tassa. "Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo." arXiv:2212.00541, 2022.
5. Kennedy, Eberhart. "Particle swarm optimization." ICNN'95, 1995.
