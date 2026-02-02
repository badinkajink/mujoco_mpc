# IDTO Planner Implementation Plan

**Target**: Port Inverse Dynamics Trajectory Optimization (IDTO) as a planner in `mjpc/planners/idto/`

**Timeline**: 2-3 weeks
**Status**: Planning Phase
**Branch**: `idto`

---

## Executive Summary

Implement IDTO as a native mjpc planner using MuJoCo's inverse dynamics and convex contact model. This leverages:
- MuJoCo's `mj_inverse` with `mjENBL_INVDISCRETE` for inverse dynamics
- Existing `mjpc/direct` infrastructure for configuration-based optimization
- IDTO's trust-region SQP solver for robust convergence
- MuJoCo's analytically-invertible contact formulation

**Key Insight**: mjpc/direct already does inverse dynamics trajectory optimization for estimation. IDTO adapts this for planning/MPC with a more sophisticated solver.

---

## Architecture Overview

```
mjpc/planners/idto/
├── planner.h              # Main planner class (implements Planner interface)
├── planner.cc             # Planner implementation
├── policy.h               # Policy representation (time-indexed + feedback)
├── policy.cc              # Policy implementation
├── settings.h             # Solver parameters and settings
├── trajectory_optimizer.h # Core IDTO optimizer
├── trajectory_optimizer.cc
├── penta_diagonal.h       # Sparse Hessian structure
├── penta_diagonal.cc
└── solver_utils.h         # Helper functions (warm start, convergence, etc.)
```

---

## Phase 1: Foundation (Days 1-3)

### 1.1 Directory Setup & Build Integration ✅
- [x] Create `mjpc/planners/idto/` directory
- [ ] Add to `mjpc/CMakeLists.txt` build system
- [ ] Create initial header files with placeholders
- [ ] Verify compilation

### 1.2 Planner Interface Skeleton
**File**: `mjpc/planners/idto/planner.h`

Implement base `Planner` interface:
```cpp
class IDTOPlanner : public Planner {
 public:
  void Initialize(mjModel* model, const Task& task) override;
  void Allocate() override;
  void Reset(int horizon, const double* initial_repeated_action) override;
  void SetState(const State& state) override;
  void OptimizePolicy(int horizon, ThreadPool& pool) override;
  void NominalTrajectory(int horizon, ThreadPool& pool) override;
  void ActionFromPolicy(double* action, const double* state,
                       double time, bool use_previous) override;
  const Trajectory* BestTrajectory() override;
  void Traces(mjvScene* scn) override;
  void GUI(mjUI& ui) override;
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
            int planner_shift, int timer_shift,
            int planning, int* shift) override;
  int NumParameters() override;
};
```

**References**:
- `mjpc/planners/ilqg/planner.h` for interface patterns
- `mjpc/planners/planner.h` for base interface

---

## Phase 2: Core Optimization Infrastructure (Days 4-7)

### 2.1 Data Structures

**File**: `mjpc/planners/idto/settings.h`
```cpp
struct IDTOSettings {
  int max_iterations = 100;
  double gradient_tolerance = 1e-6;
  double cost_tolerance = 1e-6;
  double trust_region_radius_initial = 1.0;
  double trust_region_radius_min = 1e-6;
  double trust_region_radius_max = 1e3;
  double regularization = 1e-6;
  int num_threads = 1;
};
```

**File**: `mjpc/planners/idto/penta_diagonal.h`
Port from `idto/optimizer/penta_diagonal_matrix.h`:
- Penta-diagonal sparse matrix structure
- Efficient storage for (T+1) x (T+1) blocks of size nq x nq
- Matrix-vector multiplication
- Interface for solver

**Key Difference**: Use Eigen (consistent with mjpc) instead of Drake's types

### 2.2 Trajectory State Management

**File**: `mjpc/planners/idto/trajectory_optimizer.h`
```cpp
class TrajectoryOptimizerState {
 public:
  // Decision variables
  std::vector<double> q;  // Configurations [nq * (T+1)]

  // Cached quantities (computed from q)
  std::vector<double> v;    // Velocities [nv * (T+1)]
  std::vector<double> a;    // Accelerations [nv * T]
  std::vector<double> tau;  // Generalized forces [nv * T]

  // Derivatives (computed via finite differences)
  std::vector<double> dv_dq;   // Velocity partials
  std::vector<double> dtau_dq; // Inverse dynamics partials

  // Cost and derivatives
  double cost;
  std::vector<double> gradient;  // [nq * (T+1)]
  PentaDiagonalMatrix hessian;   // Sparse structure
};
```

**References**:
- `idto/optimizer/trajectory_optimizer_state.h`
- `mjpc/direct/direct.h` for MuJoCo-specific patterns

---

## Phase 3: Inverse Dynamics Evaluation (Days 8-10)

### 3.1 Configuration to Forces Pipeline

**Leverage from mjpc/direct**:
```cpp
// In mjpc/direct/direct.cc line 58:
this->model->opt.enableflags |= mjENBL_INVDISCRETE;
```

**Implementation**:
1. Configure MuJoCo model with discrete inverse dynamics
2. Compute velocities via finite differences: `v[t] = (q[t] - q[t-1]) / dt`
3. Compute accelerations: `a[t] = (v[t+1] - v[t]) / dt`
4. Call `mj_inverse` to get forces: `tau[t] = ID(q[t+1], v[t+1], a[t])`

**File**: `mjpc/planners/idto/trajectory_optimizer.cc`
```cpp
void TrajectoryOptimizer::EvaluateTrajectory(
    TrajectoryOptimizerState& state) {
  // Set model state and call mj_inverse for each timestep
  for (int t = 0; t < horizon_; t++) {
    // Set q[t+1], v[t+1], a[t]
    mju_copy(data_->qpos, &state.q[(t+1) * nq_], nq_);
    mju_copy(data_->qvel, &state.v[(t+1) * nv_], nv_);
    mju_copy(data_->qacc, &state.a[t * nv_], nv_);

    // Compute inverse dynamics
    mj_inverse(model_, data_);

    // Extract forces
    mju_copy(&state.tau[t * nv_], data_->qfrc_inverse, nv_);
  }
}
```

### 3.2 Derivatives Computation

Use finite differences (similar to IDTO approach):

**Velocity Jacobian**: `∂v[t]/∂q[t] = I/dt`, `∂v[t]/∂q[t-1] = -I/dt`

**Inverse Dynamics Jacobian**: Finite difference around `mj_inverse`
```cpp
void ComputeInverseDynamicsJacobian(
    const mjModel* model, mjData* data,
    const double* q, const double* v, const double* a,
    double* dtau_dq, double eps = 1e-6);
```

**Reference**: `idto/optimizer/inverse_dynamics_partials.h`

---

## Phase 4: Cost Function Integration (Days 11-12)

### 4.1 Task to Quadratic Cost Adapter

IDTO expects quadratic costs:
```
L(q) = Σ[(q[t] - q_nom[t])' Q (q[t] - q_nom[t])
       + (v[t] - v_nom[t])' Qv (v[t] - v_nom[t])
       + tau[t]' R tau[t]]
     + (q[T] - q_nom[T])' Qf (q[T] - q_nom[T])
```

mjpc uses residual-based costs via `Task::Residual`.

**Approach**:
1. Evaluate residual at each timestep
2. Approximate with quadratic (Gauss-Newton):
   - `Q ≈ J_r' W J_r` where `J_r = ∂r/∂q`
   - Can use existing `mjpc/planners/cost_derivatives.h`

**File**: `mjpc/planners/idto/cost_adapter.h`
```cpp
void ComputeQuadraticCost(
    const Task& task,
    const TrajectoryOptimizerState& state,
    double* cost,
    double* gradient,  // [nq * (T+1)]
    PentaDiagonalMatrix* hessian);
```

### 4.2 Cost Derivatives

**Option 1** (simpler): Gauss-Newton approximation
- `H ≈ J' J` (always PSD, no second-order terms)
- Similar to existing mjpc planners

**Option 2** (higher fidelity): Include control cost explicitly
- Compute `∂tau/∂q` via inverse dynamics derivatives
- Add `R ∂tau/∂q' ∂tau/∂q` to Hessian

Start with Option 1, upgrade to Option 2 if needed.

---

## Phase 5: Trust-Region SQP Solver (Days 13-15)

### 5.1 Penta-Diagonal Solver

Port from `idto/optimizer/penta_diagonal_solver.h`:
- LDL' factorization for block penta-diagonal matrices
- Forward/backward substitution
- Regularization handling

**Note**: This is the performance-critical component. Structure:
```
H = [ D0  U0  V0                    ]
    [ L0  D1  U1  V1                ]
    [     L1  D2  U2  V2            ]
    [         L2  D3  U3  V3        ]
    ...
```
where each block is `nq x nq`.

### 5.2 Trust-Region Method

**File**: `mjpc/planners/idto/trajectory_optimizer.cc`

Main optimization loop:
```cpp
SolverStatus Solve(TrajectoryOptimizerState& state) {
  for (int iter = 0; iter < settings_.max_iterations; iter++) {
    // 1. Evaluate cost, gradient, Hessian
    EvaluateTrajectory(state);
    ComputeCost(state);
    ComputeGradient(state);
    ComputeHessian(state);

    // 2. Solve trust-region subproblem
    //    min_dq: g'*dq + 0.5*dq'*H*dq
    //    s.t. ||dq|| <= radius
    SolveTrustRegionSubproblem(state, &dq);

    // 3. Evaluate improvement
    double cost_new = EvaluateCostAt(state.q + dq);
    double actual_reduction = state.cost - cost_new;
    double predicted_reduction = -g'*dq - 0.5*dq'*H*dq;
    double ratio = actual_reduction / predicted_reduction;

    // 4. Update trust region and accept/reject step
    if (ratio > 0.25) {
      // Accept step
      state.q += dq;
      if (ratio > 0.75) radius *= 2.0;  // Expand
    } else {
      radius *= 0.5;  // Shrink
    }

    // 5. Check convergence
    if (||gradient|| < tol) return kConverged;
  }
  return kMaxIterations;
}
```

**Reference**: `idto/optimizer/trajectory_optimizer.cc` lines 191-250

---

## Phase 6: Policy and Action Generation (Days 16-17)

### 6.1 Policy Representation

**File**: `mjpc/planners/idto/policy.h`

IDTO produces open-loop trajectories. For real-time control:
```cpp
class IDTOPolicy {
 public:
  // Nominal trajectory
  std::vector<double> q_nominal;  // [nq * (T+1)]
  std::vector<double> v_nominal;  // [nv * (T+1)]
  std::vector<double> tau_nominal;  // [nv * T]

  // Times
  std::vector<double> times;  // [T+1]

  // Interpolate to get action at given time
  void GetAction(double* action, double time) const;

  // Optional: simple feedback gains (for robustness)
  // K[t] such that tau = tau_nom + K*(q - q_nom)
  std::vector<double> feedback_gains;  // [nu * nq * T]
};
```

### 6.2 Action from Policy

**Implementation**: Linear interpolation of nominal forces
```cpp
void IDTOPlanner::ActionFromPolicy(
    double* action, const double* state, double time, bool use_previous) {
  // Find timestep
  int t = std::lower_bound(policy_.times.begin(),
                          policy_.times.end(), time)
          - policy_.times.begin();

  // Interpolate forces
  if (t >= policy_.tau_nominal.size() / nv_) {
    // Past horizon, use last action
    mju_copy(action, &policy_.tau_nominal[(horizon_-1) * nv_], nv_);
  } else {
    // Linear interpolation
    double alpha = (time - policy_.times[t]) / dt_;
    mju_scl(action, &policy_.tau_nominal[t * nv_], 1-alpha, nv_);
    mju_addToScl(action, &policy_.tau_nominal[(t+1) * nv_], alpha, nv_);
  }
}
```

---

## Phase 7: Integration with mjpc Agent Loop (Days 18-19)

### 7.1 Warm Starting

Critical for MPC performance. On each replanning:
```cpp
void WarmStart(const State& current_state) {
  // Shift trajectory forward by one timestep
  for (int t = 0; t < horizon_; t++) {
    mju_copy(&state_.q[t * nq_], &state_.q[(t+1) * nq_], nq_);
  }

  // Last timestep: duplicate or extrapolate
  mju_copy(&state_.q[horizon_ * nq_],
           &state_.q[(horizon_-1) * nq_], nq_);

  // Set initial state
  mju_copy(&state_.q[0], current_state.state().data(), nq_);
}
```

### 7.2 Real-Time Iteration Scheme

For fast MPC (similar to IDTO paper):
```cpp
void OptimizePolicy(int horizon, ThreadPool& pool) override {
  // Do only 1-2 iterations per MPC step
  // (Rely on warm start for good initial guess)

  settings_.max_iterations = 2;  // Real-time iteration

  Solve(state_);

  // Update policy
  UpdatePolicy();
}
```

---

## Phase 8: Visualization & GUI (Days 20-21)

### 8.1 Traces

**File**: `mjpc/planners/idto/planner.cc`
```cpp
void IDTOPlanner::Traces(mjvScene* scn) override {
  // Draw nominal trajectory
  for (int t = 0; t < horizon_; t++) {
    // Get position at t
    mj_resetDataKeyframe(model_, data_[0].get(), 0);
    mju_copy(data_[0]->qpos, &state_.q[t * nq_], nq_);
    mj_forward(model_, data_[0].get());

    // Draw COM or end-effector trace
    // (similar to ilqg/planner.cc Traces implementation)
  }
}
```

### 8.2 GUI Elements

Add settings controls:
```cpp
void IDTOPlanner::GUI(mjUI& ui) override {
  mjuiDef defIDTO[] = {
    {mjITEM_SECTION, "IDTO", 1, nullptr, "AP"},
    {mjITEM_SLIDERINT, "Max iterations", 2, &settings_.max_iterations,
     "1 200"},
    {mjITEM_SLIDERDOUBLE, "Gradient tol", 2, &settings_.gradient_tolerance,
     "1e-8 1e-2"},
    {mjITEM_SLIDERDOUBLE, "Trust region", 2,
     &settings_.trust_region_radius_initial, "1e-3 10"},
    {mjITEM_END}
  };
  mjui_add(&ui, defIDTO);
}
```

---

## Code Reuse Strategy

### From `mjpc/direct/`:
- Model initialization with `mjENBL_INVDISCRETE` ✅
- Configuration trajectory evaluation patterns
- Jacobian block allocation/management
- Band matrix factorization infrastructure

### From `idto/optimizer/`:
- Penta-diagonal matrix operations
- Trust-region logic
- Warm start strategies
- Convergence criteria

### From `mjpc/planners/ilqg/`:
- Planner interface implementation patterns
- GUI setup
- Trajectory visualization
- Policy structure

---

## Testing Strategy

### Unit Tests
1. **Penta-diagonal solver** (Day 14)
   - Test against dense solver on small problems
   - Verify factorization correctness

2. **Inverse dynamics** (Day 10)
   - Compare `mj_inverse` output with `mj_forward` + `mj_step`
   - Test gradient accuracy via finite differences

3. **Cost computation** (Day 12)
   - Verify gradient via finite differences
   - Check Hessian symmetry

### Integration Tests (Day 22)
1. **Cartpole swing-up**: Simple, well-understood dynamics
2. **Particle task**: Minimal DoF, good for debugging
3. **Quadruped locomotion**: Contact-rich, performance test

### Benchmark Comparison (Day 23)
Compare IDTO vs iLQG on:
- Solve time per iteration
- Total cost achieved
- Robustness to poor initialization
- Contact-rich tasks (humanoid, manipulation)

---

## Risk Mitigation

### High Risk Items:
1. **Penta-diagonal solver bugs**: Port carefully, test extensively
   - Mitigation: Use reference implementation for validation

2. **Inverse dynamics derivatives**: MuJoCo's finite-diff accuracy
   - Mitigation: Tune perturbation size, compare with mjd derivatives

3. **Trust-region tuning**: Parameter sensitivity
   - Mitigation: Start with IDTO paper defaults, add adaptive logic

### Medium Risk:
1. **Task cost adapter**: Approximation quality
   - Mitigation: Start simple (Gauss-Newton), profile performance

2. **Real-time performance**: Iteration time budget
   - Mitigation: Profile, parallelize Jacobian computation

---

## Success Criteria

### Minimum Viable Product (End of Week 2):
- [x] Branch created
- [ ] Compiles without errors
- [ ] Solves cartpole swing-up
- [ ] Basic GUI integration
- [ ] Matches iLQG cost within 10%

### Full Success (End of Week 3):
- [ ] Solves all mjpc benchmark tasks
- [ ] Outperforms iLQG on ≥3 contact-rich tasks
- [ ] Real-time capable (>20 Hz on humanoid)
- [ ] Documentation complete
- [ ] Unit tests pass

---

## Open Questions

1. **Actuator limits**: IDTO optimizes forces directly. How to handle:
   - `tau_min < tau < tau_max` constraints?
   - Gear ratios, actuator dynamics?
   - **Answer**: Box constraints in trust-region QP (similar to iLQG boxQP)

2. **Feedback policy**: IDTO is open-loop. Add feedback?
   - Option A: Pure open-loop (simplest)
   - Option B: Simple PD around trajectory
   - Option C: Compute Riccati gains (expensive)
   - **Decision**: Start with A, evaluate need for B

3. **Unactuated DoFs**: IDTO handles via equality constraints
   - MuJoCo automatically computes constraint forces
   - Should be transparent, but verify on floating-base systems

---

## Next Actions

**Immediate** (Today):
1. Create directory structure
2. Set up build system
3. Create header skeletons
4. Verify compilation

**Tomorrow**:
1. Implement settings.h
2. Start planner.h/cc skeleton
3. Test basic initialization

---

## References

### Papers:
- [IDTO Paper](https://arxiv.org/abs/2309.01813)
- [MuJoCo Contact Model](https://ieeexplore.ieee.org/document/6907751/)
- [Smooth Contact](https://ieeexplore.ieee.org/document/5979814/)

### Code:
- `idto/optimizer/` - Reference implementation
- `mjpc/direct/` - MuJoCo inverse dynamics patterns
- `mjpc/planners/ilqg/` - Planner interface example

### Documentation:
- [mjpc Planners](../mjpc/planners/planner.h)
- [Direct Optimizer](DIRECT.md)
- [MuJoCo Inverse Dynamics](https://mujoco.readthedocs.io/en/stable/computation/index.html)

---

**Last Updated**: 2026-02-01
**Author**: Claude (with human guidance)
**Status**: Ready to begin implementation
