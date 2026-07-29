# AgileRRT in MJPC — feasibility teardown

Zero-control linearization-based steering (Caldwell & Correll, ISRR 2015) as a
candidate MJPC planner, with the triple-pendulum-cartpole corridor task it was
designed for.

- Paper: [`caldwell_isrr2015.pdf`](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/caldwell_isrr2015.pdf)
  — *Fast Sample-Based Planning for Dynamic Systems by Zero-Control
  Linearization-Based Steering*
- Reference implementation: [timocaldwell/agile_rrt](https://github.com/timocaldwell/agile_rrt)
  at commit `d0b8fe4` (~4.5 kLOC, GSL + Boost.odeint + Eigen). **Every line
  number in this document was verified against that commit.** It is third-party
  code with its own history, so it is not vendored into this repo; to follow the
  anchors locally:
  ```bash
  git clone https://github.com/timocaldwell/agile_rrt \
      mjpc/tasks/triple_pendulum_cartpole/agile_rrt   # gitignored
  git -C mjpc/tasks/triple_pendulum_cartpole/agile_rrt checkout d0b8fe4
  ```
- Task ported to MJPC: [`tasks/triple_pendulum_cartpole/task.xml`](../../tasks/triple_pendulum_cartpole/task.xml)
- Benchmark: [`benchmark/corridor_benchmark.cc`](../../tasks/triple_pendulum_cartpole/benchmark/corridor_benchmark.cc),
  renderer: [`benchmark/filmstrip.py`](../../tasks/triple_pendulum_cartpole/benchmark/filmstrip.py)

---

## Verdict

**The steering primitive ports cleanly and cheaply. The tree search does not fit
MJPC's planner contract, and should not be forced into it.**

1. **Porting the math is easy, and most of the reference code evaporates.**
   `mjd_transitionFD` replaces all 2296 lines of Mathematica-exported Jacobians
   in [`pendcart_3link.h`](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/pendcart_3link.h);
   `mj_step` replaces Boost.odeint; MuJoCo contacts replace the hand-rolled
   swept line-circle checks. Every continuous-time ODE in the paper becomes a
   discrete recursion (table in [The algorithm](#the-algorithm)). No new
   dependency: `mju_*` covers the linear algebra, so Eigen is not needed.

2. **The inner loop is fast enough; the tree is not.** Measured (NumPy
   prototype, `t_h = 1.0 s`, 200 steps): **7.8 µs** per nearest-neighbour test,
   **8.1 ms** per-vertex precompute, **3.0 ms** per extend. A 1000-vertex tree
   costs **≈15 s** — which independently reproduces the paper's own Fig. 3
   (~10 s at `t_h = 1.0 s`). MJPC's budget is one `OptimizePolicy` call per
   control step, ~10 ms. **A tree build is 10³× over a single MPC step's
   budget.** AgileRRT cannot be a receding-horizon `Planner`.

3. **The paper's headline contribution is a no-op at its own benchmark's start
   state.** Linearizing about `x_zero(t)` instead of the point `x0` only differs
   if `x_zero` drifts. The benchmark starts at the upright equilibrium, which
   *is* an equilibrium of the zero-control dynamics — measured drift over 1.0 s
   is **0.000**. The two linearizations are bit-identical there, which explains
   why the paper's own Fig. 3 shows near-identical vertex counts, and why its
   Discussion (§6, p.15) concludes single-state linearization is "more
   desirable" — a conclusion that contradicts the paper's title.

4. **From non-equilibrium vertices the contribution is real, but only at long
   horizons.** Terminal steering error after projection, median over 40 random
   vertex/target pairs:

   | `t_h` | about `x_zero(t)` | about `x0` | control saturated (`x_zero` / `x0`) |
   |---|---|---|---|
   | 0.10 s | 5.33 | **3.29** | 82% / 5% |
   | 0.25 s | 10.97 | **7.61** | 92% / 20% |
   | 0.50 s | **10.59** | 17.34 | 37% / 30% |
   | 0.75 s | **14.08** | 20.07 | 24% / 85% |
   | 1.00 s | **9.36** | 26.43 | 23% / 100% |

   Crossover at `t_h ≈ 0.4 s`; 2.8× better at 1.0 s, where the point
   linearization saturates the actuator on *100%* of attempts. So the idea earns
   its keep exactly in the long-horizon regime the paper argues for — the
   paper simply measured the wrong quantity (insertion failures, not steering
   error) and drew the wrong conclusion from it.

5. **The closed-loop reformulation (§3.2) is load-bearing, and the paper
   understates it.** Measured in MJPC's discretization at `t_h = 1.0 s`:

   | | paper (continuous) | measured (MuJoCo, dt=0.005) |
   |---|---|---|
   | `κ(W_0)` open loop | 3.33e15 | **1.52e17** |
   | `κ(W_K)` closed loop | 1.03e5 | 4.74e7 |
   | `κ(S_K)` | 2.57e6 | 6.06e7 |
   | `κ(W_K P1 W_K + S_K)` — actually inverted | — | 5.49e7 |

   `κ(W_0) = 1.5e17` exceeds double precision (2.2e-16), so Algorithm 1 is
   unusable, exactly as the paper argues. The closed-loop form is invertible but
   still burns ~8 of 16 digits: **use a Cholesky solve, never an explicit
   inverse.** The reference code calls `.inverse()`
   ([treevertex.cpp:139](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L139)) —
   see [Deviations](#deviations-from-the-paper).

6. **There is a real gap, but it is not the one it first appears to be.**
   Across 24 runs of four MJPC planners, **0 ever reach the goal state** — cart
   within 0.3 m of `x=6`, all `cos θ_i > 0.95`, `‖qvel‖ < 1`. But the corridor is
   *not* the obstacle: 23/24 runs get the cart past it. What no planner does is
   bring the pendulum to **rest** upright. MPPI sweeps *through* vertical at a
   median `‖qvel‖ = 4.68 rad/s` (1517 qualifying steps over 10 runs; only 5 below
   `0.5 rad/s`).

   This is a **terminal-capture** failure, not a search failure, and it changes
   what would fix it. MPC's cost descent has no mechanism for "arrive at rest in
   a specified unstable configuration"; an RRT's stopping test
   `‖x − x_goal‖ < δ` is over the *full state including velocities* (paper §5),
   which targets exactly that. Two further findings support the approach:
   lengthening MPPI's horizon to compensate does **not** work (open-loop rollouts
   diverge past ~1 s on this chaotic system — measured), and AgileRRT's edges are
   *closed-loop* (Eq 26 rolls out `u = ũ − K(x − x̃)`), which is what survives
   there.

   Caveat on the same evidence: because the gap is terminal stabilization, a
   *stabilizing terminal controller* (LQR/funnel about the upright equilibrium)
   may close it far more cheaply than any tree. The piece of this paper that
   matters for that is the **projection operator**, not the RRT.

**Recommendation.** Do not implement AgileRRT as a `Planner`. Given that the
measured gap is terminal capture rather than search, take the cheap test first:

- **Test the cheap hypotheses first.** A *global* velocity penalty is already
  ruled out — measured, §[A global velocity penalty is not the fix
  either](#a-global-velocity-penalty-is-not-the-fix-either): the weight that
  would enforce capture is above the weight at which the planner refuses to
  start. Two candidates remain, both far cheaper than a tree:
  **(a)** a *state-gated* capture cost (velocity charged only within ~1 m of the
  goal, so the swing-up stays unpenalized) — a ~10-line residual change;
  **(b)** an LQR terminal controller about the upright equilibrium at `x=6`,
  handed the last spline segment. If either captures the goal, the tree question
  is moot. Do these before writing any planner.

If that fails, implement AgileRRT in two separable pieces, preferring a more
modern tree for the outer loop:

- **Worth building (small, reusable):** the cached-precompute LQT steering
  primitive — Eq 23/24/25 plus the Eq 26 projection — as a standalone
  `agile_rrt/steering.{h,cc}`. It is ~300 lines against MJPC's existing
  `ModelDerivatives`, gives a principled kinodynamic distance metric at 7.8 µs
  per query, and is useful to *any* future tree planner in this repo.
- **Prefer for the outer loop: SST\*** (Li, Littlefield & Bekris, IJRR 2016)
  rather than AgileRRT's RRT. See [Better-fitting successors](#better-fitting-successors).
- **Integration pattern:** one-shot / anytime global planner producing a
  reference trajectory, with MJPC's Sampling planner tracking it — not a
  drop-in `OptimizePolicy`. See [Integration options](#integration-options).

---

## What problem it solves

MJPC is a receding-horizon sampling MPC framework: every control step it
perturbs a spline of controls, rolls out `num_trajectory` candidates open-loop
with `mj_step`, and takes a cost-weighted or best-of update. That is excellent
at local stabilization and hopeless at problems where the *cost is
non-monotone over a horizon longer than the rollout window*.

The corridor task is deliberately such a problem. The obstacle gap is
`2 × (0.85 − 0.6) = 0.5 m`; the pendulum is `1.0 m`. Reaching the goal requires
committing to a sequence that makes the cost *worse* for seconds — swing the
pendulum down from upright, traverse laid-out, then re-erect — with only one
actuator (force on the cart, `|u| ≤ 20 N`) and 4 DoF. No 1-second cost horizon
can see the payoff.

A tree search is not myopic: it explores reachable sets and keeps every branch,
so it can find the detour through cost-increasing states. That is the gap.

---

## The algorithm

Notation follows the paper. State `x ∈ R^n` (`n = 8`), control `u ∈ R^m`
(`m = 1`), cost weights `R = Rᵀ > 0`, `P1 = P1ᵀ ≥ 0`.

The chain of ideas, in one paragraph: *steering* and *distance* in a kinodynamic
RRT are both optimal control problems (Eq 2); solving them nonlinearly is too
slow, so linearize (Eq 4) — but linearize about the **zero-control trajectory**
`x_zero(t)` (Eq 3) rather than a single point, so the approximation survives a
long horizon; the resulting LQT problem has an open-loop solution whose
reachability Gramian is numerically hopeless (Eq 11, `κ ~ 1e15`), so re-pose it
in **closed loop** around a stabilizing `K` (Eq 14–17), which conditions it;
then *precompute* `K, W_K, S_K, Φ_K` once per vertex over `[0, t_h^max]` so that
steering to any target at any `t_h ≤ t_h^max` is **pure matrix algebra** with no
ODE solves (Algorithm 4); finally, the linear solution is infeasible for the
real nonlinear system, so **project** it with a feedback rollout (Eq 26).

### Continuous (paper) → discrete (as portable to MJPC)

MuJoCo gives discrete Jacobians directly (`mjd_transitionFD` returns `A, B` for
`x_{k+1} = A x_k + B u_k`), so every ODE in the paper collapses to a recursion.
This is a strict simplification — no integrator, no interpolation of
time-varying matrices, no `InterpVector` class.

| Paper | Continuous form | Discrete form to implement |
|---|---|---|
| Eq 3 | `ẋ_zero = f(x_zero, 0)`, `x_zero(0)=x0` | `mj_step` with `ctrl = 0` |
| Eq 4 | `A(t) = ∂f/∂x`, `B(t) = ∂f/∂u` along `x_zero` | `mjd_transitionFD` at each `k` |
| Eq 22 | `−Ṗ = AᵀP + PA − PBR⁻¹BᵀP + Q`, `P(t_h)=P1` | `P_k = Q + Aᵀ P_{k+1} A − Aᵀ P_{k+1} B K_k` |
| Eq 21 | `K = R⁻¹BᵀP` | `K_k = (R + Bᵀ P_{k+1} B)⁻¹ Bᵀ P_{k+1} A` |
| Eq 15 | `Ẇ_K = A_K W_K + W_K A_Kᵀ + BR⁻¹Bᵀ`, `W_K(0)=0` | `W_{k+1} = A_{K,k} W_k A_{K,k}ᵀ + B_k R⁻¹ B_kᵀ` |
| Eq 17 | `Ṡ_K = AS_K + S_KAᵀ + (KW_K − R⁻¹Bᵀ)ᵀR(KW_K − R⁻¹Bᵀ)` | `S_{k+1} = A_{K,k} S_k A_{K,k}ᵀ + M_kᵀ R M_k`, `M_k = K_k W_k − R⁻¹B_kᵀ` |
| Eq 7 | `∂Φ/∂τ = −Φ A(τ)`, `Φ(t,t)=I` | `Φ_{k+1,0} = A_{K,k} Φ_{k,0}` (store forward, compose) |

with `A_{K,k} = A_k − B_k K_k`.

### Algorithm 4 — the steering primitive (the part worth porting)

Given cached `W_K, S_K, Φ_K, K` and a target `x_des` at horizon index `k_h`:

```
e     = x_zero(k_h) − x_des                                  # Eq 23 argument
η*    = (W_K P1 W_K + S_K)⁻¹ W_K P1 · e                      # Eq 23  ← Cholesky solve
J*    = ½ η*ᵀ S_K η* + ½ (e − W_K η*)ᵀ P1 (e − W_K η*)       # Eq 25  ← the RRT metric
x̃(k)  = x_zero(k) − W_K(k) Φ_K(k_h,k)ᵀ η*                    # Eq 24
ũ(k)  = [K(k) W_K(k) − R⁻¹ B(k)ᵀ] Φ_K(k_h,k)ᵀ η*             # Eq 24
```

`J*` alone (three lines, one `n×n` solve) is the distance function — this is why
nearest-neighbour costs 7.8 µs and not milliseconds. `x̃, ũ` are only needed once
a vertex has *won* the nearest-neighbour contest.

### Eq 26 — the projection

`(x̃, ũ)` satisfies the *linearized* dynamics, so it is not a trajectory of the
real system. Hauser's projection operator makes it feasible by tracking it:

```
u(k) = ũ(k) − K(k) · (x(k) − x̃(k)),   then x(k+1) = mj_step(x(k), u(k))
```

Note this is a **feedback** rollout. It is the single most important property for
MJPC's purposes: on a chaotic system, closed-loop rollouts stay bounded where
MPPI's open-loop spline rollouts diverge (measured in
[Measured behaviour](#measured-behaviour)).

---

## Equation-to-code map

Reference implementation. All line numbers verified against the files at
upstream commit `d0b8fe4`.

### Per-vertex precompute — `TreeVertex::Initialize()`, Algorithm 6 step 1

| Paper | Quantity | Code |
|---|---|---|
| Eq 3 | `x_zero` by integrating free dynamics | [treevertex.cpp:56](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L56) |
| §6 alt. | point linearization: `x_T(t) ≡ x0` | [treevertex.cpp:60-67](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L60-L67) |
| Eq 4 | `A(t)` along the linearization | [treevertex.cpp:76](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L76) |
| Eq 4 | `B(t)` | [treevertex.cpp:79](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L79) |
| Eq 22 | Riccati for `P(t)` | [treevertex.cpp:90](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L90) |
| Eq 21 | `K = R⁻¹BᵀP` | [treevertex.cpp:102](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L102) |
| Eq 15 | `W_K` | [treevertex.cpp:108](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L108) |
| Eq 17 | `S_K` | [treevertex.cpp:114](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L114) |

### Algorithm 4 — `TreeVertex::JJLinearSteer()`

| Paper | Quantity | Code |
|---|---|---|
| — | reject `t_h > t_h_bar` | [treevertex.cpp:121](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L121) |
| Eq 23 | `P_{t_h} = (W_K P1 W_K + S_K)⁻¹ W_K P1` | [treevertex.cpp:139](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L139) |
| Eq 23 | `e = x_zero(t_h) − x_samp` | [treevertex.cpp:140](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L140) |
| Eq 23 | `η* = P_{t_h} e` | [treevertex.cpp:141](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L141) |
| Eq 25 | `J*` — the RRT distance metric | [treevertex.cpp:143](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L143) |

### Algorithm 5 / Eq 26 — `TreeVertex::Projection()`

| Paper | Quantity | Code |
|---|---|---|
| Eq 7 | closed-loop STM `Φ_K` | [treevertex.cpp:160](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L160) |
| Eq 24+26 | fused rollout + constraints + cost | [treevertex.cpp:167](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L167) |
| Eq 24 | `x̃ = x_zero − W_K Φᵀ η` | [pendcart_3link.cpp:101](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/pendcart_3link.cpp#L101) |
| Eq 24 | `ũ = (K W_K − R⁻¹Bᵀ) Φᵀ η` | [pendcart_3link.cpp:102](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/pendcart_3link.cpp#L102) |
| Eq 26 | `u = ũ − K_proj (x − x̃)` | [pendcart_3link.cpp:108](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/pendcart_3link.cpp#L108) |
| Eq 2 | running cost `½xᵀQx + ½uᵀRu` | [pendcart_3link.cpp:111](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/pendcart_3link.cpp#L111) |
| §2 | obstacle check (swept segment vs disk) | [pendcart_3link.cpp:321](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/pendcart_3link.cpp#L321) |

### Algorithm 6 — `Tree::RunRRT()`

| Paper | Step | Code |
|---|---|---|
| Alg 6.5 | `x_samp ~ U(X)`, `t_h ~ U(0, t_h^max)` | [tree.cpp:130-137](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/tree.cpp#L130-L137) |
| Alg 6.6 | `nearestneighbor(V, x_samp)` | [tree.cpp:141](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/tree.cpp#L141) |
| — | cheap Euclidean prefilter (`nndelta`) | [tree.cpp:38](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/tree.cpp#L38) |
| — | brute-force scan, `J*` per node | [tree.cpp:25](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/tree.cpp#L25), [:41-48](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/tree.cpp#L41-L48) |
| Alg 6.7-9 | steer, project, accept-or-fail | [tree.cpp:143](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/tree.cpp#L143) |
| Alg 6.10-12 | precompute for the new vertex, insert | [treevertex.cpp:202-228](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L202-L228) |
| Alg 6.4 | stop within `stopdist` of goal | [tree.cpp:159-160](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/tree.cpp#L159-L160) |

### Problem setup — `main.cpp`

| Quantity | Paper §5 | Code |
|---|---|---|
| `x0` = upright, cart at 0 | ✓ | [main.cpp:72](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/main.cpp#L72) |
| obstacles r=0.6 at (3, ±0.85) | ✓ | [main.cpp:155-157](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/main.cpp#L155-L157) |
| `Q_ii = 1/((range_i/2)²)` | ✗ paper says `P1 = I` | [main.cpp:163](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/main.cpp#L163) |
| `R = 1/((u_range/2)²) = 0.0025` | ✗ paper says `R = 0.025` | [main.cpp:167](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/main.cpp#L167) |
| goal: cart at 6, upright | ✓ | [main.cpp:196](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/main.cpp#L196) |

**Note the state ordering trap.** The reference interleaves
`(θ1, θ̇1, θ2, θ̇2, θ3, θ̇3, x, ẋ)`; MuJoCo uses `(x, θ1, θ2, θ3 | ẋ, θ̇1, θ̇2, θ̇3)`.
Any `Q`/`P1` copied from `main.cpp` must be permuted. The
[gramian experiment](#measured-behaviour) does this explicitly.

---

## Walkthrough

One RRT iteration from the benchmark's root vertex, `t_h^max = 1.0 s`,
`dt = 0.005` → 200 steps.

```
x0 = (0,0,0,0 | 0,0,0,0)          upright equilibrium, cart at origin
  │
  ├─ ONCE PER VERTEX  (Alg 6 step 1) ─────────────────────── measured 8.1 ms
  │
  ├→ Eq 3   integrate zero-control dynamics        treevertex.cpp:56
  │     x_zero(k), k=0..200
  │     ** drift over the horizon = 0.000 **  ← x0 is a zero-control equilibrium,
  │        so here x_zero(t) ≡ x0 and the paper's contribution is inert
  │
  ├→ Eq 4   Jacobians at each k                    treevertex.cpp:76,79
  │     A_k (8×8), B_k (8×1)                       [MJPC: mjd_transitionFD]
  │
  ├→ Eq 22  backward Riccati, P_200 = P1           treevertex.cpp:90
  ├→ Eq 21  K_k (1×8)                              treevertex.cpp:102
  │
  ├→ Eq 15  forward accumulate W_K(k) (8×8)        treevertex.cpp:108
  ├→ Eq 17  forward accumulate S_K(k) (8×8)        treevertex.cpp:114
  │     κ(W_K(1.0)) = 4.74e7    κ(S_K(1.0)) = 6.06e7
  │     (κ(W_0) = 1.52e17 — why the closed-loop form exists at all)
  │
  ├─ PER NEAREST-NEIGHBOUR TEST ──────────────────────────── measured 7.8 µs
  │
  │   sample x_samp ~ U(X), t_h ~ U(0,1.0)         tree.cpp:130-137
  │   prefilter ‖x_samp − x_zero(t_h)‖ < nndelta   tree.cpp:38    ← skips the solve
  │
  ├→ Eq 23  e = x_zero(k_h) − x_samp               treevertex.cpp:140
  ├→ Eq 23  η* = (W_K P1 W_K + S_K)⁻¹ W_K P1 e     treevertex.cpp:139,141
  │     one 8×8 solve — the ONLY linear solve in the inner loop
  ├→ Eq 25  J* = ½η*ᵀS_Kη* + ½‖e − W_Kη*‖²_P1      treevertex.cpp:143
  │     scalar; this is the RRT distance. argmin over the tree wins.
  │
  ├─ PER EXTEND (winner only) ───────────────────────────── measured 3.0 ms
  │
  ├→ Eq 24  x̃(k), ũ(k) over k=0..k_h               pendcart_3link.cpp:101,102
  │     the INFEASIBLE reference (satisfies linear, not nonlinear, dynamics)
  │
  ├→ Eq 26  project: roll out mj_step with          pendcart_3link.cpp:108
  │           u = ũ − K(x − x̃)                      ← feedback, not open-loop
  │     abort on bound/obstacle violation           pendcart_3link.cpp:321
  │       → "insertion failure", vertex discarded
  │
  └→ accept: new vertex at x_proj(t_h), recurse to step 1   treevertex.cpp:202
        (+ pendulum-specific angle unwrapping           treevertex.cpp:213-218)
```

The asymmetry between 7.8 µs and 8.1 ms is the whole design: precompute is
amortized over every future nearest-neighbour query against that vertex. It is
also the cost model that dooms the tree in real time — see below.

---

## Measured behaviour

### Cost model of the tree

Scripts: [`prototype/gramian_test.py`](prototype/gramian_test.py),
[`prototype/steering_test.py`](prototype/steering_test.py).
`t_h = 1.0 s`, 200 steps, `dt = 0.005`.

| Operation | Cost | Frequency |
|---|---|---|
| linearize along `x_zero` (200 × `mjd_transitionFD`) | 4.74 ms | per vertex |
| precompute `K, W_K, S_K, Φ` | 3.38 ms | per vertex |
| **Alg 4 distance only** (Eq 23 + Eq 25) | **7.8 µs** | per NN test |
| Alg 4 full (`x̃, ũ` over horizon) | 1.16 ms | per extend |
| Eq 26 projection (nonlinear rollout) | 1.81 ms | per extend |

Extrapolated to a 1000-vertex tree with brute-force nearest neighbour
(`≈5×10⁵` NN tests):

```
precompute   1000 × 8.12 ms  =  8.1 s
NN tests     5e5  × 7.8 µs   =  3.9 s
extends      1000 × 2.97 ms  =  3.0 s
                        total ≈ 15 s      (paper Fig. 3 at t_h=1.0s: ~10 s ✓)
```

At `t_h = 0.25 s` the per-vertex precompute drops to 2.7 ms, but the paper's own
Fig. 3 shows vertex count rising ~100× as `t_h` falls to 0.05 s — which is the
paper's central and *correct* trade-off claim, and it survives the port.

**Threading changes the constant, not the conclusion.** MJPC's
[`ModelDerivatives::Compute`](../model_derivatives.cc#L74-L106) already threads
`mjd_transitionFD` over the horizon *and* supports evaluating a subset with
interpolation (`skip`), which should cut the 4.74 ms to well under 1 ms. Even at
a 10× overall speedup, 1.5 s per tree is 150× a 10 ms control step.

### Gramian conditioning (reproducing paper Eq 13)

See the table in [Verdict](#verdict) item 5. Headline: `κ(W_0) = 1.52e17`
against the paper's `3.33e15` — the open-loop form is *worse* in MuJoCo's
discretization than the paper reports, reinforcing its argument. Also confirmed:
at the upright equilibrium, linearizing about `x_zero(t)` and about `x0` produce
**identical** matrices, hence identical condition numbers.

### MJPC's existing planners on this task

`corridor_benchmark`, 30 simulated seconds, 6 repeats each, 15 planner threads,
`t_h = 1.0 s`. **Goal set** = cart within 0.3 m of `x=6`, all `cos θ_i > 0.95`,
`‖qvel‖ < 1.0`. A run is *solved* if it enters the goal set without a
disqualifying collision (>2 cm penetration or >2% of steps in contact).

| Planner | reached goal set | past corridor | collided | final cart `x` | `cos θ` at end | contact % | mean cost |
|---|---|---|---|---|---|---|---|
| Sampling (MPPI) | **0/6** | 6/6 | 4/6 | 5.5–7.0 | −1.00 … +0.98 | 0.1–0.5% | **22** |
| Sample Gradient | 0/6 | 6/6 | 3/6 | 3.4–8.0 | −1.00 … +0.96 | 0.3–31% | 42–124 |
| Cross Entropy | 0/6 | 5/6 | 5/6 | 2.2–8.0 | −1.00 … −0.63 | 1.3–76% | 82–121 |
| iLQG | 0/6 | 6/6 | 6/6 | ~4.2 (3/6) | −0.89 … +0.23 | 0.7–88% | 26–30 |

Reading the failures — and note the cost column ranks MPPI *best* while it
solves no more of the task than the others:

- **MPPI** is the strongest: it threads the corridor in every run (median
  crossing at **t = 1.07 s** of the 30 s available) and gets the cart to the goal
  region. It then swings the pendulum repeatedly through vertical without ever
  capturing it. One run reached `cos θ = 0.980` at `cart = 6.18` — visually
  indistinguishable from solved in a single frame, at `‖qvel‖ = 3.42`.
- **iLQG** is the cautionary case. It parks at `x ≈ 4.2` with 43–88% of steps in
  contact — *resting against the lower obstacle*, one run with `spd_end = 0.00`.
  It achieves the lowest terminal speed of any planner by leaning on geometry it
  is supposed to avoid. Derivative-based planning against a contact
  discontinuity, behaving exactly as expected.
- **Cross Entropy** is worst: variance collapse, never enters the goal set, and
  drags along obstacles for up to 76% of steps.
- **Sample Gradient** is cleanest on obstacles in its good runs but slower to
  reach the goal region and much more variable.

**The corridor is not the bottleneck.** Over 10 instrumented MPPI runs:
crossing succeeded 10/10 at median `t = 1.07 s`; clearance during the traverse
was a median **−0.019 m** (a graze in 9/10 — the 0.5 m gap against a 1.0 m
pendulum is genuinely tight); and *after* crossing, every run reached
`cos θ > 0.95` at some point. The pendulum spends the remaining ~29 s cycling
through vertical. What is missing is capture, not search.

> **How this table was corrected.** Scored on the final state alone, every run
> reads as a flat failure. Scored on "was the goal *configuration* ever reached",
> the same runs read as "solved at t≈2.5 s, then lost balance" — which looked
> like a finding and was an artifact of ignoring velocity. Only the
> speed-gated criterion above distinguishes arriving at the goal from sweeping
> through it. The filmstrip that exposed this is reproduced under
> [Looking at the rollouts](#looking-at-the-rollouts); the discipline is written
> up as the `planner-eyes` skill.

### A global velocity penalty is not the fix either

If the failure is terminal capture, the cheapest candidate fix is to charge for
momentum. Sweeping the `Velocity` cost weight (MPPI, 30 s, 4 repeats):

| weight | past corridor | terminal `‖qvel‖` | best `cos θ` at end | outcome |
|---|---|---|---|---|
| 0.1 (default) | 4/4 | 5.4 – 8.5 | +0.85 | swings hard, never captures |
| 0.5 | 4/4 | 4.2 – 5.2 | **+0.994** | closer, still never captures |
| 1.0 | **0/4** | 0.00 | +1.000 | **planner refuses to move at all** (cost pinned at 180) |

There is a sharp bifurcation between 0.5 and 1.0: **the weight that would enforce
capture is above the weight at which the planner declines to start.** With a 1 s
horizon, the momentum needed to begin the swing-up costs more than the reachable
cost reduction, so the optimum is to sit at the start equilibrium forever.

This is the myopia in its purest form, and it is not fixable by reweighting a
global term. It requires either a **state-gated** terminal cost (charge velocity
only near the goal, leaving the swing-up unpenalized) or a planner whose goal
test is **set membership rather than cost descent** — which is exactly the RRT
stopping criterion `‖x − x_goal‖ < δ`.

```bash
# reproduce: edit the Velocity `user` weight in the build copy of task.xml
python3 - <<'EOF'
import re
p = "mjpc/tasks/triple_pendulum_cartpole/task.xml"   # or the build/ copy
s = open(p).read()
s = re.sub(r'<user name="Velocity"\s+dim="4" user="[^"]*"',
           '<user name="Velocity"  dim="4" user="0 1.0  0.0  10.0"', s)
open(p, "w").write(s)
EOF
./bin/corridor_benchmark --planner=0 --total_time=30 --repeats=4
```

### Horizon is not the fix

Extending MPPI's horizon to chase the myopia does not work — it destabilizes:

| `t_h` | outcome |
|---|---|
| 1.0 s | 3/3 past corridor; one run cart 6.31 with `cos θ = 0.966` (near-solve) |
| 2.0 s | `Rollout divergence at step …` warnings |
| 3.0 s | `WARNING: Nan, Inf or huge value in QACC at DOF 0` at t=22.5 s |

**This is the strongest argument for the paper's approach.** The system is
chaotic, so open-loop rollouts (MPPI's mechanism) lose meaning beyond ~1 s,
while AgileRRT's edges are closed-loop feedback rollouts (Eq 26) that stay
bounded. Long-horizon reasoning on this system requires stabilized edges, which
is exactly what the projection operator provides.

### Looking at the rollouts

The corrected reading above came from rendering, not from the cost table. The
frames that settled it, from one MPPI run (tracking camera, `--track`):

| step | t | cart | `cos θ_min` | what the frame shows |
|---|---|---|---|---|
| 190 | 0.95 s | 2.63 | +0.04 | pendulum laid out horizontally, entering the gap |
| 215 | 1.08 s | 3.01 | −0.71 | folded through the corridor, clearance 0.005 m |
| 400 | 2.00 s | 5.86 | −0.85 | hanging, cart already past the goal |
| 470 | 2.35 s | 6.15 | +0.90 | swinging up on the far side |
| **508** | **2.54 s** | **6.15** | **+1.00** | **fully vertical at the goal — cost 1.17** |
| 560 | 2.81 s | 6.29 | +0.33 | already falling |
| 700 | 3.50 s | 6.13 | +0.74 | swinging back up |
| 1000 | 5.00 s | 6.64 | −0.40 | fallen again |

Step 508 is why the velocity gate matters: it is a textbook picture of a solved
task, and the pendulum is passing through it at ~4 rad/s. The cycle
508 → 560 → 700 → 1000 repeats for the remaining 27 s.

```bash
# dump a trajectory, then render filmstrip + video
./bin/corridor_benchmark --planner=0 --total_time=12 --repeats=1 \
    --dump=/tmp/mppi.csv
MUJOCO_GL=egl python3 \
  ../mjpc/tasks/triple_pendulum_cartpole/benchmark/filmstrip.py \
  --dump /tmp/mppi.csv --out /tmp/mppi.png --video /tmp/mppi.mp4

# the frames above
MUJOCO_GL=egl python3 \
  ../mjpc/tasks/triple_pendulum_cartpole/benchmark/filmstrip.py \
  --dump /tmp/mppi.csv --at 190,215,400,470,508,560,700,1000 \
  --track --distance 2.6 --out /tmp/upright_fail.png
```

### Reproducing

```bash
# build
cd build && cmake . && ninja corridor_benchmark

# MJPC planners on the task (planner: 0=Sampling 2=iLQG 5=CEM 6=SampleGradient)
./bin/corridor_benchmark --planner=0 --total_time=30 --repeats=6

# the model matches the paper's physics
python3 -c "
import mujoco
m = mujoco.MjModel.from_xml_path('../mjpc/tasks/triple_pendulum_cartpole/task.xml')
print('nq',m.nq,'nv',m.nv,'nu',m.nu,'total mass',m.body_mass.sum())"
```

The two NumPy prototypes that produced the conditioning and steering numbers are
scratch scripts, not repo artifacts. They are reproduced in full in the
[appendix](#appendix-prototype-scripts) so the numbers above are checkable.

---

## Deviations from the paper

Ordered by severity. Items 1–3 matter for a port; 4–6 are latent in the
reference implementation.

1. **Explicit inverse where a Cholesky solve is required.**
   [treevertex.cpp:139](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L139)
   forms `(W_K P1 W_K + S_K).inverse()`. That matrix is symmetric positive
   definite with `κ ≈ 5.5e7`; an explicit inverse roughly squares the error
   growth versus a factor-and-solve. **In a port, use `mju_cholFactor` /
   `mju_cholSolve`.** This is the one deviation with a concrete accuracy cost.

2. **The distance metric and the realized cost measure different objectives.**
   Eq 25 (`J*`, [treevertex.cpp:143](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L143))
   has no running state term — Lemma 3 derives it for control energy plus a
   terminal penalty only. But the projection accumulates `½xᵀQx + ½uᵀRu`
   ([pendcart_3link.cpp:111](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/pendcart_3link.cpp#L111))
   with a nonzero `Q`. So the tree is *grown* by one metric and *scored* by
   another. Defensible, but must be deliberate in a port.

3. **`J_proj` is computed and never used.** The projection's accumulated cost is
   written to `JJproj_cur_`
   ([treevertex.cpp:167](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L167))
   but `treevertex.h` exposes no getter and no decision reads it — pure waste in
   the inner loop. A port should either use it (e.g. for RRT\*-style rewiring) or
   not pay for it.

4. **Cost weights differ from the paper's text.** Paper §5 says `R = [0.025]`,
   `P1 = I_n`; the code uses range-normalized `Q_ii = 1/((range_i/2)²)` and
   `R = 0.0025` ([main.cpp:163,167](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/main.cpp#L163-L167)),
   a 10× difference in `R`. The published numbers are therefore not reproducible
   from the published code as-is. (Immaterial for `κ(W_0)`: with `m = 1`, `R`
   scales `W_0` by a scalar and cancels out of the condition number.)

5. **Latent gain-matrix inconsistency.** The Riccati integrates with
   `RRlqr_inv_` ([treevertex.cpp:90](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L90))
   but `K` is then formed with `RRinv_`
   ([treevertex.cpp:102](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L102)) —
   the *cost* `R`, not the *LQR* `R`. Dormant only because `main.cpp:90` passes
   the same matrices for both roles. It would silently produce a non-optimal `K`
   for anyone who separated them, which the API explicitly invites.

6. **`get_x0` is a no-op.**
   [treevertex.h:69](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.h#L69) —
   `void get_x0(ode_state_type * x0) {x0 = &x0_;}` assigns to the local
   parameter. Its only caller is commented out
   ([tree.cpp:153-157](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/tree.cpp#L153-L157)),
   which is presumably how it survived.

**Deviations MJPC *improves* on**, for free:

- **Collision geometry.** The reference checks only the 3 link endpoints, swept
  as line segments against disks
  ([pendcart_3link.cpp:321](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/pendcart_3link.cpp#L321)) —
  the rods themselves can pass through an obstacle. MuJoCo collides the actual
  capsules. The port is *stricter*, so paper vertex-count numbers will not
  transfer exactly.
- **Jacobians.** 2296 lines of Mathematica output, valid for exactly this
  3-link model, versus one `mjd_transitionFD` call valid for any model.
- **Angle wrapping.** Hand-rolled and explicitly marked
  `SPECIAL CODE FOR THE PENDULUM`
  ([treevertex.cpp:213-218](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/treevertex.cpp#L213-L218)).
  MJPC's `cos θ − 1` residual convention sidesteps it.

---

## Failure modes

| Symptom | Cause | Mitigation |
|---|---|---|
| `η*` is garbage / NaN; steering error explodes | inverting `W_0` (open loop), `κ ≈ 1e17` | never use Alg 1/2; closed-loop `W_K` only |
| Steering error grows with `t_h` past a point | linearization no longer valid; `x_zero` has diverged (chaotic drift measured at 8–14 over 1 s) | cap `t_h^max`; the `nndelta` prefilter ([tree.cpp:38](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/tree.cpp#L38)) exists for this |
| Control saturates on nearly every extend | targets outside the reachable ellipsoid `X̃_{V_δv}` | this is *expected* — "inexact" steering. Measured 23% at `t_h=1.0` even with `x_zero` linearization, 100% with point linearization |
| Insertion failure rate → 1, tree stops growing | long edges + dense obstacles (paper §6 predicts exactly this) | shorten `t_h`, accepting ~100× more vertices |
| Tree grows but never nears the goal | brute-force NN with a uniform sampler wastes effort | goal-biased sampling; a spatial index; or switch to SST\* |
| Per-step budget blown in MJPC | a tree build is ~10³ control steps of work | do not run it per step — see Integration options |

---

## Parameters

| Parameter | Reference default | What it does | How to tell it is wrong |
|---|---|---|---|
| `t_h_max_upper` | 1.0 s | horizon the precompute covers; edges may use any `t_h ≤` this | too large → insertion failures dominate; too small → vertex count explodes (~100× from 1.0→0.05 s) |
| `usezerotraj` | true | Eq 3 linearization vs point | **irrelevant at equilibrium vertices** (drift 0.0); at `t_h < 0.4 s` false is measurably better; at `t_h ≥ 0.5 s` true is 1.6–2.8× better |
| `nndelta` | ∞ (README suggests 15) | Euclidean prefilter radius before the `J*` solve | too small → `samplemiss` climbs, no candidate found; too large → pays 7.8 µs on hopeless nodes |
| `stopdist` | 0.0 (off) | terminate when within this cost-ball of the goal | — |
| `max_cnt` / `max_miss` / `max_samplemiss` | 1000/1000/10000 | budget caps | `miss` hitting its cap first ⇒ `t_h^max` too long for the obstacle density |
| `R` | 0.0025 (paper text: 0.025) | control energy weight | drives `K` aggressiveness; does not affect `κ(W_0)` when `m=1` |
| `Q`, `P1` | range-normalized | metric shape | must be permuted for MuJoCo's state ordering |

MJPC-side, in [`task.xml`](../../tasks/triple_pendulum_cartpole/task.xml):
`residual_Goal` (6.0) is the goal cart position; `residual_Clearance` (0.08 m)
is the margin at which the avoidance cost starts charging, giving the sampler a
gradient *before* contact — contact alone is a discontinuity and provides none.

---

## Integration options

Given the 10³× budget gap, three viable shapes, in increasing effort:

1. **Offline reference + MJPC tracking** *(recommended first step)*.
   Run the tree once as a standalone binary (~1–15 s), dump the best branch as a
   reference trajectory, and add a tracking residual so MJPC's Sampling planner
   follows it. Zero changes to the `Planner` interface. Directly tests whether
   the global plan closes the gap the benchmark exposes.

2. **Anytime tree inside a `Planner`.** Persist the tree across
   `OptimizePolicy` calls, growing it by a fixed ~10 ms budget per step and
   re-rooting as the state advances. Fits the interface, but re-rooting
   invalidates per-vertex precompute (each vertex's `x_zero`, `K`, `W_K`, `S_K`
   are all relative to *its* `x0`), so most of the 8.1 ms/vertex is thrown away
   each step. Substantial work for uncertain benefit.

3. **Steering primitive only, as shared infrastructure.** Land
   `agile_rrt/steering.{h,cc}` — Eq 23/24/25 + Eq 26 projection over MJPC's
   `ModelDerivatives` — with a unit test that reproduces the conditioning table.
   ~300 lines, independently useful as a kinodynamic distance metric, and a
   prerequisite for options 1 and 2 anyway. **Start here.**

---

## Better-fitting successors

The paper is 11 years old (ISRR 2015) and was, as the user suspected, little
built upon directly. Its ideas were largely superseded — in two opposite
directions.

**Direction 1: make steering unnecessary.** The paper spends its entire
technical budget making `steer(x0, x_des)` tractable for an unstable nonlinear
system. The dominant modern answer is to *not steer at all*:

- **SST / SST\*** — Li, Littlefield & Bekris, IJRR 2016 (the paper cites their
  2014 workshop version as [13,14] but predates the journal result).
  Propagates *random controls* forward, prunes with a best-first rule plus a
  witness set, and achieves asymptotic near-optimality **with no steering
  function, no linearization, and no Gramians.** It needs only a forward
  simulator — which is exactly what `mj_step` is. For MJPC this is a
  substantially better fit: it deletes items 1, 2, 4, 5 from the
  [Deviations](#deviations-from-the-paper) list by construction, and every
  numerical hazard in [Failure modes](#failure-modes) that involves `κ`.
  **If the goal is to close the long-horizon gap in MJPC, implement SST\*, not
  AgileRRT.**

**Direction 2: make the local certificate rigorous.** AgileRRT's
"linearize, steer, project with LQR feedback" is an *uncertified* tube — nothing
bounds how far the projected trajectory can be from the reference:

- **LQR-Trees / funnel libraries** — Tedrake et al. 2010; Majumdar & Tedrake,
  IJRR 2017. Precompute verified regions of attraction ("funnels") and compose
  them sequentially. This is the principled version of what AgileRRT
  approximates, and it directly addresses the insertion-failure problem: a funnel
  tells you *in advance* whether an edge is safe.
- **Control Contraction Metrics** — Manchester & Tedrake, TAC 2017. Yields a
  tracking controller with a certified contraction tube around any feasible
  trajectory. The natural modern replacement for Eq 26's ad-hoc LQR projection.
- **Graph of Convex Sets** — Marcucci et al., 2023–24. Global optimality for
  trajectory optimization through obstacle fields; kinodynamic extensions are
  active work. Different trade-off (needs convex decomposition), but the right
  reference point for "global, not local".

**What is genuinely worth keeping from this paper.** Two things, both narrow:

1. **The precompute-and-cache structure** (§3.4 / Algorithm 4). Amortizing
   `K, W_K, S_K, Φ_K` per vertex so that any `(x_des, t_h)` query is pure matrix
   algebra is a good idea independent of RRT, and the measured 7.8 µs proves it
   works. It would give an MJPC SST\* or RRT\* a real kinodynamic distance metric
   for the price of a Cholesky solve.
2. **The empirical horizon/vertex-count trade-off** (§6, Fig. 3). "Longer
   feedback-stabilized edges ⇒ far fewer vertices ⇒ faster search, until obstacle
   density makes edges fail" is the paper's most durable contribution, and it is
   orthogonal to the linearization machinery.

The zero-control linearization itself — the title contribution — is the weakest
part: inert at equilibria, helpful only past `t_h ≈ 0.4 s`, and disowned by the
paper's own Discussion.

---

## References

1. T. M. Caldwell and N. Correll. *Fast Sample-Based Planning for Dynamic
   Systems by Zero-Control Linearization-Based Steering.* ISRR, 2015.
   [local copy](https://github.com/timocaldwell/agile_rrt/blob/d0b8fe4/caldwell_isrr2015.pdf)
2. J. Hauser. *A projection operator approach to the optimization of trajectory
   functionals.* IFAC World Congress, 2002. — Eq 26
3. J. Hauser and D. G. Meyer. *The trajectory manifold of a nonlinear control
   system.* CDC, 1998.
4. Y. Li, Z. Littlefield, K. E. Bekris. *Asymptotically optimal sampling-based
   kinodynamic planning.* IJRR 35(5), 2016. — SST / SST\*
5. A. Majumdar and R. Tedrake. *Funnel libraries for real-time robust feedback
   motion planning.* IJRR 36(8), 2017.
6. I. R. Manchester and R. Tedrake. *Control contraction metrics: convex and
   intrinsic criteria for nonlinear feedback design.* IEEE TAC 62(6), 2017.
7. A. Perez et al. *LQR-RRT\*: Optimal sampling-based motion planning with
   automatically derived extension heuristics.* ICRA, 2012.
8. D. J. Webb and J. van den Berg. *Kinodynamic RRT\*.* ICRA, 2013.
9. T. Marcucci et al. *Motion planning around obstacles with convex
   optimization.* Science Robotics, 2023.

---

## Appendix: prototype scripts

The measurements in this document come from two NumPy prototypes that implement
the discrete restatement above directly against `mujoco`. They are committed so
the numbers are checkable, but are prototypes, not build targets — option 3 in
[Integration options](#integration-options) supersedes them with tested C++.

- [`prototype/gramian_test.py`](prototype/gramian_test.py) — reproduces paper
  Eq 13. Rolls out `x_zero`, collects `A_k, B_k` via `mjd_transitionFD`, runs the
  discrete Riccati and the `W_0`, `W_K`, `S_K` recursions, reports condition
  numbers, and checks the zero-control drift that makes the two linearizations
  coincide at equilibrium.
- [`prototype/steering_test.py`](prototype/steering_test.py) — implements
  Algorithm 4 and the Eq 26 projection, then measures (a) terminal steering error
  for `x_zero` vs point linearization across `t_h`, and (b) the per-operation
  timings in [Cost model of the tree](#cost-model-of-the-tree).

Both resolve `task.xml` relative to their own location, so they run from
anywhere:

```bash
python3 mjpc/planners/agile_rrt/prototype/gramian_test.py
python3 mjpc/planners/agile_rrt/prototype/steering_test.py   # ~2 min
```

Requirements: `mujoco` and `numpy` only (no MJPC build needed).
