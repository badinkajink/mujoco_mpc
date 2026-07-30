"""Prototype of the AgileRRT steering primitive on top of MuJoCo.

This implements, in discrete time against mj_step / mjd_transitionFD:
  - Eq 3   zero-control trajectory x_zero
  - Eq 4   LTV linearization (A_k, B_k) along it
  - Eq 21/22 LQR gains K_k (backward Riccati)
  - Eq 15  W_K,  Eq 17  S_K   (forward Gramian accumulation)
  - Alg 4  efficient fixed-t_h inexact linear steering (pure matrix algebra)
  - Eq 26  Hauser projection onto feasible trajectories (nonlinear rollout with
           u = u_tilde - K (x - x_tilde))

and then answers the two questions that decide whether the paper's core idea is
worth porting:

  Q1  Does linearizing about x_zero(t) beat linearizing about the single point
      x0, measured as terminal steering error after projection?
  Q2  How does steering accuracy decay with the time horizon t_h -- i.e. is the
      "long horizon" premise (t_h up to 1.0s) actually supported?
"""
import numpy as np
import pathlib

import mujoco

def _default_xml():
    """Resolve task.xml relative to this file, so the script runs from anywhere."""
    import os
    here = pathlib.Path(__file__).resolve()
    # .../mjpc/planners/agile_rrt/prototype/ -> .../mjpc/tasks/triple_pendulum_cartpole/
    candidate = here.parents[3] / "tasks" / "triple_pendulum_cartpole" / "task.xml"
    return str(candidate)


XML = _default_xml()
EPS, CENTERED = 1e-6, 1
U_MAX = 20.0

model = mujoco.MjModel.from_xml_path(XML)
NQ, NV, NU = model.nq, model.nv, model.nu
N = 2 * NV
DT = model.opt.timestep


def set_state(data, x):
    data.qpos[:] = x[:NQ]
    data.qvel[:] = x[NQ:]


def rollout_zero(x0, steps):
    """Eq 3: x_zero, plus discrete (A_k, B_k) along it (Eq 4)."""
    data = mujoco.MjData(model)
    set_state(data, x0)
    mujoco.mj_forward(model, data)
    xs, As, Bs = [], [], []
    for _ in range(steps):
        xs.append(np.concatenate([data.qpos, data.qvel]).copy())
        A, B = np.zeros((N, N)), np.zeros((N, NU))
        data.ctrl[:] = 0.0
        mujoco.mjd_transitionFD(model, data, EPS, CENTERED, A, B, None, None)
        As.append(A); Bs.append(B)
        mujoco.mj_step(model, data)
    xs.append(np.concatenate([data.qpos, data.qvel]).copy())
    return np.array(xs), np.array(As), np.array(Bs)


def freeze(x0, steps):
    """The single-point (LTI) alternative: A, B evaluated once at x0, and
    x_T(t) == x0 for all t."""
    data = mujoco.MjData(model)
    set_state(data, x0)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    A, B = np.zeros((N, N)), np.zeros((N, NU))
    mujoco.mjd_transitionFD(model, data, EPS, CENTERED, A, B, None, None)
    xs = np.tile(np.concatenate([data.qpos, data.qvel]), (steps + 1, 1))
    return xs, np.array([A] * steps), np.array([B] * steps)


def precompute(As, Bs, Q, R, P1lqr):
    """Alg 6 step 1 / Alg 4's prerequisites: K_k, W_K(k), S_K(k), Phi_K(k_h, k).

    Everything here is done ONCE per tree vertex, for the longest horizon
    t_h_max; steering to any target at any t_h <= t_h_max then reuses the
    cached slices.  Phi is stored as Phi_K(k, 0) so that
    Phi_K(k_h, k) = Phi_K(k_h, 0) @ inv(Phi_K(k, 0)) -- see the note in Sec 3.4.
    """
    steps = len(As)
    Rinv = np.linalg.inv(R)

    # ---- Eq 22: backward Riccati -> Eq 21: K_k ----
    P = P1lqr.copy()
    Ks = [None] * steps
    for k in range(steps - 1, -1, -1):
        A, B = As[k], Bs[k]
        S = R + B.T @ P @ B
        Ks[k] = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ Ks[k]
        P = 0.5 * (P + P.T)

    # ---- Eq 15: W_K,  Eq 17: S_K,  closed-loop STM Phi_K ----
    # Eq 15 is a continuous-time integral over B_c R^-1 B_c', but
    # mjd_transitionFD returns the discrete B_d ~ B_c dt. Dividing by dt
    # recovers B_c; the matching dt on M' R M keeps the cost integral
    # consistent. Without these the Gramian is a factor dt (200x here) too
    # small and the steering it produces uses ~1% of the available actuator.
    inv_dt = 1.0 / DT
    W = np.zeros((N, N)); S = np.zeros((N, N))
    WKs, SKs, Acls = [W.copy()], [S.copy()], []
    for k in range(steps):
        A, B, K = As[k], Bs[k], Ks[k]
        Acl = A - B @ K
        M = K @ W - Rinv @ B.T * inv_dt
        S = Acl @ S @ Acl.T + M.T @ R @ M * DT;  S = 0.5 * (S + S.T)
        W = Acl @ W @ Acl.T + B @ Rinv @ B.T * inv_dt;  W = 0.5 * (W + W.T)
        Acls.append(Acl)
        WKs.append(W.copy()); SKs.append(S.copy())
    return dict(Ks=Ks, WKs=WKs, SKs=SKs, Acls=Acls, Rinv=Rinv, R=R,
                inv_dt=inv_dt)


def steer(pre, xzero, x_des, kh, P1):
    """Alg 4: efficient fixed-t_h inexact linear steering. No ODE solves --
    only matrix algebra on cached quantities.

    Returns (J_approx, x_tilde, u_tilde): Eq 25 cost and the *infeasible*
    reference trajectory that the projection will track.
    """
    WK, SK = pre["WKs"][kh], pre["SKs"][kh]
    Rinv, R, Ks = pre["Rinv"], pre["R"], pre["Ks"]

    # Eq 23: P_th = (W_K P1 W_K + S_K)^-1 W_K P1 ; eta* = P_th (x_zero(t_h) - x_des)
    err = xzero[kh] - x_des
    Pth = np.linalg.solve(WK @ P1 @ WK + SK, WK @ P1)
    eta = Pth @ err

    # Eq 25: the approximate cost, used as the RRT distance metric
    resid = err - WK @ eta
    J = 0.5 * eta @ (SK @ eta) + 0.5 * resid @ (P1 @ resid)

    # Eq 24: z*(t) = -W_K(t) Phi_K(t_h,t)' eta,  v*(t) = [K W_K - Rinv B'] Phi_K' eta
    #
    # w_k = Phi_K(kh,k)' eta is accumulated backwards from w_kh = eta via
    # w_{k-1} = Acl_{k-1}' w_k. The reference implementation instead forms
    # Phi_K(kh,0) inv(Phi_K(k,0)); that inverse throws "Singular matrix" here,
    # because a stable closed loop makes the 200-step product Phi_K(k,0)
    # numerically singular. The recursion is the same quantity without it.
    x_tilde = np.zeros((kh + 1, N))
    u_tilde = np.zeros((kh, NU))
    Acls = pre["Acls"]
    w = eta.copy()
    for k in range(kh, -1, -1):
        x_tilde[k] = xzero[k] - pre["WKs"][k] @ w
        if k < kh:
            u_tilde[k] = (Ks[k] @ pre["WKs"][k]
                          - Rinv @ Bs_cache[k].T * pre["inv_dt"]) @ w
        if k > 0:
            w = Acls[k - 1].T @ w
    return J, x_tilde, u_tilde


def project(x0, x_tilde, u_tilde, Ks, kh):
    """Eq 26 (Hauser projection): roll out the true nonlinear dynamics under
    u = u_tilde - K (x - x_tilde). Result is feasible by construction.
    Returns the realized trajectory and whether it stayed in bounds."""
    data = mujoco.MjData(model)
    set_state(data, x0)
    mujoco.mj_forward(model, data)
    xs = np.zeros((kh + 1, N))
    saturated = False
    for k in range(kh):
        x = np.concatenate([data.qpos, data.qvel])
        xs[k] = x
        u = u_tilde[k] - Ks[k] @ (x - x_tilde[k])
        if np.any(np.abs(u) > U_MAX):
            saturated = True
        data.ctrl[:] = np.clip(u, -U_MAX, U_MAX)
        mujoco.mj_step(model, data)
    xs[kh] = np.concatenate([data.qpos, data.qvel])
    return xs, saturated


# ---------------------------------------------------------------- experiments

rng = np.random.default_rng(0)


def sample_vertex_state():
    """A plausible non-equilibrium tree vertex: modest angles, real velocity."""
    x = np.zeros(NQ + NV)
    x[0] = rng.uniform(0.0, 4.0)                 # cart position
    x[1:4] = rng.uniform(-1.0, 1.0, 3)           # link angles (rad)
    x[4] = rng.uniform(-1.5, 1.5)                # cart velocity
    x[5:8] = rng.uniform(-2.0, 2.0, 3)           # link angular velocities
    return x


def sample_target(x0, scale):
    x = x0.copy()
    x[0] += rng.uniform(-scale, scale)
    x[1:4] += rng.uniform(-scale, scale, 3)
    x[4] += rng.uniform(-scale, scale)
    x[5:8] += rng.uniform(-2 * scale, 2 * scale, 3)
    return x


Q = np.zeros((N, N))
R = np.array([[0.025]])
P1 = np.eye(N)

print("=" * 78)
print("Q1: zero-control-trajectory linearization vs single-point linearization")
print("    metric = ||x_proj(t_h) - x_des|| after Eq-26 projection (lower better)")
print("=" * 78)

for t_h in (0.1, 0.25, 0.5, 0.75, 1.0):
    kh = int(round(t_h / DT))
    err_zero, err_pt, sat_zero, sat_pt, drift = [], [], [], [], []
    for trial in range(40):
        x0 = sample_vertex_state()
        x_des = sample_target(x0, 0.6)

        for tag, builder in (("zero", rollout_zero), ("pt", freeze)):
            xs, As, Bs = builder(x0, kh)
            globals()["Bs_cache"] = Bs
            pre = precompute(As, Bs, Q, R, P1)
            try:
                J, x_tilde, u_tilde = steer(pre, xs, x_des, kh, P1)
                xproj, sat = project(x0, x_tilde, u_tilde, pre["Ks"], kh)
            except np.linalg.LinAlgError:
                continue
            e = np.linalg.norm(xproj[kh] - x_des)
            if not np.isfinite(e):
                e = np.inf
            (err_zero if tag == "zero" else err_pt).append(e)
            (sat_zero if tag == "zero" else sat_pt).append(sat)
        xs, _, _ = rollout_zero(x0, kh)
        drift.append(np.linalg.norm(xs[kh] - xs[0]))

    def med(v):
        v = [x for x in v if np.isfinite(x)]
        return np.median(v) if v else np.inf

    print(f"  t_h={t_h:4.2f}s ({kh:3d} steps)  "
          f"median err: x_zero {med(err_zero):8.3f} | x0 {med(err_pt):8.3f}   "
          f"|| ctrl saturated: {np.mean(sat_zero)*100:3.0f}% / {np.mean(sat_pt)*100:3.0f}%"
          f"   || median zero-ctrl drift {np.median(drift):6.2f}")

print()
print("=" * 78)
print("Q2: cost of one steering evaluation (the RRT inner loop)")
print("=" * 78)
import time

for t_h in (0.25, 1.0):
    kh = int(round(t_h / DT))
    x0 = sample_vertex_state()
    x_des = sample_target(x0, 0.6)

    t = time.perf_counter()
    xs, As, Bs = rollout_zero(x0, kh)
    globals()["Bs_cache"] = Bs
    t_lin = time.perf_counter() - t

    t = time.perf_counter()
    pre = precompute(As, Bs, Q, R, P1)
    t_pre = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(20):
        J, x_tilde, u_tilde = steer(pre, xs, x_des, kh, P1)
    t_steer = (time.perf_counter() - t) / 20

    # the distance-only part of Alg 4 (steps 1,2,5) -- what nearest-neighbour needs
    WK, SK = pre["WKs"][kh], pre["SKs"][kh]
    t = time.perf_counter()
    for _ in range(200):
        err = xs[kh] - x_des
        eta = np.linalg.solve(WK @ P1 @ WK + SK, WK @ P1) @ err
        resid = err - WK @ eta
        _ = 0.5 * eta @ (SK @ eta) + 0.5 * resid @ (P1 @ resid)
    t_dist = (time.perf_counter() - t) / 200

    t = time.perf_counter()
    project(x0, x_tilde, u_tilde, pre["Ks"], kh)
    t_proj = time.perf_counter() - t

    print(f"  t_h={t_h:4.2f}s ({kh:3d} steps)")
    print(f"    linearize along x_zero (mjd_transitionFD x{kh}) : {t_lin*1e3:8.2f} ms  [per vertex]")
    print(f"    precompute K, W_K, S_K, Phi                     : {t_pre*1e3:8.2f} ms  [per vertex]")
    print(f"    Alg 4 distance only (Eq 23 + Eq 25)             : {t_dist*1e3:8.4f} ms  [per NN test]")
    print(f"    Alg 4 full (+ x_tilde, u_tilde over horizon)    : {t_steer*1e3:8.2f} ms  [per extend]")
    print(f"    Eq 26 projection (nonlinear rollout)            : {t_proj*1e3:8.2f} ms  [per extend]")
