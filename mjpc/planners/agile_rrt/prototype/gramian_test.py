"""Reproduce the central numerical claim of Caldwell & Correll (ISRR 2015) inside
MuJoCo's discretization, using mjd_transitionFD instead of Mathematica-exported
closed-form Jacobians.

Paper claims (Eq 13 / Sec 3.2), continuous time, t_h = 1.0s, n-link pendulum on
a cart at the upright equilibrium:

    open loop   kappa(W_0)   1-link 1.32e6   2-link 1.08e8   3-link 3.33e15
    closed loop kappa(W_K)   1-link 5.84     2-link 5.49e2   3-link 1.03e5
                kappa(S_K)   1-link 3.80e4   2-link 2.52e6   3-link 2.57e6

If the open-loop Gramian is as ill-conditioned as claimed, Algorithm 1
(open-loop exact steering) is unusable and the closed-loop form of Sec 3.2 is
load-bearing rather than a refinement.  That is the claim under test.
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

T_H = 1.0
EPS = 1e-6          # mjd_transitionFD finite-difference epsilon
CENTERED = 1


def linearize_along_zero_control(model, x0, t_h):
    """Roll out the zero-control trajectory from x0 and return discrete-time
    (A_k, B_k) along it -- the discrete analogue of Eq 3 + Eq 4."""
    data = mujoco.MjData(model)
    nv, nu = model.nv, model.nu
    n = 2 * nv                       # dim_state_derivative, = NS = 8 here
    steps = int(round(t_h / model.opt.timestep))

    As, Bs, xs = [], [], []
    data.qpos[:] = x0[:model.nq]
    data.qvel[:] = x0[model.nq:]
    mujoco.mj_forward(model, data)

    for _ in range(steps):
        xs.append(np.concatenate([data.qpos, data.qvel]).copy())
        A = np.zeros((n, n))
        B = np.zeros((n, nu))
        data.ctrl[:] = 0.0           # zero-control linearization
        mujoco.mjd_transitionFD(model, data, EPS, CENTERED, A, B, None, None)
        As.append(A)
        Bs.append(B)
        mujoco.mj_step(model, data)  # advance along x_zero
    xs.append(np.concatenate([data.qpos, data.qvel]).copy())
    return np.array(As), np.array(Bs), np.array(xs)


def linearize_at_point(model, x0, t_h):
    """The alternative used by refs [2,3,15] in the paper: freeze the
    linearization at the single state x0 (LTI)."""
    data = mujoco.MjData(model)
    nv, nu = model.nv, model.nu
    n = 2 * nv
    steps = int(round(t_h / model.opt.timestep))
    data.qpos[:] = x0[:model.nq]
    data.qvel[:] = x0[model.nq:]
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    A = np.zeros((n, n))
    B = np.zeros((n, nu))
    mujoco.mjd_transitionFD(model, data, EPS, CENTERED, A, B, None, None)
    return np.array([A] * steps), np.array([B] * steps), None


def discrete_lqr_gains(As, Bs, Q, R, P1):
    """Backward Riccati recursion -- discrete analogue of Eq 22, gains Eq 21."""
    N = len(As)
    P = P1.copy()
    Ks = [None] * N
    for k in range(N - 1, -1, -1):
        A, B = As[k], Bs[k]
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)      # K_k = (R + B'PB)^-1 B'PA
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        P = 0.5 * (P + P.T)
        Ks[k] = K
    return Ks


def gramians(As, Bs, Rinv, Ks=None):
    """Forward accumulation of the R^-1-weighted reachability Gramian.
    Discrete analogue of Eq 15 (W_K) with Ks given, or Lemma 1's W_0 without.
        W(k+1) = Acl_k W(k) Acl_k' + B_k Rinv B_k',  W(0) = 0
    """
    n = As[0].shape[0]
    W = np.zeros((n, n))
    for k in range(len(As)):
        A, B = As[k], Bs[k]
        Acl = A - B @ Ks[k] if Ks is not None else A
        W = Acl @ W @ Acl.T + B @ Rinv @ B.T
        W = 0.5 * (W + W.T)
    return W


def sk_gramian(As, Bs, Ks, R, Rinv):
    """Discrete analogue of Eq 17: the control-energy Gramian S_K, accumulated
    with the same closed-loop transition as W_K but weighted by
    (K W_K - Rinv B')' R (K W_K - Rinv B')."""
    n = As[0].shape[0]
    W = np.zeros((n, n))
    S = np.zeros((n, n))
    for k in range(len(As)):
        A, B, K = As[k], Bs[k], Ks[k]
        Acl = A - B @ K
        M = K @ W - Rinv @ B.T          # (nu x n)
        S = Acl @ S @ Acl.T + M.T @ R @ M
        S = 0.5 * (S + S.T)
        W = Acl @ W @ Acl.T + B @ Rinv @ B.T
        W = 0.5 * (W + W.T)
    return S


def kappa(M):
    s = np.linalg.svd(M, compute_uv=False)
    return s[0] / s[-1] if s[-1] > 0 else np.inf


def report(name, model, x0, Q, R, P1, use_zero_traj=True):
    Rinv = np.linalg.inv(R)
    lin = linearize_along_zero_control if use_zero_traj else linearize_at_point
    As, Bs, xs = lin(model, x0, T_H)

    W0 = gramians(As, Bs, Rinv)
    Ks = discrete_lqr_gains(As, Bs, Q, R, P1)
    WK = gramians(As, Bs, Rinv, Ks)
    SK = sk_gramian(As, Bs, Ks, R, Rinv)

    print(f"  {name}")
    print(f"    kappa(W_0) = {kappa(W0):.3e}   (open loop,   Lemma 1)")
    print(f"    kappa(W_K) = {kappa(WK):.3e}   (closed loop, Eq 15)")
    print(f"    kappa(S_K) = {kappa(SK):.3e}   (Eq 17)")

    # The matrix Algorithm 4 step 1 actually inverts: (W_K P1 W_K + S_K)
    M = WK @ P1 @ WK + SK
    print(f"    kappa(W_K P1 W_K + S_K) = {kappa(M):.3e}   <-- inverted in Alg 4 step 1")
    if xs is not None:
        print(f"    zero-control drift over {T_H}s: |x_zero(t_h) - x0|_inf = "
              f"{np.abs(xs[-1] - xs[0]).max():.3f}")
    return W0, WK, SK


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    print(f"model: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"dt={model.opt.timestep} -> {int(T_H/model.opt.timestep)} steps over {T_H}s")
    n = 2 * model.nv
    x0 = np.zeros(model.nq + model.nv)    # upright equilibrium, cart at 0

    print("\n=== paper's stated weights: R = [0.025], P1 = I_8, Q = 0 ===")
    report("3-link, linearized about x_zero(t)", model, x0,
           np.zeros((n, n)), np.array([[0.025]]), np.eye(n), True)
    report("3-link, linearized about x0 (LTI)", model, x0,
           np.zeros((n, n)), np.array([[0.025]]), np.eye(n), False)

    # The reference implementation actually uses range-normalized weights
    # (main.cpp SpecifyCostGains): Q_ii = 1/((range_i/2)^2), R = 1/((40/2)^2).
    print("\n=== reference implementation's weights (main.cpp SpecifyCostGains) ===")
    bd = 10.0
    xr = np.array([2*np.pi, bd*np.pi*0.5, 2*np.pi, bd*np.pi*0.5,
                   2*np.pi, bd*np.pi*0.5, 4.5, bd*1.5])
    # note: reference orders state as (th1,dth1,th2,dth2,th3,dth3,x,dx);
    # MuJoCo orders it (x,th1,th2,th3,dx,dth1,dth2,dth3). Reorder the weights.
    ref_half_ranges = np.array([xr[6], xr[0], xr[2], xr[4],
                                xr[7], xr[1], xr[3], xr[5]])
    Q = np.diag(1.0 / ref_half_ranges**2)
    R = np.array([[1.0 / 20.0**2]])
    report("3-link, linearized about x_zero(t)", model, x0, Q, R, Q, True)


if __name__ == "__main__":
    main()
