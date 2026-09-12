#!/usr/bin/env python3
"""Correlation time of the sampled perturbation of the executed action.

For each spline representation (MJPC's TimeSpline: zero-order hold, linear,
cubic Hermite with finite-difference slopes, one-sided at the ends) and knot
noise (white, or iCEM's AR(1) with coefficient alpha and a cold start), draw
knot perturbations, evaluate the perturbation of the executed action delta_u(tau)
over the horizon, and integrate its normalized autocorrelation with delta_u(0):

    tau_p = integral_0^H  max(0, corr(delta_u(0), delta_u(tau))) dtau

and also the lag at which that correlation first falls below 0.8, t80. Over the
16 spline/noise arms at 33 plans/s, t80 orders completion and tau_p does not
(a 4-knot hold, tau_p 0.25 s, completes 5/6; the 3-knot cubic, tau_p 0.28 s,
0/6): the arms that complete keep the executed perturbation at >= 80 % for
>= 0.22 s, the arms that fail lose it by 0.20 s. Knot times follow the
planner: hold knots at t + i H/n, linear/cubic knots at t + i H/(n-1).
"""
import numpy as np

H = 1.0


def knots(n, kind):
    return np.arange(n) * (H / n if kind == "zero" else H / (n - 1))


def sample_spline(times, vals, kind, t):
    """MJPC TimeSpline::Sample for one channel; vals shape (n,), t scalar."""
    n = len(times)
    if t >= times[-1]:
        return vals[-1]
    i = int(np.searchsorted(times, t, side="right")) - 1
    i = max(0, min(i, n - 2))
    t0, t1 = times[i], times[i + 1]
    s = (t - t0) / (t1 - t0)
    if kind == "zero":
        return vals[i]
    if kind == "linear":
        return vals[i] * (1 - s) + vals[i + 1] * s

    def slope(j):
        if j == 0:
            return (vals[1] - vals[0]) / (times[1] - times[0])
        if j == n - 1:
            return (vals[n - 1] - vals[n - 2]) / (times[n - 1] - times[n - 2])
        return 0.5 * (vals[j + 1] - vals[j]) / (times[j + 1] - times[j]) + \
               0.5 * (vals[j] - vals[j - 1]) / (times[j] - times[j - 1])
    dt = t1 - t0
    h00 = 2 * s**3 - 3 * s**2 + 1; h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2;     h11 = s**3 - s**2
    return h00 * vals[i] + h10 * dt * slope(i) + h01 * vals[i + 1] + h11 * dt * slope(i + 1)


def tau_p(n, kind, alpha=0.0, samples=4000, dt=0.01, seed=0):
    rng = np.random.RandomState(seed)
    times = knots(n, kind)
    taus = np.arange(0, H + 1e-9, dt)
    U = np.zeros((samples, len(taus)))
    for k in range(samples):
        eps = rng.randn(n)
        if alpha > 0:
            v = np.zeros(n); v[0] = np.sqrt(1 - alpha**2) * eps[0]
            for j in range(1, n):
                v[j] = alpha * v[j - 1] + np.sqrt(1 - alpha**2) * eps[j]
        else:
            v = eps
        U[k] = [sample_spline(times, v, kind, t) for t in taus]
    u0 = U[:, 0]
    rho = np.array([np.corrcoef(u0, U[:, j])[0, 1] if U[:, j].std() > 1e-12 else 0.0 for j in range(len(taus))])
    return float(np.trapezoid(np.clip(rho, 0, None), taus)), rho, taus


def t_below(rho, taus, level=0.8):
    """lag at which the correlation first drops below `level` (H if never)."""
    below = rho < level
    return float(taus[np.argmax(below)]) if below.any() else float(H)


CONFIGS = {  # arm -> (knots, kind, alpha)
    "hold, 3 knots": (3, "zero", 0.0), "hold, 6 knots": (6, "zero", 0.0), "hold, 2 knots": (2, "zero", 0.0),
    "cubic, 3 knots, white": (3, "cubic", 0.0), "cubic, 6 knots, white": (6, "cubic", 0.0),
    "cubic, 2 knots, white": (2, "cubic", 0.0), "linear, 3 knots, white": (3, "linear", 0.0),
    "cubic, 3 knots, AR(1) 0.3": (3, "cubic", 0.3), "cubic, 3 knots, AR(1) 0.7": (3, "cubic", 0.7),
    "cubic, 3 knots, AR(1) 0.95": (3, "cubic", 0.95), "hold, 3 knots, AR(1) 0.7": (3, "zero", 0.7),
}

if __name__ == "__main__":
    for name, (n, kind, a) in CONFIGS.items():
        tp, rho, taus = tau_p(n, kind, a)
        print("%-30s tau_p = %.3f s   rho(0.2 s) = %.2f   t80 = %.2f s" % (name, tp, rho[20], t_below(rho, taus)))
