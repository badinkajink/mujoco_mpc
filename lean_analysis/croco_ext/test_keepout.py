#!/usr/bin/env python3
"""The C++ keep-out must be the SAME function as the Python one, not a faster one.

A speed-up that quietly changes the optimisation problem invalidates every
number the study has already published against the Python activation, so this
checks agreement before anything is allowed to use it:

  * the raw SDF and its gradient on random points, including points placed
    exactly on the box faces, edges and corners -- the branches of `sdf_box`
    that a random sample almost never hits;
  * `a_value` and `Ar` out of a real crocoddyl ActivationData, driven through
    both models;
  * `Arr`, remembering that crocoddyl stores it as a DIAGONAL matrix, so the
    Python model's np.outer(g, g) was only ever contributing its diagonal.

usage: croco_ext/test_keepout.py
"""

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL
import croco_geom as cg

crocoddyl = cb.import_crocoddyl()
import croco_keepout as ck         # noqa: E402  (must follow the RTLD_GLOBAL import)


HALF = np.array([0.3, 0.6, 0.02])
RMIN = 0.015


def sample_points(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    pts = list(rng.uniform(-1.0, 1.0, (n, 3)) * (HALF * 3 + 0.05))
    # the branch boundaries: faces, edges, corners, centre, and points a hair
    # inside/outside each of them
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                p = np.array([sx, sy, sz], float) * HALF
                for e in (0.0, 1e-9, -1e-9, 1e-4, -1e-4):
                    pts.append(p * (1 + e) if np.any(p) else p + e)
    return pts


def main():
    worst_s = worst_g = 0.0
    for p in sample_points():
        s_py = cg.sdf_box(p, HALF)
        s_cc = ck.sdf_box(p[0], p[1], p[2], *HALF)
        worst_s = max(worst_s, abs(s_py - s_cc))
        g_py = cg.sdf_box_grad(p, HALF)
        g_cc = np.array(ck.sdf_box_grad(p[0], p[1], p[2], *HALF))
        worst_g = max(worst_g, float(np.max(np.abs(g_py - g_cc))))
    print(f"sdf        max |py - c++| = {worst_s:.3e}")
    print(f"sdf grad   max |py - c++| = {worst_g:.3e}")

    a_py = cg.ActivationModelBoxKeepOut(HALF, RMIN)
    a_cc = ck.ActivationModelBoxKeepOut(HALF[0], HALF[1], HALF[2], RMIN)
    d_py, d_cc = a_py.createData(), a_cc.createData()
    worst_v = worst_ar = worst_arr = 0.0
    n_active = 0
    for p in sample_points(2000, seed=1):
        a_py.calc(d_py, p);  a_py.calcDiff(d_py, p)
        a_cc.calc(d_cc, p);  a_cc.calcDiff(d_cc, p)
        worst_v = max(worst_v, abs(d_py.a_value - d_cc.a_value))
        worst_ar = max(worst_ar, float(np.max(np.abs(d_py.Ar - d_cc.Ar))))
        worst_arr = max(worst_arr, float(np.max(np.abs(
            np.diag(d_py.Arr) - np.diag(d_cc.Arr)))))
        n_active += d_py.a_value > 0
    print(f"a_value    max |py - c++| = {worst_v:.3e}   "
          f"({n_active} of 2027 points inside the inflated box)")
    print(f"Ar         max |py - c++| = {worst_ar:.3e}")
    print(f"Arr diag   max |py - c++| = {worst_arr:.3e}")

    ok = max(worst_s, worst_g, worst_v, worst_ar, worst_arr) < 1e-12
    print("\nAGREE" if ok else "\nDISAGREE -- do not use the C++ path")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
