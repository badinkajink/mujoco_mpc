#!/usr/bin/env python3
"""press_brace_check.py -- can the robot PRESS on the table instead of SAGGING onto it?

THE QUESTION. The brace pose is solved by brace_bow_solve.py, whose design
constraint is explicit in its own docstring: "CoM is driven PAST THE TOE ... At
that CoM the pose CANNOT stand on its own -- the forearm on the slab is the only
thing holding it." That guarantees the brace is load-bearing (it fixed the real
2026-08-01 phantom-brace failure) but it also guarantees the pose is
unrecoverable: releasing the arm from a configuration defined as "cannot stand
without the arm" is not a tuning problem.

Measured feet-only feasible region (feasible_region.py): forward limit +0.142 m
from ankle-mid. Toe is +0.133. So a pose at CoM 0.18 is outside it by design.

THE THIRD OPTION, never built: load-bearing by PRESSING. CoM stays INSIDE the
feet-only region, and the arm pushes down on the slab using shoulder torque --
like leaning a hand on a table while standing upright. You can press hard without
your weight leaving your feet.

WHAT THIS SCRIPT ASKS, per pose:

    maximise   f_forearm . n
    subject to   static equilibrium
                 CWC (friction, mu=1.0)
                 AWP (0.9 x TAU_ESTOP, ankle L1 diamond)
                 c_x <= com_cap          <- the CoM stays where the feet can hold it

If the answer is >= the brace force target (40 N design, 85 N seen), the press
brace is feasible at that geometry and the pose family should be redesigned.
If it is ~0, sagging is the only way to load the arm here, and we need the
(table_x, table_z) sweep to find a geometry where pressing works.
"""
import argparse
import numpy as np
import mujoco
from mujoco import mjtObj
from scipy.optimize import linprog

import feasible_region as FR


def max_press(m, d, pts, normals, n_fore, com_cap):
    """Max achievable forearm normal force with the CoM capped at com_cap (world x)."""
    Aeq, beq, Aub, bub, nf, nx = FR.build_lp(m, d, pts, normals,
                                             use_actuation=True, ankle_diamond=True)
    Aub = list(Aub); bub = list(bub)
    # CoM cap: c_x <= com_cap
    r = np.zeros(nx); r[nf + 0] = 1.0
    Aub.append(r); bub.append(com_cap)

    # objective: maximise the summed normal component over the forearm contacts
    c = np.zeros(nx)
    nc = len(pts)
    for i in range(nc - n_fore, nc):
        c[3 * i:3 * i + 3] = -normals[i] / np.linalg.norm(normals[i])
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub),
                  A_eq=Aeq, b_eq=beq, bounds=[(None, None)] * nx, method="highs")
    if res.status != 0:
        return None, None
    f = res.x[:nf].reshape(-1, 3)
    press = sum(float(f[i] @ (normals[i] / np.linalg.norm(normals[i])))
                for i in range(nc - n_fore, nc))
    return press, res.x[nf:nf + 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--qpos-npz", required=True)
    ap.add_argument("--time", type=float, default=40.0)
    ap.add_argument("--caps", default="0.10,0.12,0.142,0.16,0.18,0.21")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.model)
    d = mujoco.MjData(m)
    z = np.load(a.qpos_npz)
    k = int(np.argmin(np.abs(z["t"] - a.time)))
    d.qpos[:] = m.qpos0
    d.qpos[0:7] = z["base_qpos"][k]
    d.qpos[7:7 + 27] = z["q"][k]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    feet = [p for nm in ("left_ankle_roll_link", "right_ankle_roll_link")
            for p in FR.sole_hull(m, d, nm)]
    fore = list(FR.forearm_patch(m, d))
    pts = np.array(feet + fore)
    normals = [np.array([0, 0, 1.0])] * len(pts)

    amid = 0.5 * (d.xpos[mujoco.mj_name2id(m, mjtObj.mjOBJ_BODY, "left_ankle_roll_link")][0] +
                  d.xpos[mujoco.mj_name2id(m, mjtObj.mjOBJ_BODY, "right_ankle_roll_link")][0])
    actual = float(z["f_fore"][k])
    print(f"pose t={a.time:.0f}s   ankle-mid x={amid:.3f}   measured forearm force={actual:.1f} N")
    print(f"{'CoM cap (rel ankle-mid)':>26}{'max forearm press (N)':>24}")
    for capr in [float(s) for s in a.caps.split(",")]:
        press, cxy = max_press(m, d, pts, normals, len(fore), amid + capr)
        if press is None:
            print(f"{capr:>+26.3f}{'INFEASIBLE':>24}")
        else:
            print(f"{capr:>+26.3f}{press:>24.1f}")


if __name__ == "__main__":
    main()
