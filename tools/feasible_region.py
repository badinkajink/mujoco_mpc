#!/usr/bin/env python3
"""feasible_region.py -- actuation-aware feasible region Y_fa for the H1-2 brace.

Implements the Feasible Region formulation (arXiv 1903.07999):

    CWC   Contact Wrench Cone         B f <= 0            (friction)
    AWP   Actuation Wrench Polytope   tau_lo <= g(q) - J^T f <= tau_hi
    Y_fa  = { c_xy | exists f s.t. equilibrium AND CWC AND AWP }

The torque constraint lives INSIDE the projection LP. It is NOT a post-hoc
filter: Y_fa is a STRICT subset of Y_f ∩ Y_a because projection and intersection
do not commute (paper, Sec. III).

Robot-specific constraints that the paper does not model, all measured on this
robot -- see docs/feasibility_map_design_2026-08-04.md:

  * ANKLE L1 DIAMOND. The H1-2 ankle is a 4-bar parallel linkage driven by two
    motors, so its feasible torque set is |tP|/tPmax + |tR|/tRmax <= 1, NOT an
    independent box per axis. A box overstates the diagonal corner ~2x, and a
    one-armed brace is inherently laterally loaded, so roll demand steals the
    pitch authority the un-lean needs.
  * FEET ARE SOLE HULLS, not ankle origins. `foot_left_pos` is a framepos on the
    ankle BODY ORIGIN and throws away 134 mm of foot.
  * TORQUE LIMITS ARE 0.9 x TAU_ESTOP (the deploy node's clamp on TOTAL torque),
    not the operational URDF values.
  * g(q) COMES FROM qfrc_bias. mj_inverse includes contact forces and returns
    nonsense on a touching pose (measured 85-96 Nm ankle demand vs a 48.6 limit).

Usage:
    feasible_region.py --model <lean xml> [--qpos-npz run.npz --time 40]
                       [--contacts feet|feet+forearm] [--no-actuation]
"""
import argparse
import numpy as np
import mujoco
from mujoco import mjtObj
from scipy.optimize import linprog

# 0.9 x TAU_ESTOP -- the deploy node clamp (h12_control_node.cc:205,
# deploy_common.h:69 kClampRatio). NOT the operational URDF limits.
TAU_ESTOP = [60, 130, 200, 300, 54, 36,  60, 130, 200, 300, 54, 36,  40,
             32, 32, 14.4, 14.4, 9.5, 9.5, 9.5,
             32, 32, 14.4, 14.4, 9.5, 9.5, 9.5]
CLAMP = 0.9
ROBOT_MASS = 68.7          # kg. NOT sum(body_mass) -- that includes the table.
G = 9.81
MU = 1.0                   # geom_friction[0] on feet, table and floor

ANKLE_PAIRS = [(4, 5), (10, 11)]   # (pitch, roll) actuator indices per leg


def sole_hull(m, d, body_name, n_edge=5):
    """Contact points around the true sole, not the ankle body origin.

    Takes the lowest-z AABB corners of every geom on the foot body and returns a
    convex-hull-ish ring of n_edge points at the sole plane.
    """
    b = mujoco.mj_name2id(m, mjtObj.mjOBJ_BODY, body_name)
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b:
            continue
        lo = m.geom_aabb[g, :3] - m.geom_aabb[g, 3:]
        hi = m.geom_aabb[g, :3] + m.geom_aabb[g, 3:]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        R = R.reshape(3, 3)
        for cx in (lo[0], hi[0]):
            for cy in (lo[1], hi[1]):
                for cz in (lo[2], hi[2]):
                    local = m.geom_pos[g] + R @ np.array([cx, cy, cz])
                    pts.append(d.xpos[b] + d.xmat[b].reshape(3, 3) @ local)
    if not pts:
        return np.zeros((0, 3))
    P = np.array(pts)
    zmin = P[:, 2].min()
    sole = P[P[:, 2] < zmin + 0.02]          # points within 2 cm of the sole plane
    if len(sole) <= n_edge:
        return sole
    ctr = sole.mean(axis=0)
    ang = np.arctan2(sole[:, 1] - ctr[1], sole[:, 0] - ctr[0])
    idx = [int(np.argmin(np.abs(((ang - a + np.pi) % (2 * np.pi)) - np.pi)))
           for a in np.linspace(-np.pi, np.pi, n_edge, endpoint=False)]
    return sole[sorted(set(idx))]


def forearm_patch(m, d, n=3):
    """The braced forearm contact: a ~20 mm patch 12-15% along a 126 mm forearm,
    i.e. PROXIMAL, near the elbow -- not at the forearm midpoint."""
    b = mujoco.mj_name2id(m, mjtObj.mjOBJ_BODY, "left_elbow_link")
    if b < 0:
        return np.zeros((0, 3))
    R = d.xmat[b].reshape(3, 3)
    return np.array([d.xpos[b] + R @ np.array([0.13 * 0.126, 0.0, s * 0.010])
                     for s in np.linspace(-1, 1, n)])


def build_lp(m, d, pts, normals, use_actuation=True, ankle_diamond=True):
    """Return the LP pieces over x = [f (3 per contact), cx, cy]."""
    nc = len(pts)
    nf = 3 * nc
    nx = nf + 2
    W = ROBOT_MASS * G

    # --- equilibrium: forces and moments about the world origin -------------
    Aeq = np.zeros((6, nx))
    beq = np.zeros(6)
    for i in range(nc):
        Aeq[0:3, 3 * i:3 * i + 3] = np.eye(3)
        px, py, pz = pts[i]
        Aeq[3:6, 3 * i:3 * i + 3] = np.array([[0, -pz, py],
                                              [pz, 0, -px],
                                              [-py, px, 0]])
    beq[2] = W
    # Moment about the ORIGIN:  sum(p_i x f_i) + c x (-W e_z) = 0
    #   c x (W e_z) = W*(cy, -cx, 0)
    # so  sum(p x f)_x = +W*cy   ->  row_x has  -W  on cy
    #     sum(p x f)_y = -W*cx   ->  row_y has  +W  on cx
    Aeq[3, nf + 1] = -W
    Aeq[4, nf + 0] = +W

    # --- CWC: linearised friction pyramid at each contact -------------------
    Aub, bub = [], []
    for i in range(nc):
        n = normals[i] / np.linalg.norm(normals[i])
        t1 = np.array([1.0, 0, 0]) - n * n[0]
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.array([0, 1.0, 0]) - n * n[1]
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        row = np.zeros(nx); row[3 * i:3 * i + 3] = -n
        Aub.append(row.copy()); bub.append(0.0)          # f . n >= 0
        for t in (t1, t2):
            for s in (1, -1):
                row = np.zeros(nx)
                row[3 * i:3 * i + 3] = s * t - MU * n
                Aub.append(row.copy()); bub.append(0.0)  # |f.t| <= mu f.n

    # --- AWP: tau_lo <= g(q) - J^T f <= tau_hi ------------------------------
    if use_actuation:
        Jt = np.zeros((m.nv, nf))
        for i in range(nc):
            Jp = np.zeros((3, m.nv)); Jr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, Jp, Jr, pts[i], _body_of(m, d, pts[i]))
            Jt[:, 3 * i:3 * i + 3] = Jp.T
        g = d.qfrc_bias.copy()          # gravity term. NOT mj_inverse.
        for a in range(m.nu):
            dof = m.jnt_dofadr[m.actuator_trnid[a, 0]]
            lim = CLAMP * TAU_ESTOP[a]
            if ankle_diamond and any(a in p for p in ANKLE_PAIRS):
                continue                # handled below as a coupled constraint
            r = np.zeros(nx); r[:nf] = -Jt[dof, :]
            Aub.append(r.copy());  bub.append(lim - g[dof])       # tau <= +lim
            Aub.append(-r.copy()); bub.append(lim + g[dof])       # tau >= -lim

        if ankle_diamond:
            for (ap, ar) in ANKLE_PAIRS:
                dp = m.jnt_dofadr[m.actuator_trnid[ap, 0]]
                dr = m.jnt_dofadr[m.actuator_trnid[ar, 0]]
                Lp = CLAMP * TAU_ESTOP[ap]
                Lr = CLAMP * TAU_ESTOP[ar]
                # |tP|/Lp + |tR|/Lr <= 1, tau = g - J^T f  ->  4 sign combos
                for sp in (1, -1):
                    for sr in (1, -1):
                        r = np.zeros(nx)
                        r[:nf] = -(sp * Jt[dp, :] / Lp + sr * Jt[dr, :] / Lr)
                        Aub.append(r.copy())
                        bub.append(1.0 - (sp * g[dp] / Lp + sr * g[dr] / Lr))

    return Aeq, beq, np.array(Aub), np.array(bub), nf, nx


def _body_of(m, d, p):
    """Nearest body to a world point -- used to pick the Jacobian frame."""
    dist = np.linalg.norm(d.xpos - p, axis=1)
    return int(np.argmin(dist))


def region(m, d, pts, normals, use_actuation=True, ndir=24, ankle_diamond=True):
    Aeq, beq, Aub, bub, nf, nx = build_lp(m, d, pts, normals,
                                          use_actuation, ankle_diamond)
    verts = []
    for th in np.linspace(0, 2 * np.pi, ndir, endpoint=False):
        c = np.zeros(nx)
        c[nf + 0] = -np.cos(th)
        c[nf + 1] = -np.sin(th)
        r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                    bounds=[(None, None)] * nx, method="highs")
        if r.status == 0:
            verts.append(r.x[nf:nf + 2])
    return np.array(verts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--qpos-npz")
    ap.add_argument("--time", type=float, default=40.0)
    ap.add_argument("--contacts", default="feet+forearm",
                    choices=["feet", "feet+forearm"])
    ap.add_argument("--no-actuation", action="store_true")
    ap.add_argument("--no-diamond", action="store_true")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.model)
    d = mujoco.MjData(m)
    if a.qpos_npz:
        z = np.load(a.qpos_npz)
        k = int(np.argmin(np.abs(z["t"] - a.time)))
        d.qpos[:] = m.qpos0
        d.qpos[0:7] = z["base_qpos"][k]
        d.qpos[7:7 + 27] = z["q"][k]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    pts = [p for nm in ("left_ankle_roll_link", "right_ankle_roll_link")
           for p in sole_hull(m, d, nm)]
    normals = [np.array([0, 0, 1.0])] * len(pts)
    if a.contacts == "feet+forearm":
        fp = forearm_patch(m, d)
        pts += list(fp)
        normals += [np.array([0, 0, 1.0])] * len(fp)
    pts = np.array(pts)

    amid = 0.5 * (d.xpos[mujoco.mj_name2id(m, mjtObj.mjOBJ_BODY, "left_ankle_roll_link")][0] +
                  d.xpos[mujoco.mj_name2id(m, mjtObj.mjOBJ_BODY, "right_ankle_roll_link")][0])
    print(f"contacts: {len(pts)}  ({a.contacts})   ankle-mid x = {amid:.3f}")

    V = region(m, d, pts, normals, use_actuation=not a.no_actuation,
               ankle_diamond=not a.no_diamond)
    if len(V) == 0:
        print("INFEASIBLE: no static equilibrium satisfies both cones")
        return
    print(f"region x: {V[:,0].min()-amid:+.3f} .. {V[:,0].max()-amid:+.3f} m "
          f"(rel ankle-mid)   y: {V[:,1].min():+.3f} .. {V[:,1].max():+.3f}")
    print(f"forward margin from ankle-mid: {V[:,0].max()-amid:+.3f} m")


if __name__ == "__main__":
    main()
