#!/usr/bin/env python3
"""Static contact selection for the H12 braced lean.

For a reach target (x, y, z) and a candidate set of bracing contacts on the right
arm, this:

  1. solves IK for a lean pose that puts the LEFT hand on the target, keeps both
     feet pinned, and places each selected bracing site on the table top;
  2. solves a static-equilibrium QP at that pose for the contact forces and joint
     torques, subject to linearized friction cones, unilateral normal forces, and
     the real actuator torque limits;
  3. reports feasibility and cost for every non-empty subset of the candidate
     contacts, plus the legs-only case.

Formulation
-----------
Static equilibrium of the floating-base robot (qvel = qacc = 0):

    S' tau  +  sum_i J_i' lambda_i  =  g(q)

with g(q) = d.qfrc_bias at rest (pure gravity).  NOT mj_inverse, which folds in
qfrc_constraint -- whatever the pose is penetrating.  Splitting the
6 unactuated floating-base rows from the 27 actuated rows:

    base   :  sum_i J_i[:, :6]'  lambda_i  =  g[:6]        (hard constraint)
    joints :  tau = g[6:] - sum_i J_i[:, 6:]' lambda_i     (bounded by tau_max)

Decision variables are the contact forces lambda_i in R^3 (world frame), one per
contact point: 4 corners per foot (8) plus one per selected bracing site.

Objective:  w_tau * ||tau / tau_max||^2  +  w_lam * ||lambda||^2
The first term is the thing we actually care about -- normalized actuator effort,
so a knee at 129/300 counts the same as a wrist at 4/9.5.  The second is the
min-norm regularizer that makes load distribution across redundant contacts
well-posed (and is what should make triple contact emerge).
"""

import itertools
import json
import sys

import numpy as np
import mujoco
from scipy.optimize import lsq_linear

import os
_LEAN_DIR = ("/home/humanoid/Programs/mjpc_icra2026/build/mjpc/tasks/"
             "humanoid_bench/lean/")
# LEAN_MODEL=magpie selects the variant that carries the real magpie grippers
# (0.506 kg each, with box collision proxies + jaws).  The base model is the
# handless robot: its "palm" is a wrist capsule + sphere, NOT a gripper.
_VARIANTS = {
    "handless": "Lean_H12.xml",              # no grippers at all
    "magpie":   "Lean_H12_Magpie.xml",       # box proxies + jaw boxes (current)
    "sphere":   "Lean_H12_Magpie_sphere.xml",# single r=32mm sphere per gripper
    "mesh":     "Lean_H12_Magpie_mesh.xml",  # visual mount mesh IS the collision mesh
}
MODEL = _LEAN_DIR + _VARIANTS.get(os.environ.get("LEAN_MODEL", "handless"),
                                  "Lean_H12.xml")

# Right arm is the bracing arm (reach_hand=2 -> left reaches).  Points are in
# body frame; see the geom dump in the plan file for provenance.
SITES = {
    # NOTE: body-frame z of the elbow JOINT on the upper-arm link is -0.182.
    # The first version of this used -0.250, taken from the collision geom's
    # `geom_size`, which for a MESH is the BOUNDING-BOX half-extent, not the
    # shape -- so the point floated ~68 mm past the joint in empty space and the
    # site was placeable only 1/12.  At the joint it places 12/12.
    "elbow":   ("right_shoulder_yaw_link", np.array([0.002,  0.007, -0.182])),
    "forearm": ("right_elbow_link",        np.array([0.050,  0.021, -0.007])),
    "palm":    ("right_wrist_yaw_link",    np.array([0.130,  0.000,  0.000])),
    # Trunk sites (added S10).  The pose audit found the hips touching the table
    # edge in ~80% of solved poses and the torso in ~60%, all unmodelled -- so
    # they are promoted to first-class candidates.  Both anchors are taken from
    # PRIMITIVE geoms on torso_link, where `size` is the true shape (capsule
    # radius / box half-extent), NOT a mesh bounding box -- see the S6.1 elbow bug.
    #   "hip"   : front surface of the `hip` capsule  (local [0,0,-0.05], r=0.05)
    #   "torso" : lower front edge of the `torso` box (local [0,0,0.25], hx=0.07)
    "hip":     ("torso_link",              np.array([0.050,  0.000, -0.050])),
    "torso":   ("torso_link",              np.array([0.070,  0.000,  0.050])),
}
ARM_SITES = ("elbow", "forearm", "palm")
TRUNK_SITES = ("hip", "torso")
REACH_BODY = "left_wrist_yaw_link"
REACH_OFF = np.array([0.13, 0.0, 0.0])
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]

IK_MARGIN = 0.025   # collision look-ahead for the IK [m]
# CL_ARM_TORQUE=1 overrides the lean model's stale arm actuatorfrcrange with the
# CL_Assets / h12_safety_layer values (S10.2).
CL_TORQUE_LIMITS = os.environ.get("CL_ARM_TORQUE") == "1"
_CL_TAU = {"shoulder_pitch": 40.0, "shoulder_roll": 40.0, "shoulder_yaw": 18.0,
           "elbow": 18.0, "wrist_roll": 19.0, "wrist_pitch": 19.0,
           "wrist_yaw": 19.0}
MU = 0.6            # friction coefficient (table & floor)
N_ROBOT_DOF = 33    # 6 floating base + 27 joints; DOFs 33+ are the object anchor
N_ACT = 27


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #
def load(ik_margin=IK_MARGIN):
    """Load the model.  `ik_margin` inflates every geom's collision margin so
    MuJoCo reports near-contacts BEFORE they overlap.  Without it the IK only
    learns about a collision once it is already inside the table, and repairing
    that within one trust-region step is often infeasible -- measured 245 of 250
    iterations infeasible, i.e. non-penetration silently dropped.  With a margin
    the constraint is predictive and the solver never enters the hole."""
    m = mujoco.MjModel.from_xml_path(MODEL)
    if ik_margin:
        m.geom_margin[:] = np.maximum(m.geom_margin, ik_margin)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_up")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    return m, d


def bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def point_world(m, d, body, offset):
    b = bid(m, body)
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ offset


def point_jac(m, d, body, offset):
    """3 x nv positional Jacobian of a body-fixed point."""
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, None, point_world(m, d, body, offset), bid(m, body))
    return jacp


def foot_corners(m, d, foot):
    """4 world-frame corner points of a foot, on its sole."""
    b = bid(m, foot)
    # foot collision proxy is centred at (0.033, 0, -0.019) in the ankle frame;
    # sole plane sits at the lowest z the body reaches.
    half_x, half_y = 0.10, 0.04
    z = -0.055
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append((foot, np.array([0.033 + sx * half_x, sy * half_y, z])))
    return out


def table_top_z(m, d):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    return d.geom_xpos[g][2] + m.geom_size[g][2]


def table_x_range(m, d):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    cx = d.geom_xpos[g][0]
    return cx - m.geom_size[g][0], cx + m.geom_size[g][0]


# --------------------------------------------------------------------------- #
# IK: lean pose for a reach target + a set of bracing sites on the table
# --------------------------------------------------------------------------- #
TORSO_DOF = 18          # dofadr of torso_joint
W_FOOT, W_REACH, W_SITE = 300.0, 4.0, 6.0   # feet are pinned: they must not slide
W_COLL = 40.0           # collision rows dominate: non-penetration is not optional
W_POST = 0.6            # nullspace pull back toward the seed pose
PEN_TOL = 0.002         # allowed contact depth before a collision row fires
TRUST = 0.05            # per-step trust region [rad/m], enforced INSIDE the QP
LEVENBERG = 1e-6        # keeps the QP Hessian positive definite


def collision_rows(m, d, allow_bodies):
    """One row per penetrating robot contact, pushing along the contact normal.

    Uses MuJoCo's own narrowphase (mj_forward has already run), so it sees exactly
    the collision proxies the planner sees.

    NOTE: robot bodies are never exempted, not even the feet or the selected
    bracing site.  Exempting a site's body lets the whole link sink through the
    table (measured 47 mm).  Instead a 2 mm tolerance lets a resting contact sit
    at dist ~ 0 without fighting the site-on-plane task.
    """
    rows, errs = [], []
    for c in range(d.ncon):
        con = d.contact[c]
        if con.dist >= IK_MARGIN:
            continue                        # outside the look-ahead band
        b1 = m.geom_bodyid[con.geom[0]]
        b2 = m.geom_bodyid[con.geom[1]]
        if _is_env(m, b1) and _is_env(m, b2):
            continue                                    # scenery vs scenery
        # which side is the robot?
        robot_b, sign = (b1, -1.0) if b1 != 0 and not _is_env(m, b1) else (b2, 1.0)
        if _is_env(m, robot_b):
            continue
        n = con.frame[:3].copy()                        # contact normal, world
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, con.pos.copy(), robot_b)
        rows.append(sign * (n @ jacp).reshape(1, -1))
        errs.append(np.array([-con.dist]))              # push out to depth 0
    return rows, errs


_ENV_CACHE = {}


def _is_env(m, b):
    """True for world/table bodies (everything that is not the robot)."""
    if b in _ENV_CACHE:
        return _ENV_CACHE[b]
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    v = (b == 0) or ("table" in n) or ("object" in n)
    _ENV_CACHE[b] = v
    return v


def solve_ik(m, d, target, subset, iters=250, tol=2e-3, lock_torso=False):
    """Gauss-Newton IK over the 33 robot DOFs, one QP per step.

    Each step solves, for the joint-velocity-like step dq:

        min  1/2 dq' G dq - a' dq
        s.t. J_foot dq = e_foot                     (feet are pinned: HARD)
             n' J_c dq >= -(dist + PEN_TOL)         (non-penetration: INEQUALITY)
             -TRUST <= dq_i <= TRUST                (trust region, in the QP)

    with G / a assembled from the weighted objective terms (reach, bracing sites,
    posture toward the seed) plus a Levenberg term.  This replaces the previous
    weighted normal equations + post-hoc step clipping, which did not descend
    (see plan file S3.3).  Weights are NOT the lever here and should not be tuned.

    `lock_torso=True` freezes torso_joint at its seed value -- the "assume we do
    not use the torso" case.  The shipped model does NOT lock it (only penalized,
    weight 30), so False is the faithful default.

    Returns dict of per-task residuals: reach, foot, site_max, penetration, ok.
    """
    import quadprog

    q_seed = d.qpos.copy()
    foot_targets = []
    for f in FEET:
        b = bid(m, f)
        foot_targets.append((d.xpos[b].copy(), d.xquat[b].copy()))
    z_tab = table_top_z(m, d)
    x_lo, x_hi = table_x_range(m, d)
    jnt_dofs = np.arange(6, N_ROBOT_DOF)
    n = N_ROBOT_DOF

    parts = {"reach": np.inf, "foot": np.inf, "site": {}}
    stats = {"qp_fallback": 0}
    for _ in range(iters):
        mujoco.mj_forward(m, d)

        # ---- hard equality: both feet hold their seed pose -------------------
        Jeq, eeq, foot_err = [], [], []
        for f, (p_t, q_t) in zip(FEET, foot_targets):
            b = bid(m, f)
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, d.xpos[b], b)
            ep = p_t - d.xpos[b]
            foot_err.append(np.linalg.norm(ep))
            Jeq.append(jp[:, :n]); eeq.append(ep)
            qerr = np.zeros(3); dq4 = np.zeros(4)
            mujoco.mju_mulQuat(dq4, q_t, np.array([d.xquat[b][0], -d.xquat[b][1],
                                                   -d.xquat[b][2], -d.xquat[b][3]]))
            mujoco.mju_quat2Vel(qerr, dq4, 1.0)
            Jeq.append(jr[:, :n]); eeq.append(qerr)
        Jeq = np.vstack(Jeq); eeq = np.concatenate(eeq)
        # only ask the step to remove what it can this iteration
        eeq = np.clip(eeq, -TRUST, TRUST)

        # ---- objective terms -------------------------------------------------
        G = np.zeros((n, n)); a = np.zeros(n)

        p = point_world(m, d, REACH_BODY, REACH_OFF)
        parts["reach"] = float(np.linalg.norm(target - p))
        Jr = point_jac(m, d, REACH_BODY, REACH_OFF)[:, :n]
        er = np.clip(target - p, -TRUST, TRUST)
        G += (W_REACH ** 2) * Jr.T @ Jr;  a += (W_REACH ** 2) * Jr.T @ er

        parts["site"] = {}
        for st in subset:
            body, off = SITES[st]
            ps = point_world(m, d, body, off)
            Js = point_jac(m, d, body, off)[:, :n]
            parts["site"][st] = float(abs(z_tab - ps[2]))
            rows_s = [Js[2:3]]; errs_s = [np.array([z_tab - ps[2]])]
            if ps[0] > x_hi:
                rows_s.append(Js[0:1]); errs_s.append(np.array([x_hi - ps[0]]))
            elif ps[0] < x_lo:
                rows_s.append(Js[0:1]); errs_s.append(np.array([x_lo - ps[0]]))
            Js2 = np.vstack(rows_s); es2 = np.clip(np.concatenate(errs_s), -TRUST, TRUST)
            G += (W_SITE ** 2) * Js2.T @ Js2;  a += (W_SITE ** 2) * Js2.T @ es2

        # posture pull toward the seed, on joint DOFs only
        dq_post = np.zeros(n)
        wp = np.zeros(n)
        for dof in jnt_dofs:
            j = int(np.argmax(m.jnt_dofadr == dof))
            adr = m.jnt_qposadr[j]
            dq_post[dof] = np.clip(q_seed[adr] - d.qpos[adr], -TRUST, TRUST)
            wp[dof] = W_POST
        if lock_torso:
            wp[TORSO_DOF] *= 100.0
        G += np.diag(wp ** 2);  a += (wp ** 2) * dq_post

        G += LEVENBERG * np.eye(n)          # keeps G positive definite

        # ---- inequalities: non-penetration + trust region ---------------------
        Cin, bin_ = [], []
        for r, er_c in zip(*collision_rows(m, d, set())):
            # want dist + n'J dq >= -PEN_TOL, i.e. n'J dq >= depth - PEN_TOL.
            # Demanding more than the trust region can deliver makes the QP
            # infeasible, so ask for at most half of it per step and let
            # successive iterations walk a deep penetration out.
            want = min(float(er_c[0]) - PEN_TOL, 0.5 * TRUST)
            Cin.append(r[0, :n]); bin_.append(want)
        Cin.append(np.eye(n));  bin_.append(np.full(n, -TRUST))
        Cin.append(-np.eye(n)); bin_.append(np.full(n, -TRUST))
        Cin = np.vstack([c.reshape(-1, n) if c.ndim == 1 else c for c in Cin])
        bin_ = np.concatenate([np.atleast_1d(x) for x in bin_])

        C = np.vstack([Jeq, Cin]).T          # quadprog wants C' x >= b, C is (n, m)
        bvec = np.concatenate([eeq, bin_])
        meq = Jeq.shape[0]

        parts["foot"] = float(max(foot_err))
        if (parts["reach"] < tol and parts["foot"] < tol
                and max(parts["site"].values(), default=0.0) < tol):
            break

        # Feet enter the objective at W_FOOT=300 rather than as a hard equality:
        # a hard equality plus the collision inequalities is frequently infeasible
        # inside the trust region, and every such failure used to silently DROP
        # non-penetration.  Soft feet at this weight hold to <1e-3 m in practice
        # (verified), and it lets non-penetration stay hard, which is what we
        # actually cannot compromise on.
        Gs = G + (W_FOOT ** 2) * Jeq.T @ Jeq
        asf = a + (W_FOOT ** 2) * Jeq.T @ eeq
        nbox = 2 * n
        try:
            dq = quadprog.solve_qp(Gs, asf, Cin.T, bin_, 0)[0]
        except ValueError:
            stats["qp_fallback"] += 1
            try:
                dq = quadprog.solve_qp(Gs, asf, Cin[-nbox:].T, bin_[-nbox:], 0)[0]
            except ValueError:
                dq = np.zeros(n)

        full = np.zeros(m.nv)
        full[:n] = dq
        mujoco.mj_integratePos(m, d.qpos, full, 1.0)

        for j in range(1, m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                adr_j = m.jnt_qposadr[j]
                d.qpos[adr_j] = np.clip(d.qpos[adr_j],
                                        m.jnt_range[j][0], m.jnt_range[j][1])
        if lock_torso:
            d.qpos[m.jnt_qposadr[13]] = q_seed[m.jnt_qposadr[13]]

    mujoco.mj_forward(m, d)
    pen = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        # object<->table is scenery penetrating scenery (42.5 mm at the seed
        # keyframe) and has nothing to do with the robot's pose.
        if _is_env(m, b1) and _is_env(m, b2):
            continue
        pen = min(pen, con.dist)
    parts["penetration"] = float(-pen)
    parts["qp_fallback"] = stats["qp_fallback"]
    parts["site_max"] = float(max(parts.get("site", {}).values(), default=0.0))

    # A bracing site counts as PLACED when its link is actually in contact with
    # the table, not when the nominal body-frame point reaches the table plane:
    # the point sits inside the link, so a resting contact still leaves ~35 mm of
    # link thickness between the point and the surface.
    tbl = bid(m, "table")
    touching = set()
    for c in range(d.ncon):
        con = d.contact[c]
        if con.dist > IK_MARGIN * 0.5:
            continue
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if b1 == tbl:
            touching.add(b2)
        elif b2 == tbl:
            touching.add(b1)
    parts["sites_placed"] = {st: (bid(m, SITES[st][0]) in touching) for st in subset}
    parts["all_placed"] = all(parts["sites_placed"].values())
    parts["ok"] = (parts["reach"] < 0.03 and parts["foot"] < 0.02
                   and parts["penetration"] < 0.01 and parts["all_placed"])
    return parts


# --------------------------------------------------------------------------- #
# static equilibrium QP
# --------------------------------------------------------------------------- #
def equilibrium_qp(m, d, subset, w_tau=1.0, w_lam=1e-4):
    """Solve for contact forces + torques at the current pose.

    Returns dict with feasibility, tau, lambda, and the normalized effort.
    """
    # Pure gravity bias.  NOT mj_inverse: qfrc_inverse folds in qfrc_constraint,
    # i.e. whatever contacts the pose happens to be penetrating (20 kN at the
    # seed keyframe), which swamps the 668 N we are actually balancing.
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_forward(m, d)
    g = d.qfrc_bias[:N_ROBOT_DOF].copy()

    contacts = []
    for f in FEET:
        contacts += foot_corners(m, d, f)
    for s in subset:
        contacts.append(SITES[s])

    J = [point_jac(m, d, b, o)[:, :N_ROBOT_DOF] for b, o in contacts]
    nc = len(contacts)
    A = np.hstack([Ji.T for Ji in J])          # (33, 3*nc)

    tau_max = m.actuator_forcerange[:, 1].copy()
    if CL_TORQUE_LIMITS:
        # The lean model's actuatorfrcrange is STALE: shoulder 32 vs the real 40,
        # shoulder_yaw/elbow 14.4 vs 18, wrist 9.5 vs 19 (2x).  CL_Assets is the
        # source of truth (see _gen_h12_base_limits.py, which cross-checks against
        # h12_safety_layer/core/joint_limits.py).  The limit import propagated
        # `range` but not `actuatorfrcrange`, so override here.
        for i in range(m.nu):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for key, val in _CL_TAU.items():
                if key in n:
                    tau_max[i] = val
                    break

    # ---- variables: lambda (3*nc) ------------------------------------------
    # base rows are a hard equality; joint rows define tau.
    Ab, gb = A[:6], g[:6]
    Aj, gj = A[6:], g[6:]

    # least-squares with the base equality enforced by heavy weighting, plus
    # bounds; friction handled by an iterative clamp (pyramid) -- see note.
    W_BASE = 1e6
    rows = [W_BASE * Ab, (np.diag(1.0 / tau_max) @ Aj) * np.sqrt(w_tau)]
    rhs = [W_BASE * gb, (gj / tau_max) * np.sqrt(w_tau)]
    rows.append(np.eye(3 * nc) * np.sqrt(w_lam))
    rhs.append(np.zeros(3 * nc))
    M = np.vstack(rows)
    b = np.concatenate(rhs)

    # bounds: normal (z) force >= 0; tangential bounded by the pyramid at a
    # nominal normal load, then re-clamped against the solution below.
    lo = np.full(3 * nc, -np.inf)
    hi = np.full(3 * nc, np.inf)
    for i in range(nc):
        lo[3 * i + 2] = 0.0                     # unilateral: push only

    sol = lsq_linear(M, b, bounds=(lo, hi), max_iter=200)
    lam = sol.x

    # friction-cone repair: project tangential onto the pyramid, re-solve once
    for _ in range(6):
        viol = False
        for i in range(nc):
            n = lam[3 * i + 2]
            t = lam[3 * i:3 * i + 2]
            lim = MU * max(n, 0.0) / np.sqrt(2)
            if np.abs(t).max() > lim + 1e-9:
                viol = True
                hi[3 * i:3 * i + 2] = lim
                lo[3 * i:3 * i + 2] = -lim
        if not viol:
            break
        sol = lsq_linear(M, b, bounds=(lo, hi), max_iter=200)
        lam = sol.x

    tau = gj - Aj @ lam
    base_res = float(np.linalg.norm(Ab @ lam - gb))
    ratio = np.abs(tau) / tau_max
    return dict(
        subset=list(subset),
        base_residual=base_res,
        feasible=bool(base_res < 1.0 and ratio.max() <= 1.0),
        tau=tau, tau_ratio=ratio,
        max_ratio=float(ratio.max()),
        effort=float(np.sum(ratio ** 2)),
        lam=lam,
        brace_force=float(np.linalg.norm(lam[24:])) if len(subset) else 0.0,
        knee_l=float(abs(tau[3])), knee_r=float(abs(tau[9])),
    )


# --------------------------------------------------------------------------- #
def evaluate(target, verbose=False, lock_torso=False):
    """All 8 contact subsets for one reach target."""
    out = []
    for k in range(4):
        for subset in itertools.combinations(("elbow", "forearm", "palm"), k):
            m, d = load()
            ik_res, pen = solve_ik(m, d, np.asarray(target), subset,
                                   lock_torso=lock_torso)
            r = equilibrium_qp(m, d, subset)
            r["ik_residual"] = ik_res
            r["penetration"] = pen
            r["ik_ok"] = (ik_res < 0.05) and (pen < 0.01)
            r["target"] = list(map(float, target))
            out.append(r)
            if verbose:
                print(f"  {str(subset):40s} ik={ik_res:.3f} pen={pen*1000:5.1f}mm "
                      f"base_res={r['base_residual']:7.2f} "
                      f"max|tau|/lim={r['max_ratio']:.2f} effort={r['effort']:.3f} "
                      f"knee_L={r['knee_l']:6.1f} brace={r['brace_force']:6.1f} "
                      f"{'OK' if r['feasible'] and r['ik_ok'] else '--'}")
    return out


if __name__ == "__main__":
    m, d = load()
    print("table top z =", round(table_top_z(m, d), 3),
          " x range =", np.round(table_x_range(m, d), 3))
    print("left hand at seed:", np.round(point_world(m, d, REACH_BODY, REACH_OFF), 3))
    for s, (b, o) in SITES.items():
        print(f"  site {s:8s} -> {np.round(point_world(m, d, b, o), 3)}")

    tgt = [float(x) for x in sys.argv[1:4]] if len(sys.argv) >= 4 else [1.20, 0.15, 0.85]
    print(f"\nreach target {tgt}")
    evaluate(tgt, verbose=True)
