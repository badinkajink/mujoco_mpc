#!/usr/bin/env python3
"""crocoddyl MVP: plan stand -> braced contact mode for the H1-2 lean.

WHAT THIS IS FOR.  Today the braced lean is authored by hand: eight keyframes, a
14 s weight ramp, and a 10-sample Cross-Entropy MPC with a 1 s horizon tracking
the setpoint they drag past it.  The keyframes ARE the plan; CEM is a
short-horizon stabilizer.  The offline study separately certifies a pose q* and a
contact subset by enumeration + a static-equilibrium LP/QP, and the hand-authored
ramp lands ~1.5 rad away from it.  This closes that gap the other way: take the
enumeration's (q*, contact schedule) and ask a DDP to produce the trajectory
that gets there, with the contact schedule PRESCRIBED -- which is the input
crocoddyl wants and the thing the enumeration already produces.

STRUCTURE.  Three phases over a prescribed schedule:

    A  approach   feet only.        Lower the CoM and swing the bracing arm out
                                    over the table, sites kept ABOVE the table
                                    plane by a barrier.
    I  impact     impulse at the    Zero-velocity touchdown of the bracing sites.
       brace sites
    B  braced     feet + brace.     Settle onto q*, reach the target, come to
                                    rest inside the friction cones.

WHAT IS AND IS NOT MODELLED.  Honest list, because these are the costs that decide
whether the result transfers:

  * Contacts are RIGID and prescribed.  This is not MuJoCo's soft contact model,
    so predicted forces will not match MuJoCo's exactly -- `croco_replay.py`
    measures the difference rather than assuming it away.
  * The table is a contact reference and a half-space barrier on the brace sites
    and reaching hand.  It is NOT a collision body: nothing stops an ELBOW from
    clipping the table edge on the way in.  The barrier covers the sites we care
    about, which is less than full collision avoidance.
  * The feet are 6D-pinned at their placement in the START pose, not at q*'s.
    The study's IK pinned feet at the brace SEED keyframe, whose stance is 19 mm
    narrower than `stand`; a rigid contact cannot be in two places, so the start
    pose wins and q* enters as a posture target rather than a hard terminal
    constraint.  How far the result actually lands from q* is measured, not
    assumed -- see the report at the end of a run.
  * Torque limits are the study's `clamp` basis, enforced as hard box bounds by
    SolverBoxFDDP.
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import pinocchio as pin
import mujoco

import contact_select as cs
import croco_bridge as cb

# NOT `import crocoddyl` -- see croco_bridge.import_crocoddyl for why a plain
# import segfaults inside ShootingProblem.
crocoddyl = cb.import_crocoddyl()

# The feet enter as 6D contacts at the SOLE frames (see croco_bridge.SOLES), which
# is where a wrench cone has to be written for its centre-of-pressure rows to mean
# anything.
FEET = ["sole_left", "sole_right"]

# Baumgarte contact stabilisation gains (Kp, Kd) and the KKT inverse-damping used
# by the contact forward dynamics.  Module-level so they can be swept.
CONTACT_GAINS = np.array([0.0, 50.0])

# INV_DAMPING is not a nuisance parameter -- it is what makes the braced phase
# solvable at all.  Two 6D feet plus the bracing sites is 18 constraint rows on a
# 33-DOF robot, and the contacts are REDUNDANT: many force splits between feet and
# brace produce the same motion, so the KKT system is singular and the force the
# friction-cone cost differentiates is arbitrary.  Measured: with damping 0 or
# 1e-6 the braced problem fails from every cone weight and min-normal-force tried
# (stop stalls at 0.1-2.5 after 80 iterations); at 1e-4 all six of those
# configurations converge to stop < 4e-5.  This is the same regularizer the
# study's static QP already needs for the same reason -- its w_lam min-norm term
# on lambda, which its docstring says "is what should make triple contact emerge".
INV_DAMPING = 1e-4


# --------------------------------------------------------------------------- #
def _joint_bounds(model_mj):
    """Joint position bounds (27 actuated joints) straight from the MJCF."""
    lb, ub = [], []
    for name in cb.ACTUATED:
        j = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model_mj.jnt_limited[j]:
            lb.append(model_mj.jnt_range[j][0])
            ub.append(model_mj.jnt_range[j][1])
        else:
            lb.append(-np.inf)
            ub.append(np.inf)
    return np.array(lb), np.array(ub)


class LeanOCP:
    def __init__(self, subset, q_star_mj, q0_mj, mu=0.6, table_z=None,
                 reach_target=None, clearance=0.02, cones=True):
        self.rmodel = cb.build_pin()
        self.sites = cb.mj_site_frames(self.rmodel)
        self.rdata = self.rmodel.createData()
        self.subset = list(subset)

        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.actuation.nu

        self.q0 = cb.mj_to_pin(q0_mj)
        self.q_star = cb.mj_to_pin(q_star_mj)
        self.x0 = np.concatenate([self.q0, np.zeros(self.rmodel.nv)])
        self.x_star = np.concatenate([self.q_star, np.zeros(self.rmodel.nv)])
        self.mu = mu
        self.cones = cones
        self.table_z = table_z
        self.clearance = clearance
        self.reach_target = reach_target

        # Contact references.  Feet from the START pose (see the module docstring);
        # bracing sites from q*, i.e. the places on the table the static QP
        # certified and solved forces for.
        pin.forwardKinematics(self.rmodel, self.rdata, self.q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        self.foot_ref = {f: self.rdata.oMf[self.sites[f]].copy() for f in FEET}
        pin.forwardKinematics(self.rmodel, self.rdata, self.q_star)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        self.site_ref = {s: self.rdata.oMf[self.sites[s]].translation.copy()
                         for s in self.subset}
        self.reach_ref = self.rdata.oMf[self.sites["reach"]].translation.copy()

        m_mj, _ = cb.mj_model()
        self.tau_lim = cs.torque_limits(m_mj)
        self.jlb, self.jub = _joint_bounds(m_mj)

        # State-regularisation weights.  The floating base is NOT regularised to
        # q* (the base pose is an outcome of the lean, not a target); joints are,
        # so q* biases the solution without fighting the contacts.
        #
        # These are FLAT across the joints on purpose.  Weighting the bracing arm
        # 10x, to pull it onto the collision-free certified pose and keep the
        # gripper out of the table, destroys the solve outright: the staged
        # continuation goes back to the stalled signature (17 iterations, no step
        # accepted, cost stuck at its warm-start value).  The gripper penetration
        # is real and still open -- see the write-up -- but per-joint reweighting
        # is not the lever that fixes it.
        wq = np.concatenate([np.zeros(6), np.ones(self.rmodel.nv - 6)])
        wv = np.ones(self.rmodel.nv)
        self.w_state = np.concatenate([wq, wv]) ** 2

    # ----------------------------------------------------------------- costs --
    def _contacts(self, braced):
        contacts = crocoddyl.ContactModelMultiple(self.state, self.nu)
        for f in FEET:
            contacts.addContact(f, crocoddyl.ContactModel6D(
                self.state, self.sites[f], self.foot_ref[f],
                pin.LOCAL_WORLD_ALIGNED, self.nu, CONTACT_GAINS))
        if braced:
            for s in self.subset:
                contacts.addContact(f"brace_{s}", crocoddyl.ContactModel3D(
                    self.state, self.sites[s], self.site_ref[s],
                    pin.LOCAL_WORLD_ALIGNED, self.nu, CONTACT_GAINS))
        return contacts

    def _base_costs(self, braced, w_state=1e-1, w_ctrl=1e-3, cones=None):
        costs = crocoddyl.CostModelSum(self.state, self.nu)
        costs.addCost("stateReg", crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ActivationModelWeightedQuad(self.w_state),
            crocoddyl.ResidualModelState(self.state, self.x_star, self.nu)),
            w_state)
        costs.addCost("ctrlReg", crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControl(self.state, self.nu)),
            w_ctrl)

        # Joint limits, as a two-sided barrier on the state.  The reference MUST be
        # state.zero() and not a zeroed copy of x_star: zeroing x_star zeroes the
        # floating base QUATERNION too, and every state.diff against a (0,0,0,0)
        # quaternion normalises by zero.  crocoddyl does not check for it -- the
        # process segfaults while the ShootingProblem is being assembled.
        nv = self.state.nv
        lb = np.concatenate([self.state.lb[1:nv + 1], self.state.lb[-nv:]])
        ub = np.concatenate([self.state.ub[1:nv + 1], self.state.ub[-nv:]])
        costs.addCost("jointLim", crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(lb, ub)),
            crocoddyl.ResidualModelState(self.state, self.state.zero(), self.nu)),
            1e1)

        # Contact cones.  mu is the study's 0.6, and min_nforce > 0 is what makes
        # these UNILATERAL -- without it a "contact" is free to pull the robot down.
        #
        # The feet get a WRENCH cone and the bracing sites a FRICTION cone, and the
        # pairing is forced: a wrench cone is the 6D object (3 force rows + the
        # centre-of-pressure rows that keep the CoP inside the 0.20 x 0.08 m sole),
        # a friction cone is the 3D one.  Handing a 5-row friction cone to a 6D
        # contact does not raise a type error -- crocoddyl mis-sizes an allocation
        # and dies with a bare MemoryError, which is how this was found.
        if cones is None:
            cones = self.cones
        if not cones:
            return costs
        wrench = crocoddyl.WrenchCone(np.eye(3), self.mu,
                                      np.array(cb.FOOT_HALF) * 2.0,
                                      4, False, 1.0, 1e4)
        wact = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(wrench.lb, wrench.ub))
        for f in FEET:
            costs.addCost(f"cone_{f}", crocoddyl.CostModelResidual(
                self.state, wact, crocoddyl.ResidualModelContactWrenchCone(
                    self.state, self.sites[f], wrench, self.nu)), 1e1)
        if braced:
            cone = crocoddyl.FrictionCone(np.eye(3), self.mu, 4, False, 1.0, 1e4)
            act = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub))
            for s in self.subset:
                costs.addCost(f"cone_{s}", crocoddyl.CostModelResidual(
                    self.state, act, crocoddyl.ResidualModelContactFrictionCone(
                        self.state, self.sites[s], cone, self.nu)), 1e1)
        return costs

    def _table_barrier(self, costs, nu, weight=5e2):
        """Keep the whole bracing arm and the reaching hand ABOVE the table plane.

        The table is not a collision body in this OCP (see module docstring), so
        without this the arm is free to sweep straight through the slab.

        The barrier covers EVERY arm site -- elbow, forearm and palm -- not just
        the ones in the contact subset, and that distinction is the whole point.
        A site that is not selected as a contact is not thereby harmless: with the
        subset elbow+forearm and the barrier applied only to those two, the
        unconstrained magpie GRIPPER drove 80 mm through the tabletop (measured in
        MuJoCo off the planned trajectory; the resulting contact force was 154 kN).
        The links that are not bracing still have to not be inside the table.
        Checked against q*: every arm site clears table + 20 mm there -- palm by
        48 mm, elbow 23, forearm 14 -- so this constrains nothing the certified
        pose was relying on.  z-only: x and y are unbounded.
        """
        if self.table_z is None:
            return
        lo = np.array([-np.inf, -np.inf, self.table_z + self.clearance])
        hi = np.array([np.inf, np.inf, np.inf])
        act = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(lo, hi))
        keep_out = sorted(set(self.subset) | set(cs.ARM_SITES) | {"reach"})
        for s in keep_out:
            fid = self.sites[s]
            costs.addCost(f"above_{s}", crocoddyl.CostModelResidual(
                self.state, act, crocoddyl.ResidualModelFrameTranslation(
                    self.state, fid, np.zeros(3), nu)), weight)

    # --------------------------------------------------------------- phases --
    def build(self, dt=0.01, n_approach=120, n_braced=80, impulse=False,
              cones=None, w_terminal=1e0):
        approach = []
        for k in range(n_approach):
            costs = self._base_costs(braced=False, cones=cones)
            self._table_barrier(costs, self.nu)
            # Guide each bracing site toward its certified landing spot, ramped in
            # over the phase so the pull is weak while the robot is still upright
            # and strong just before touchdown.
            w = 5e1 * (k / max(n_approach - 1, 1)) ** 2
            if w > 0:
                for s in self.subset:
                    tgt = self.site_ref[s] + np.array(
                        [0, 0, self.clearance * (1 - k / max(n_approach - 1, 1))])
                    costs.addCost(f"land_{s}", crocoddyl.CostModelResidual(
                        self.state, crocoddyl.ResidualModelFrameTranslation(
                            self.state, self.sites[s], tgt, self.nu)), w)
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, self._contacts(False), costs, INV_DAMPING, True)
            approach.append(crocoddyl.IntegratedActionModelEuler(dmodel, dt))

        # Optional impulse node: the bracing sites touch down at zero velocity.
        #
        # OFF by default, and the default is the honest one for this maneuver.  An
        # impulse node has nu = 0, which makes the control trajectory ragged and,
        # more to the point, makes SolverBoxFDDP misbehave here -- it returns after
        # a handful of iterations with an uninitialised `stop` value and a
        # trajectory identical to the warm start.  Physically the cost of dropping
        # it is small: the brace is a hand placed deliberately on a table, so the
        # touchdown velocity the impulse would absorb is near zero by construction,
        # and the contact's Baumgarte gains acquire it over the first few braced
        # nodes.  Turning it on is `--impulse`, and the ragged-control handling
        # downstream is kept so that path still runs.
        impact = []
        if impulse:
            impulses = crocoddyl.ImpulseModelMultiple(self.state)
            for s in self.subset:
                impulses.addImpulse(f"brace_{s}", crocoddyl.ImpulseModel3D(
                    self.state, self.sites[s], pin.LOCAL_WORLD_ALIGNED))
            icosts = crocoddyl.CostModelSum(self.state, 0)
            icosts.addCost("stateReg", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ActivationModelWeightedQuad(self.w_state),
                crocoddyl.ResidualModelState(self.state, self.x_star, 0)), 1e-1)
            impact = [crocoddyl.ActionModelImpulseFwdDynamics(
                self.state, impulses, icosts)]

        braced = []
        for k in range(n_braced):
            costs = self._base_costs(braced=True, w_state=1e0, cones=cones)
            self._table_barrier(costs, self.nu)
            if self.reach_target is not None:
                costs.addCost("reach", crocoddyl.CostModelResidual(
                    self.state, crocoddyl.ResidualModelFrameTranslation(
                        self.state, self.sites["reach"], self.reach_target,
                        self.nu)), 1e2)
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, self._contacts(True), costs, INV_DAMPING, True)
            braced.append(crocoddyl.IntegratedActionModelEuler(dmodel, dt))

        # terminal: at q*, at rest, hand on target
        tcosts = self._base_costs(braced=True, w_state=w_terminal, w_ctrl=0.0,
                                  cones=cones)
        tcosts.removeCost("ctrlReg")
        self._table_barrier(tcosts, self.nu)
        if self.reach_target is not None:
            tcosts.addCost("reach", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ResidualModelFrameTranslation(
                    self.state, self.sites["reach"], self.reach_target, self.nu)),
                1e3)
        tdmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, self._contacts(True), tcosts, INV_DAMPING, True)
        terminal = crocoddyl.IntegratedActionModelEuler(tdmodel, 0.0)

        # Hand the torque limits to the solver as BOX BOUNDS.  SolverBoxFDDP only
        # enforces limits that the action models actually carry; without this it
        # silently degrades to plain FDDP and the `clamp` basis is documented but
        # not applied.
        for mdl in approach + braced:
            mdl.u_lb = -self.tau_lim
            mdl.u_ub = self.tau_lim

        self.dt = dt
        self.n_approach, self.n_braced = n_approach, n_braced
        return crocoddyl.ShootingProblem(
            self.x0, approach + impact + braced, terminal)


# --------------------------------------------------------------------------- #
def solve_staged(ocp, dt, n_approach, n_braced, iters=200, impulse=False,
                 w_terminal=1e0, verbose=True):
    """Two-stage continuation: solve without cones, then re-solve with them.

    Handing the cone costs an infeasible warm start does not work.  The
    interpolated guess has zero velocity everywhere, so the forces it implies are
    nothing like the ones that carry the motion, and the cone barriers evaluate at
    ~3150 against a total cost of ~2.8 for everything else -- three orders of
    magnitude of the objective is a constraint the guess was never going to
    satisfy.  From there FDDP's backward pass fails at every trial step, the
    regularization walks 1e-8 -> 1e9, and it gives up after 17 iterations having
    moved nothing.  Measured at four cone weights and two min-normal-forces: all
    fail identically.

    Solving the same problem with the cones OFF converges in a handful of
    iterations, and its trajectory is a warm start that already has plausible
    contact forces -- so the cones then enter as a correction rather than as the
    dominant term.  Stage 1 is not a throwaway: it is the only thing that makes
    stage 2 well-posed.
    """
    stage1 = ocp.build(dt=dt, n_approach=n_approach, n_braced=n_braced,
                       impulse=impulse, cones=False, w_terminal=w_terminal)
    s1, ok1, t1 = solve(stage1, ocp, iters=iters, verbose=verbose)
    if not ocp.cones:
        return s1, ok1, t1, None

    stage2 = ocp.build(dt=dt, n_approach=n_approach, n_braced=n_braced,
                       impulse=impulse, cones=True, w_terminal=w_terminal)
    s2 = crocoddyl.SolverBoxFDDP(stage2)
    if verbose:
        s2.setCallbacks([crocoddyl.CallbackVerbose()])
    t0 = time.time()
    ok2 = s2.solve(list(s1.xs), list(s1.us), iters, False, 1e-9)
    return s2, ok2, t1 + (time.time() - t0), (ok1, float(s1.cost))


def solve(problem, ocp, iters=200, verbose=True):
    """FDDP with a warm start that interpolates stand -> q*.

    Warm-starting every node at x0 (the usual crocoddyl boilerplate) is a bad
    initial guess here: the whole point of the problem is a large, slow
    configuration change, so a constant guess puts every node at the far end of
    the very nonlinearity the rollout has to cross, and the line search spends its
    iterations at step 2e-3 without ever leaving the initial point.

    The interpolation has to run over the WHOLE configuration via pin.interpolate,
    not just the 27 joint angles.  Interpolating joints while holding the floating
    base at its stand value swings the legs underneath a pelvis that never
    descends, which lifts both feet ~0.39 m off the floor by the final node -- a
    guess that violates the very contacts the problem is built around, and a worse
    starting point than the constant one it replaced.  pin.interpolate also does
    the right thing on the base quaternion, which a componentwise lerp does not.
    """
    solver = crocoddyl.SolverBoxFDDP(problem)
    if verbose:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    xs = []
    for k in range(problem.T + 1):
        q = pin.interpolate(ocp.rmodel, ocp.q0, ocp.q_star, k / problem.T)
        xs.append(np.concatenate([q, np.zeros(ocp.rmodel.nv)]))
    us = problem.quasiStatic(xs[:-1])
    t0 = time.time()
    ok = solver.solve(xs, us, iters, False, 1e-9)
    return solver, ok, time.time() - t0


def report(solver, ocp):
    """Measure the plan against the things it was supposed to achieve."""
    xs = np.array(solver.xs)
    q_end = xs[-1][:ocp.rmodel.nq]
    v_end = xs[-1][ocp.rmodel.nq:]
    pin.forwardKinematics(ocp.rmodel, ocp.rdata, q_end)
    pin.updateFramePlacements(ocp.rmodel, ocp.rdata)

    out = {"cost": float(solver.cost), "iters": int(solver.iter),
           "stop": float(solver.stop), "is_feasible": bool(solver.isFeasible)}
    out["site_err"] = {s: float(np.linalg.norm(
        ocp.rdata.oMf[ocp.sites[s]].translation - ocp.site_ref[s]))
        for s in ocp.subset}
    if ocp.reach_target is not None:
        out["reach_err"] = float(np.linalg.norm(
            ocp.rdata.oMf[ocp.sites["reach"]].translation - ocp.reach_target))
    # NB the sole frames are registered as "site_sole_left"/"site_sole_right", so
    # they have to be looked up through ocp.sites, not getFrameId(f).
    out["foot_err"] = {f: float(np.linalg.norm(
        ocp.rdata.oMf[ocp.sites[f]].translation - ocp.foot_ref[f].translation))
        for f in FEET}
    out["terminal_vel_norm"] = float(np.linalg.norm(v_end))
    # distance from the certified pose, over the 27 actuated joints only
    out["q_err_vs_qstar_rad"] = float(np.linalg.norm(q_end[7:] - ocp.q_star[7:]))
    out["q_err_max_joint_rad"] = float(np.max(np.abs(q_end[7:] - ocp.q_star[7:])))
    # The impulse node carries nu = 0, so solver.us is a ragged list: one entry is
    # a length-0 array and np.array() on the whole thing gives an object array.
    us = np.array([u for u in solver.us if len(u) == ocp.nu])
    out["max_torque_ratio"] = float(np.max(np.abs(us) / ocp.tau_lim))
    out["torque_ratio_argmax_joint"] = cb.ACTUATED[
        int(np.argmax(np.max(np.abs(us) / ocp.tau_lim, axis=0)))]
    return out
