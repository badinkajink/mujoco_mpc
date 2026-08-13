#!/usr/bin/env python3
"""Replay a crocoddyl plan in MuJoCo under five controllers, score it, render it.

crocoddyl solved the plan against RIGID, PRESCRIBED contacts on a Pinocchio
model.  MuJoCo has soft contacts, a real table it can collide with anywhere, and
position-servo actuators.  Whether the plan survives that is the only question
that matters, and it is not answerable from the solver's own cost.

THE LADDER.  S12 compared two things -- "position tracking" and "torque
inversion" -- found both failed, and concluded that "the missing piece is a
tracking controller".  Neither of those two closes a loop around the BASE, which
is the only loop that matters here, so this runs the whole ladder and lets "does
closing the loop help" be a measurement.  Every mode below sends the command the
real robot takes, (q_des, kp, kd, tau_ff); they differ only in tau_ff and q_des:

  kinematic    no dynamics; drive MuJoCo onto q_plan.  What crocoddyl BELIEVES
               it planned.  Contact forces here are meaningless by construction.
  hold         q_des = the START pose, forever.  The control experiment: it says
               how long this robot stays up doing NOTHING, which turns out to be
               the number every other row has to be read against.
  position     q_des = q_plan, tau_ff = 0.  The servo is the whole controller --
               a naive WBC handoff, with no knowledge of the planned torques.
  ff           q_des = q_plan, tau_ff = tau_plan.  The deployment-faithful
               handoff, and what S12 called "torque inversion" -- note it is NOT
               open-loop torque: the servo's PD term is still underneath it.
               (`torque` is kept as an alias for S12's name.)
  torque_meas  invert the servo at the MEASURED state, which cancels it exactly
               and leaves pure open-loop tau_plan.  The ablation that says how
               much of a replay's survival is the plan and how much is the servo.
  riccati      tau_ff = tau_plan - K_k (x_meas (-) x_plan).  The gains are the
               DDP's OWN Riccati gains, computed on the way to its step and
               otherwise thrown away: a time-varying LQR about the optimal
               trajectory, for free, and the first row that feeds back the base.
  mpc          re-solve a short-horizon OCP from the measured state every control
               step, warm-started from the last solution; apply its first control
               AND its first predicted state as the servo's setpoint.  Refinement
               and stabilisation together.

TWO THINGS THIS GETS RIGHT, both of which have burned this study before:

 1. THE ACTUATORS ARE POSITION SERVOS, not torque sources: tau = kp (ctrl - q)
    - kd qvel, with kp/kd read off the model rather than assumed.  Writing the
    plan's torques into d.ctrl commands a joint angle of a few hundred radians.
 2. THE PLAN IS AT dt = 0.02 AND MuJoCo RUNS AT 0.002.  Each plan node is held
    for the intervening substeps rather than one MuJoCo step per node, which
    would replay a 2 s trajectory in 0.2 s and call the resulting flailing a
    tracking failure.

usage: croco_replay.py --tag s13 [--ctrl riccati] [--video] [--push 120]
"""

import argparse
import json
import os
import time

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL (see that module)
import contact_select as cs
import croco_plan as cp
import mujoco

W, H = 960, 720
CTRLS = ["kinematic", "hold", "position", "ff", "torque", "torque_meas",
         "riccati", "mpc"]

# Per-site marker colours for the rendered contacts.  Same hues as the docpage's
# --s1..--s5 series so a video and a plot of the same run agree about which dot
# is the elbow.
SITE_RGBA = {
    "elbow":   (0.15, 0.45, 0.85, 0.95),
    "forearm": (0.90, 0.55, 0.10, 0.95),
    "palm":    (0.20, 0.65, 0.35, 0.95),
    "hip":     (0.60, 0.35, 0.75, 0.95),
    "torso":   (0.85, 0.25, 0.35, 0.95),
}
FOOT_RGBA = (0.45, 0.45, 0.50, 0.75)
TARGET_RGBA = (0.95, 0.15, 0.45, 0.85)
FORCE_SCALE = 0.0015          # metres of arrow per newton (100 N -> 15 cm)

# UNPLANNED table contacts (2026-08-07).  The first version of this drew only
# contacts that walked up to a known brace SITE and dropped everything else,
# which meant the one thing the grid replays most needed to show was invisible:
# in most cells the largest force on the table does not go through the brace at
# all -- it goes through the torso, or the reaching arm, or the gripper of the
# arm that is not bracing (croco_why quantifies this).  A video that hides those
# contacts shows a robot leaning on nothing and gives no clue why it stays up.
# They are drawn in MAGENTA, one colour for all of them, because the message is
# not "which unplanned part" but "this is not the brace".
UNPLANNED_RGBA = (0.95, 0.25, 0.85, 0.95)

# Camera presets.  `wide` is the S13 framing and is kept so the new videos are
# comparable to the ones already on the docpage; `brace` is a close-up on the
# tabletop, which is the only framing in which a 12 mm contact marker is legible.
CAMERAS = {
    "wide":  dict(lookat=(0.75, 0.00, 0.90), distance=3.0, azimuth=120,
                  elevation=-14),
    "brace": dict(lookat=(0.72, 0.00, 1.05), distance=1.4, azimuth=70,
                  elevation=-20),
    "table": dict(lookat=(0.70, 0.10, 1.02), distance=1.2, azimuth=160,
                  elevation=-15),
}


def show_gripper(m):
    """Make the magpie's collision jaws VISIBLE without giving them dynamics.

    Answering a question this study kept asking of the videos: the gripper's only
    rendered geom is the `h12_mount` mesh (group 1), while the parts that
    actually touch the table -- the 171 mm jaw boxes and the wrist brace pad --
    are class="collision", i.e. group 3, which the default renderer hides.  So
    every replay video showed the arm ending in a stub roughly 100 mm short of
    the hardware that was bracing on the wood.

    Promoting those geoms to group 2 is a RENDER-ONLY change to the MjModel: it
    touches `geom_group` and `geom_rgba`, not contype/conaffinity, mass, or
    anything the integrator reads.  And it is worth saying plainly that there are
    no gripper DOFs to remove -- `left_magpie_gripper` is a rigidly welded body
    with `jntnum = 0`, so it already costs the planner nothing but the frames its
    keep-out points hang off.  Nothing is being traded here for the picture.
    """
    for arm in ("left", "right"):
        for suffix in ("gripper_collision", "gripper_jaw_a", "gripper_jaw_b"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                  f"{arm}_{suffix}")
            if g >= 0:
                m.geom_group[g] = 2
                m.geom_rgba[g] = (0.25, 0.25, 0.28, 1.0)
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{arm}_wrist_pad")
        if g >= 0:
            m.geom_group[g] = 2
            m.geom_rgba[g] = (0.25, 0.25, 0.28, 1.0)


def ghost_arm(m, arm=None, alpha=0.45):
    """Make the BRACING arm translucent in the render.

    The contact points this session cares about are on the UNDERSIDE of the
    elbow, forearm and gripper, i.e. between the link and the tabletop, so a
    marker drawn where the contact actually is gets hidden by the link that is
    making it -- a video of a brace shows an arm on a table and nothing about
    which part of it is loaded.  Dropping the bracing arm's visual alpha lets the
    markers read without moving them somewhere they are not.

    Render-only: `geom_rgba` alone, and only on group 1/2 (visual) geoms, so the
    collision geometry, the mass and the contact model are untouched.
    """
    arm = arm or cs.BRACE_ARM
    links = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
             "wrist_roll", "wrist_pitch", "wrist_yaw"]
    bodies = {cs.bid(m, f"{arm}_{l}_link") for l in links}
    bodies.add(cs.bid(m, f"{arm}_magpie_gripper"))
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in bodies and m.geom_group[g] in (1, 2):
            m.geom_rgba[g][3] = alpha


def _marker(scene, pos, rgba, size=0.02, gtype=mujoco.mjtGeom.mjGEOM_SPHERE):
    if scene.ngeom >= scene.maxgeom:
        return 0
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype,
                        np.full(3, size), np.asarray(pos, float),
                        np.eye(3).ravel(), np.array(rgba, np.float32))
    scene.ngeom += 1
    return 1


def draw_refs(scene, target, site_ref):
    """The reach target and the certified brace landing spots, as ghost markers.

    Without these a viewer can see the hand move but not whether it arrived, and
    "the brace is 10 mm above where it was certified" -- the S12 defect this
    whole session turns on -- is invisible in a render that draws only what the
    robot is doing and not what it was aiming at.
    """
    n = _marker(scene, target, TARGET_RGBA, 0.022)
    for s, p in site_ref.items():
        rgba = list(SITE_RGBA.get(s, FOOT_RGBA))
        rgba[3] = 0.55                       # ghost: this is a reference, not a fact
        n += _marker(scene, p, rgba, 0.012)
    return n


def draw_contacts(scene, m, d, site_body, tbl, feet):
    """Mark the live table contacts on the rendered scene, coloured by identity.

    MuJoCo's own mjVIS_CONTACTPOINT/CONTACTFORCE flags draw every contact in one
    colour, which in this maneuver is 8 foot corners plus whatever the arm is
    doing -- and the whole question the videos are being asked is WHICH of the
    elbow, forearm and palm is carrying load at any moment.  So the markers are
    drawn per contact instead: a sphere at the contact point in the site's own
    colour, and an arrow along the contact normal whose length is the normal
    force.  Foot contacts get the same treatment in grey, because "is the robot
    still on its heels" is the other thing a viewer wants from the picture.

    Returns the number of geoms added, so the caller can size max_geom.
    """
    buf = np.zeros(6)
    added = 0
    # `hip` and `torso` are both sites on torso_link, so a naive inversion loses
    # one of them.  The ARM sites are the ones this render exists for, so they
    # are written last and win the collision.
    body_site = {}
    for s in ("hip", "torso", "elbow", "forearm", "palm"):
        if s in site_body:
            body_site[site_body[s]] = s
    for c in range(d.ncon):
        if scene.ngeom + 2 > scene.maxgeom:
            break
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if tbl in (b1, b2):
            rb = b2 if b1 == tbl else b1
            rgba = None
            while rb > 0 and rgba is None:            # walk up to a known site
                rgba = SITE_RGBA.get(body_site.get(rb))
                rb = m.body_parentid[rb]
            if rgba is None:
                rgba = UNPLANNED_RGBA
        elif b1 in feet or b2 in feet:
            rgba = FOOT_RGBA
        else:
            continue
        mujoco.mj_contactForce(m, d, c, buf)
        fn = abs(float(buf[0]))
        p = con.pos.copy()
        added += _marker(scene, p, rgba, 0.018)
        if fn > 1.0:
            # frame[0] is the contact NORMAL, pointing from geom[0] into geom[1]
            # (checked against a resting contact: table/object reads +z).  The
            # arrow shows the force ON THE ROBOT, so it runs along +normal when
            # the robot is the second geom and -normal when it is the first.
            # Getting this from "is geom[0] the table" instead does not work for
            # the FEET, whose partner is `world`, not `table` -- their arrows
            # then point down through the floor.
            n = con.frame[:3].copy()
            if not _under(m, b2, 1):
                n = -n
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                                np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.008,
                                 p, p + n * (FORCE_SCALE * fn))
            scene.ngeom += 1
            added += 1
    return added


def servo_gains(m):
    """(kp, kd) per actuator, read off the MJCF position actuators."""
    kp = m.actuator_gainprm[:m.nu, 0].copy()
    kd = -m.actuator_biasprm[:m.nu, 2].copy()
    if not np.allclose(-m.actuator_biasprm[:m.nu, 1], kp):
        raise RuntimeError("actuators are not the expected position servos "
                           "(biasprm[1] != -gainprm[0]); check the model")
    return kp, kd


def support_margin(m, d, contacts_only=True):
    """Signed distance from the CoM to the edge of the FOOT support polygon [m].

    Deliberately the foot polygon and not stability.equilibrium_region: this is
    evaluated on every one of ~200 frames of every replay, and the region is 36
    LP ray-shoots.  The region is the certificate and croco_score computes it on
    the frames that matter; this is the cheap running signal, and on the
    feet-only phases the two agree about what "leaving support" means.
    """
    pts = []
    for f in cs.FEET:
        for body, off in cs.foot_corners(m, d, f):
            pts.append(cs.point_world(m, d, body, off)[:2])
    pts = np.array(pts)
    # convex hull of 8 coplanar points -> axis-aligned is exact for two rectangles
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = d.subtree_com[1][:2]
    inside = np.all((c > lo) & (c < hi))
    dist = min(np.min(c - lo), np.min(hi - c))
    return float(dist if inside else -abs(dist))


class MPC:
    """Receding-horizon crocoddyl about the plan's own models.

    The horizon slides over the SAME action models the plan was built from, so
    every cost the MPC sees -- landing spots, the reach target, the cones, the
    keep-out -- is the one the plan was solved against.  Nothing is re-authored
    for the online problem, which is the point: this measures what re-solving
    from the measured state buys, not what a different cost function buys.

    `iters` is small on purpose.  A warm-started DDP that takes one or two
    iterations per control step is the standard MPC construction (the previous
    solution is a very good guess when the state has moved 20 ms), and it is what
    makes the per-step solve time reportable rather than embarrassing.
    """

    def __init__(self, ocp, models, terminal, horizon=40, iters=2,
                 xs_plan=None, us_plan=None, n_alphas=0):
        cro = cb.import_crocoddyl()
        self.n_alphas = n_alphas
        self.ocp, self.models, self.terminal = ocp, models, terminal
        self.H = min(horizon, len(models))
        self.iters = iters
        self.xs_plan, self.us_plan = xs_plan, us_plan
        self.solve_times = []
        # Line-search diagnostics.  FDDP's forward pass is a full nonlinear
        # rollout PER TRIAL STEP, and crocoddyl's default alpha ladder has 10
        # rungs, so a step that walks to the bottom costs ten rollouts -- as much
        # again as the backward pass it followed.  Whether that happens is not
        # inferable from the mean solve time, so it is recorded.
        self.step_lengths = []
        # ONE problem, rotated with circularAppend, not a fresh ShootingProblem
        # per control step.  Rebuilding costs an allocateData over the whole
        # horizon every 20 ms, and with the box keep-out that is 25 nodes x 86
        # Python activation datas -- measured at 336 ms per step, seventeen times
        # the control period, which makes the "is this real-time" question
        # unanswerable for reasons that have nothing to do with the solve.
        self.datas = [m.createData() for m in models]
        self.problem = cro.ShootingProblem(
            ocp.x0, list(models[:self.H]), terminal)
        self.solver = self._make_solver(cro, self.problem)
        self.head = self.H            # index of the next model to append
        self.xs = self.us = None

    def _make_solver(self, cro, problem):
        """A BoxFDDP with, optionally, a TRUNCATED line-search ladder.

        FDDP's forward pass is a full nonlinear rollout per trial step, and
        crocoddyl's default ladder is ten rungs down to alpha = 2^-9.  Measured
        here the median step is 0.125-0.1875, i.e. ~4 rollouts -- but the tail of
        the distribution is what decides whether a control period is met, and the
        p95 is 40% above the mean for exactly this reason.  Capping the ladder at
        `n_alphas` rungs BOUNDS the rollouts per iteration, which is the shape a
        real-time budget wants: a step that would have needed a tenth of a rung
        is not taken at all, the regularisation goes up, and the controller
        re-applies its shifted previous solution for that period.  Whether that
        costs anything is the `alphas` column of croco_speed.py sweep.
        """
        solver = cro.SolverBoxFDDP(problem)
        if self.n_alphas:
            solver.alphas = [2.0 ** -i for i in range(self.n_alphas)]
        return solver

    def __call__(self, k, x_meas):
        """Returns (first control, first predicted state) or (None, None)."""
        if k >= len(self.models):
            return None, None
        if k + self.H <= len(self.models):
            # Window still fits: slide it by rotating the existing problem.
            while self.head < k + self.H:
                self.problem.circularAppend(self.models[self.head],
                                            self.datas[self.head])
                self.head += 1
        else:
            # TAIL: the window would run past the end, and circularAppend can
            # only rotate, never shrink.  Left alone the window simply stops
            # sliding, so the MPC keeps solving a stale set of models while the
            # robot advances past them -- and for H equal to the full horizon it
            # never slides at all, which is not "MPC with a long horizon" but
            # "restart the whole maneuver every 20 ms".  Measured: that
            # configuration stands still and misses the target by 856 mm.  So the
            # tail rebuilds a shrinking-horizon problem instead; it costs an
            # allocateData per step over the last H steps only.
            cro = cb.import_crocoddyl()
            self.problem = cro.ShootingProblem(
                x_meas, list(self.models[k:]), self.terminal)
            self.solver = self._make_solver(cro, self.problem)
            self.head = len(self.models)
        self.problem.x0 = x_meas
        H = self.problem.T
        if self.xs is None:
            # First call: warm-start from the OFFLINE PLAN, not from a constant.
            # The plan is the best guess that exists for this window, and one
            # DDP iteration from a constant guess is not an MPC, it is noise.
            xs = [x_meas] + [np.array(x) for x in self.xs_plan[k + 1:k + H + 1]]
            us = [np.array(u) for u in self.us_plan[k:k + H]]
        else:                                   # shift the previous solution
            xs = [x_meas] + list(self.xs[2:]) + [self.xs[-1]]
            us = list(self.us[1:]) + [self.us[-1]]
        while len(xs) < H + 1:
            xs.append(xs[-1])
        while len(us) < H:
            us.append(us[-1])
        xs, us = xs[:H + 1], us[:H]
        t0 = time.time()
        self.solver.solve(xs, us, self.iters, False, 1e-9)
        self.solve_times.append(time.time() - t0)
        self.step_lengths.append(float(self.solver.stepLength))
        self.xs, self.us = list(self.solver.xs), list(self.solver.us)
        return np.array(self.solver.us[0]), np.array(self.solver.xs[1])


def certified_sites(plan, run_dir, m, d):
    """World positions of the bracing sites AT q*, for the render's ghost markers.

    Evaluated in MuJoCo off the saved q*, then the state is restored -- so this
    is the same number the static QP certified and the same one the OCP's
    `land_`/`hold_` costs are written against, read out of the model the replay
    is about to run rather than re-derived.
    """
    if not plan["subset"]:
        return {}
    modes = json.load(open(os.path.join(run_dir, "modes.json")))
    entry = next(e for e in modes["modes"] if e["name"] == plan["mode"])
    q_star = np.loadtxt(os.path.join(run_dir, entry["qpos_file"]))
    keep_q, keep_v = d.qpos.copy(), d.qvel.copy()
    d.qpos[:len(q_star)] = q_star
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    ref = {s: cs.point_world(m, d, *cs.SITES[s]).copy() for s in plan["subset"]}
    d.qpos[:], d.qvel[:] = keep_q, keep_v
    mujoco.mj_forward(m, d)
    return ref


def build_ocp(plan, run_dir):
    """Rebuild the OCP the plan came from, for the MPC and the Riccati state map."""
    modes = json.load(open(os.path.join(run_dir, "modes.json")))
    entry = next(e for e in modes["modes"] if e["name"] == plan["mode"])
    q_star = np.loadtxt(os.path.join(run_dir, entry["qpos_file"]))
    m, d = cs.load()
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, plan["start"])
    q0 = m.key_qpos[kid].copy()
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    ocp = cp.LeanOCP(plan["subset"], q_star, q0, mu=plan["mu"],
                     table_z=cs.table_top_z(m, d),
                     reach_target=np.array(plan["target"]),
                     cones=plan["cones"], legacy=plan.get("legacy", False),
                     keepout=bool(plan.get("keepout", 0)),
                     com_margin=plan.get("com_margin", 0.03),
                     w_land=plan.get("w_land", 5e3), w_hold=plan.get("w_hold", 1e3),
                     w_com=plan.get("w_com", 1e3))
    return ocp, q_star


def replay(tag, ctrl_mode="riccati", dt_plan=0.02, run_dir=".", video=None,
           fps=30, speed=1.0, push=0.0, push_at=0.5, push_dir=(1, 0, 0),
           push_hold=0.2, mpc_horizon=40, mpc_iters=2, seed=None, q_noise=0.0,
           cam="wide", ghost=True, qpos_out=None, mpc_cones=True,
           mpc_alphas=0):
    xs = np.load(os.path.join(run_dir, f"xs_{tag}.npy"))
    us = np.load(os.path.join(run_dir, f"us_{tag}.npy"))
    with open(os.path.join(run_dir, f"plan_{tag}.json")) as fh:
        plan = json.load(fh)
    subset = plan["subset"]
    Kpath = os.path.join(run_dir, f"K_{tag}.npy")
    Ks = np.load(Kpath) if os.path.exists(Kpath) else None
    if ctrl_mode == "riccati" and Ks is None:
        raise SystemExit(f"{Kpath} missing -- re-run croco_run.py to save gains")

    ocp = mpc = state = None
    if ctrl_mode in ("riccati", "mpc"):
        ocp, _ = build_ocp(plan, run_dir)
        state = ocp.state
        if ctrl_mode == "mpc":
            # The ONLINE problem may drop the cone costs while the OFFLINE plan
            # keeps them.  That is a real change to what the MPC optimises and it
            # is opt-in for that reason -- but it is the largest remaining item
            # in the step: the five cone terms are 36 us of a 131 us braced node,
            # and they are also the only costs that read the contact-force
            # derivatives, so dropping them lets `enable_force` go off too, for
            # 50 us total.  The plan the horizon slides over was solved WITH the
            # cones, so the trajectory being tracked is still cone-feasible; the
            # question this makes measurable is whether re-solving needs to
            # re-check that every 20 ms.  `--mpc-no-cones` and the (H, iters,
            # cones) grid in croco_speed.py sweep report what it costs.
            problem = ocp.build(dt=plan["dt"], n_approach=plan["n_approach"],
                                n_braced=plan["n_braced"],
                                n_return=plan.get("n_return", 0),
                                dwell=plan.get("dwell", 0),
                                cones=plan["cones"] and mpc_cones)
            mpc = MPC(ocp, list(problem.runningModels), problem.terminalModel,
                      horizon=mpc_horizon, iters=mpc_iters,
                      xs_plan=xs, us_plan=us, n_alphas=mpc_alphas)

    # NO inflated collision margin.  cs.load defaults to margin = 25 mm so the
    # IK can see collisions before it enters them; in a DYNAMICS replay that
    # same inflation makes every geom generate contact forces 25 mm before it
    # touches anything, which is not the contact model the plan is being tested
    # against.  S12's replays ran with it on.
    m, d = cs.load(ik_margin=0.0)
    kp, kd = servo_gains(m)
    nsub = max(1, int(round(dt_plan / m.opt.timestep)))

    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, plan["start"])
    qpos_full = m.key_qpos[kid].copy()
    nq = cb.NQ_ROBOT
    d.qpos[:] = cb.pin_to_mj(xs[0][:nq], qpos_full)
    d.qvel[:] = 0
    if q_noise:
        rng = np.random.default_rng(seed)
        d.qpos[7:nq] += rng.normal(0.0, q_noise, nq - 7)
    mujoco.mj_forward(m, d)

    renderer = frames = None
    if video:
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H)
        show_gripper(m)
        if ghost:
            ghost_arm(m)
        # +256 geoms of headroom for the contact markers (2 per contact).  The
        # default max_geom is sized from the model, so appending to the scene
        # without this silently drops the markers once the arm is down and there
        # are a dozen live contacts -- exactly the frames they exist for.
        renderer = mujoco.Renderer(m, H, W, max_geom=m.ngeom + 256)
        frames = []
        frame_every = max(1, int(round(speed / (fps * dt_plan))))
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        preset = CAMERAS[cam]
        camera.lookat[:] = preset["lookat"]
        camera.distance = preset["distance"]
        camera.azimuth = preset["azimuth"]
        camera.elevation = preset["elevation"]

    tbl = cs.bid(m, "table")
    brace_bodies = {s: cs.bid(m, cs.SITES[s][0]) for s in subset}
    # For the render: EVERY arm site, not just the ones in the contact schedule.
    # A video that only marks the planned contacts cannot show the failure this
    # study spent S12 chasing -- a link that is touching the table while the plan
    # says it is not.
    all_site_bodies = {s: cs.bid(m, cs.SITES[s][0]) for s in SITE_RGBA
                       if s in cs.SITES}
    feet = [cs.bid(m, f) for f in cs.FEET]
    tau_lim = cs.torque_limits(m)
    target = np.array(plan["target"])
    site_ref = certified_sites(plan, run_dir, m, d) if video else {}
    push_k = int(round(push_at * len(us)))
    log = []
    # Full qpos trace, saved alongside the log as .npy.  The JSON log carries
    # scalars only, which is enough to say a replay reached the target and not
    # enough to ask anything else of the pose it reached -- and the CoM-region
    # comparison (croco_region.py) needs the POSE, not a summary of it.  41
    # doubles x ~200 nodes is 66 kB of binary, so it is saved for every replay
    # rather than for the ones that turn out to be interesting.
    qtrace = []

    for k in range(len(us)):
        q_plan = xs[k][:nq]
        v_plan = xs[k][nq:]
        # Pinocchio joint order == MJCF joint order for the 27 actuated joints,
        # and both put them after the free joint, so the slice is direct.
        qj, vj = q_plan[7:], v_plan[6:]

        if ctrl_mode == "kinematic":
            d.qpos[:] = cb.pin_to_mj(q_plan, qpos_full)
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            tau_cmd = us[k]
        else:
            # THE COMMAND THE ROBOT ACTUALLY TAKES.  h12_control_node sends
            # (q_des, kp, kd, tau_ff) and the motor driver forms
            #     tau = kp (q_des - q) + kd (v_des - v) + tau_ff
            # downstream of it.  A MuJoCo <position> actuator emits
            #     tau = kp (ctrl - q) - kd v,
            # so the two are identical under
            #     ctrl = q_des + (tau_ff + kd v_des) / kp,
            # which is the inversion S12 called "torque" -- and the important
            # consequence is that it is NOT open-loop torque: the servo's own
            # PD term is still there, doing the stabilising.  Every mode below
            # except `torque_meas` is that command with a different tau_ff, so
            # the ladder isolates the feedforward and leaves the interface fixed.
            q_des, v_des = qj, vj
            if ctrl_mode == "hold":
                q_des, v_des = xs[0][7:nq], np.zeros(nq - 7)
                tau_ff = np.zeros(m.nu)
            elif ctrl_mode == "position":
                tau_ff = np.zeros(m.nu)
            elif ctrl_mode in ("ff", "torque"):
                tau_ff = us[k]
            else:
                x_meas = np.concatenate([
                    cb.mj_to_pin(d.qpos),
                    cb.mj_to_pin_v(d.qvel, d.xmat[1].reshape(3, 3))])
                if ctrl_mode == "torque_meas":
                    tau_ff = us[k]
                elif ctrl_mode == "riccati":
                    # u = u* - K (x (-) x*).  crocoddyl's K is defined with that
                    # sign; state.diff(x*, x) is the tangent step FROM the plan TO
                    # the measurement, which is the argument the gain expects.
                    dx = state.diff(xs[k], x_meas)
                    tau_ff = us[k] - Ks[min(k, len(Ks) - 1)] @ dx
                else:                                    # mpc
                    u0, xs1 = mpc(k, x_meas)
                    if u0 is None:
                        tau_ff = us[k]
                    else:
                        tau_ff = u0
                        # The MPC re-plans the reference too: its own next state
                        # is a better setpoint for the servo than the offline
                        # plan's, and using it is what "refinement" means here.
                        q_des, v_des = xs1[:nq][7:], xs1[nq:][6:]
                tau_ff = np.clip(tau_ff, -tau_lim, tau_lim)
                if ctrl_mode == "torque_meas":
                    # ABLATION: invert at the MEASURED state instead, which
                    # cancels the servo exactly and leaves pure open-loop torque.
                    # Kept because it is the only way to see how much of a
                    # replay's survival is the plan and how much is the servo.
                    d.ctrl[:] = (d.qpos[7:nq]
                                 + (tau_ff + kd * d.qvel[6:nq - 1]) / kp)
                    q_des = None
            tau_cmd = tau_ff
            if q_des is not None:
                d.ctrl[:] = q_des + (tau_ff + kd * v_des) / kp
            # DISTURBANCE.  A sustained horizontal force on the pelvis, held for
            # `push_hold` seconds, rather than a velocity impulse on the base.
            # The impulse version is nearly free to reject -- both feet are on the
            # ground, so the added base velocity is absorbed within a step, and
            # 0.5 m/s changed the reach error by 1 mm -- while a force in newtons
            # is directly comparable to stability.max_push, the study's own
            # "how hard can something shove this pose" metric.
            if push:
                on = push_k <= k < push_k + max(1, int(round(push_hold / dt_plan)))
                d.xfrc_applied[1, 0:3] = np.asarray(push_dir, float) * push if on \
                    else 0.0
            for _ in range(nsub):
                mujoco.mj_step(m, d)

        reach_p = cs.point_world(m, d, cs.REACH_BODY, cs.REACH_OFF)
        rec = dict(k=k, t=k * dt_plan,
                   pelvis_z=float(d.qpos[2]),
                   tracking_err=float(np.linalg.norm(d.qpos[7:nq] - qj)),
                   reach_err=float(np.linalg.norm(reach_p - target)),
                   com=[float(v) for v in d.subtree_com[1]],
                   support_margin=support_margin(m, d),
                   tau_ratio=float(np.max(np.abs(d.actuator_force[:m.nu]) / tau_lim)),
                   cmd_ratio=float(np.max(np.abs(tau_cmd) / tau_lim)))
        f_brace = {s: 0.0 for s in subset}
        f_feet = 0.0
        deepest = 0.0
        buf = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
            mujoco.mj_contactForce(m, d, c, buf)
            fn = abs(float(buf[0]))
            if tbl in (b1, b2):
                rb = b2 if b1 == tbl else b1
                if _under(m, rb, 1):
                    deepest = min(deepest, float(con.dist))
            for s, bid_ in brace_bodies.items():
                if bid_ in (b1, b2) and tbl in (b1, b2):
                    f_brace[s] += fn
            if b1 in feet or b2 in feet:
                f_feet += fn
        rec.update({f"F_{s}": f_brace[s] for s in subset})
        rec["F_feet"] = f_feet
        rec["F_brace_total"] = float(sum(f_brace.values()))
        rec["penetration"] = deepest
        log.append(rec)
        qtrace.append(d.qpos.copy())

        if renderer is not None and k % frame_every == 0:
            renderer.update_scene(d, camera=camera)
            draw_refs(renderer.scene, target, site_ref)
            draw_contacts(renderer.scene, m, d, all_site_bodies, tbl, feet)
            frames.append(renderer.render())

    if renderer is not None:
        import imageio.v2 as imageio
        imageio.mimsave(video, frames, fps=fps, quality=8, macro_block_size=1)
        print(f"wrote {video}  ({len(frames)} frames)")

    if qpos_out:
        np.save(qpos_out, np.array(qtrace))

    if mpc is not None:
        sl = np.array(mpc.step_lengths)
        plan["mpc_solve_ms"] = dict(
            mean=float(1000 * np.mean(mpc.solve_times)),
            p95=float(1000 * np.percentile(mpc.solve_times, 95)),
            horizon=mpc.H, iters=mpc.iters,
            step_length_median=float(np.median(sl)),
            step_length_min=float(sl.min()),
            # A rung of crocoddyl's alpha ladder is a rollout, so log2(1/alpha)
            # rollouts were thrown away before one was accepted.
            trials_median=float(np.median(np.log2(1.0 / sl)) + 1),
            frac_full_step=float(np.mean(sl >= 1.0)))
    return log, plan


def _under(m, b, root):
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


def summarise(log, plan, ctrl_mode, verbose=True):
    last = log[-1]
    n_ret = plan.get("n_return", 0)
    k_score = len(log) - 1 - n_ret          # end of the braced phase
    braced = log[plan["n_approach"]:len(log) - n_ret]
    fell = min(r["pelvis_z"] for r in log) < 0.55
    res = dict(
        ctrl=ctrl_mode, fell=bool(fell),
        final_pelvis_z=last["pelvis_z"],
        min_pelvis_z=min(r["pelvis_z"] for r in log),
        max_tracking_err=max(r["tracking_err"] for r in log),
        final_tracking_err=last["tracking_err"],
        reach_err_at_brace_end=log[k_score]["reach_err"],
        reach_err_best=min(r["reach_err"] for r in braced) if braced else None,
        reach_err_braced_mean=float(np.mean([r["reach_err"] for r in braced]))
        if braced else None,
        reach_err_braced_std=float(np.std([r["reach_err"] for r in braced]))
        if braced else None,
        brace_force_peak={s: max(r[f"F_{s}"] for r in log) for s in plan["subset"]},
        brace_force_braced_mean={
            s: float(np.mean([r[f"F_{s}"] for r in braced])) if braced else 0.0
            for s in plan["subset"]},
        brace_total_braced_mean=float(np.mean([r["F_brace_total"] for r in braced]))
        if braced else 0.0,
        feet_force_final=last["F_feet"],
        min_support_margin=min(r["support_margin"] for r in log),
        support_margin_at_brace_end=log[k_score]["support_margin"],
        max_tau_ratio=max(r["tau_ratio"] for r in log),
        worst_penetration=min(r["penetration"] for r in log),
    )
    if "mpc_solve_ms" in plan:
        res["mpc_solve_ms"] = plan["mpc_solve_ms"]
    if not verbose:
        return res
    print(f"\n--- MuJoCo replay ({ctrl_mode}) ---")
    print(f"  steps                {len(log)}   ({last['t']:.2f} s)")
    print(f"  pelvis z  final      {res['final_pelvis_z']:.3f} m   "
          f"min {res['min_pelvis_z']:.3f} m   "
          f"{'TOPPLED' if fell else 'upright'}")
    print(f"  joint tracking error final {res['final_tracking_err']:.4f} rad, "
          f"max {res['max_tracking_err']:.4f} rad")
    print(f"  reach error at brace end   {res['reach_err_at_brace_end']*1000:7.1f} mm"
          f"   (best over braced phase {res['reach_err_best']*1000:.1f} mm)")
    for s in plan["subset"]:
        print(f"  brace {s:9s} force  braced-mean "
              f"{res['brace_force_braced_mean'][s]:7.1f} N   "
              f"peak {res['brace_force_peak'][s]:7.1f} N")
    print(f"  feet normal force    final {res['feet_force_final']:7.1f} N   "
          f"(robot weight 673.6 N)")
    print(f"  CoM support margin   min {res['min_support_margin']*1000:+7.1f} mm  "
          f"at brace end {res['support_margin_at_brace_end']*1000:+7.1f} mm")
    print(f"  max |tau|/limit      {res['max_tau_ratio']:.3f} (measured)")
    print(f"  worst table penetration {res['worst_penetration']*1000:7.1f} mm")
    if "mpc_solve_ms" in res:
        s = res["mpc_solve_ms"]
        print(f"  MPC solve            mean {s['mean']:.1f} ms, p95 {s['p95']:.1f} ms "
              f"(H={s['horizon']}, {s['iters']} iters, control period "
              f"{plan['dt']*1000:.0f} ms)")
        if "step_length_median" in s:
            print(f"  MPC line search      median step {s['step_length_median']:.4f}"
                  f" ({s['trials_median']:.0f} rollouts), "
                  f"{100*s['frac_full_step']:.0f}% took the full step")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="elbow+forearm")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ctrl", default="riccati", choices=CTRLS)
    ap.add_argument("--dir", default="runs/2026-08-06_session13")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--push", type=float, default=0.0,
                    help="horizontal force [N] on the pelvis, mid-trajectory")
    ap.add_argument("--push-at", type=float, default=0.5,
                    help="fraction of the trajectory at which the push starts")
    ap.add_argument("--push-hold", type=float, default=0.2,
                    help="how long the push is held [s]")
    ap.add_argument("--push-dir", default="-1,0,0",
                    help="unit direction of the push; default -x, i.e. AWAY from "
                         "the table, which is the direction the plant already "
                         "falls (the planned CoP rides the heel edge)")
    ap.add_argument("--q-noise", type=float, default=0.0,
                    help="std [rad] of joint noise on the initial state")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mpc-horizon", type=int, default=40)
    ap.add_argument("--mpc-iters", type=int, default=2)
    ap.add_argument("--mpc-alphas", type=int, default=0,
                    help="truncate the FDDP line-search ladder to this many "
                         "rungs (0 = crocoddyl's default 10).  Bounds the "
                         "rollouts per iteration, and so the worst-case step.")
    ap.add_argument("--mpc-no-cones", action="store_true",
                    help="drop the friction/wrench cone costs from the ONLINE "
                         "problem only (the plan keeps them).  Also switches "
                         "enable_force off, since the cones are the only costs "
                         "that read the contact-force derivatives.")
    ap.add_argument("--cam", default="wide", choices=sorted(CAMERAS),
                    help="camera preset for --video")
    ap.add_argument("--no-ghost", action="store_true",
                    help="keep the bracing arm opaque in the video (the default "
                         "makes it translucent so the contact markers under it "
                         "are visible)")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    tag = args.tag or args.mode.replace("+", "_")
    name = f"replay_{tag}_{args.ctrl}{args.suffix}"
    vid = os.path.join(args.dir, name + ".mp4") if args.video else None
    log, plan = replay(tag, ctrl_mode=args.ctrl, run_dir=args.dir, video=vid,
                       qpos_out=os.path.join(args.dir, name + "_q.npy"),
                       fps=args.fps, speed=args.speed, push=args.push,
                       push_at=args.push_at, push_hold=args.push_hold,
                       push_dir=[float(v) for v in args.push_dir.split(",")],
                       seed=args.seed, q_noise=args.q_noise,
                       mpc_horizon=args.mpc_horizon, mpc_iters=args.mpc_iters,
                       mpc_cones=not args.mpc_no_cones,
                       mpc_alphas=args.mpc_alphas,
                       cam=args.cam, ghost=not args.no_ghost,
                       dt_plan=plan_dt(args.dir, tag))
    res = summarise(log, plan, args.ctrl)
    out = os.path.join(args.dir, name + ".json")
    with open(out, "w") as fh:
        json.dump({"mode": plan["mode"], "tag": tag, "ctrl": args.ctrl,
                   "push": args.push, "push_hold": args.push_hold,
                   "q_noise": args.q_noise, "seed": args.seed,
                   "summary": res, "log": log}, fh, indent=1)
    print(f"\nwrote {out}")


def plan_dt(run_dir, tag):
    with open(os.path.join(run_dir, f"plan_{tag}.json")) as fh:
        return json.load(fh)["dt"]


if __name__ == "__main__":
    main()
