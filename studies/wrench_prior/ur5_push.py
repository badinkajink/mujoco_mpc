"""UR5 + MAGPIE wrench-controlled push, predictive-sampling MPC, in Python.

The model is the MJPC `UR5 Magpie` task (build_cmake/mjpc/tasks/ur5/task_magpie.xml):
the arm joints are passive (damping 2, gravcomp) and the six controls are a
Cartesian wrench applied at the `eeff` site, +-10 N per force axis and +-1 N.m per
torque axis. So the planner's decision variable IS the wrist wrench, which is the
object "wrench prompting" talks about.

Two models are built from one spec: the PLANT carries the object's true mass and
floor friction; the PLANNER carries the belief (m_hat, mu_hat) and the wrench cap
F_max in its cost. The gap between the two is the quantity a VLM prior has to
close, and this file exists to measure how much that gap costs.

Predictive sampling follows mjpc/planners/sampling: K knots over the horizon,
linear interpolation, N Gaussian samples around the nominal, keep the best,
shift the nominal by one plan period. Rollouts are mujoco.rollout on the planner
model from the plant's state.
"""
from __future__ import annotations

import dataclasses
import math
import os
import time

import mujoco
import mujoco.rollout as mj_rollout
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
BUILT_XML = os.path.join(ROOT, "build_cmake/mjpc/tasks/ur5/task_magpie.xml")
G = 9.81


@dataclasses.dataclass
class Obj:
    """The pushed object as one model has it."""
    mass: float = 0.5           # kg
    mu: float = 0.4             # sliding friction against the floor (geom priority 1)
    half: tuple = (0.03, 0.03, 0.10)   # box half extents, m (a 6x6x20 cm bottle)
    f_break: float = math.inf   # N of gripper->object contact force that counts as damage
    solref: float = 0.01        # contact time constant (s) of the object's geom

    @property
    def f_slide(self):
        return self.mu * self.mass * G

    def f_tip(self, h_contact):
        """Horizontal force at contact height h (above the floor) that tips the box."""
        return self.mass * G * self.half[0] / max(h_contact, 1e-3)


@dataclasses.dataclass
class Cfg:
    horizon: float = 1.0        # s
    plan_dt: float = 0.03       # s between plans (10 plant steps of 3 ms)
    knots: int = 4
    n_samples: int = 64
    sigma: float = 0.35         # ctrl units (ctrl in [-1, 1]); 0.35 = 3.5 N / 0.35 N.m
    nthread: int = 4
    seed: int = 0
    # cost weights (MJPC style: weight x norm(residual))
    w_bring: float = 20.0       # box centre -> target (xy), smooth-abs
    w_reach: float = 10.0       # hand -> push point, smooth-abs
    w_upright: float = 30.0     # 1 - box z-axis . world z
    w_orient: float = 10.0      # 1 + eeff z-axis . world z (gripper pointing down)
    w_ctrl: float = 0.1         # |u|^2 on the six wrench controls
    w_cap: float = 40.0         # one-sided: max(0, F_touch_pred - f_max)^2 (F in N; scaled by /10)
    k_bring_quad: float = 5.0   # adds k*err^2 to the smooth-abs bring residual
    w_terminal: float = 20.0    # last-step multiplier on bring + box velocity
    w_vel: float = 60.0         # |box linvel|^2 (the overshoot term: a 0.5 kg box at 0.3 m/s slides 1 cm past)
    w_hvel: float = 2.0         # |hand linvel|^2 (impact term)
    f_max: float = math.inf     # N, the planner-side wrench ceiling
    # push geometry: where the hand should sit relative to the box
    d_behind: float = 0.020     # m behind the box face along the push direction (plate half 0.008 + 12 mm gap)
    z_hand: float = 0.010       # m, TCP height above the floor (plate spans TCP+0.003 .. TCP+0.115)
    # pre-push servo (phase A)
    servo_kp: float = 8.0       # ctrl units per m (x10 N/m)
    servo_kd: float = 3.0       # ctrl units per m/s
    servo_kp_rot: float = 3.0   # ctrl units per rad (x1 N.m/rad)
    servo_kd_rot: float = 0.5   # ctrl units per rad/s
    servo_t_max: float = 6.0
    servo_lift_t: float = 2.0   # s spent travelling at pre_lift height before descending
    pre_lift: float = 0.12      # m above the push point during the lateral approach
    # episode (phase B, the MPC push)
    t_max: float = 8.0
    tol: float = 0.03           # m, box centre within this of the target (xy)
    hold: float = 0.5           # s inside tol to count as done
    push_dist: float = 0.15     # m
    box_xy: tuple = (0.0, 0.45)
    push_dir_deg: float = 0.0   # world yaw of the push direction


GRIPPER_BODIES = ["gripper_attachment", "base_bot", "base_top", "left_crank",
                  "left_finger_combined", "left_rocker", "right_crank",
                  "right_finger_combined", "right_rocker"]


def build_model(obj: Obj) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(BUILT_XML)
    for g in spec.geoms:
        if g.name == "box":
            g.size = list(obj.half)
            g.mass = obj.mass
            g.friction = [obj.mu, 0.01, 0.003]
            g.density = 0.0
            # a rubber-padded plate on a real bottle is not a 1 ms contact; the
            # task XML's global solref .001 makes every first touch a 50 N spike
            g.solref = [obj.solref, 1.0]
    for s in spec.sites:
        if s.name in ("box1", "box2", "target1", "target2"):
            s.pos = [0, math.copysign(obj.half[1] - 0.004, s.pos[1]), 0]
            s.size = [obj.half[0] + 0.001, 0.005, obj.half[2] + 0.001]
    # Touch sensors on the finger PLATES (the outer faces do the pushing; the
    # existing touch_left/right sites cover only the inner pads). A touch sensor
    # sums the normal force of every contact on its body inside the site volume,
    # and the fingers collide with nothing but the object (world and finger-
    # finger pairs are excluded in the task XML), so plate_left + plate_right is
    # the normal force the gripper puts on the object -- the quantity a wrist
    # F/T sensor reports on the real arm, and the one a fragility bound is about.
    # Sized from the compiled mesh AABB, so it needs one compile to read it.
    # Only the finger plates may touch the object. With the cranks, rockers and
    # base left collidable the planner pushed the box with a rocker to keep the
    # plate touch sensors under the cap (measured: 6.4 N on the box, 0.8 N on the
    # plates). A cap on the object force has to see every contact with it.
    finger_bodies = {"left_finger_combined", "right_finger_combined"}
    for b in spec.bodies:
        if b.name in GRIPPER_BODIES and b.name not in finger_bodies:
            for g in b.geoms:
                g.contype = 0
                g.conaffinity = 0
    # The gripper linkage jams against itself when the fingers are commanded
    # fully closed (crank-rocker 220 N, pad-crank 134 N at ctrl 2.7 rad) and the
    # plate touch sensors would read that internal load as a push. Exclude every
    # pair inside the gripper subtree (finger-finger and finger-world already are).
    for i, b1 in enumerate(GRIPPER_BODIES):
        for b2 in GRIPPER_BODIES[i + 1:]:
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = b1, b2
    m0 = spec.compile()
    for side in ("left", "right"):
        body = f"{side}_finger_combined"
        bid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, body)
        gid = next(g for g in range(m0.ngeom) if m0.geom_bodyid[g] == bid
                   and m0.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m0.geom_contype[g] != 0)
        aabb = m0.geom_aabb[gid]          # centre(3) + half-size(3) in the geom frame
        gq = m0.geom_quat[gid]
        centre_local = np.zeros(3)
        mujoco.mju_rotVecQuat(centre_local, aabb[:3], gq)
        sb = spec.body(body)
        site = sb.add_site()
        site.name = f"plate_{side}"
        site.type = mujoco.mjtGeom.mjGEOM_BOX
        site.pos = (m0.geom_pos[gid] + centre_local).tolist()
        site.quat = gq.tolist()
        site.size = (aabb[3:] + 0.004).tolist()
        site.group = 4
        site.rgba = [1, 0, 0, 0.3]
    for name, typ, ot, on in [
            ("box_linvel", mujoco.mjtSensor.mjSENS_FRAMELINVEL, mujoco.mjtObj.mjOBJ_BODY, "box"),
            ("box_zaxis", mujoco.mjtSensor.mjSENS_FRAMEZAXIS, mujoco.mjtObj.mjOBJ_BODY, "box"),
            ("eeff_zaxis", mujoco.mjtSensor.mjSENS_FRAMEZAXIS, mujoco.mjtObj.mjOBJ_SITE, "eeff"),
            ("hand_linvel", mujoco.mjtSensor.mjSENS_FRAMELINVEL, mujoco.mjtObj.mjOBJ_SITE, "eeff"),
            ("plate_left", mujoco.mjtSensor.mjSENS_TOUCH, mujoco.mjtObj.mjOBJ_SITE, "plate_left"),
            ("plate_right", mujoco.mjtSensor.mjSENS_TOUCH, mujoco.mjtObj.mjOBJ_SITE, "plate_right")]:
        s = spec.add_sensor()
        s.name, s.type, s.objtype, s.objname = name, typ, ot, on
    m = spec.compile()
    return m


class Sens:
    """Sensor addresses in sensordata, resolved once per model."""
    def __init__(self, m):
        def adr(n):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
            return m.sensor_adr[i], m.sensor_dim[i]
        self.hand = adr("hand")[0]
        self.box = adr("box")[0]
        self.target = adr("target")[0]
        self.box1, self.box2 = adr("box1")[0], adr("box2")[0]
        self.target1, self.target2 = adr("target1")[0], adr("target2")[0]
        self.box_linvel = adr("box_linvel")[0]
        self.box_zaxis = adr("box_zaxis")[0]
        self.hand_linvel = adr("hand_linvel")[0]
        self.eeff_zaxis = adr("eeff_zaxis")[0]
        self.plate_left = adr("plate_left")[0]
        self.plate_right = adr("plate_right")[0]


def smooth_abs(x, p=0.01):
    return np.sqrt(x * x + p * p) - p


class Cost:
    def __init__(self, m, cfg: Cfg, obj_half):
        self.s = Sens(m)
        self.cfg = cfg
        self.half = np.asarray(obj_half)

    def push_point(self, box_pos, target_pos):
        """Where the TCP should sit: behind the box along the FIXED push direction, near
        the floor. (Using the box->target bearing here flips the point to the far side
        the moment the box overshoots, and the hand then pushes it further away.)"""
        yaw = math.radians(self.cfg.push_dir_deg)
        r = np.array([math.cos(yaw), math.sin(yaw)])
        pp = np.empty_like(box_pos)
        pp[..., :2] = box_pos[..., :2] - r * (self.half[0] + self.cfg.d_behind)
        pp[..., 2] = self.cfg.z_hand
        return pp

    def evaluate(self, sens, ctrl, breakdown=False):
        """sens: (nb, T, nsensordata); ctrl: (nb, T, nu). Returns (nb,) total cost."""
        s, c = self.s, self.cfg
        hand = sens[..., s.hand:s.hand + 3]
        box = sens[..., s.box:s.box + 3]
        tgt = sens[..., s.target:s.target + 3]
        vel = sens[..., s.box_linvel:s.box_linvel + 3]
        zax = sens[..., s.box_zaxis:s.box_zaxis + 3]
        hvel = sens[..., s.hand_linvel:s.hand_linvel + 3]
        ez = sens[..., s.eeff_zaxis:s.eeff_zaxis + 3]
        orient = 1.0 + ez[..., 2]
        # the task XML's Bring residual: two sites on the box vs two on the target,
        # so the box's yaw is in the cost and an off-centre push gets corrected
        e1 = np.linalg.norm(sens[..., s.box1:s.box1 + 3] - sens[..., s.target1:s.target1 + 3], axis=-1)
        e2 = np.linalg.norm(sens[..., s.box2:s.box2 + 3] - sens[..., s.target2:s.target2 + 3], axis=-1)
        err = 0.5 * (e1 + e2)
        bring = smooth_abs(err) + c.k_bring_quad * err * err
        reach = smooth_abs(np.linalg.norm(hand - self.push_point(box, tgt), axis=-1))
        upright = 1.0 - zax[..., 2]
        u = ctrl[..., :6]
        effort = np.sum(u * u, axis=-1)
        # the cap is on the PREDICTED contact force on the object (plate touch
        # sensors), not on the commanded wrist wrench: with joint damping 2 N.m.s/rad
        # most of the commanded force goes into moving the arm, not the object
        f_obj = sens[..., s.plate_left] + sens[..., s.plate_right]
        cap = np.maximum(0.0, f_obj - c.f_max) / 10.0
        cap = cap * cap
        v2 = np.sum(vel * vel, axis=-1)
        hv2 = np.sum(hvel * hvel, axis=-1)
        per_step = (c.w_bring * bring + c.w_reach * reach + c.w_upright * upright
                    + c.w_orient * orient
                    + c.w_ctrl * effort + c.w_cap * cap + c.w_vel * v2 + c.w_hvel * hv2)
        # terminal emphasis on where the box ENDS UP (a pusher cannot pull it back)
        per_step[..., -1] += c.w_terminal * (c.w_bring * bring[..., -1] + c.w_vel * v2[..., -1])
        if breakdown:
            return {"bring": (c.w_bring * bring).sum(-1), "reach": (c.w_reach * reach).sum(-1),
                    "upright": (c.w_upright * upright).sum(-1), "orient": (c.w_orient * orient).sum(-1),
                    "ctrl": (c.w_ctrl * effort).sum(-1),
                    "cap": (c.w_cap * cap).sum(-1), "vel": (c.w_vel * v2).sum(-1),
                    "hvel": (c.w_hvel * hv2).sum(-1),
                    "terminal": c.w_terminal * (c.w_bring * bring[..., -1] + c.w_vel * v2[..., -1]),
                    "total": per_step.sum(-1)}
        return per_step.sum(axis=-1)


class Planner:
    """Predictive sampling over the six wrench controls; fingers held closed."""
    def __init__(self, m_plan, cfg: Cfg, obj_half, finger_ctrl=(1.5, -1.5)):
        self.m = m_plan
        self.cfg = cfg
        self.cost = Cost(m_plan, cfg, obj_half)
        self.datas = [mujoco.MjData(m_plan) for _ in range(cfg.nthread)]
        self.rng = np.random.default_rng(cfg.seed)
        self.dt = m_plan.opt.timestep
        self.T = int(round(cfg.horizon / self.dt))
        self.K = cfg.knots
        self.knot_t = np.linspace(0.0, cfg.horizon, self.K)
        self.step_t = np.arange(self.T) * self.dt
        self.U = np.zeros((self.K, 6))          # nominal knots, ctrl units
        self.finger = np.array(finger_ctrl)
        self.nstate = mujoco.mj_stateSize(m_plan, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        # mjSTATE_FULLPHYSICS carries no mocap pose, and mujoco.rollout RESETS each
        # data before applying the state, so a mocap_pos written into the thread
        # datas is discarded and every rollout sees the XML's target (0.6 0.4 0.05).
        # The target has to ride along in the control array under a control_spec
        # that includes MOCAP_POS. Measured, not guessed: the sensor came back at the
        # XML value until this was done.
        self.ctrl_spec = mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_MOCAP_POS
        self.nctrl = mujoco.mj_stateSize(m_plan, self.ctrl_spec)
        self.target = None
        self.last = {}
        self.debug = False

    def interp(self, knots):
        """knots (nb, K, 6) -> (nb, T, 6), linear in time."""
        nb = knots.shape[0]
        out = np.empty((nb, self.T, 6))
        for j in range(6):
            for b in range(nb):
                out[b, :, j] = np.interp(self.step_t, self.knot_t, knots[b, :, j])
        return out

    def full_ctrl(self, u6):
        """(nb, T, 6) -> (nb, T, nctrl): fingers appended, then the mocap target."""
        nb, T, _ = u6.shape
        u = np.zeros((nb, T, self.nctrl))
        u[..., :6] = u6
        u[..., 6] = self.finger[0]
        u[..., 7] = self.finger[1]
        u[..., self.m.nu:self.m.nu + 3] = self.target
        return u

    def set_target(self, target):
        self.target = np.asarray(target, dtype=float)

    def plan(self, state):
        c = self.cfg
        assert self.target is not None, "set_target() before plan()"
        eps = self.rng.normal(0.0, c.sigma, size=(c.n_samples, self.K, 6))
        cand = np.concatenate([self.U[None], self.U[None] + eps], axis=0)
        cand = np.clip(cand, -1.0, 1.0)
        u6 = self.interp(cand)
        ctrl = self.full_ctrl(u6)
        init = np.tile(state, (cand.shape[0], 1))
        _, sens = mj_rollout.rollout(self.m, self.datas, init, ctrl,
                                     control_spec=self.ctrl_spec, nstep=self.T)
        J = self.cost.evaluate(sens, ctrl)
        best = int(np.argmin(J))
        self.U = cand[best]
        self.last = {"J_best": float(J[best]), "J_nominal": float(J[0]),
                     "improved": best != 0}
        if self.debug:
            bd = self.cost.evaluate(sens[best:best + 1], ctrl[best:best + 1], breakdown=True)
            self.last["breakdown"] = {k: float(v[0]) for k, v in bd.items()}
        return self.U.copy()

    def action(self, t_since_plan):
        u = np.array([np.interp(t_since_plan, self.knot_t, self.U[:, j]) for j in range(6)])
        return np.concatenate([u, self.finger])

    def shift(self, dt):
        """Advance the nominal spline by dt so the next plan warm-starts from it."""
        t = self.knot_t + dt
        self.U = np.stack([np.interp(t, self.knot_t, self.U[:, j]) for j in range(6)], axis=1)


def contact_wrench_on_box(m, d, box_geom, gripper_bodies):
    """World-frame force the gripper applies to the box (N), summed over contacts."""
    f = np.zeros(3)
    f6 = np.zeros(6)
    n = 0
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = con.geom1, con.geom2
        if g1 == box_geom:
            other, sign = g2, -1.0     # force returned is ON geom2; the box is geom1
        elif g2 == box_geom:
            other, sign = g1, +1.0
        else:
            continue
        if m.geom_bodyid[other] not in gripper_bodies:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fr = con.frame.reshape(3, 3)
        f += sign * (f6[0] * fr[0] + f6[1] * fr[1] + f6[2] * fr[2])
        n += 1
    return f, n


def run_episode(obj_true: Obj, obj_belief: Obj, cfg: Cfg, log=None, render=None):
    """One push. Returns a summary dict; `log` (list) receives per-step rows."""
    m_plant = build_model(obj_true)
    m_plan = build_model(obj_belief)
    d = mujoco.MjData(m_plant)
    mujoco.mj_resetDataKeyframe(m_plant, d, 0)
    box_jnt = mujoco.mj_name2id(m_plant, mujoco.mjtObj.mjOBJ_JOINT, "")  # unnamed free joint
    box_body = mujoco.mj_name2id(m_plant, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom = mujoco.mj_name2id(m_plant, mujoco.mjtObj.mjOBJ_GEOM, "box")
    qadr = m_plant.jnt_qposadr[m_plant.body_jntadr[box_body]]
    yaw = math.radians(cfg.push_dir_deg)
    r = np.array([math.cos(yaw), math.sin(yaw)])
    d.qpos[qadr:qadr + 3] = [cfg.box_xy[0], cfg.box_xy[1], obj_true.half[2] + 0.001]
    d.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    d.qvel[:] = 0
    target = np.array([cfg.box_xy[0] + cfg.push_dist * r[0], cfg.box_xy[1] + cfg.push_dist * r[1],
                       obj_true.half[2]])
    d.mocap_pos[0] = target
    d.mocap_quat[0] = [1, 0, 0, 0]
    d.ctrl[6], d.ctrl[7] = 1.5, -1.5
    mujoco.mj_forward(m_plant, d)

    gripper_bodies = {mujoco.mj_name2id(m_plant, mujoco.mjtObj.mjOBJ_BODY, n) for n in GRIPPER_BODIES}
    sens_plant = Sens(m_plant)
    planner = Planner(m_plan, cfg, obj_true.half)
    planner.set_target(target)
    planner.debug = log is not None

    # ---- phase A: Cartesian servo to the pre-push pose (not the MPC's job) ----
    # A PD on the TCP through the same wrench actuators, saturated at the actuator
    # range, until the plate sits d_behind behind the box face, slow. On the real
    # arm this is the moveL to the pre-push pose; the study starts when it ends.
    eeff = mujoco.mj_name2id(m_plant, mujoco.mjtObj.mjOBJ_SITE, "eeff")
    pre = planner.cost.push_point(d.xpos[box_body].copy(), target)
    pre_z = pre[2] + cfg.pre_lift
    t_servo0 = d.time
    for i in range(int(cfg.servo_t_max / m_plant.opt.timestep)):
        p = d.site_xpos[eeff]
        v = np.zeros(6); mujoco.mj_objectVelocity(m_plant, d, mujoco.mjtObj.mjOBJ_SITE, eeff, v, 0)
        goal = pre.copy()
        goal[2] = pre_z if (d.time - t_servo0) < cfg.servo_lift_t else pre[2]
        e = goal - p
        f = np.clip(cfg.servo_kp * e - cfg.servo_kd * v[3:6], -1.0, 1.0)
        d.ctrl[:3] = f
        # orientation: keep the gripper pointing down (eeff z-axis = -world z).
        # Without this the passive wrist tumbles during the approach and the
        # push is made with whatever part of the gripper happens to face the box.
        R = d.site_xmat[eeff].reshape(3, 3)
        z_cur = R[:, 2]
        z_des = np.array([0.0, 0.0, -1.0])
        e_rot = np.cross(z_cur, z_des)
        tau = np.clip(cfg.servo_kp_rot * e_rot - cfg.servo_kd_rot * v[0:3], -1.0, 1.0)
        d.ctrl[3:6] = tau
        d.ctrl[6], d.ctrl[7] = 1.5, -1.5
        mujoco.mj_step(m_plant, d)
        if (d.time - t_servo0) > cfg.servo_lift_t + 0.5 and np.linalg.norm(pre - p) < 0.008 \
                and np.linalg.norm(v[3:6]) < 0.03:
            break
    t_mpc0 = d.time
    servo_err = float(np.linalg.norm(pre - d.site_xpos[eeff]))
    box_moved_in_servo = float(np.linalg.norm(d.xpos[box_body][:2] - np.asarray(cfg.box_xy)))
    state = np.zeros(planner.nstate)
    steps_per_plan = max(1, int(round(cfg.plan_dt / m_plant.opt.timestep)))
    n_steps = int(round(cfg.t_max / m_plant.opt.timestep))

    t_in_tol = 0.0
    outcome, t_done = "timeout", -1.0
    peak_f, peak_tilt = 0.0, 0.0
    # damage is judged on a 50 ms low-passed contact force, not the 3 ms impact
    # spike a stiff plate makes on first touch (30-50 N at 0.2 m/s, every seed)
    f_lp, peak_f_lp, tau_lp = 0.0, 0.0, 0.05
    ft_lp, peak_ft_lp = 0.0, 0.0
    peak_box_v = 0.0
    push_impulse = 0.0
    plan_wall = []
    t_since = 0.0
    for i in range(n_steps):
        if i % steps_per_plan == 0:
            if i > 0:
                planner.shift(cfg.plan_dt)
            mujoco.mj_getState(m_plant, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            w0 = time.perf_counter()
            planner.plan(state)
            plan_wall.append(time.perf_counter() - w0)
            t_since = 0.0
        d.ctrl[:] = planner.action(t_since)
        mujoco.mj_step(m_plant, d)
        t_since += m_plant.opt.timestep

        box_pos = d.xpos[box_body]
        zax = d.xmat[box_body].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, zax[2]))))
        f, ncon = contact_wrench_on_box(m_plant, d, box_geom, gripper_bodies)
        fmag = float(np.linalg.norm(f))
        f_touch = float(d.sensordata[sens_plant.plate_left] + d.sensordata[sens_plant.plate_right])
        peak_f = max(peak_f, fmag)
        f_lp += (fmag - f_lp) * (m_plant.opt.timestep / tau_lp)
        peak_f_lp = max(peak_f_lp, f_lp)
        ft_lp += (f_touch - ft_lp) * (m_plant.opt.timestep / tau_lp)
        peak_ft_lp = max(peak_ft_lp, ft_lp)
        push_impulse += fmag * m_plant.opt.timestep
        peak_box_v = max(peak_box_v, float(np.linalg.norm(d.cvel[box_body][3:6])))
        peak_tilt = max(peak_tilt, tilt)
        err = float(np.linalg.norm(box_pos[:2] - target[:2]))
        if log is not None and i % 5 == 0:
            log.append({"t": d.time - t_mpc0, "box_x": box_pos[0], "box_y": box_pos[1], "box_z": box_pos[2],
                        "tilt_deg": tilt, "err": err,
                        "hand_x": d.site("eeff").xpos[0], "hand_y": d.site("eeff").xpos[1],
                        "hand_z": d.site("eeff").xpos[2],
                        "u": d.ctrl[:6].copy(), "f_box": f.copy(), "f_mag": fmag, "f_lp": f_lp, "ncon": ncon,
                        "f_touch": f_touch,
                        "J": planner.last.get("J_best", float("nan")),
                        "bd": planner.last.get("breakdown")})
        if render is not None and i % render["every"] == 0:
            render["frames"].append(render["fn"](m_plant, d))
        if f_lp > obj_true.f_break:
            outcome, t_done = "damaged", d.time - t_mpc0
            break
        if tilt > 45.0:
            outcome, t_done = "tipped", d.time - t_mpc0
            break
        if box_pos[2] < 0.0:
            outcome, t_done = "off_table", d.time - t_mpc0
            break
        if err < cfg.tol and tilt < 10.0:
            t_in_tol += m_plant.opt.timestep
            if t_in_tol >= cfg.hold:
                outcome, t_done = "success", d.time - t_mpc0
                break
        else:
            t_in_tol = 0.0
    return {"outcome": outcome, "t_done": t_done, "t_end": d.time - t_mpc0,
            "servo_err": servo_err, "box_moved_in_servo": box_moved_in_servo,
            "t_servo": t_mpc0 - t_servo0,
            "final_err": float(np.linalg.norm(d.xpos[box_body][:2] - target[:2])),
            "peak_f": peak_f, "peak_f_lp": peak_f_lp, "peak_ft_lp": peak_ft_lp, "push_impulse": push_impulse,
            "peak_box_v": peak_box_v, "peak_tilt": peak_tilt,
            "plan_wall_mean": float(np.mean(plan_wall)) if plan_wall else float("nan"),
            "n_plans": len(plan_wall)}
