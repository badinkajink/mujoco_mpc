#!/usr/bin/env python3
"""base_estimator_node_v3.py -- the battle-tested rw-ekf core + the audit guards.

WHY v3 EXISTS (2026-07-29). The literature-audit fixes were first shipped as v2
around the kf12 reference filter -- but kf12 has now been falsified twice on
this bench (07-16 A/B: rw-ekf WINS; 07-29 120s hold: v2 fell at 24.6s vs
rw-ekf's 56.1s even with propagation forced), and today's runs exposed two
kf12-specific fragilities (trust self-collapse through the estimated-foot-height
gate; post-fall bias poisoning). The rw-ekf, meanwhile, has never lost a bench.

v3 therefore = the SHIPPED rw-ekf loop, byte-equivalent math, plus only the
fixes that are filter-agnostic or already proven by v1's own selftests:

  1. WAIST FIX ALWAYS ON. The imu site lives in torso_link behind torso_joint;
     feeding the raw torso quat to the leg FK rotates the estimated base
     velocity by the waist angle -- v1's selftest measures 0.14 m/s of phantom
     at 0.6 rad. v1 ships the fix but withholds it from the rw-ekf path
     (--waist-fix defaults to kf12-only). v3 applies pelvis_from_torso always
     (--no-waist-fix restores the shipped defect for A/B).

  2. UPRIGHT-GATED ZUPT, ON BY DEFAULT. The documented long-hold ceiling is an
     estimator drift spike (held run v_err peak 0.05 m/s vs fell run 0.70) and
     in a planted double-support stand 0 IS the truth -- v1 has the ZUPT but
     ships it off. v3 turns it on, and adds the gate the 07-29 120s run proved
     necessary: a FALLEN robot also has loaded static legs and a quiet gyro, so
     ZUPT (and anything else that learns from "static") must additionally
     require the robot to be UPRIGHT (kinematic height near nominal + small
     tilt), or it fuses post-fall garbage.

  3. SLIP MONITOR (from v2, detection reused verbatim). Two loaded feet are two
     independent measurements of one base velocity; sustained disagreement
     means at least one planted-foot assumption is a lie. rw-ekf's Mahalanobis
     gate judges each foot against the PRIOR -- slow coherent slip inside the
     gate passes. When the low-passed ||v_L - v_R|| exceeds the threshold, BOTH
     feet's measurement noise is inflated (we know someone lies, not who).

  4. STALE GUARD (from v2, verbatim). No more re-publishing a dead lowstate as
     fresh state forever: past --stale-ms the node stops publishing and says
     so. Silence trips --require_sportstate downstream; a frozen state seeding
     every rollout does not.

DELIBERATELY NOT PORTED from v2: accel propagation (that IS kf12 -- falsified),
the bias learner (rw-ekf never integrates accel, so there is no drift to
de-bias), force-accept removal (without propagation, prolonged gating = frozen
estimate; the hatch is rw-ekf's only escape), measured-GRF contact FSM (the
straight-knee-blind torque proxy soft-weighting has beaten it on every bench).

STRUCTURE mirrors v2: the whole per-tick pipeline is `V3Core.step()` so the
offline harness (estimator_ab.py --est v3) runs EXACTLY the live code; main()
is DDS plumbing; build_parser()/default_args() are the single source of
defaults.

Run in the twin venv (has mujoco + unitree_sdk2py):
  cd ~/Desktop/h12/h1_mujoco
  .venv/bin/python .../base_estimator_node_v3.py --selftest        # offline checks
  .venv/bin/python .../base_estimator_node_v3.py                   # live
  .venv/bin/python .../base_estimator_node_v3.py --out-topic rt/sportmodestate_est --compare
"""
import argparse
import copy
import time

import numpy as np
import mujoco

import base_estimator_node as v1
from base_estimator_node import (
    NJ, IMU_OFFSET, TORSO_MOTOR,
    pelvis_from_torso, zupt_hold, zupt_fuse_v,
    load_model_and_calibrate, _pick_iface, _quat2mat, _mat2quat, _rz,
)
from base_estimator_node_v2 import StaleGuard, SlipMonitor


GYRO_BIAS_CONST_DEFAULT = "-0.0007,0.0043,0.0050"


def _parse_vec3(s):
    """'bx,by,bz' (or a 3-sequence) -> float ndarray(3)."""
    if isinstance(s, str):
        v = [float(x) for x in s.replace(" ", "").split(",") if x != ""]
    else:
        v = [float(x) for x in s]
    if len(v) != 3:
        raise ValueError(f"expected 3 comma-separated values, got {s!r}")
    return np.array(v, float)


def _quat_delta_rotvec(q_prev, q_now):
    """Rotation vector (rad) of q_prev^-1 * q_now, expressed in the q_prev BODY
    frame (= the IMU sensor frame for two successive IMU quaternions). Handles
    the q/-q double cover (sign flips between samples)."""
    w0, x0, y0, z0 = q_prev
    w1, x1, y1, z1 = q_now
    # conj(q_prev) * q_now  (Hamilton, wxyz)
    w = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
    x = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
    y = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
    z = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
    if w < 0.0:                       # sign flip: same rotation, shorter arc
        w, x, y, z = -w, -x, -y, -z
    v = np.array([x, y, z])
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return 2.0 * v
    return v * (2.0 * np.arctan2(s, w) / s)


class V3Core:
    """The shipped rw-ekf per-tick pipeline + waist fix + upright-gated ZUPT +
    slip monitor. DDS-free; one instance per robot; step() once per sample.

    imu_is_pelvis: offline harnesses that feed the TRUE pelvis quat (twin free
    joint) set True -- applying the waist correction there would INJECT the
    error it removes on the real (torso-mounted) IMU."""

    def __init__(self, a, m, data, foot_ids, height_C, nominal_z,
                 imu_is_pelvis=False, verbose=True):
        self.a = a
        self.m, self.data, self.foot_ids = m, data, foot_ids
        self.height_C = height_C
        self.nominal_z = float(nominal_z)
        self.imu_is_pelvis = imu_is_pelvis
        self.verbose = verbose
        self.LOAD_IDX = [(3, 4) if "left" in nm else (9, 10) for _, nm in foot_ids]
        self.I3 = np.eye(3)
        self.v = np.zeros(3)
        self.P = self.I3 * 0.04
        self.rej = 0
        self.pos_xy = np.array(getattr(a, "init_xy", [0.0, 0.0]), float)
        self.slip = SlipMonitor(a.slip_thresh, 1.0)   # detection only; fac unused here
        self.cos_tilt = float(np.cos(np.deg2rad(a.upright_tilt_deg)))
        self.zupt_n = 0
        self.zupt_blocked = 0        # gates open but robot not upright (the post-fall case)
        self.slip_infl_n = 0
        self.n = 0
        self.base_height = self.nominal_z
        self.upright = True
        self._res = np.zeros(6)
        # --- 2026-10-15 FIX 1: raw-gyro bias (sensor/torso-IMU frame) ----------
        # The Unitree gyro carries a constant bias ~(-0.7, +4.3, +5.0) mrad/s
        # (torso/IMU frame, stable 08-11..08-30). Through the omega x r lever arm
        # of the leg-odometry measurement it is ~4 mm/s of phantom base velocity
        # = the ~260 mm/min internal drift. 'off' (default) = untouched raw gyro.
        self.gyro_bias_mode = getattr(a, "gyro_bias", "off")
        self.gyro_bias = (np.zeros(3) if self.gyro_bias_mode == "off" else
                          _parse_vec3(getattr(a, "gyro_bias_const", GYRO_BIAS_CONST_DEFAULT)))
        self._gb_qprev = None
        self.gyro_bias_upd = 0       # online updates accepted
        self.gyro_bias_skip = 0      # online updates refused (not upright / insane delta)
        self.last_gyro = None        # gyro actually used this tick (bias-corrected if on)
        # --- 2026-10-15 FIX 2: which feet must be loaded for the ZUPT ---------
        self.zupt_load_gate = getattr(a, "zupt_load_gate", "both")
        self.zupt_static_n = 0       # ticks the static gates passed (before upright)

    def step(self, q, dq, tau, quat, gyro, step_dt):
        """One proprioceptive sample -> (pelvis) estimate.
        q/dq/tau 27-vector motor order; quat wxyz + gyro body frame (torso IMU,
        or pelvis if imu_is_pelvis). No accelerometer: rw-ekf never integrates it.
        Returns dict(base_height, v, xy, upright, slip_active)."""
        a = self.a
        q = np.asarray(q, float)
        dq = np.asarray(dq, float)
        tau = np.asarray(tau, float)
        quat = np.asarray(quat, float)
        gyro = np.asarray(gyro, float)
        # fix 1: subtract the gyro bias in the SENSOR frame, BEFORE the waist
        # fix rotates omega into the pelvis frame ('off' = the raw array, as-is)
        gyro = self._debias_gyro(gyro, quat, step_dt)
        self.last_gyro = gyro

        # --- pelvis frame (fix #1: waist correction ALWAYS, unless disabled) ------
        if self.imu_is_pelvis or not a.waist_fix:
            R_wp, omega_p, quat_fk = _quat2mat(quat), gyro, quat
        else:
            R_wp, omega_p = pelvis_from_torso(quat, gyro, q[TORSO_MOTOR], dq[TORSO_MOTOR])
            quat_fk = _mat2quat(R_wp)

        # --- leg odometry (byte-equivalent to v1's rw-ekf path) -------------------
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[3:7] = quat_fk
        d.qpos[7:7 + NJ] = q
        d.qvel[:] = 0.0
        d.qvel[3:6] = omega_p
        d.qvel[6:6 + NJ] = dq
        mujoco.mj_forward(self.m, d)
        self.base_height = -min(float(d.xpos[bid][2])
                                for bid, _ in self.foot_ids) + self.height_C
        vfeet = []
        for bid, _ in self.foot_ids:
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, bid,
                                     self._res, 0)
            vfeet.append(-self._res[3:6].copy())
        vfeet = np.array(vfeet)

        # --- fix #3: feet cross-check BEFORE the per-foot updates -----------------
        loads = [abs(tau[i1]) + abs(tau[i2]) for i1, i2 in self.LOAD_IDX]
        slip_active = False
        if a.slip_monitor:
            both_loaded = all(l >= a.zupt_load for l in loads)
            self.slip.step({"vs": [vfeet[0], vfeet[1]]}, [1.0, 1.0],
                           [both_loaded, both_loaded], step_dt)
            slip_active = self.slip.active
            if slip_active:
                self.slip_infl_n += 1

        # --- the shipped RW-EKF update loop (math unchanged from v1) --------------
        self.P = self.P + ((a.ekf_amax * step_dt) ** 2) * self.I3
        for k in range(len(vfeet)):
            rk = (a.ekf_r0 + a.ekf_r1 / max(loads[k], a.ekf_lfloor)) ** 2
            if slip_active:
                rk *= a.slip_infl            # someone is lying; trust both feet less
            S = self.P + rk * self.I3
            inn = vfeet[k] - self.v
            if float(inn @ np.linalg.solve(S, inn)) > a.ekf_chi2 and rej_ok(self.rej, a):
                self.rej += 1
                continue
            self.rej = 0
            K = self.P @ np.linalg.inv(S)
            self.v = self.v + K @ inn
            self.P = (self.I3 - K) @ self.P

        # --- fix #2: upright-gated ZUPT -------------------------------------------
        self.upright = (self.base_height > a.upright_z_frac * self.nominal_z
                        and float(R_wp[2, 2]) > self.cos_tilt)
        if a.zupt and self._zupt_static(tau, dq, gyro, loads):
            self.zupt_static_n += 1
            if self.upright:
                self.v, self.P = zupt_fuse_v(self.v, self.P, a.zupt_vr)
                self.zupt_n += 1
            else:
                self.zupt_blocked += 1       # fallen-but-static: fusing v==0 here
                                             # would launder post-fall garbage

        # --- odometric xy (kinematically consistent (pos, vel) pair) --------------
        self.pos_xy += self.v[0:2] * step_dt
        self.n += 1
        return dict(base_height=self.base_height, v=self.v.copy(),
                    xy=self.pos_xy.copy(), upright=self.upright,
                    slip_active=slip_active)

    # --- fix 1 ---------------------------------------------------------------
    def _debias_gyro(self, gyro_raw, quat, step_dt):
        """Raw IMU gyro -> bias-corrected gyro (sensor frame). 'off' returns the
        input object untouched (byte-identical legacy path)."""
        if self.gyro_bias_mode == "off":
            return gyro_raw
        if self.gyro_bias_mode == "online":
            self._update_gyro_bias(gyro_raw, quat, step_dt)
        return gyro_raw - self.gyro_bias

    def _update_gyro_bias(self, gyro_raw, quat, step_dt):
        """b <- low-pass(gyro_raw - omega_quat), omega_quat from successive IMU
        quaternions (same IMU, same body frame). The Unitree quaternion is
        bias-compensated (yaw drifts ~1 deg/min vs 17 deg/min the raw z-bias
        would give) but filtered/LAGGED, so only the long time constant is
        meaningful. Increment form  b += (g*dt - dtheta - b*dt)/tau  is the
        same EMA as  b += dt/tau*(g - dtheta/dt - b)  but immune to repeated
        quaternion samples and dt jitter (the integrals still telescope)."""
        a = self.a
        qn = np.asarray(quat, float)
        nq = float(np.linalg.norm(qn))
        if not np.isfinite(nq) or abs(nq - 1.0) > 0.05:
            self._gb_qprev = None                    # garbage quat: restart the chain
            self.gyro_bias_skip += 1
            return
        qn = qn / nq
        qp, self._gb_qprev = self._gb_qprev, qn
        if qp is None or not (0.0 < step_dt <= 0.1) or not np.all(np.isfinite(gyro_raw)):
            self.gyro_bias_skip += 1
            return
        if self.n == 0 or not self.upright:          # only learn while upright
            self.gyro_bias_skip += 1
            return
        dth = _quat_delta_rotvec(qp, qn)
        if float(np.linalg.norm(dth)) > getattr(a, "gyro_bias_max_dth", 0.2):
            self.gyro_bias_skip += 1                 # quat jump / reset, not motion
            return
        tau_b = max(float(getattr(a, "gyro_bias_tau", 30.0)), step_dt)
        self.gyro_bias = self.gyro_bias + (gyro_raw * step_dt - dth
                                           - self.gyro_bias * step_dt) / tau_b
        c = float(getattr(a, "gyro_bias_clamp", 0.02))
        self.gyro_bias = np.clip(self.gyro_bias, -c, c)
        self.gyro_bias_upd += 1

    # --- fix 2 ---------------------------------------------------------------
    def _zupt_static(self, tau, dq, gyro, loads):
        """Static-base gates for the ZUPT. 'both' (default) = the shipped v1
        zupt_hold (each foot >= --zupt-load). 'sum' = L+R >= --zupt-load-sum;
        'either' = any foot >= --zupt-load. dq/gyro gates unchanged; gyro here
        is the bias-corrected one when fix 1 is on."""
        a = self.a
        gate = self.zupt_load_gate
        if gate == "both":
            return zupt_hold(tau, dq, gyro, self.LOAD_IDX, a)
        if gate == "sum":
            loaded = (loads[0] + loads[1]) >= getattr(a, "zupt_load_sum", 25.0)
        elif gate == "either":
            loaded = max(loads) >= a.zupt_load
        else:
            raise ValueError(f"unknown zupt_load_gate {gate!r}")
        return (loaded
                and float(np.linalg.norm(dq)) < a.zupt_dq
                and float(np.linalg.norm(gyro)) < a.zupt_gyro)

    def telemetry(self):
        line = (f"rej={self.rej}"
                f" upright={'Y' if self.upright else 'N'}")
        if self.a.zupt:
            line += (f" zupt={100.0 * self.zupt_n / max(self.n, 1):.0f}%"
                     + (f" (blocked {self.zupt_blocked})" if self.zupt_blocked else ""))
        if self.a.slip_monitor and (self.slip.active or self.slip.events):
            line += (f" slip={'ACTIVE' if self.slip.active else 'ok'}"
                     f"({self.slip.events} ev, ema {self.slip.ema:.2f})")
        if self.gyro_bias_mode != "off":
            b = self.gyro_bias * 1e3
            line += f" gb[{self.gyro_bias_mode}]=[{b[0]:+.2f},{b[1]:+.2f},{b[2]:+.2f}]mrad/s"
            if self.gyro_bias_mode == "online":
                line += f"(upd {100.0 * self.gyro_bias_upd / max(self.n, 1):.0f}%)"
        return line


def rej_ok(rej, a):
    """The v1 force-accept hatch, kept: without accel propagation, prolonged
    gating = a frozen estimate; the hatch is rw-ekf's only escape."""
    return rej < a.ekf_rejcap


def _selftest(m, data, foot_ids, height_C, home_q, a):
    ok = True
    dt = 0.005
    nominal_z = float(home_q[2])
    qj = np.array(home_q[7:7 + NJ])
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    # torque pattern that passes the ZUPT load gate (knee+ankleP >= 15 Nm/leg)
    tau_loaded = np.zeros(NJ)
    for i in (3, 4, 9, 10):
        tau_loaded[i] = 10.0

    # 1. static home: v ~ 0, height ~ home (the v1 selftest, through V3Core).
    c = V3Core(a, m, data, foot_ids, height_C, nominal_z, imu_is_pelvis=True)
    out = None
    for _ in range(400):
        out = c.step(qj, np.zeros(NJ), tau_loaded, ident, np.zeros(3), dt)
    good = (np.linalg.norm(out["v"]) < 1e-3
            and abs(out["base_height"] - nominal_z) < 0.02 and out["upright"])
    ok &= good
    print(f"[selftest] v3 static home  : |v|={np.linalg.norm(out['v']):.2e} "
          f"base_z={out['base_height']:.3f} (expect ~{nominal_z:.3f}) "
          f"upright={out['upright']} zupt={c.zupt_n > 0} ({'ok' if good else 'FAIL'})")

    # 2. joint moving -> v nonzero (leg odo alive).
    c2 = V3Core(a, m, data, foot_ids, height_C, nominal_z, imu_is_pelvis=True)
    dqm = np.zeros(NJ)
    dqm[3] = 0.5
    for _ in range(100):
        out = c2.step(qj, dqm, tau_loaded, ident, np.zeros(3), dt)
    good = float(np.linalg.norm(out["v"])) > 1e-4
    ok &= good
    print(f"[selftest] v3 joint moving : |v|={np.linalg.norm(out['v']):.4f} "
          f"(expect >0) ({'ok' if good else 'FAIL'})")

    # 3. WAIST FIX: velocity estimate invariant to torso_joint (the 0.14 m/s
    #    defect v1 measures on its own legacy path must be ABSENT here).
    ref = None
    worst = 0.0
    for theta in (0.0, 0.6, -1.2):
        c3 = V3Core(a, m, data, foot_ids, height_C, nominal_z, imu_is_pelvis=False)
        q_t = qj.copy()
        q_t[TORSO_MOTOR] = theta
        quat_t = _mat2quat(_rz(theta))       # pelvis = I => torso IMU reads Rz(theta)
        for _ in range(100):
            out = c3.step(q_t, dqm, tau_loaded, quat_t, np.zeros(3), dt)
        if ref is None:
            ref = out["v"]
        else:
            worst = max(worst, float(np.linalg.norm(out["v"] - ref)))
    good = worst < 1e-6
    ok &= good
    print(f"[selftest] v3 waist fix    : worst |dv| over waist sweep = {worst:.2e} "
          f"(v1 legacy path: 0.14) ({'ok' if good else 'FAIL'})")

    # 4. UPRIGHT GATE: a FALLEN robot (tilted 90 deg) with loaded static legs
    #    must BLOCK the ZUPT, not fuse v==0.
    c4 = V3Core(a, m, data, foot_ids, height_C, nominal_z, imu_is_pelvis=True)
    quat_fallen = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # 90 deg pitch
    for _ in range(200):
        out = c4.step(qj, np.zeros(NJ), tau_loaded, quat_fallen, np.zeros(3), dt)
    good = c4.zupt_n == 0 and c4.zupt_blocked > 0 and not out["upright"]
    ok &= good
    print(f"[selftest] v3 upright gate : fallen+static -> zupt fired {c4.zupt_n} "
          f"blocked {c4.zupt_blocked} upright={out['upright']} "
          f"({'ok' if good else 'FAIL'})")

    # 5. SLIP MONITOR: force the two feet to disagree (one leg's joints moving,
    #    the other static) with both LOADED -> the monitor must latch.
    c5 = V3Core(a, m, data, foot_ids, height_C, nominal_z, imu_is_pelvis=True)
    dq_slip = np.zeros(NJ)
    dq_slip[3] = 1.2                          # left knee sweeping: feet disagree
    for _ in range(100):
        out = c5.step(qj, dq_slip, tau_loaded, ident, np.zeros(3), dt)
    good = c5.slip.events >= 1 and c5.slip_infl_n > 0
    ok &= good
    print(f"[selftest] v3 slip monitor : events={c5.slip.events} "
          f"inflated_ticks={c5.slip_infl_n} ema={c5.slip.ema:.2f} "
          f"({'ok' if good else 'FAIL'})")

    # 6. StaleGuard (imported from v2; quick regression).
    sg = StaleGuard(0.3)
    sg.beat()
    good = sg.fresh(0.0) and not sg.fresh(0.4)
    sg.beat()
    good = good and sg.fresh(0.45)
    ok &= good
    print(f"[selftest] v3 stale guard  : fresh->stale->recovered "
          f"({'ok' if good else 'FAIL'})")

    # 7. GYRO BIAS OFF == LEGACY: off-mode must ignore every new knob and hand
    #    the raw gyro through untouched (bit-identical state on a noisy run).
    rng = np.random.default_rng(1)
    seq = []
    for i in range(300):
        dqn = rng.normal(0.0, 0.02, NJ)
        g = rng.normal(0.0, 0.01, 3) + np.array([-0.0007, 0.0043, 0.0050])
        seq.append((dqn, g))
    a_off = copy.copy(a)
    a_off.gyro_bias = "off"
    a_off.zupt_load_gate = "both"
    a_knob = copy.copy(a_off)
    a_knob.gyro_bias_const = "0.3,-0.2,0.1"
    a_knob.gyro_bias_tau = 0.5
    a_knob.zupt_load_sum = 1.0

    def run_seq(aa, gyro_add=np.zeros(3)):
        c = V3Core(aa, m, data, foot_ids, height_C, nominal_z, imu_is_pelvis=False)
        vs, xs = [], []
        for dqn, g in seq:
            o = c.step(qj, dqn, tau_loaded, ident, g + gyro_add, dt)
            vs.append(o["v"])
            xs.append(o["xy"])
        return np.array(vs), np.array(xs), c

    v_off, x_off, c_off = run_seq(a_off)
    v_kn, x_kn, _ = run_seq(a_knob)
    good = (np.array_equal(v_off, v_kn) and np.array_equal(x_off, x_kn)
            and np.array_equal(c_off.gyro_bias, np.zeros(3)))
    ok &= good
    print(f"[selftest] v3 gyro-bias off: knobs ignored, bit-identical state "
          f"({'ok' if good else 'FAIL'})")

    # 8. CONST BIAS cancels an injected bias: const(b) on gyro+b == off on gyro.
    b_inj = _parse_vec3(GYRO_BIAS_CONST_DEFAULT)
    a_c = copy.copy(a_off)
    a_c.gyro_bias = "const"
    a_c.gyro_bias_const = GYRO_BIAS_CONST_DEFAULT
    v_c, x_c, _ = run_seq(a_c, gyro_add=b_inj)
    worst = max(float(np.abs(v_c - v_off).max()), float(np.abs(x_c - x_off).max()))
    good = worst < 1e-9
    ok &= good
    print(f"[selftest] v3 gyro-bias const: |state - off(unbiased)| = {worst:.1e} "
          f"({'ok' if good else 'FAIL'})")

    # 9. ONLINE BIAS converges: constant yaw rotation, quaternion from the
    #    TRUE rate (with random q/-q sign flips and repeated samples), gyro =
    #    true + b_true (b_true != const init). After 8 tau, |b - b_true| small.
    a_o = copy.copy(a_off)
    a_o.gyro_bias = "online"
    a_o.gyro_bias_tau = 5.0
    b_true = np.array([0.003, -0.002, 0.006])
    w_true = np.array([0.0, 0.0, 0.2])            # body-frame yaw rate (stays upright)
    c9 = V3Core(a_o, m, data, foot_ids, height_C, nominal_z, imu_is_pelvis=True)
    qcur = ident.copy()
    th = np.linalg.norm(w_true * dt)
    ax_ = w_true / np.linalg.norm(w_true)
    dqq = np.concatenate([[np.cos(th / 2)], np.sin(th / 2) * ax_])
    for i in range(int(40.0 / dt)):
        if i % 3 != 2:                            # every 3rd tick: repeated quat
            w0, x0, y0, z0 = qcur
            w1, x1, y1, z1 = dqq
            qcur = np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1, w0*x1 + x0*w1 + y0*z1 - z0*y1,
                             w0*y1 - x0*z1 + y0*w1 + z0*x1, w0*z1 + x0*y1 - y0*x1 + z0*w1])
            qcur /= np.linalg.norm(qcur)
        qfeed = -qcur if rng.random() < 0.5 else qcur
        # quaternion repeated 1/3 of ticks -> its rate is 2/3 of w_true there; gyro
        # reports what the quat implies on average so the telescoped sum is exact
        c9.step(qj, np.zeros(NJ), tau_loaded, qfeed,
                w_true * (2.0 / 3.0) + b_true, dt)
    # quat-implied mean rate = w_true*2/3 -> bias estimate should equal b_true
    err = float(np.abs(c9.gyro_bias - b_true).max())
    good = err < 3e-4 and c9.gyro_bias_upd > 0
    ok &= good
    print(f"[selftest] v3 gyro-bias online: b={np.round(c9.gyro_bias * 1e3, 3)} mrad/s "
          f"vs true {np.round(b_true * 1e3, 3)} (err {err * 1e3:.3f}) "
          f"({'ok' if good else 'FAIL'})")

    # 10. ZUPT LOAD GATE: 'both' == the v1 zupt_hold; one foot loaded -> only
    #     'either' (and 'sum' if L+R >= thresh) fires.
    tau_one = np.zeros(NJ)
    tau_one[3] = tau_one[4] = 15.0                # left 30 Nm, right 0
    res = {}
    for gate in ("both", "sum", "either"):
        aa = copy.copy(a_off)
        aa.zupt_load_gate = gate
        c = V3Core(aa, m, data, foot_ids, height_C, nominal_z, imu_is_pelvis=True)
        for _ in range(50):
            c.step(qj, np.zeros(NJ), tau_one, ident, np.zeros(3), dt)
        res[gate] = c.zupt_n
    same_v1 = (V3Core(a_off, m, data, foot_ids, height_C, nominal_z)
               ._zupt_static(tau_one, np.zeros(NJ), np.zeros(3), [30.0, 0.0])
               == zupt_hold(tau_one, np.zeros(NJ), np.zeros(3),
                            [(3, 4), (9, 10)], a_off))
    good = res["both"] == 0 and res["either"] == 50 and res["sum"] == 50 and same_v1
    ok &= good
    print(f"[selftest] v3 zupt load gate: one foot 30 Nm -> fired both {res['both']} "
          f"sum {res['sum']} either {res['either']} /50, both==v1 {same_v1} "
          f"({'ok' if good else 'FAIL'})")

    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="H1-2 MuJoCo scene (default: handless twin scene)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--no-auto-iface", action="store_true")
    ap.add_argument("--rate", type=float, default=200.0, help="publish Hz")
    ap.add_argument("--tick-dt", type=float, default=0.0,
                    help="PLANT-CLOCK the filter (see v1 --tick-dt). 0 = wall-clocked.")
    ap.add_argument("--lowstate-topic", default="rt/lowstate")
    ap.add_argument("--out-topic", default="rt/sportmodestate")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--truth-topic", default="rt/sportmodestate")
    ap.add_argument("--selftest", action="store_true")

    g = ap.add_argument_group("rw-ekf (v1 defaults, math unchanged)")
    g.add_argument("--ekf-amax", type=float, default=3.0)
    g.add_argument("--ekf-r0", type=float, default=0.02)
    g.add_argument("--ekf-r1", type=float, default=0.3)
    g.add_argument("--ekf-lfloor", type=float, default=5.0)
    g.add_argument("--ekf-chi2", type=float, default=11.34)
    g.add_argument("--ekf-rejcap", type=int, default=50)

    v = ap.add_argument_group("v3 guards")
    v.add_argument("--no-waist-fix", dest="waist_fix", action="store_false", default=True,
                   help="restore the shipped defect (raw torso quat into the leg FK; "
                        "0.14 m/s phantom at 0.6 rad waist) -- A/B only")
    v.add_argument("--no-zupt", dest="zupt", action="store_false", default=True,
                   help="v3 default ON (v1 ships it off)")
    v.add_argument("--zupt-load", type=float, default=15.0)
    v.add_argument("--zupt-dq", type=float, default=0.15)
    v.add_argument("--zupt-gyro", type=float, default=0.05)
    v.add_argument("--zupt-vr", type=float, default=0.01)
    v.add_argument("--zupt-load-gate", choices=["both", "sum", "either"], default="both",
                   help="2026-10-15: which feet must carry load for the ZUPT. 'both' "
                        "(default, shipped) = each foot |tau_knee|+|tau_ankleP| >= "
                        "--zupt-load; 'sum' = L+R >= --zupt-load-sum; 'either' = any "
                        "foot >= --zupt-load. dq/gyro/upright gates unchanged.")
    v.add_argument("--zupt-load-sum", type=float, default=25.0,
                   help="'sum' gate threshold on L+R load proxy (Nm)")
    v.add_argument("--upright-z-frac", type=float, default=0.7,
                   help="ZUPT additionally requires kinematic base height > this fraction "
                        "of the nominal stand height (a fallen robot is also 'static')")
    v.add_argument("--upright-tilt-deg", type=float, default=25.0,
                   help="...and base tilt below this (R[2,2] > cos)")
    v.add_argument("--slip-thresh", type=float, default=0.15,
                   help="low-passed ||v_L - v_R|| between two LOADED feet above which "
                        "at least one planted-foot assumption is a lie")
    v.add_argument("--slip-infl", type=float, default=10.0,
                   help="measurement-noise inflation on BOTH feet while slip is active")
    v.add_argument("--no-slip-monitor", dest="slip_monitor", action="store_false",
                   default=True)
    gb = ap.add_argument_group("gyro bias (2026-10-15; default off = legacy raw gyro)")
    gb.add_argument("--gyro-bias", choices=["off", "const", "online"], default="off",
                    help="subtract a raw-gyro bias in the IMU SENSOR (torso) frame "
                         "before the waist fix. const = --gyro-bias-const; online = "
                         "low-pass of (gyro - quaternion rate), initialised from the "
                         "const, learned only while upright")
    gb.add_argument("--gyro-bias-const", default=GYRO_BIAS_CONST_DEFAULT,
                    help="bx,by,bz rad/s, IMU/torso frame (08-11..08-30 data: "
                         "x -0.5..-0.9, y 4.2..4.3, z 4.7..5.3 mrad/s)")
    gb.add_argument("--gyro-bias-tau", type=float, default=30.0,
                    help="online: low-pass time constant (s)")
    gb.add_argument("--gyro-bias-clamp", type=float, default=0.02,
                    help="online: per-axis |b| clamp (rad/s)")
    gb.add_argument("--gyro-bias-max-dth", type=float, default=0.2,
                    help="online: reject a quaternion step rotating more than this "
                         "(rad) as a glitch/reset")
    v.add_argument("--stale-ms", type=float, default=300.0,
                   help="stop publishing when rt/lowstate goes this stale (0 = never)")
    v.add_argument("--init-xy", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("X", "Y"),
                   help="initial odometric xy belief (bench tool: inject a "
                        "believed base offset; leg odom never observes absolute "
                        "position, so it persists)")
    v.add_argument("--vel-noise", type=float, default=0.0,
                   help="bench tool: 1-sigma m/s Gaussian noise added to the "
                        "PUBLISHED base velocity (emulates the real leg-odom "
                        "velocity noise floor, ~0.009 median speed)")
    v.add_argument("--pos-noise", type=float, default=0.0,
                   help="bench tool: 1-sigma m white xy jitter on the PUBLISHED "
                        "position (the node FDs position for qvel; ~0.0001 "
                        "reproduces the real node-side velocity noise floor)")
    return ap


def default_args():
    a = build_parser().parse_args([])
    a.filter = "rw-ekf"
    return a


def main():
    a = build_parser().parse_args()

    m, data, foot_ids, height_C, home_q = load_model_and_calibrate(a.scene)

    if a.selftest:
        raise SystemExit(_selftest(m, data, foot_ids, height_C, home_q, a))

    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelSubscriber,
                                             ChannelPublisher)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

    if a.no_auto_iface:
        iface, why = None, "forced autodetermine (--no-auto-iface; twin same-host)"
    else:
        iface, why = _pick_iface(a.iface)
    print(f"[est] DDS interface = {iface or 'autodetermine'} ({why}), domain {a.domain}")
    if iface:
        ChannelFactoryInitialize(a.domain, iface)
    else:
        ChannelFactoryInitialize(a.domain)

    stale = StaleGuard(a.stale_ms * 1e-3 if a.stale_ms > 0 else float("inf"))
    latest = {"ls": None}

    def _on_ls(msg_in):
        latest["ls"] = msg_in
        stale.beat()

    ls_sub = ChannelSubscriber(a.lowstate_topic, LowState_)
    ls_sub.Init(_on_ls, 10)

    truth = {"sp": None}
    if a.compare:
        truth_sub = ChannelSubscriber(a.truth_topic, SportModeState_)
        truth_sub.Init(lambda msg_in: truth.__setitem__("sp", msg_in), 10)
        if a.out_topic == a.truth_topic:
            print("[est] WARN: --compare with out-topic==truth-topic; use "
                  "--out-topic rt/sportmodestate_est", flush=True)

    pub = ChannelPublisher(a.out_topic, SportModeState_)
    pub.Init()
    msg = unitree_go_msg_dds__SportModeState_()

    core = V3Core(a, m, data, foot_ids, height_C, float(home_q[2]),
                  imu_is_pelvis=False, verbose=True)
    vel_noise_std = float(getattr(a, "vel_noise", 0.0))
    pos_noise_std = float(getattr(a, "pos_noise", 0.0))
    vel_noise_rng = np.random.default_rng(0)
    if vel_noise_std > 0.0 or pos_noise_std > 0.0:
        print(f"[est] BENCH noise ACTIVE: vel sigma {vel_noise_std*1000:.1f} mm/s, "
              f"pos sigma {pos_noise_std*1000:.2f} mm on published stream")
    print(f"[est] v3 publishing -> '{a.out_topic}' @ {a.rate:.0f}Hz | RW-EKF "
          f"(amax={a.ekf_amax} r0={a.ekf_r0} r1={a.ekf_r1} chi2={a.ekf_chi2} "
          f"rejcap={a.ekf_rejcap}) | waist-fix={'ON' if a.waist_fix else 'OFF'} | "
          f"zupt={'ON (upright-gated %g/%g deg)' % (a.upright_z_frac, a.upright_tilt_deg) if a.zupt else 'off'} | "
          f"slip-mon={'%.2fm/s x%g' % (a.slip_thresh, a.slip_infl) if a.slip_monitor else 'off'} | "
          f"stale={a.stale_ms:.0f}ms | zupt-load-gate={a.zupt_load_gate} | "
          f"gyro-bias={a.gyro_bias}"
          + (f"({a.gyro_bias_const}"
             + (f", tau {a.gyro_bias_tau:g}s" if a.gyro_bias == "online" else "") + ")"
             if a.gyro_bias != "off" else "")
          + (f" | COMPARE vs '{a.truth_topic}'" if a.compare else ""))
    print("[est] waiting for rt/lowstate ...", flush=True)

    dt = 1.0 / a.rate
    plant_clocked = a.tick_dt > 0.0
    last_tick = None
    n = 0
    t0 = time.time()
    next_t = t0
    seen = False
    while True:
        now = time.time()
        if now < next_t:
            time.sleep(min(dt, next_t - now))
            continue
        next_t += dt
        ls = latest["ls"]
        if ls is None:
            continue
        publish_ok = stale.fresh(now)
        step_dt = dt
        if plant_clocked:
            tick = int(ls.tick)
            if tick == last_tick:
                continue
            step_dt = (a.tick_dt if last_tick is None
                       else min(max(tick - last_tick, 1) * a.tick_dt, 0.5))
            last_tick = tick
        if not seen:
            print("[est] rt/lowstate up -> estimating + publishing."
                  + (f" PLANT-CLOCKED: tick_dt={a.tick_dt:g}s" if plant_clocked else ""),
                  flush=True)
            seen = True

        q = np.array([ls.motor_state[i].q for i in range(NJ)])
        dq = np.array([ls.motor_state[i].dq for i in range(NJ)])
        tau = np.array([ls.motor_state[i].tau_est for i in range(NJ)])
        quat = np.array(list(ls.imu_state.quaternion))       # wxyz (TORSO orientation)
        gyro = np.array(list(ls.imu_state.gyroscope))        # body frame

        out = core.step(q, dq, tau, quat, gyro, step_dt)

        # pelvis -> IMU site (v1 legacy convention: R here IS R_torso, exact at
        # any waist angle because torso_link.body_pos == 0 -- verified in v1)
        R = _quat2mat(quat)
        roff = R @ IMU_OFFSET
        omega_w = R @ (gyro if core.gyro_bias_mode == "off" else core.last_gyro)
        site_p = np.array([out["xy"][0], out["xy"][1], out["base_height"]]) + roff
        site_v = out["v"] + np.cross(omega_w, roff)
        if vel_noise_std > 0.0:
            # bench tool (--vel-noise): emulate the REAL leg-odom velocity
            # noise floor (~9 mm/s median) on the published velocity only --
            # the internal EKF state stays clean, like the real robot where
            # the noise lives in the leg-kinematic measurement chain.
            site_v = site_v + vel_noise_rng.normal(0.0, vel_noise_std, 3)
        if pos_noise_std > 0.0:
            # bench tool (--pos-noise): white xy jitter on the PUBLISHED
            # position. The node derives base qvel by finite-differencing this
            # stream through a 30 ms LPF (deploy_common.cc), so ~0.1 mm here at
            # 500 Hz reproduces the real ~9 mm/s node-side velocity floor.
            site_p[0:2] = site_p[0:2] + vel_noise_rng.normal(0.0, pos_noise_std, 2)

        if publish_ok:
            for k in range(3):
                msg.position[k] = float(site_p[k])
                msg.velocity[k] = float(site_v[k])
            pub.Write(msg)

        n += 1
        if n % int(a.rate) == 0:
            v = out["v"]
            line = (f"[est] {now - t0:6.1f}s  base_z={out['base_height']:5.3f}  "
                    f"xy=[{out['xy'][0]:+.2f},{out['xy'][1]:+.2f}]  "
                    f"v=[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}] m/s  | "
                    + core.telemetry())
            if stale.stale:
                line += " STALE(not publishing)"
            if a.compare and truth["sp"] is not None:
                tp = np.array(list(truth["sp"].position))
                tv = np.array(list(truth["sp"].velocity))
                line += (f" | truth_z={tp[2]:5.3f} pos_err={np.linalg.norm(site_p - tp):.3f}m"
                         f" vel_err={np.linalg.norm(site_v - tv):.3f}m/s")
            print(line, flush=True)


if __name__ == "__main__":
    main()
