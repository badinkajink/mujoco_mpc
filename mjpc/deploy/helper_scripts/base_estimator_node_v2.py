#!/usr/bin/env python3
"""base_estimator_node_v2.py -- the perfected proprioceptive floating-base estimator.

V2 = v1's reference-design kf12 promoted to THE filter (no legacy rw-ekf /
complementary paths -- v1 keeps those for A/B), with perfected defaults
(sole tracking, waist fix, accel auto-probe, hang prior, v-clamp, ZUPT ON),
plus the three gaps the 2026-07-29 literature audit left open in v1:

  1. STALENESS GUARD (--stale-ms). v1 has no freshness contract: if rt/lowstate
     dies, the loop keeps re-publishing the last sample as current state forever
     -- the planner then seeds every rollout with yesterday's robot. v2 counts
     subscriber callbacks; past the deadline it STOPS publishing and says so.
     Silence is a failure the downstream node can see (--require_sportstate
     trips); a confidently frozen state is not.

  2. STATIC ACCEL-BIAS LEARNING (--bias-tau / --no-bias-learn). kf12 integrates
     u = R@a + g with NO bias state, so residual world accel -- 2.1 deg of
     attitude error == 0.36 m/s^2, the measured 51-metre hang divergence --
     accrues as velocity drift until a ZUPT clips the symptom. v2 learns the
     residual itself: during PROVABLY static intervals (the same both-feet-
     loaded + joints-static + base-not-rotating gate the ZUPT uses, so no new
     failure modes) the propagation input should read exactly zero; whatever it
     reads instead IS the bias, learned with a ~2s time constant and subtracted
     from u thereafter. Static-interval bias observability is the standard EKF
     result (Bloesch RSS'12 carries b_a in the state; this is the cheap gated
     equivalent that cannot be corrupted by motion -- the ice-skating blindspot
     is excluded because a foot must be LOADED, not merely still, to open the
     gate... and a base sliding with planted loaded feet is exactly what the
     leg odometry DOES see, unlike the old |dq|-only static test).

  3. FEET-DISAGREEMENT SLIP MONITOR (--slip-thresh / --no-slip-monitor). Two
     planted feet are two independent measurements of the SAME base velocity;
     their disagreement is free evidence that at least one contact assumption
     is a lie (slip, heel peel, calibration). v1 lets the Mahalanobis gate
     judge each foot against the PRIOR -- slow coherent slip inside the gate
     passes. v2 additionally judges the feet against EACH OTHER: when the
     low-passed disagreement exceeds the threshold, both trusts are scaled
     down (we know someone is lying, not who) and the event is logged.

Everything load-bearing -- KF12, FootTrust (measured contact + blind-knee
hold), AccelProbe, pelvis_from_torso, kin_snapshot, grf_from_torque, ZUPT,
hang prior, iface pinning, model calibration -- is IMPORTED from
base_estimator_node.py (same folder) rather than forked: a fix there lands
here. v1 stays byte-identical as the rollback / A/B baseline.

STRUCTURE: the whole per-tick pipeline lives in `EstimatorCore.step()` so
offline harnesses (estimator_ab.py --est v2) run EXACTLY the code the live
node runs; main() is only DDS plumbing around it. `build_parser()` is the
single source of defaults for both.

Run in the twin venv (has mujoco + unitree_sdk2py):
  cd ~/Desktop/h12/h1_mujoco
  # offline self-test (no DDS, no robot) -- runs v1's kf12 suite + the v2 additions:
  .venv/bin/python ~/Desktop/h12/mujoco_mpc/mujoco_mpc/mjpc/deploy/helper_scripts/base_estimator_node_v2.py --selftest
  # live (auto-pins the robot NIC), publishes rt/sportmodestate:
  .venv/bin/python .../base_estimator_node_v2.py
  # twin validation against ground truth:
  .venv/bin/python .../base_estimator_node_v2.py --out-topic rt/sportmodestate_est --compare
"""
import argparse
import time

import numpy as np
import mujoco

import base_estimator_node as v1
from base_estimator_node import (
    NJ, TORSO_MOTOR, LEG_MOTOR, LEG_DOF, ANKLE_PITCH_MOTOR, ANKLE_ROLL_MOTOR,
    KF12, FootTrust, AccelProbe,
    pelvis_from_torso, kin_snapshot, grf_from_torque,
    zupt_hold, zupt_fuse_kf12,
    load_model_and_calibrate, _kf12_cfg, _pick_iface, _quat2mat, _mat2quat, _rz,
)


class StaleGuard:
    """Freshness contract on rt/lowstate. `beat()` is called from the DDS callback;
    `fresh(now)` answers whether the newest sample is recent enough to publish on.
    Warns once per outage, announces recovery. Clock-injectable for the selftest."""

    def __init__(self, stale_sec):
        self.stale_sec = stale_sec
        self.n = 0
        self._seen_n = -1
        self._last_new = None
        self.stale = False

    def beat(self):
        self.n += 1                                   # DDS thread: count arrivals only

    def fresh(self, now):
        if self.n != self._seen_n:                    # new sample since last check
            self._seen_n = self.n
            self._last_new = now
        if self._last_new is None:
            return False
        was = self.stale
        self.stale = (now - self._last_new) > self.stale_sec
        if self.stale and not was:
            print(f"[est] STALE: no new rt/lowstate for >{self.stale_sec * 1e3:.0f}ms -> "
                  f"NOT publishing (a frozen state seeding the planner is worse than silence)",
                  flush=True)
        elif was and not self.stale:
            print("[est] rt/lowstate recovered -> publishing again", flush=True)
        return not self.stale


class BiasLearner:
    """World-frame residual of the propagation input, learned ONLY while the robot
    is provably static (the ZUPT gate). At true rest u must be exactly zero; what
    it reads instead is attitude-leak + accel bias -- the very residual that
    integrated 0.975m -> 51m on the hanging robot. First-order learn, hard norm cap
    (a residual bigger than bias_max means the attitude is broken, not biased --
    do not launder that into a 'calibration')."""

    def __init__(self, tau, bmax):
        self.tau = tau
        self.bmax = bmax
        self.b = np.zeros(3)
        self.learned_t = 0.0

    def step(self, u, dt, static):
        if static and self.tau > 0.0:
            self.b += (dt / max(self.tau, dt)) * (u - self.b)
            n = float(np.linalg.norm(self.b))
            if n > self.bmax:
                self.b *= self.bmax / n
            self.learned_t += dt
        return u - self.b


class SlipMonitor:
    """Cross-check the two planted feet against each other. Each planted foot
    implies a base velocity; ||v_L - v_R|| should be ~0. A sustained gap means at
    least one planted-foot assumption is false (slip / heel peel / calibration) --
    we cannot tell WHICH, so both trusts are scaled down and the filter leans on
    its prior + propagation until the feet agree again. Low-passed so a single
    noisy tick cannot trip it; only consulted when both feet are in confident
    stance (a swing foot disagreeing is normal and already handled by trust)."""

    def __init__(self, thresh, fac, tau=0.10):
        self.thresh = thresh
        self.fac = fac
        self.tau = tau
        self.ema = 0.0
        self.active = False
        self.events = 0

    def step(self, snap, trusts, contacts, dt):
        if not (contacts[0] and contacts[1] and min(trusts) > 0.5):
            self.ema += (dt / (self.tau + dt)) * (0.0 - self.ema)
            self.active = False
            return trusts
        d = float(np.linalg.norm(snap["vs"][0] - snap["vs"][1]))
        self.ema += (dt / (self.tau + dt)) * (d - self.ema)
        was = self.active
        self.active = self.ema > self.thresh
        if self.active and not was:
            self.events += 1
        if self.active:
            return [t * self.fac for t in trusts]
        return trusts


class EstimatorCore:
    """The entire v2 per-tick pipeline, DDS-free, so offline harnesses run the
    EXACT live code. One instance per robot; call step() once per lowstate sample.

    imu_is_pelvis: the live robot's IMU rides the TORSO (far side of torso_joint),
    so quat/gyro need the waist correction. An offline harness that feeds the
    TRUE pelvis orientation (e.g. the twin's free-joint quat) sets this True and
    the correction is skipped -- applying it there would INJECT the waist error
    it exists to remove."""

    def __init__(self, a, m, data, foot_ids, height_C, imu_is_pelvis=False,
                 verbose=True):
        self.a = a
        self.m, self.data, self.foot_ids = m, data, foot_ids
        self.kcfg = _kf12_cfg(a, m)
        self.imu_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        if self.imu_sid < 0:
            raise SystemExit("[est] v2 needs an 'imu' site in the model")
        if len(foot_ids) != 2:
            raise SystemExit(f"[est] v2 is a 2-foot port; model has {len(foot_ids)} feet")
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.sole_off = height_C if a.sole else 0.0
        self.ground_ref = 0.0 if a.sole else height_C
        self.kf = KF12(self.kcfg, self.ground_ref)
        self.trust_fsm = [FootTrust("left" if "left" in nm else "right", self.kcfg)
                          for _, nm in foot_ids]
        self.probe = AccelProbe(a.accel_mode, a.static_dq, a.accel_probe_sec)
        self.bias = BiasLearner(a.bias_tau if a.bias_learn else 0.0, a.bias_max)
        self.slip = SlipMonitor(a.slip_thresh, a.slip_fac)
        self.LOAD_IDX = [(3, 4) if "left" in nm else (9, 10) for _, nm in foot_ids]
        self.imu_is_pelvis = imu_is_pelvis
        self.verbose = verbose
        self.dt_nom = 1.0 / a.rate
        self.seeded = False
        self.noc_t = 0.0
        self.hanging = False
        self.hang_warned = False
        self.v_clamp_warned = False
        self.zupt_n = 0
        self.n = 0
        self.fz_dbg, self.fzs_dbg = [0.0, 0.0], [0.0, 0.0]
        self._a_world_last = np.zeros(3)
        self.trusts, self.contacts = [0.0, 0.0], [False, False]

    def step(self, q, dq, tau, quat, gyro, accel, step_dt):
        """One proprioceptive sample -> the estimate.
        q/dq/tau: 27-vector motor order; quat wxyz + gyro body-frame (torso IMU, or
        pelvis if imu_is_pelvis); accel body-frame specific force.
        Returns dict(site_p, site_v, base_height, roff, static, hanging)."""
        a = self.a
        q = np.asarray(q, float)
        dq = np.asarray(dq, float)
        tau = np.asarray(tau, float)
        quat = np.asarray(quat, float)
        gyro = np.asarray(gyro, float)
        R = _quat2mat(quat)
        if self.imu_is_pelvis:
            R_wp, omega_p = R, gyro
        else:
            R_wp, omega_p = pelvis_from_torso(quat, gyro, q[TORSO_MOTOR], dq[TORSO_MOTOR])
        kdt = min(step_dt, self.dt_nom * a.dt_clamp)   # a big dt in B@u is a velocity KICK
        snap = kin_snapshot(self.m, self.data, self.foot_ids, self.imu_sid, q, dq,
                            R_wp, omega_p, self.jacp, self.jacr, self.sole_off)

        # --- propagation input + static-gated bias learning ----------------------
        a_world = R @ np.asarray(accel, float)
        if not np.all(np.isfinite(a_world)) or np.linalg.norm(a_world) > a.accel_clamp_g * 9.81:
            a_world = self._a_world_last              # spike / NaN -> reuse last
        else:
            self._a_world_last = a_world
        self.probe.step(a_world, float(np.linalg.norm(dq)), kdt)
        static = zupt_hold(tau, dq, gyro, self.LOAD_IDX, a)
        u = self.bias.step(self.probe.input(a_world), kdt,
                           static and self.probe.resolved is not None)

        # --- per-foot contact + trust (measured; no gait clock exists) ------------
        trusts, contacts, rolling = [], [], []
        for k, (bid, nm) in enumerate(self.foot_ids):
            side = "left" if "left" in nm else "right"
            f, fz_sig = grf_from_torque(self.m, self.data, bid, LEG_DOF[side],
                                        tau[LEG_MOTOR[side]], self.jacp, self.jacr,
                                        self.kcfg.tau_sigma)
            self.fz_dbg[k] = float(f[2])
            self.fzs_dbg[k] = fz_sig
            foot_z = self.kf.x[2] + snap["pf"][k][2]
            t = self.trust_fsm[k].step(kdt, self.fz_dbg[k], fz_sig,
                                       float(tau[ANKLE_PITCH_MOTOR[side]]),
                                       float(tau[ANKLE_ROLL_MOTOR[side]]),
                                       foot_z, float(np.linalg.norm(snap["vfw"][k])),
                                       self.ground_ref)
            trusts.append(t)
            contacts.append(self.trust_fsm[k].contact)
            rolling.append(self.trust_fsm[k].rolling)

        if a.slip_monitor:
            trusts = self.slip.step(snap, trusts, contacts, kdt)

        if not self.seeded:
            bh = -min(float(snap["pf"][k][2] + snap["roff"][2])
                      for k in range(len(self.foot_ids))) + self.ground_ref
            self.kf.x[0:3] = np.array([0.0, 0.0, bh]) + snap["roff"]
            self.kf.reset_feet(snap["pf"])
            self.seeded = True
            if self.verbose:
                print(f"[est] v2 seeded: base_z={bh:.3f} site_z={self.kf.x[2]:.3f} "
                      f"fz=[{self.fz_dbg[0]:.0f},{self.fz_dbg[1]:.0f}]N "
                      f"(expect ~{self.kcfg.mg/2:.0f} each standing) "
                      f"fz_sigma=[{self.fzs_dbg[0]:.0f},{self.fzs_dbg[1]:.0f}]N"
                      + ("  *** KNEES TOO STRAIGHT: f_z is noise, holding contact on kinematics"
                         if (self.trust_fsm[0].blind or self.trust_fsm[1].blind) else ""),
                      flush=True)

        # --- no-contact divergence guard (51m bug) --------------------------------
        self.noc_t = 0.0 if any(contacts) else self.noc_t + kdt
        self.hanging = a.noc_coast_sec > 0.0 and self.noc_t > a.noc_coast_sec
        if self.hanging:
            u = np.zeros(3)
        self.kf.predict(u, kdt, trusts, contacts)
        self.kf.update(snap, trusts, contacts, rolling)
        if self.hanging:
            bh_kin = -min(float(snap["pf"][k][2] + snap["roff"][2])
                          for k in range(len(self.foot_ids))) + self.ground_ref
            self.kf.hang_prior(bh_kin + snap["roff"][2], a.zupt_r)
            if not self.hang_warned and self.verbose:
                self.hang_warned = True
                print(f"[est] no contact >{a.noc_coast_sec:g}s -> NOT flying; safe fallback "
                      f"(v->0, z->kinematic). Feet unloaded / harness?", flush=True)
        else:
            self.hang_warned = False
        if self.kf.clamp_v(a.v_clamp) and not self.v_clamp_warned:
            self.v_clamp_warned = True
            if self.verbose:
                print(f"[est] WARN: |v| hit the {a.v_clamp} m/s backstop -- filter diverging; "
                      f"do NOT trust this estimate in-loop.", flush=True)

        if a.zupt and static:
            zupt_fuse_kf12(self.kf, a.zupt_vr)
            self.zupt_n += 1

        self.trusts, self.contacts = trusts, contacts
        self.n += 1
        site_p = self.kf.x[0:3].copy()
        return dict(site_p=site_p, site_v=self.kf.x[3:6].copy(),
                    base_height=float(site_p[2] - snap["roff"][2]),
                    roff=snap["roff"].copy(), static=static, hanging=self.hanging)

    def telemetry(self):
        ct = ("L" if self.trust_fsm[0].contact else "-") + \
             ("R" if self.trust_fsm[1].contact else "-")
        blind = "".join("B" if t.blind else "." for t in self.trust_fsm)
        line = (f"{ct} trust=[{self.trusts[0]:.2f},{self.trusts[1]:.2f}]"
                f" fz=[{self.fz_dbg[0]:4.0f},{self.fz_dbg[1]:4.0f}]"
                f"+-[{self.fzs_dbg[0]:3.0f},{self.fzs_dbg[1]:3.0f}]N"
                f" blind={blind} gate=[{int(self.kf.gated[0])}{int(self.kf.gated[1])}]"
                f" acc={self.probe.resolved or 'probing'}")
        if self.a.zupt:
            line += f" zupt={100.0 * self.zupt_n / max(self.n, 1):.0f}%"
        if self.a.bias_learn and self.bias.learned_t > 0.0:
            b = self.bias.b
            line += f" bias=[{b[0]:+.2f},{b[1]:+.2f},{b[2]:+.2f}]"
        if self.a.slip_monitor and (self.slip.active or self.slip.events):
            line += (f" slip={'ACTIVE' if self.slip.active else 'ok'}"
                     f"({self.slip.events} ev, ema {self.slip.ema:.2f})")
        if self.hanging:
            line += f" HANG({self.noc_t:.1f}s)"
        return line


def _selftest_v2(m, data, foot_ids, height_C, home_q, a):
    """v2-only offline checks (v1's kf12 suite runs separately)."""
    ok = True
    dt = 0.005

    # 1. BiasLearner: a constant static residual must be learned away, and a
    #    residual beyond the cap must be refused (broken attitude != bias).
    resid = np.array([0.12, 0.36, 0.115])            # the measured 51m-bug residual
    bl = BiasLearner(a.bias_tau, a.bias_max)
    u = resid
    for _ in range(int(6 * a.bias_tau / dt)):        # 6 time constants, static
        u = bl.step(resid, dt, static=True)
    good = float(np.linalg.norm(u)) < 0.05 * float(np.linalg.norm(resid))
    ok &= good
    print(f"[selftest] v2 bias learn   : residual {np.round(resid, 2)} -> corrected "
          f"|u|={np.linalg.norm(u):.4f} m/s^2 ({'ok' if good else 'FAIL'})")
    bl2 = BiasLearner(a.bias_tau, a.bias_max)
    for _ in range(int(6 * a.bias_tau / dt)):
        bl2.step(np.array([0.0, 0.0, 5.0]), dt, static=True)
    good = abs(float(np.linalg.norm(bl2.b)) - a.bias_max) < 1e-6
    ok &= good
    print(f"[selftest] v2 bias cap     : 5.0 m/s^2 'residual' capped at |b|="
          f"{np.linalg.norm(bl2.b):.2f} (cap {a.bias_max}) ({'ok' if good else 'FAIL'})")
    bl3 = BiasLearner(a.bias_tau, a.bias_max)
    for _ in range(400):
        bl3.step(resid, dt, static=False)            # moving -> must NOT learn
    good = float(np.linalg.norm(bl3.b)) < 1e-12
    ok &= good
    print(f"[selftest] v2 bias gated   : non-static samples learned |b|="
          f"{np.linalg.norm(bl3.b):.1e} (expect 0) ({'ok' if good else 'FAIL'})")

    # 2. SlipMonitor: agreement passes trust through; sustained disagreement in
    #    confident double stance scales it down; a swing foot never trips it.
    sm = SlipMonitor(a.slip_thresh, a.slip_fac)
    agree = {"vs": [np.array([0.1, 0.0, 0.0]), np.array([0.11, 0.0, 0.0])]}
    for _ in range(100):
        t = sm.step(agree, [1.0, 1.0], [True, True], dt)
    good = t == [1.0, 1.0] and not sm.active
    ok &= good
    print(f"[selftest] v2 slip agree   : trusts untouched={t == [1.0, 1.0]} "
          f"ema={sm.ema:.3f} ({'ok' if good else 'FAIL'})")
    slip = {"vs": [np.array([0.1, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])]}
    for _ in range(100):
        t = sm.step(slip, [1.0, 1.0], [True, True], dt)
    good = sm.active and t[0] == a.slip_fac and sm.events == 1
    ok &= good
    print(f"[selftest] v2 slip caught  : 0.4 m/s disagreement -> active={sm.active} "
          f"trust {1.0}->{t[0]:.2f} events={sm.events} ({'ok' if good else 'FAIL'})")
    sm2 = SlipMonitor(a.slip_thresh, a.slip_fac)
    for _ in range(100):
        t = sm2.step(slip, [1.0, 0.0], [True, False], dt)   # right foot swinging
    good = not sm2.active and t == [1.0, 0.0]
    ok &= good
    print(f"[selftest] v2 slip vs swing: swing-foot disagreement ignored={not sm2.active} "
          f"({'ok' if good else 'FAIL'})")

    # 3. StaleGuard: fresh while beating, stale past the deadline, recovers on beat.
    sg = StaleGuard(0.3)
    t0 = 100.0
    sg.beat()
    f1 = sg.fresh(t0)                                # just arrived -> fresh
    f2 = sg.fresh(t0 + 0.2)                          # 200ms, no beat -> still fresh
    f3 = sg.fresh(t0 + 0.4)                          # 400ms, no beat -> STALE
    sg.beat()
    f4 = sg.fresh(t0 + 0.45)                         # new beat -> fresh again
    good = f1 and f2 and (not f3) and f4
    ok &= good
    print(f"[selftest] v2 stale guard  : fresh={f1} fresh@200ms={f2} stale@400ms={not f3} "
          f"recovered={f4} ({'ok' if good else 'FAIL'})")

    # 4. END-TO-END filter math: planted + static with a deliberate accel residual:
    #    the bias must be learned and |v| must settle ~0.
    cfg = _kf12_cfg(a, m)
    imu_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    sole_off = height_C if a.sole else 0.0
    ground_ref = 0.0 if a.sole else height_C
    qj = np.array(home_q[7:7 + NJ])
    R_wp, om_p = pelvis_from_torso(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 0.0, 0.0)
    snap = kin_snapshot(m, data, foot_ids, imu_sid, qj, np.zeros(NJ), R_wp, om_p,
                        jacp, jacr, sole_off)
    bh = -min(float(snap["pf"][k][2] + snap["roff"][2]) for k in range(2)) + ground_ref
    kf = KF12(cfg, ground_ref)
    kf.x[0:3] = np.array([0.0, 0.0, bh]) + snap["roff"]
    kf.reset_feet(snap["pf"])
    bl = BiasLearner(a.bias_tau, a.bias_max)
    for _ in range(int(10.0 / dt)):                  # 10s planted+static, biased accel
        u = bl.step(resid, dt, static=True)          # static gate open the whole time
        kf.predict(u, dt, [1.0, 1.0], [True, True])
        kf.update(snap, [1.0, 1.0], [True, True], [False, False])
    vn = float(np.linalg.norm(kf.x[3:6]))
    good = vn < 5e-3
    ok &= good
    print(f"[selftest] v2 e2e biased   : 10s static under {np.round(resid, 2)} m/s^2 "
          f"residual -> |v|={vn:.4f} (expect ~0, bias absorbed) ({'ok' if good else 'FAIL'})")

    # 5. EstimatorCore.step END-TO-END: the exact object offline harnesses drive.
    #    Bent-knee stance (f_z observable), synthetic tau for mg/2 per foot, rest
    #    accel = pure gravity reaction. Expect: probe resolves SPECIFIC, both feet
    #    latch contact, |v| ~ 0, base_height ~ FK height.
    core = EstimatorCore(a, m, data, foot_ids, height_C, imu_is_pelvis=True,
                         verbose=False)
    q_st = np.zeros(NJ)
    for s in ("left", "right"):
        q_st[LEG_MOTOR[s][1]] = -0.16
        q_st[LEG_MOTOR[s][3]] = 0.36
        q_st[LEG_MOTOR[s][4]] = -0.20
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    # synthesize the leg torques a real mg/2 support force produces at this pose
    snap_st = kin_snapshot(m, data, foot_ids, imu_sid, q_st, np.zeros(NJ),
                           np.eye(3), np.zeros(3), jacp, jacr, sole_off)
    tau_st = np.zeros(NJ)
    for bid, nm in foot_ids:
        side = "left" if "left" in nm else "right"
        f_true = np.array([0.0, 0.0, cfg.mg / 2.0])
        mujoco.mj_jacBody(m, data, jacp, jacr, bid)
        tau_st[LEG_MOTOR[side]] = data.qfrc_bias[LEG_DOF[side]] - \
            jacp[:, LEG_DOF[side]].T @ f_true
    accel_rest = np.array([0.0, 0.0, 9.81])          # identity attitude, at rest
    out = None
    for _ in range(600):                             # 3 s
        out = core.step(q_st, np.zeros(NJ), tau_st, ident, np.zeros(3),
                        accel_rest, dt)
    bh_expect = -min(float(snap_st["pf"][k][2] + snap_st["roff"][2])
                     for k in range(2)) + core.ground_ref
    vn = float(np.linalg.norm(out["site_v"]))
    good = (core.probe.resolved == "specific" and all(core.contacts)
            and vn < 5e-3 and abs(out["base_height"] - bh_expect) < 0.02)
    ok &= good
    print(f"[selftest] v2 core 3s stand: acc={core.probe.resolved} contacts={core.contacts} "
          f"|v|={vn:.4f} base_z={out['base_height']:.3f} (expect ~{bh_expect:.3f}) "
          f"({'ok' if good else 'FAIL'})")

    return ok


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="H1-2 MuJoCo scene (default: handless twin scene)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None, help="pin NIC; default = auto-pin robot subnet")
    ap.add_argument("--no-auto-iface", action="store_true",
                    help="force SDK autodetermine (use for the same-host TWIN)")
    ap.add_argument("--rate", type=float, default=200.0, help="publish Hz")
    ap.add_argument("--tick-dt", type=float, default=0.0,
                    help="PLANT-CLOCK the filter: seconds of plant time per lowstate tick "
                         "(see v1's --tick-dt rationale). 0 = wall-clocked.")
    ap.add_argument("--lowstate-topic", default="rt/lowstate")
    ap.add_argument("--out-topic", default="rt/sportmodestate")
    ap.add_argument("--static-dq", type=float, default=0.3,
                    help="joint-vel-norm below which a sample counts as static (accel probe)")
    ap.add_argument("--compare", action="store_true",
                    help="subscribe the TRUTH sportmodestate and print estimate-vs-truth error")
    ap.add_argument("--truth-topic", default="rt/sportmodestate")
    ap.add_argument("--selftest", action="store_true",
                    help="offline math checks (v1 kf12 suite + v2 additions), then exit")

    g = ap.add_argument_group("v2 additions")
    g.add_argument("--stale-ms", type=float, default=300.0,
                   help="stop publishing when rt/lowstate goes this stale. The planner "
                        "re-seeds every rollout from this state; 300ms-old state is a "
                        "different robot. 0 = v1 behaviour (publish forever).")
    g.add_argument("--bias-tau", type=float, default=2.0,
                   help="accel-residual learning time constant (s) during provably-static "
                        "intervals (the ZUPT gate). 0 = disable learning.")
    g.add_argument("--no-bias-learn", dest="bias_learn", action="store_false", default=True)
    g.add_argument("--bias-max", type=float, default=1.0,
                   help="hard cap on the learned residual norm (m/s^2). 0.36 was the "
                        "measured 2.1-deg-tilt leak; anything near 1.0 means the attitude "
                        "is broken, not biased.")
    g.add_argument("--slip-thresh", type=float, default=0.15,
                   help="low-passed ||v_L - v_R|| (m/s) between two confident planted feet "
                        "above which at least one contact assumption is a lie")
    g.add_argument("--slip-fac", type=float, default=0.3,
                   help="trust multiplier applied to BOTH feet while the slip monitor is "
                        "active (we know someone lies, not who)")
    g.add_argument("--no-slip-monitor", dest="slip_monitor", action="store_false", default=True)
    g.add_argument("--no-zupt", dest="zupt", action="store_false", default=True,
                   help="v2 default ON (v1: opt-in). The gyro gate makes it safe for the "
                        "rigid-topple case; see v1's --zupt-gyro note.")

    # ---- kf12 knobs: same names + defaults as v1 (consumed by _kf12_cfg / suite) ----
    k = ap.add_argument_group("kf12 (see v1 for full rationale)")
    k.add_argument("--accel-mode", choices=["auto", "specific", "linear", "off"], default="auto")
    k.add_argument("--accel-probe-sec", type=float, default=1.0)
    k.add_argument("--suspect", type=float, default=100.0)
    k.add_argument("--p0", type=float, default=100.0)
    k.add_argument("--q-pos", type=float, default=0.02)
    k.add_argument("--q-vel", type=float, default=0.02)
    k.add_argument("--q-foot", type=float, default=0.002)
    k.add_argument("--r-fpos", type=float, default=0.01)
    k.add_argument("--r-fvel", type=float, default=1.0)
    k.add_argument("--r-fh", type=float, default=1.0)
    k.add_argument("--kf-chi2", type=float, default=16.27)
    k.add_argument("--fz-hi", type=float, default=0.15)
    k.add_argument("--fz-lo", type=float, default=0.05)
    k.add_argument("--strike-ms", type=float, default=5.0)
    k.add_argument("--min-stance-ms", type=float, default=50.0)
    k.add_argument("--tau-sigma", type=float, default=1.0)
    k.add_argument("--fz-sigma-max", type=float, default=25.0)
    k.add_argument("--td-ramp-ms", type=float, default=35.0)
    k.add_argument("--kin-h", type=float, default=0.03)
    k.add_argument("--kin-v", type=float, default=0.15)
    k.add_argument("--kin-fail", type=float, default=0.3)
    k.add_argument("--no-sole", dest="sole", action="store_false", default=True)
    k.add_argument("--cop", action="store_true")
    k.add_argument("--cop-margin", type=float, default=0.15)
    k.add_argument("--cop-fail", type=float, default=0.5)
    k.add_argument("--roll-infl", type=float, default=10.0)
    k.add_argument("--sole-fwd", type=float, default=0.133)
    k.add_argument("--sole-back", type=float, default=0.079)
    k.add_argument("--sole-half-y", type=float, default=0.04)
    k.add_argument("--noc-coast-sec", type=float, default=0.5)
    k.add_argument("--zupt-load", type=float, default=15.0)
    k.add_argument("--zupt-dq", type=float, default=0.15)
    k.add_argument("--zupt-gyro", type=float, default=0.05)
    k.add_argument("--zupt-vr", type=float, default=0.01)
    k.add_argument("--zupt-r", type=float, default=1.0)
    k.add_argument("--v-clamp", type=float, default=4.0)
    k.add_argument("--accel-clamp-g", type=float, default=4.0)
    k.add_argument("--dt-clamp", type=float, default=1.5)
    return ap


def default_args():
    """The perfected v2 defaults as a namespace (for offline harnesses)."""
    a = build_parser().parse_args([])
    a.filter = "kf12"
    return a


def main():
    a = build_parser().parse_args()
    a.filter = "kf12"                                # v1 helpers key off this

    m, data, foot_ids, height_C, home_q = load_model_and_calibrate(a.scene)

    if a.selftest:
        ok1 = v1._selftest_kf12(m, data, foot_ids, height_C, home_q, a)
        ok2 = _selftest_v2(m, data, foot_ids, height_C, home_q, a)
        print(f"[selftest] {'PASS' if (ok1 and ok2) else 'FAIL'} "
              f"(v1 kf12 suite={'ok' if ok1 else 'FAIL'}, v2 additions={'ok' if ok2 else 'FAIL'})")
        raise SystemExit(0 if (ok1 and ok2) else 1)

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
            print("[est] WARN: --compare with out-topic==truth-topic -> you'll hear your own "
                  "publish; use --out-topic rt/sportmodestate_est", flush=True)

    pub = ChannelPublisher(a.out_topic, SportModeState_)
    pub.Init()
    msg = unitree_go_msg_dds__SportModeState_()

    core = EstimatorCore(a, m, data, foot_ids, height_C, imu_is_pelvis=False,
                         verbose=True)
    print(f"[est] v2 publishing -> '{a.out_topic}' @ {a.rate:.0f}Hz | KF12 "
          f"(suspect={core.kcfg.suspect:g} chi2={core.kcfg.chi2:g} accel={a.accel_mode}) | "
          f"mg={core.kcfg.mg:.1f}N strike/release {core.kcfg.fz_hi:.0f}/{core.kcfg.fz_lo:.0f}N | "
          f"zupt={'ON' if a.zupt else 'off'} | stale={a.stale_ms:.0f}ms | "
          f"bias-learn={'tau%.1fs cap%.1f' % (a.bias_tau, a.bias_max) if a.bias_learn else 'off'} | "
          f"slip-mon={'%.2fm/s x%.1f' % (a.slip_thresh, a.slip_fac) if a.slip_monitor else 'off'}"
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
        publish_ok = stale.fresh(now)                # check BEFORE consuming the sample
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
        accel = np.array(list(ls.imu_state.accelerometer))   # body frame (specific force)

        out = core.step(q, dq, tau, quat, gyro, accel, step_dt)

        if publish_ok:
            for k2 in range(3):
                msg.position[k2] = float(out["site_p"][k2])
                msg.velocity[k2] = float(out["site_v"][k2])
            pub.Write(msg)

        n += 1
        if n % int(a.rate) == 0:
            sp, sv = out["site_p"], out["site_v"]
            line = (f"[est] {now - t0:6.1f}s  base_z={out['base_height']:5.3f}  "
                    f"xy=[{sp[0]:+.2f},{sp[1]:+.2f}]  "
                    f"v=[{sv[0]:+.3f},{sv[1]:+.3f},{sv[2]:+.3f}] m/s  | "
                    + core.telemetry())
            if stale.stale:
                line += " STALE(not publishing)"
            if a.compare and truth["sp"] is not None:
                tp = np.array(list(truth["sp"].position))
                tv = np.array(list(truth["sp"].velocity))
                line += (f" | truth_z={tp[2]:5.3f} pos_err={np.linalg.norm(sp - tp):.3f}m"
                         f" vel_err={np.linalg.norm(sv - tv):.3f}m/s")
            print(line, flush=True)


if __name__ == "__main__":
    main()
