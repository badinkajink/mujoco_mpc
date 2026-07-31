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
        self.pos_xy = np.zeros(2)
        self.slip = SlipMonitor(a.slip_thresh, 1.0)   # detection only; fac unused here
        self.cos_tilt = float(np.cos(np.deg2rad(a.upright_tilt_deg)))
        self.zupt_n = 0
        self.zupt_blocked = 0        # gates open but robot not upright (the post-fall case)
        self.slip_infl_n = 0
        self.n = 0
        self.base_height = self.nominal_z
        self.upright = True
        self._res = np.zeros(6)

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
        if a.zupt and zupt_hold(tau, dq, gyro, self.LOAD_IDX, a):
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

    def telemetry(self):
        line = (f"rej={self.rej}"
                f" upright={'Y' if self.upright else 'N'}")
        if self.a.zupt:
            line += (f" zupt={100.0 * self.zupt_n / max(self.n, 1):.0f}%"
                     + (f" (blocked {self.zupt_blocked})" if self.zupt_blocked else ""))
        if self.a.slip_monitor and (self.slip.active or self.slip.events):
            line += (f" slip={'ACTIVE' if self.slip.active else 'ok'}"
                     f"({self.slip.events} ev, ema {self.slip.ema:.2f})")
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
    v.add_argument("--stale-ms", type=float, default=300.0,
                   help="stop publishing when rt/lowstate goes this stale (0 = never)")
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
    print(f"[est] v3 publishing -> '{a.out_topic}' @ {a.rate:.0f}Hz | RW-EKF "
          f"(amax={a.ekf_amax} r0={a.ekf_r0} r1={a.ekf_r1} chi2={a.ekf_chi2} "
          f"rejcap={a.ekf_rejcap}) | waist-fix={'ON' if a.waist_fix else 'OFF'} | "
          f"zupt={'ON (upright-gated %g/%g deg)' % (a.upright_z_frac, a.upright_tilt_deg) if a.zupt else 'off'} | "
          f"slip-mon={'%.2fm/s x%g' % (a.slip_thresh, a.slip_infl) if a.slip_monitor else 'off'} | "
          f"stale={a.stale_ms:.0f}ms"
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
        omega_w = R @ gyro
        site_p = np.array([out["xy"][0], out["xy"][1], out["base_height"]]) + roff
        site_v = out["v"] + np.cross(omega_w, roff)

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
