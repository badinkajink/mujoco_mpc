#!/usr/bin/env python3
"""base_estimator_node_v4.py -- v3 + the auxiliary-measurement bus.

V4 = the v3 deploy candidate (rw-ekf core, waist fix, upright-gated ZUPT, slip
monitor, stale guard -- all unchanged) plus ONE new seam: any external process
may publish a timestamped world-frame odometry measurement on a DDS topic
(default rt/aux_odom, SportModeState_ idl reused: position + velocity fields),
and v4 fuses it. With no aux publisher running, v4 behaves identically to v3 --
the selftest proves byte-equivalence. That makes it safe to deploy first and
light up feeders one at a time:

    lio_bridge_node.py   FAST-LIO /Odometry (Livox MID-360)  -> rt/aux_odom
    (future) NMN head    learned velocity from proprioception -> rt/aux_odom
    (future) foot IMUs   slip/contact verification            -> rt/foot_imu

WHAT GETS FUSED (and what deliberately does not):
  velocity   EKF update on the rw-ekf velocity state, chi2-gated (aux_chi2)
             with noise aux-rv. This is the main prize: an independent,
             drift-free velocity source that does not depend on the
             planted-foot assumption -- it keeps working through slip, brace,
             and single-support, exactly where leg odometry lies.
  xy         gentle anchor pull (first-order, tau aux-xy-tau) toward
             aux_xy + offset, where offset is latched on the FIRST fresh aux
             sample. The odometric xy integral stays continuous (no jumps a
             planner would see as teleports) but stops drifting relative to
             the aux frame. NOT a Kalman update: the odometric xy carries no
             covariance, and pretending it does would be theater.
  z          NOT fused. Kinematic height through a planted foot is accurate
             (mm) and lidar z is the noisier signal indoors. Flag if needed.
  yaw        NOT fused here. The estimator seam publishes position+velocity
             only -- the control node reads orientation straight from the
             IMU on rt/lowstate, so a yaw correction cannot ride this seam.
             Instead the BRIDGE rotates aux measurements into the IMU-world
             frame continuously (see lio_bridge_node.py), which transfers
             LIO's yaw stability into frame-consistent velocities.

FRESHNESS: an aux sample older than --aux-age-max (default 200 ms) is ignored
-- a stale correction seeds the same wrong-state failure the StaleGuard exists
to prevent. A dead bridge therefore degrades v4 to exactly v3, silently and
safely; the telemetry shows aux=IDLE.

Run in the twin venv:
  .venv/bin/python .../base_estimator_node_v4.py --selftest
  .venv/bin/python .../base_estimator_node_v4.py                 # live (aux optional)
"""
import argparse
import time

import numpy as np
import mujoco

from base_estimator_node import (
    NJ, IMU_OFFSET, load_model_and_calibrate, _pick_iface, _quat2mat,
)
from base_estimator_node_v2 import StaleGuard
import base_estimator_node_v3 as v3
from base_estimator_node_v3 import V3Core


class V4Core(V3Core):
    """V3Core + aux fusion. accept_aux() may be called from any thread at any
    rate; step() consumes the newest sample once (fuse-once latch) if fresh."""

    def __init__(self, a, m, data, foot_ids, height_C, nominal_z,
                 imu_is_pelvis=False, verbose=True):
        super().__init__(a, m, data, foot_ids, height_C, nominal_z,
                         imu_is_pelvis=imu_is_pelvis, verbose=verbose)
        self._aux = None            # dict(xy, v, t_mono) -- newest sample
        self._aux_consumed = True
        self.aux_offset_xy = None   # est_xy - aux_xy, latched on first fusion
        self.aux_used = 0
        self.aux_gated = 0
        self.aux_stale = 0
        self.aux_fresh = False
        # REGIME GATE (--aux-gate planted, the default): world-referenced aux
        # (LIO velocity, tag anchor) is only fused while leg odometry is BLIND
        # -- i.e. the feet are NOT trustworthy. Measured 2026-07-30 in-loop:
        # with healthy planted feet, aux re-references the estimate to the
        # WORLD, so real foot creep (invisible+harmless to feet-referenced leg
        # odom, which the controller's trim/balance loops were tuned on) enters
        # the state as apparent drift -> the trim leans the robot to "cancel"
        # it -> forced lean / v1-style foot shuffle (v3 198s vs v4-aux ~50s
        # stands). While deferring, the aux<->est offset keeps re-latching so
        # engagement is seamless when the feet unload (brace load-bearing --
        # the regime the aux sources exist for).
        self.aux_engaged = False
        self.aux_deferred = 0
        self._unplanted_t = 0.0
        self._planted_t = 0.0

    def accept_aux(self, pos_xy, vel_world, t_mono, pos_only=False):
        """Newest-wins mailbox (single ref swap: GIL-atomic, no lock needed).
        pos_only=True (wire: SportModeState_.mode == 2) = a POSITION-ONLY source
        (e.g. tag_bridge: fiducial PnP gives cm-accurate position but 30 Hz
        differenced velocity would be noise) -> only the xy anchor is applied,
        the velocity state is never touched."""
        self._aux = dict(xy=np.asarray(pos_xy, float)[:2].copy(),
                         v=np.asarray(vel_world, float)[:3].copy(),
                         t=float(t_mono), pos_only=bool(pos_only))
        self._aux_consumed = False

    def step(self, q, dq, tau, quat, gyro, step_dt, now=None):
        out = super().step(q, dq, tau, quat, gyro, step_dt)
        a = self.a
        # -- regime gate: are the feet trustworthy right now? ------------------
        loads = [abs(tau[i1]) + abs(tau[i2]) for i1, i2 in self.LOAD_IDX]
        # ONE solidly loaded foot is enough for leg odometry (the EKF weights
        # per-foot trust by load) -- requiring BOTH made every ordinary weight
        # shift open the gate and fuse world-frame aux during exactly the
        # delicate moments (measured 2026-07-30 run: aux windows at each lean).
        # Leg odom is only BLIND when NO foot is loaded (brace/fall/slip).
        foot_loaded = self.upright and any(l >= a.zupt_load for l in loads)
        planted = foot_loaded and not out["slip_active"]
        if planted:
            self._planted_t += step_dt
            self._unplanted_t = 0.0
        else:
            self._unplanted_t += step_dt
            self._planted_t = 0.0
        if getattr(a, "aux_gate", "planted") == "always":
            self.aux_engaged = True
        else:                        # hysteresis: no chatter at the boundary
            if self.aux_engaged and self._planted_t >= 0.5:
                self.aux_engaged = False
            elif not self.aux_engaged and self._unplanted_t >= 0.1:
                self.aux_engaged = True
        aux = self._aux
        self.aux_fresh = False
        if aux is not None and not self._aux_consumed:
            age = (time.monotonic() if now is None else now) - aux["t"]
            if age > a.aux_age_max:
                self.aux_stale += 1
            elif not self.aux_engaged:
                # feet trusted -> defer to leg odometry, but keep the anchor
                # calibration FRESH so engagement continues seamlessly from the
                # trusted state instead of yanking toward a stale offset
                self._aux_consumed = True
                self.aux_offset_xy = (np.zeros(2) if getattr(a, "aux_abs", False)
                                      else self.pos_xy - aux["xy"])
                self.aux_deferred += 1
            else:
                self._aux_consumed = True             # fuse each sample once
                self.aux_fresh = True
                vel_ok = False
                if aux.get("pos_only"):
                    vel_ok = True                     # nothing to gate: no velocity claim
                    self.aux_used += 1
                else:
                    # --- velocity: a real EKF update, chi2-gated ---------------
                    Rv = (a.aux_rv ** 2) * self.I3
                    S = self.P + Rv
                    inn = aux["v"] - self.v
                    d2 = float(inn @ np.linalg.solve(S, inn))
                    if d2 > a.aux_chi2:
                        self.aux_gated += 1           # wild aux: LIO glitch/frame slip
                    else:
                        K = self.P @ np.linalg.inv(S)
                        self.v = self.v + K @ inn
                        self.P = (self.I3 - K) @ self.P
                        self.aux_used += 1
                        vel_ok = True
                # --- xy: continuous anchor pull (kills odometric drift) --------
                # position-only sources get a sanity gate instead of the velocity
                # chi2: an anchor >1 m from the current belief is a mis-solve
                # (wrong tag id / frame flip), not a correction.
                if vel_ok and a.aux_xy_tau > 0.0:
                    if getattr(a, "aux_abs", False):
                        self.aux_offset_xy = np.zeros(2)
                    elif self.aux_offset_xy is None:
                        self.aux_offset_xy = self.pos_xy - aux["xy"]
                    target = aux["xy"] + self.aux_offset_xy
                    if aux.get("pos_only") and \
                            float(np.linalg.norm(target - self.pos_xy)) > 1.0:
                        self.aux_gated += 1
                    else:
                        # TWO-TIER pull: gate opened by SLIP while a foot is
                        # still loaded (stand) -> the anchor only NUDGES
                        # (tau_soft). A brace-aggressive world-frame pull
                        # mid-stand feeds the controller's feet-tuned trim
                        # loop (measured 2026-07-30: a 14 s slip window fused
                        # at tau 2.0 during an in-loop stand). The full-rate
                        # pull is reserved for feet fully UNLOADED
                        # (brace/fall), where leg odometry is blind and aux
                        # is the only truth. 'always' mode keeps the single
                        # configured tau (bench / ride-along semantics).
                        tau_xy = a.aux_xy_tau
                        if (getattr(a, "aux_gate", "planted") != "always"
                                and foot_loaded):
                            tau_xy = getattr(a, "aux_xy_tau_soft", 30.0)
                        alpha = step_dt / (tau_xy + step_dt)
                        self.pos_xy += alpha * (target - self.pos_xy)
                        # ★ 2026-08-28 CONTINUOUS GLIDE: remember the accepted
                        # target so the pull keeps acting EVERY step below.
                        self._aux_glide_target = target.copy()
                        self._aux_glide_tau = tau_xy
                        self._aux_glide_t = (time.monotonic() if now is None else now)
        # ★ 2026-08-28 CONTINUOUS ANCHOR GLIDE (abs-mode real-robot fix). The
        # pull above fires ONCE PER ANCHOR MESSAGE (~6 Hz on the real head-cam
        # bridge) with a per-200Hz-step alpha, so `--aux-xy-tau 2.0` behaved
        # like ~60 s (standing leg-odometry creep won, base walked 14 cm off
        # the anchor), and forcing tau 0.1 made the base JUMP ~1 cm per
        # message -> the node's finite-diff base velocity saw phantom spikes
        # -> planner stood stiff-legged and refused to dive (29_26/29_27).
        # Fix: glide toward the last accepted target on every step with the
        # configured tau (true 2 s time constant, no steps). Stops when the
        # anchor goes stale (5x aux_age_max) or is not engaged.
        tgt = getattr(self, "_aux_glide_target", None)
        if tgt is not None and self.aux_engaged and a.aux_xy_tau > 0.0:
            now_t = (time.monotonic() if now is None else now)
            if now_t - self._aux_glide_t <= 5.0 * a.aux_age_max:
                alpha_g = step_dt / (self._aux_glide_tau + step_dt)
                self.pos_xy += alpha_g * (tgt - self.pos_xy)
            else:
                self._aux_glide_target = None
        out["v"] = self.v.copy()
        out["xy"] = self.pos_xy.copy()
        return out

    def telemetry(self):
        line = super().telemetry()
        if self._aux is None:
            line += " aux=IDLE"
        else:
            state = "ok" if self.aux_fresh else (
                "DEFER" if not self.aux_engaged else "...")
            line += (f" aux={state}"
                     f"(used {self.aux_used}, deferred {self.aux_deferred}, "
                     f"gated {self.aux_gated}, stale {self.aux_stale})")
            # ★ 2026-08-29 operator readout: distance est pelvis <-> last
            # accepted anchor. SETTLED = ready to launch the node.
            tgt = getattr(self, "_aux_glide_target", None)
            if tgt is not None:
                gap = float(np.linalg.norm(tgt - self.pos_xy))
                line += (f" anchor_gap={gap * 100:.1f}cm "
                         + ("SETTLED" if gap < 0.03 else "settling..."))
        return line


def _selftest(m, data, foot_ids, height_C, home_q, a):
    ok = True
    dt = 0.005
    nominal_z = float(home_q[2])
    qj = np.array(home_q[7:7 + NJ])
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    tau_loaded = np.zeros(NJ)
    for i in (3, 4, 9, 10):
        tau_loaded[i] = 10.0

    def mk(cls):
        return cls(a, m, data, foot_ids, height_C, nominal_z, imu_is_pelvis=True,
                   verbose=False)

    # 1. NO AUX => byte-identical to v3 (same inputs, same trajectory).
    c3, c4 = mk(V3Core), mk(V4Core)
    dqm = np.zeros(NJ)
    dqm[3] = 0.4
    worst = 0.0
    for i in range(300):
        o3 = c3.step(qj, dqm, tau_loaded, ident, np.zeros(3), dt)
        o4 = c4.step(qj, dqm, tau_loaded, ident, np.zeros(3), dt, now=0.0)
        worst = max(worst, float(np.linalg.norm(o3["v"] - o4["v"])),
                    float(np.linalg.norm(o3["xy"] - o4["xy"])))
    good = worst == 0.0
    ok &= good
    print(f"[selftest] v4 no-aux == v3 : worst state diff over 300 ticks = {worst:.1e} "
          f"({'ok' if good else 'FAIL'})")

    # Tests 2-6 exercise the FUSION MATH -> pin the regime gate open (their
    # loaded-static scenario counts as planted, which would defer everything).
    a.aux_gate = "always"

    # 2. AUX VELOCITY pulls the estimate: leg odo says moving (joint sweep) but a
    #    fresh trusted aux says v = 0 -> the fused v must be much smaller.
    c4a, c4b = mk(V4Core), mk(V4Core)
    for i in range(300):
        c4b.accept_aux(np.zeros(2), np.zeros(3), t_mono=0.0)
        oa = c4a.step(qj, dqm, tau_loaded, ident, np.zeros(3), dt, now=0.0)
        ob = c4b.step(qj, dqm, tau_loaded, ident, np.zeros(3), dt, now=0.0)
    va, vb = np.linalg.norm(oa["v"]), np.linalg.norm(ob["v"])
    good = vb < 0.5 * va and c4b.aux_used > 0
    ok &= good
    print(f"[selftest] v4 aux velocity : |v| leg-odo-only {va:.3f} -> with aux {vb:.3f} "
          f"(used {c4b.aux_used}) ({'ok' if good else 'FAIL'})")

    # 3. XY ANCHOR: odometric xy drifts with the phantom velocity; aux holds xy
    #    at 0 -> the anchored xy must stay far closer to the latched origin.
    xya, xyb = np.linalg.norm(oa["xy"]), np.linalg.norm(ob["xy"])
    good = xyb < 0.5 * xya
    ok &= good
    print(f"[selftest] v4 aux xy anchor: |xy| drift {xya:.3f} -> anchored {xyb:.3f} "
          f"({'ok' if good else 'FAIL'})")

    # 4. STALE AUX ignored: same as (2) but the aux sample is 1s old.
    c4s = mk(V4Core)
    for i in range(300):
        c4s.accept_aux(np.zeros(2), np.zeros(3), t_mono=-1.0)   # age 1s at now=0
        os_ = c4s.step(qj, dqm, tau_loaded, ident, np.zeros(3), dt, now=0.0)
    good = c4s.aux_used == 0 and c4s.aux_stale > 0 and \
        abs(np.linalg.norm(os_["v"]) - va) < 1e-9
    ok &= good
    print(f"[selftest] v4 stale aux    : used {c4s.aux_used} stale {c4s.aux_stale}, "
          f"estimate untouched ({'ok' if good else 'FAIL'})")

    # 5. WILD AUX gated: a 5 m/s aux against a settled static estimate must be
    #    chi2-rejected, not swallowed.
    c4w = mk(V4Core)
    for i in range(200):                                  # settle static first
        c4w.step(qj, np.zeros(NJ), tau_loaded, ident, np.zeros(3), dt, now=0.0)
    c4w.accept_aux(np.zeros(2), np.array([5.0, 0.0, 0.0]), t_mono=0.0)
    ow = c4w.step(qj, np.zeros(NJ), tau_loaded, ident, np.zeros(3), dt, now=0.0)
    good = c4w.aux_gated == 1 and np.linalg.norm(ow["v"]) < 0.01
    ok &= good
    print(f"[selftest] v4 wild aux gate: gated {c4w.aux_gated}, |v|={np.linalg.norm(ow['v']):.4f} "
          f"(expect ~0) ({'ok' if good else 'FAIL'})")

    # 6. POSITION-ONLY AUX (tag_bridge flavor): anchors xy, never touches v,
    #    and a >1 m anchor jump is rejected as a mis-solve.
    c4p = mk(V4Core)
    for i in range(300):
        c4p.accept_aux(np.zeros(2), np.zeros(3), t_mono=0.0, pos_only=True)
        op = c4p.step(qj, dqm, tau_loaded, ident, np.zeros(3), dt, now=0.0)
    v_untouched = abs(np.linalg.norm(op["v"]) - va) < 1e-9
    # weaker than the velocity-fused arm BY DESIGN: the anchor pulls xy but the
    # phantom velocity keeps pushing between pulls -- expect improvement, not cure
    xy_anchored = np.linalg.norm(op["xy"]) < 0.8 * xya
    c4p.accept_aux(np.array([50.0, 0.0]), np.zeros(3), t_mono=0.0, pos_only=True)
    before = c4p.pos_xy.copy()
    c4p.step(qj, dqm, tau_loaded, ident, np.zeros(3), dt, now=0.0)
    jump_rejected = np.linalg.norm(c4p.pos_xy - before) < 0.01 and c4p.aux_gated >= 1
    good = v_untouched and xy_anchored and jump_rejected
    ok &= good
    print(f"[selftest] v4 pos-only aux : v untouched={v_untouched} xy anchored="
          f"{np.linalg.norm(op['xy']):.3f} (vs {xya:.3f}) 50m-jump rejected={jump_rejected} "
          f"({'ok' if good else 'FAIL'})")

    # 7. REGIME GATE (default 'planted'): while upright with loaded feet, aux is
    #    DEFERRED and the state stays byte-identical to v3; once the feet unload
    #    (brace load-bearing), aux engages and gets used.
    a.aux_gate = "planted"
    c3g, c4g = mk(V3Core), mk(V4Core)
    wrong_aux_v = np.array([0.3, 0.0, 0.0])    # a claim that WOULD move the state
    wrong_aux_xy = np.array([0.5, 0.0])
    worst_g = 0.0
    for i in range(200):     # planted phase: QUIET loaded stand (dq=0 -- a
        # sweeping joint trips the slip monitor, which correctly un-plants)
        c4g.accept_aux(wrong_aux_xy, wrong_aux_v, t_mono=0.0)
        o3g = c3g.step(qj, np.zeros(NJ), tau_loaded, ident, np.zeros(3), dt)
        o4g = c4g.step(qj, np.zeros(NJ), tau_loaded, ident, np.zeros(3), dt,
                       now=0.0)
        worst_g = max(worst_g, float(np.linalg.norm(o3g["v"] - o4g["v"])),
                      float(np.linalg.norm(o3g["xy"] - o4g["xy"])))
    used_planted = c4g.aux_used
    deferred_ok = used_planted == 0 and c4g.aux_deferred > 0 and worst_g == 0.0
    # feet unload (weight onto the arm) AND the pelvis creeps +x at 0.1 m/s --
    # exactly what blind leg odometry misses. Aux (tag-style pos-only) reports
    # it; the anchor must drag xy along. (The deferral re-latch makes the
    # ENGAGEMENT itself seamless -- zero jump -- so only NEW motion pulls.)
    tau_unloaded = np.zeros(NJ)
    xy0 = c4g.pos_xy.copy()
    for i in range(200):
        c4g.accept_aux(np.array([0.1 * i * dt, 0.0]), np.zeros(3),
                       t_mono=0.0, pos_only=True)
        og = c4g.step(qj, np.zeros(NJ), tau_unloaded, ident, np.zeros(3), dt,
                      now=0.0)
    moved = float(np.linalg.norm(c4g.pos_xy - xy0))
    engaged_ok = c4g.aux_engaged and c4g.aux_used > 0 and moved > 0.01
    good = deferred_ok and engaged_ok
    ok &= good
    print(f"[selftest] v4 regime gate  : planted -> deferred {c4g.aux_deferred} "
          f"used {used_planted}, ==v3 diff {worst_g:.1e}; feet unloaded + creep "
          f"-> engaged={c4g.aux_engaged} used {c4g.aux_used} xy followed "
          f"{moved:.3f}m ({'ok' if good else 'FAIL'})")

    # 8. TWO-TIER PULL: gate opened by SLIP with a foot still LOADED (stand)
    #    -> the anchor only nudges (tau_soft); feet fully UNLOADED -> the
    #    full-rate pull. Same anchor step (+0.3 m in y, latched at origin
    #    first), same duration: the unloaded arm must track it far faster.
    a.aux_gate = "planted"
    anchor = np.array([0.0, 0.30])
    dq_slip = np.zeros(NJ)
    dq_slip[3] = 1.2                          # feet disagree -> slip latches

    def pull_after_step(dq_vec, tau_vec):
        c = mk(V4Core)
        for i in range(400):
            c.accept_aux(np.zeros(2) if i < 100 else anchor, np.zeros(3),
                         t_mono=0.0, pos_only=True)
            if i == 99:
                y0 = float(c.pos_xy[1])
            c.step(qj, dq_vec, tau_vec, ident, np.zeros(3), dt, now=0.0)
        return float(c.pos_xy[1]) - y0, c

    soft_dy, c_soft = pull_after_step(dq_slip, tau_loaded)      # slip, loaded
    hard_dy, c_hard = pull_after_step(np.zeros(NJ), np.zeros(NJ))  # unloaded
    good = (c_soft.aux_engaged and c_hard.aux_engaged
            and hard_dy > 0.10 and 0.0 < soft_dy < 0.5 * hard_dy)
    ok &= good
    print(f"[selftest] v4 two-tier pull: slip+loaded dy={soft_dy:+.3f}m (tau "
          f"{a.aux_xy_tau_soft:.0f}s) vs unloaded dy={hard_dy:+.3f}m (tau "
          f"{a.aux_xy_tau:.1f}s) over 1.5s ({'ok' if good else 'FAIL'})")

    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_parser():
    ap = v3.build_parser()
    g = ap.add_argument_group("v4 aux-measurement bus")
    g.add_argument("--aux-topic", default="rt/aux_odom",
                   help="DDS topic external odometry arrives on (SportModeState_ idl: "
                        "position = world pelvis pos, velocity = world pelvis vel, "
                        "already rotated into the IMU-world frame by the bridge)")
    g.add_argument("--aux-rv", type=float, default=0.05,
                   help="aux velocity measurement noise (m/s, 1-sigma). FAST-LIO "
                        "velocity is good to a few cm/s indoors.")
    g.add_argument("--aux-chi2", type=float, default=11.34,
                   help="chi2 gate (3-dof, 99%%) on the aux velocity innovation")
    g.add_argument("--aux-age-max", type=float, default=0.2,
                   help="ignore aux samples older than this (s); a stale correction "
                        "is the StaleGuard failure in miniature")
    g.add_argument("--aux-xy-tau", type=float, default=2.0,
                   help="time constant of the xy anchor pull toward aux (s); 0 = "
                        "velocity fusion only, xy stays pure odometric")
    g.add_argument("--aux-xy-tau-soft", type=float, default=30.0,
                   help="xy anchor time constant (s) used when the regime gate "
                        "engaged via SLIP but a foot is still loaded (standing): "
                        "gentle enough that the world reference cannot fight the "
                        "controller's feet-tuned trim loop. The full-rate "
                        "--aux-xy-tau applies only with feet fully unloaded.")
    g.add_argument("--aux-abs", action="store_true",
                   help="2026-08-13 ABSOLUTE ANCHOR: the aux source publishes "
                        "absolute model-world xy (tag_bridge --abs-world); "
                        "bypass the bring-up offset latch (offset ≡ 0) so the "
                        "belief IS the true robot-vs-table pose. Without this "
                        "flag the latch preserves whatever offset existed at "
                        "first fusion (the flat_7 lottery).")
    g.add_argument("--aux-gate", choices=["planted", "always"], default="planted",
                   help="'planted' (default): fuse aux ONLY while leg odometry is "
                        "blind (feet unloaded / slipping / not upright) -- while "
                        "the feet are trusted, aux is deferred and only the "
                        "anchor offset tracks. Measured 2026-07-30: world-"
                        "referenced aux in quiet stand feeds real foot creep "
                        "into the controller's feet-tuned trim loop -> forced "
                        "lean. 'always' = pre-gate behavior (bench/ride-along).")
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

    core = V4Core(a, m, data, foot_ids, height_C, float(home_q[2]),
                  imu_is_pelvis=False, verbose=True)

    def _on_aux(msg_in):
        core.accept_aux(np.array(list(msg_in.position))[:2],
                        np.array(list(msg_in.velocity)), time.monotonic(),
                        pos_only=(int(msg_in.mode) == 2))

    aux_sub = ChannelSubscriber(a.aux_topic, SportModeState_)
    aux_sub.Init(_on_aux, 10)

    truth = {"sp": None}
    if a.compare:
        truth_sub = ChannelSubscriber(a.truth_topic, SportModeState_)
        truth_sub.Init(lambda msg_in: truth.__setitem__("sp", msg_in), 10)

    pub = ChannelPublisher(a.out_topic, SportModeState_)
    pub.Init()
    msg = unitree_go_msg_dds__SportModeState_()

    print(f"[est] v4 publishing -> '{a.out_topic}' @ {a.rate:.0f}Hz | v3 core "
          f"(waist-fix={'ON' if a.waist_fix else 'off'} zupt={'ON' if a.zupt else 'off'} "
          f"slip-mon={'on' if a.slip_monitor else 'off'} stale={a.stale_ms:.0f}ms) | "
          f"AUX BUS on '{a.aux_topic}' (rv={a.aux_rv} chi2={a.aux_chi2} "
          f"age<={a.aux_age_max * 1e3:.0f}ms xy-tau={a.aux_xy_tau}s)")
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
            print("[est] rt/lowstate up -> estimating + publishing.", flush=True)
            seen = True

        q = np.array([ls.motor_state[i].q for i in range(NJ)])
        dq = np.array([ls.motor_state[i].dq for i in range(NJ)])
        tau = np.array([ls.motor_state[i].tau_est for i in range(NJ)])
        quat = np.array(list(ls.imu_state.quaternion))
        gyro = np.array(list(ls.imu_state.gyroscope))

        out = core.step(q, dq, tau, quat, gyro, step_dt)

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
