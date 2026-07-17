#!/usr/bin/env python3
"""base_estimator_node.py -- proprioceptive leg-odometry floating-base estimator.

DEBUG MODE has no high-level estimator, so rt/sportmodestate (the base world
pose/velocity the MJPC control node needs) is silent. This node SYNTHESISES it
from REAL proprioception ONLY -- the robot's IMU (orientation + gyro) and joint
encoders (q, dq) on rt/lowstate -- via leg odometry, and publishes
rt/sportmodestate. The MJPC node then runs UNMODIFIED (default
--require_sportstate=true); it can't tell this apart from the factory estimator.

AUTHENTIC: uses ONLY real robot sensors. No simulation ground truth, no motion
capture. Base linear velocity is NEVER directly measured on any legged robot --
it is always estimated; the factory rt/sportmodestate.velocity is itself an
estimator output. This is the same class of computed quantity, just transparent.

METHOD -- leg odometry, planted-foot constraint:
  A foot in contact is stationary in the world, so the base velocity is whatever
  cancels the joint + angular motion at that foot:
      base_world_linvel = -(foot world linear velocity, computed with base linear
                            velocity zeroed)         [averaged over planted feet]
  evaluated on the H1-2 MuJoCo kinematics (mj_forward + mj_objectVelocity). Base
  HEIGHT comes from the same kinematics (lowest foot on the floor), auto-calibrated
  so the home pose matches the model's home base height. Base XY is dead-reckoned by
  integrating the velocity estimate (a FROZEN xy against a nonzero velocity is
  kinematically inconsistent and destabilises the planner -- see the odometric-xy note
  in main()). Orientation + gyro the node reads straight from rt/lowstate, so
  sportmodestate only needs position + velocity.

FILTERS (--filter):
  kf12          the reverse-engineered reference design -- RECOMMENDED, opt-in.
                A 2-foot port of the contact-aided linear Kalman filter that Unitree's
                own open-source unitree_guide runs (extraction proved it is the MIT
                Cheetah-Software LinearKFPositionVelocityEstimator, re-tuned: same
                18-state [p,v,p_feet], same u = R*a + g propagation, same 28-row
                measurement incl. the omega x p Coriolis term, same 0.2 trust window,
                same (1 + (1-trust)*100) inflation). Ours: state 12 = [p,v,p_LF,p_RF],
                measurement 14. What it fixes vs rw-ekf:
                  * IMU PROPAGATION exists at all (rw-ekf only grows P -- the state is
                    never propagated, so when contact is lost the estimate FREEZES
                    while the truth accelerates at 9.81 m/s^2)
                  * swing feet are REJECTED (~100x), not merely down-weighted ~3x
                  * no force-accept hatch (rw-ekf capitulates after 0.25s of gating,
                    i.e. exactly during a trot swing)
                  * base POSITION is a filter state anchored by foot-height rows,
                    not an instantaneous "lowest foot must be on the floor" readout
                  * the TORSO-mounted IMU is corrected through the waist yaw joint
                Contact is measured (tau_est -> GRF -> Schmitt + kinematic gates),
                because our sampling MPC has no gait clock to read a phase off.
  rw-ekf        legacy default. Random-walk velocity EKF, per-foot leg-odo updates
                with torque-load-scaled noise + Mahalanobis gate. Kept byte-identical
                so kf12 can be A/B'd against it; rollback = drop --filter kf12.
  complementary the earlier min-speed + horizontal-IMU fusion (lower RMS, keeps spikes).

  We publish the IMU-SITE pose (pelvis + R*IMU_OFFSET), matching the twin's
  convention, so the node's internal site->pelvis back-out recovers the right base.

Run in the twin venv (has mujoco + unitree_sdk2py):
  cd ~/Desktop/h12/h1_mujoco
  # offline self-test (no DDS, no robot):
  .venv/bin/python ~/Desktop/h12/dds_tools/base_estimator_node.py --selftest
  # live (auto-pins the robot NIC), publishes rt/sportmodestate:
  .venv/bin/python ~/Desktop/h12/dds_tools/base_estimator_node.py
  # twin validation -- measure the estimate vs the twin's GROUND-TRUTH sportmodestate
  # (publish to a side topic so we don't clash with the twin's publisher):
  .venv/bin/python ~/Desktop/h12/dds_tools/base_estimator_node.py \
      --out-topic rt/sportmodestate_est --compare
"""
import argparse
import os
import subprocess
import time

import numpy as np
import mujoco

NJ = 27                                              # H1-2 handless actuated joints (== lowstate motor order)
IMU_OFFSET = np.array([-0.04452, -0.01891, 0.27756])  # pelvis -> IMU site (matches h12_control_node.cc)
ROBOT_SUBNET = "192.168.123."
_DEFAULT_SCENE = "~/Desktop/h12/h1_mujoco/unitree_robots/h1_2/scene_handless.xml"

# ---------------------------------------------------------------------------- #
#  KF12 -- the MIT-Cheetah / unitree_guide contact-aided linear KF, 2-foot port
#  (see docs/estimator_robustness_plan.md for the full derivation + citations)
# ---------------------------------------------------------------------------- #
GRAVITY = np.array([0.0, 0.0, -9.81])

# Motor indices in the rt/lowstate ordering (VERIFIED against the model 2026-07-16:
# motor i <-> jnt_qposadr 7+i; legs 0-11, torso_joint 12, arms 13-26).
TORSO_MOTOR = 12                        # torso_joint: z-axis waist yaw, range +-2.35 rad
LEG_MOTOR = {"left": list(range(0, 6)), "right": list(range(6, 12))}
# Model dof index for motor i is 6+i (free base occupies dofs 0..5). VERIFIED:
# left_hip_yaw dof=6 .. left_ankle_roll dof=11; right 12..17; torso 18.
LEG_DOF = {"left": list(range(6, 12)), "right": list(range(12, 18))}
ANKLE_PITCH_MOTOR = {"left": 4, "right": 10}
ANKLE_ROLL_MOTOR = {"left": 5, "right": 11}

# State  x(12) = [ p_site(3) ; v_site(3) ; p_LF(3) ; p_RF(3) ]   (WORLD frame)
# Meas   y(14) = [ dp_LF(3) ; dp_RF(3) ; dv_LF(3) ; dv_RF(3) ; z_LF ; z_RF ]
KF_NX, KF_NY = 12, 14
_FOOT_STATE = {0: slice(6, 9), 1: slice(9, 12)}     # p_LF, p_RF within x
_FPOS_ROWS = {0: slice(0, 3), 1: slice(3, 6)}       # relative-foot-POSITION rows
_FVEL_ROWS = {0: slice(6, 9), 1: slice(9, 12)}      # relative-foot-VELOCITY rows
_FH_ROW = {0: 12, 1: 13}                            # foot-height rows

# Reference point: we track the IMU SITE, not the pelvis. The accelerometer IS the
# site's specific force, so propagation needs no lever-arm (alpha x r + w x (w x r))
# term -- at r=0.28m those are O(1 m/s^2) during a trot, i.e. not negligible. The
# site is also exactly what we publish and what deploy_common consumes.
# torso_link.body_pos == (0,0,0) and body_quat == identity (VERIFIED in the MJCF),
# so p_site = p_pelvis + R_torso @ IMU_OFFSET holds at ANY waist angle.


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return q


def pelvis_from_torso(quat, gyro, theta, dtheta):
    """Undo the waist yaw between the TORSO-mounted IMU and the PELVIS.

    The `imu` site lives in `torso_link`, which hangs off the pelvis through
    `torso_joint` (z-axis, +-2.35 rad, motor idx 12) -- VERIFIED in h1_2_handless.xml.
    So the IMU quaternion is the TORSO's orientation, not the pelvis's, and the legs
    hang off the PELVIS. Feeding the raw IMU quat to the free joint (what every other
    path in this file does) rotates the whole leg FK -- and therefore the estimated
    base velocity -- by the waist angle.

        R_wt = R_wp @ Rz(theta)            =>  R_wp = R_wt @ Rz(-theta)
        w_pelvis_world = w_torso_world - R_wp @ (dtheta * z)
        w_pelvis_LOCAL = R_wp^T @ w_pelvis_world = Rz(theta) @ gyro - dtheta*z

    Returns (R_world_pelvis, omega_pelvis_LOCAL). Local, because MuJoCo's free joint
    takes qvel[0:3] as world-frame linear and qvel[3:6] as BODY-frame angular.
    """
    R_wt = _quat2mat(quat)
    R_wp = R_wt @ _rz(-theta)
    omega_p = _rz(theta) @ np.asarray(gyro, dtype=float) - np.array([0.0, 0.0, dtheta])
    return R_wp, omega_p


class AccelProbe:
    """Resolve the accelerometer convention, then hand back the propagation input.

    A specific-force sensor (real IMU, and MuJoCo's <accelerometer>, which the docs
    define as "including gravity") reads ~9.81 at REST and ~0 in FREE FALL. A
    gravity-free/coordinate-acceleration stream reads ~0 in BOTH. The old one-shot
    2s-at-boot probe could not tell those apart -- which is very likely why this file
    concluded "twin accel is unusable" and dropped IMU propagation entirely. The twin
    scene does carry a real <accelerometer> (`imu_acc`, mjSENS_ACCELEROMETER) and the
    bridge indexes it correctly, so that verdict looks like a probe artifact.

    Fix: only classify on samples where the robot is proprioceptively STATIC, and
    check the signal is ALIVE (a dead/zeroed stream has ~no variance). Until resolved
    we return None -> the KF simply does not propagate (== the old behaviour), so an
    unresolved probe degrades gracefully instead of injecting 9.81 m/s^2 of error.
    """

    def __init__(self, mode, static_dq, probe_sec):
        self.mode = mode
        self.static_dq = static_dq
        self.probe_sec = probe_sec
        self.resolved = None if mode == "auto" else mode
        self._acc, self._t, self._peak = [], 0.0, 0.0
        self._warned = False

    def step(self, a_world, dq_norm, dt):
        if self.resolved is not None:
            return self.resolved
        if dq_norm < self.static_dq:
            self._acc.append(np.asarray(a_world, dtype=float))
            self._peak = max(self._peak, float(np.max(np.abs(a_world))))
            self._t += dt
            if self._t >= self.probe_sec:
                self._classify()
        return self.resolved

    def _classify(self):
        mag = float(np.linalg.norm(np.mean(self._acc, axis=0)))
        spread = float(np.std(np.linalg.norm(self._acc, axis=1)))
        if self._peak < 1e-6:
            self.resolved, why = "off", "sensor is IDENTICALLY ZERO -> dead; propagation DISABLED"
        elif 8.5 < mag < 11.0:
            self.resolved, why = "specific", "gravity-inclusive specific force -> u = R@a + g"
        elif mag < 1.0:
            self.resolved, why = "linear", "gravity-FREE coordinate accel -> u = R@a"
        else:
            self.resolved, why = ("specific",
                                  f"UNUSUAL magnitude {mag:.2f} -- assuming specific force; VERIFY")
        print(f"[est] accel probe: |mean R@a|={mag:.2f} m/s^2 spread={spread:.3f} peak={self._peak:.2f} "
              f"over {self._t:.1f}s static -> {self.resolved.upper()}: {why}", flush=True)

    def input(self, a_world):
        """Propagation input u (world accel of the IMU site). Zeros = do not propagate."""
        if self.resolved == "specific":
            return a_world + GRAVITY
        if self.resolved == "linear":
            return np.asarray(a_world, dtype=float)
        if self.resolved is None and not self._warned:
            self._warned = True
            print("[est] accel probe UNRESOLVED (robot never static yet) -> KF not propagating; "
                  "park the robot for a moment, or force --accel-mode specific", flush=True)
        return np.zeros(3)


def kin_snapshot(m, data, foot_ids, imu_sid, q, dq, R_wp, omega_p, jacp, jacr, sole_off=0.0):
    """ONE mj_forward -> every kinematic/dynamic quantity the KF12 needs.

    The base is parked at the origin with the CORRECTED pelvis orientation and zero
    base linear velocity, so every returned world quantity is relative-to-base and
    each foot's velocity is purely the joint + angular contribution.

    ★ sole_off > 0 tracks the SOLE CONTACT POINT instead of the ankle_roll body origin.
    This is not cosmetic. The planted-foot constraint says the CONTACT POINT is
    stationary -- but the sole sits sole_off (== height_C ~ 0.047 m) BELOW the ankle
    body, so the two velocities differ by omega_foot x r_sole. Standing, foot omega ~ 0
    and they coincide (which is why the stand tracks to 3 mm/s). In a TROT the foot
    rotates hard through heel-strike/toe-off: several rad/s x 0.047 m is 0.3-1.0 m/s --
    the same order as the measured phantoms. You cannot filter your way out of a BIASED
    measurement. Tracking the sole also makes the foot-height measurement 0 (the sole
    IS on the floor), exactly like the point-foot reference. (Credit: the --sole
    finding already carried by the estimator_ab harness.)

    Returns dict with, per foot k:
      pf[k]   world-oriented foot position RELATIVE TO THE IMU SITE
      vs[k]   the IMU-site world velocity implied IF foot k is planted
      vfw[k]  the foot's own world speed contribution (kinematic gate)
    plus roff (pelvis->site offset, world).
    """
    quat_p = _mat2quat(R_wp)
    data.qpos[:] = 0.0
    data.qpos[3:7] = quat_p
    data.qpos[7:7 + NJ] = q
    data.qvel[:] = 0.0
    data.qvel[3:6] = omega_p          # free joint: angular vel is BODY-local
    data.qvel[6:6 + NJ] = dq
    mujoco.mj_forward(m, data)

    res = np.zeros(6)
    roff = np.array(data.site_xpos[imu_sid])         # == R_torso @ IMU_OFFSET (base at origin)
    mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_SITE, imu_sid, res, 0)
    v_site_contrib = res[3:6].copy()

    pf, vs, vfw = [], [], []
    for bid, _ in foot_ids:
        p_foot = np.array(data.xpos[bid])
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
        w_foot, v_foot_contrib = res[0:3].copy(), res[3:6].copy()
        if sole_off > 0.0:
            r_sole = data.xmat[bid].reshape(3, 3) @ np.array([0.0, 0.0, -sole_off])
            p_foot = p_foot + r_sole
            v_foot_contrib = v_foot_contrib + np.cross(w_foot, r_sole)
        pf.append(p_foot - roff)
        # planted foot: 0 = v_pelvis + v_foot_contrib  =>  v_pelvis = -v_foot_contrib
        #               v_site = v_pelvis + v_site_contrib
        vs.append(v_site_contrib - v_foot_contrib)
        vfw.append(v_foot_contrib)
    return {"pf": pf, "vs": vs, "vfw": vfw, "roff": roff}


def grf_from_torque(m, data, foot_bid, leg_dof, tau_leg, jacp, jacr, tau_sigma=1.0):
    """Estimate the ground reaction force on one foot from joint torques.

    Camurri RA-L 2017 (the recipe HyQ shipped with NO foot force sensors). The H1-2
    exposes no foot force at all in low-level mode (hg LowState has no such field), so
    this is the only contact signal available to us.

    MuJoCo EOM:  M qacc + qfrc_bias = tau + J^T f_ext
    Quasi-static (qacc ~ 0):          J^T f_ext = qfrc_bias - tau
    J^T is 6x3 -> least squares. qfrc_bias comes free from the mj_forward we already
    ran (gravity + Coriolis), so this costs one Jacobian.

    ALSO returns fz_sigma: the expected 1-sigma error on f_z given per-joint torque
    noise tau_sigma, from cov(f) = tau_sigma^2 * inv(J J^T). This matters far more than
    it looks. J[2,:] = d(foot_z)/d(joints) COLLAPSES as the knee straightens (measured
    on this model: |J[2,:]| ~ 0.2*knee), because a straight leg cannot move its foot
    vertically by moving joints -- so a vertical force produces no joint torque and
    tau_est goes BLIND to f_z:
        knee 0.60 -> |J[2,:]|=0.118 -> ~8 N error per N.m of torque noise
        knee 0.36 -> |J[2,:]|=0.081 -> ~12 N   (unitree_rl_gym's H1-2 default stance)
        knee 0.08 -> |J[2,:]|=0.016 -> ~62 N   (LOCKSTAND, strat 26 -- unusable!)
        knee 0.00 -> |J[2,:]|=0      -> blind  (the model's singular 'home' pose)
    Against 100 N strike / 33 N release thresholds, a locked knee makes this detector
    pure noise. The caller must gate on fz_sigma rather than trust f_z blindly.
    """
    mujoco.mj_jacBody(m, data, jacp, jacr, foot_bid)
    J = jacp[:, leg_dof]                                  # 3 x 6, world frame
    rhs = data.qfrc_bias[leg_dof] - np.asarray(tau_leg, dtype=float)
    f, *_ = np.linalg.lstsq(J.T, rhs, rcond=None)
    try:
        G = np.linalg.inv(J @ J.T)
        fz_sigma = tau_sigma * float(np.sqrt(max(G[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        fz_sigma = float("inf")
    return f, fz_sigma                                    # world-frame force (N), f_z 1-sigma (N)


class FootTrust:
    """Phase-free contact detection + continuous trust for one foot.

    THE piece we cannot copy from the reference. Both unitree_guide and MIT
    Cheetah-Software take contact+phase from the controller's GAIT CLOCK
    (WaveGenerator / setContactPhase) and never measure it. Our sampling MPC has no
    commanded phase -- the planner discovers contact through physics -- so
    windowFunc(phase, 0.2) has no input and the trust signal must be rebuilt from
    measurements. Every stage below is precedented:
      force Schmitt+dwell  : Pronto's Atlas strike detector (20-30N discontinuity >5ms)
      kinematic gates      : Bledt ICRA'18 / Lin CoRL'21 feature sets
      touchdown ramp       : Pronto/Camurri impact-window covariance inflation
      CoP interior-ness    : Piperakis IROS'22 Eq.4-5 (banks Rotella's flat-foot result)
    The output plugs into the reference's own (1 + (1-trust)*100) noise scaling.
    """

    def __init__(self, side, cfg):
        self.side = side
        self.cfg = cfg
        self.contact = False
        self.hi_t = 0.0
        self.stance_t = 0.0
        self.trust = 0.0
        self.cop = np.zeros(2)
        self.rolling = False
        self.blind = False          # tau->GRF map is near-singular in z (straight knee)

    def step(self, dt, f_z, fz_sigma, tau_ap, tau_ar, foot_z, foot_speed, ground_z):
        c = self.cfg
        # --- is f_z even meaningful right now? --------------------------------
        # A straightening knee collapses d(foot_z)/d(joints), so f_z's error blows up
        # (knee 0.08 => ~62 N against a 33 N release threshold). When that happens the
        # force signal is NOISE: HOLD the contact state rather than let noise strike or
        # release a foot, and fall back to the kinematic gates for trust.
        self.blind = not np.isfinite(fz_sigma) or fz_sigma > c.fz_sigma_max
        if self.blind:
            if not self.contact:
                self.trust, self.rolling = 0.0, False
                return 0.0
            self.stance_t += dt
            t = float(np.clip(self.stance_t / max(c.td_ramp_sec, 1e-6), 0.0, 1.0))
            if not (foot_z < ground_z + c.kin_h and foot_speed < c.kin_v):
                t *= c.kin_fail
            self.trust, self.rolling = float(np.clip(t, 0.0, 1.0)), False
            return self.trust

        # --- Schmitt trigger with dwell ---------------------------------------
        if not self.contact:
            if f_z > c.fz_hi:
                self.hi_t += dt
                if self.hi_t >= c.strike_sec:
                    self.contact, self.stance_t = True, 0.0
            else:
                self.hi_t = 0.0
        else:
            self.stance_t += dt
            if f_z < c.fz_lo and self.stance_t >= c.min_stance_sec:
                self.contact = False
                self.hi_t = 0.0

        if not self.contact:
            self.trust, self.rolling = 0.0, False
            return 0.0

        # --- force ramp -------------------------------------------------------
        t = float(np.clip((f_z - c.fz_lo) / max(c.fz_hi - c.fz_lo, 1e-6), 0.0, 1.0))
        # --- touchdown (impact-rejection) ramp --------------------------------
        t *= float(np.clip(self.stance_t / max(c.td_ramp_sec, 1e-6), 0.0, 1.0))
        # --- kinematic plausibility gates (phase-free) ------------------------
        if not (foot_z < ground_z + c.kin_h and foot_speed < c.kin_v):
            t *= c.kin_fail
        # --- flat-foot CoP: pinned at toe/heel => the contact point is MIGRATING
        self.rolling = False
        if c.cop and f_z > max(c.fz_lo, 1e-6):
            self.cop[0] = -tau_ap / f_z
            self.cop[1] = tau_ar / f_z
            span = c.sole_fwd + c.sole_back
            marg = c.cop_margin * span
            if not (-c.sole_back + marg < self.cop[0] < c.sole_fwd - marg
                    and abs(self.cop[1]) < c.sole_half_y - c.cop_margin * 2 * c.sole_half_y):
                t *= c.cop_fail
                self.rolling = True
        self.trust = float(np.clip(t, 0.0, 1.0))
        return self.trust


class KF12:
    """2-foot port of the MIT-Cheetah / unitree_guide contact-aided linear KF.

    x(12) = [p_site, v_site, p_LF, p_RF] world;  u = world accel of the site.
      p += v*dt ;  v += u*dt ;  feet constant       (feet move via their process noise)
    Measurements, per foot: relative position (NEVER inflated -- that is what keeps a
    swinging foot's STATE fresh so it is already correct at touchdown), relative
    velocity (the planted-foot constraint; inflated to mute a swing foot), and foot
    height (the z anchor).

    Constants use MIT's dt-scaled Q0 (self-scales to our 200Hz; unitree_guide's fixed
    3e-4 was tuned at their 500Hz) with unitree_guide's empirically-identified R
    magnitudes (measured on a real robot, unlike MIT's flat 0.1).
    """

    def __init__(self, cfg, foot_h):
        self.cfg = cfg
        # measured height of the tracked foot point when planted: 0 with --sole (we
        # track the contact point), height_C without (we track the ankle body).
        self.foot_h = foot_h
        self.x = np.zeros(KF_NX)
        self.P = np.eye(KF_NX) * cfg.p0
        self.gated = [False, False]
        self._C = np.zeros((KF_NY, KF_NX))
        for k in (0, 1):
            self._C[_FPOS_ROWS[k], 0:3] = -np.eye(3)          # predicts p_foot - p_site
            self._C[_FPOS_ROWS[k], _FOOT_STATE[k]] = np.eye(3)
            self._C[_FVEL_ROWS[k], 3:6] = np.eye(3)           # predicts v_site
        self._C[_FH_ROW[0], 8] = 1.0                          # z of p_LF
        self._C[_FH_ROW[1], 11] = 1.0                         # z of p_RF

    def reset_feet(self, pf):
        """Seed the foot states from FK (called once the site position is known)."""
        for k in (0, 1):
            self.x[_FOOT_STATE[k]] = self.x[0:3] + pf[k]

    def _Q(self, dt, trusts, contacts):
        c = self.cfg
        Q = np.zeros((KF_NX, KF_NX))
        Q[0:3, 0:3] = np.eye(3) * (dt / 20.0) * c.q_pos
        Q[3:6, 3:6] = np.eye(3) * (dt * 9.8 / 20.0) * c.q_vel
        for k in (0, 1):
            s = _FOOT_STATE[k]
            base = np.eye(3) * dt * c.q_foot
            if not contacts[k]:
                Q[s, s] = np.eye(3) * c.suspect          # SWING: absolute large variance
            else:
                Q[s, s] = base * (1.0 + (1.0 - trusts[k]) * c.suspect)
        return Q

    def _R(self, trusts, contacts, rolling):
        c = self.cfg
        R = np.zeros((KF_NY, KF_NY))
        for k in (0, 1):
            # relative-foot-POSITION rows are NEVER inflated (both reference codebases)
            R[_FPOS_ROWS[k], _FPOS_ROWS[k]] = np.eye(3) * c.r_fpos
            if not contacts[k]:
                R[_FVEL_ROWS[k], _FVEL_ROWS[k]] = np.eye(3) * c.suspect
                R[_FH_ROW[k], _FH_ROW[k]] = c.suspect
            else:
                s = 1.0 + (1.0 - trusts[k]) * c.suspect
                R[_FVEL_ROWS[k], _FVEL_ROWS[k]] = np.eye(3) * c.r_fvel * s
                R[_FH_ROW[k], _FH_ROW[k]] = c.r_fh * s
            if rolling[k]:
                R[_FPOS_ROWS[k], _FPOS_ROWS[k]] *= c.roll_infl
        return R

    def predict(self, u, dt, trusts, contacts):
        A = np.eye(KF_NX)
        A[0:3, 3:6] = np.eye(3) * dt
        self.x[0:3] += self.x[3:6] * dt
        self.x[3:6] += np.asarray(u, dtype=float) * dt
        self.P = A @ self.P @ A.T + self._Q(dt, trusts, contacts)

    def update(self, snap, trusts, contacts, rolling):
        c = self.cfg
        C = self._C
        y = np.zeros(KF_NY)
        for k in (0, 1):
            y[_FPOS_ROWS[k]] = snap["pf"][k]
            y[_FVEL_ROWS[k]] = snap["vs"][k]
            y[_FH_ROW[k]] = self.foot_h            # 0 with --sole: the contact point IS the floor
        R = self._R(trusts, contacts, rolling)

        # MIT pseudo-measurement blend: an untrusted row degenerates to a
        # self-consistent no-op instead of fighting the prior.
        for k in (0, 1):
            t = trusts[k] if contacts[k] else 0.0
            y[_FVEL_ROWS[k]] = (1.0 - t) * self.x[3:6] + t * y[_FVEL_ROWS[k]]
            y[_FH_ROW[k]] = (1.0 - t) * self.x[_FOOT_STATE[k]][2] + t * y[_FH_ROW[k]]

        inn = y - C @ self.x
        # per-foot Mahalanobis gate on the velocity block. NO force-accept hatch:
        # covariance growth re-admits genuine motion on its own (the references need none).
        CPC = C @ self.P @ C.T
        for k in (0, 1):
            sl = _FVEL_ROWS[k]
            Sk = CPC[sl, sl] + R[sl, sl]
            try:
                d2 = float(inn[sl] @ np.linalg.solve(Sk, inn[sl]))
            except np.linalg.LinAlgError:
                d2 = 0.0
            self.gated[k] = d2 > c.chi2
            if self.gated[k]:
                R[sl, sl] = np.eye(3) * c.suspect * 10.0

        S = CPC + R
        try:
            K = self.P @ C.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ inn
        IKC = np.eye(KF_NX) - K @ C
        self.P = IKC @ self.P @ IKC.T + K @ R @ K.T      # Joseph form (unitree_guide)
        self.P = 0.5 * (self.P + self.P.T)               # symmetrize (MIT)
        # planar position is UNOBSERVABLE from proprioception (Bloesch RSS'12,
        # Rotella IROS'14) -- P(x,y) grows without bound. MIT's clamp.
        if np.linalg.det(self.P[0:2, 0:2]) > 1e-6:
            self.P[0:2, 2:] = 0.0
            self.P[2:, 0:2] = 0.0
            self.P[0:2, 0:2] /= 10.0

    def hang_prior(self, site_z_kin, r):
        """NO-CONTACT FALLBACK -- the guard the references do not have and we need.

        Both unitree_guide and MIT simply let the filter degrade to IMU dead-reckoning
        when no foot is trusted; on a walking quadruped you are never contact-free for
        long, so it never bites. We ARE: a robot hanging on the harness at bring-up is
        a NORMAL state, and there the integration is unbounded nonsense.

        MEASURED ON THE REAL ROBOT (2026-07-16, suspended, feet at 13% load so contact
        correctly never latched): base_z 0.975 -> 50.99 m and v -> [4.8, 10.9, 3.4] m/s
        in THIRTY SECONDS. The cause is ~0.36 m/s^2 of residual world accel == 2.1 deg
        of attitude error leaking gravity (0.17 m/s per s per degree) -- irreducible.

        So past any plausible flight phase we stop integrating and fall back to the SAFE
        estimate: zero velocity + the kinematic height (lowest foot on the floor). That
        is exactly what rw-ekf reports hanging, and what Unitree's own onboard estimator
        reports when craned (sdk2_python#135). Reading ~0 when you have NO information
        beats confidently reporting 51 m. This is a fail-safe, not an accuracy feature.
        """
        C = np.zeros((4, KF_NX))
        C[0:3, 3:6] = np.eye(3)            # v -> 0
        C[3, 2] = 1.0                      # p_z -> kinematic height
        y = np.array([0.0, 0.0, 0.0, site_z_kin])
        R = np.eye(4) * r
        S = C @ self.P @ C.T + R
        try:
            K = self.P @ C.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ (y - C @ self.x)
        IKC = np.eye(KF_NX) - K @ C
        self.P = IKC @ self.P @ IKC.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def clamp_v(self, vmax):
        """Hard backstop. This estimate seeds every planner rollout; a numerically
        blown-up velocity must never reach it, whatever else failed."""
        n = float(np.linalg.norm(self.x[3:6]))
        if vmax > 0.0 and n > vmax:
            self.x[3:6] *= vmax / n
            return True
        return False


def _kf12_cfg(a, m):
    """Resolve kf12's runtime config: body-weight fractions -> Newtons, ms -> seconds."""
    mg = float(m.body_mass.sum()) * 9.81
    return argparse.Namespace(
        suspect=a.suspect, p0=a.p0, chi2=a.kf_chi2,
        q_pos=a.q_pos, q_vel=a.q_vel, q_foot=a.q_foot,
        r_fpos=a.r_fpos, r_fvel=a.r_fvel, r_fh=a.r_fh,
        mg=mg, fz_hi=a.fz_hi * mg, fz_lo=a.fz_lo * mg, sole=a.sole,
        tau_sigma=a.tau_sigma, fz_sigma_max=a.fz_sigma_max,
        strike_sec=a.strike_ms * 1e-3, min_stance_sec=a.min_stance_ms * 1e-3,
        td_ramp_sec=a.td_ramp_ms * 1e-3,
        kin_h=a.kin_h, kin_v=a.kin_v, kin_fail=a.kin_fail,
        cop=a.cop, cop_margin=a.cop_margin, cop_fail=a.cop_fail, roll_infl=a.roll_infl,
        sole_fwd=a.sole_fwd, sole_back=a.sole_back, sole_half_y=a.sole_half_y,
    )


def _ipv4_ifaces():
    """[(name, ipv4)] for every up interface, via getifaddrs-equivalent ioctls.
    `ip -o -4 addr show` is absent in some containers, which silently degraded
    the auto-pin below to "twin only" while the C++ core (getifaddrs) bound the
    robot link fine. Keep the subprocess as a fallback for exotic libc setups."""
    out = []
    try:
        import fcntl
        import socket
        import struct
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for _, name in socket.if_nameindex():
                try:
                    packed = fcntl.ioctl(s.fileno(), 0x8915,  # SIOCGIFADDR
                                         struct.pack("256s", name[:15].encode()))
                    out.append((name, socket.inet_ntoa(packed[20:24])))
                except OSError:
                    pass                                       # no IPv4 on this NIC
        finally:
            s.close()
    except Exception:
        pass
    if out:
        return out
    try:
        raw = subprocess.run(["ip", "-o", "-4", "addr", "show"],
                             capture_output=True, text=True, timeout=4).stdout
    except Exception:
        raw = ""
    for line in raw.splitlines():
        p = line.split()
        if len(p) >= 4:
            out.append((p[1], p[3].split("/")[0]))
    return out


def _pick_iface(explicit):
    """Auto-pin the 192.168.123.x robot-subnet NIC (so empty binds the robot link,
    not WiFi/Tailscale), mirroring dds_topic_check/dds_live_recorder."""
    ifaces = _ipv4_ifaces()
    if explicit:
        names = [n for n, _ in ifaces]
        if not names or explicit in names:
            return explicit, "explicit --iface"
        # A NIC name baked into a config is host-specific; don't take the whole
        # node down on a machine that names its robot link something else.
        print(f"[est] WARNING: --iface '{explicit}' not present (have: "
              f"{', '.join(names)}) -> ignoring, falling back to auto-detect")
    for name, ip in ifaces:
        if name != "lo" and ip.startswith(ROBOT_SUBNET):
            return name, f"auto-detected ({ip} on robot subnet)"
    return None, "no robot-subnet NIC -> SDK autodetermine (twin only)"


def _quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def load_model_and_calibrate(scene):
    """Load the H1-2 model, find the foot bodies, and calibrate the ankle->floor
    height constant so the home pose reproduces the model's home base height."""
    scene = os.path.expanduser(scene or _DEFAULT_SCENE)
    d_dir = os.path.dirname(os.path.abspath(scene))
    cwd = os.getcwd()
    os.chdir(d_dir)
    m = mujoco.MjModel.from_xml_path(os.path.basename(scene))
    os.chdir(cwd)
    data = mujoco.MjData(m)
    assert m.nq >= 7 + NJ, f"model nq={m.nq} too small (need free base + {NJ} joints)"

    foot_ids = []
    for bid in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "ankle_roll" in nm:
            foot_ids.append((bid, nm))
    if not foot_ids:
        for bid in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if "ankle" in nm or "foot" in nm:
                foot_ids.append((bid, nm))
    if not foot_ids:
        raise RuntimeError("no foot bodies found (looked for *ankle_roll* / *ankle* / *foot*)")

    home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        # key_qpos is (nkey, nq)-shaped in the python bindings -- row-index it.
        # (Flat-slicing returned ROWS, so scenes with a "home" keyframe crashed at
        # float(home_q[2]); the twin scene has no keyframes, which hid this.)
        home_q = np.array(m.key_qpos[home])
    else:
        home_q = np.array(m.qpos0)
    home_base_z = float(home_q[2])

    # base at origin + identity orientation + home joints -> lowest ankle z.
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    data.qpos[7:7 + NJ] = home_q[7:7 + NJ]
    mujoco.mj_forward(m, data)
    ankle_z_home = min(float(data.xpos[bid][2]) for bid, _ in foot_ids)
    height_C = home_base_z + ankle_z_home   # base_height = -min(ankle_z) + C ; at home -> home_base_z

    print(f"[est] model nq={m.nq} nv={m.nv} | feet: {[nm for _, nm in foot_ids]}")
    print(f"[est] height calib: home_base_z={home_base_z:.3f} ankle_z_home={ankle_z_home:.3f} C={height_C:.3f}")
    return m, data, foot_ids, height_C, home_q


def leg_odometry(m, data, foot_ids, height_C, q, dq, quat, gyro, res):
    """Return (base_height, base_world_linvel[3]) from one proprioceptive sample.
    Base linvel left at 0 in qvel, so each foot's computed world velocity is the
    joint+angular contribution; the base velocity that makes a planted foot
    stationary is its negation (base translation adds 1:1 to every body)."""
    data.qpos[:] = 0.0
    data.qpos[3:7] = quat                 # base at origin (xy nominal), real orientation
    data.qpos[7:7 + NJ] = q
    data.qvel[:] = 0.0
    data.qvel[3:6] = gyro                 # base angvel = body gyro (base linvel stays 0)
    data.qvel[6:6 + NJ] = dq
    mujoco.mj_forward(m, data)

    base_height = -min(float(data.xpos[bid][2]) for bid, _ in foot_ids) + height_C

    vests = []
    for bid, _ in foot_ids:
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
        vests.append(-res[3:6].copy())    # world linear velocity of foot (base-lin=0), negated
    base_v = np.mean(vests, axis=0)
    return base_height, base_v


# ---------------------------------------------------------------------------
# ZUPT -- zero-velocity update (--zupt, OFF by default => byte-identical)
# ---------------------------------------------------------------------------
def zupt_hold(tau, dq, gyro, load_idx, a):
    """True when the base PROVABLY is not moving: both feet carrying load AND
    the joints static AND the base not rotating.

    The gyro term is NOT optional. A humanoid toppling RIGIDLY about the ankle
    has |dq| ~ 0 while the base genuinely moves, and leg odometry perceives that
    motion ONLY through the gyro (qvel[3:6]). Gating on joint motion alone would
    zero exactly the capture-point signal (subcom + 0.3*subcomvel) the planner
    needs to catch that fall -- the ZUPT would cause the very topple it is meant
    to survive. Contrast pedestrian/foot-mounted INS, where a ZUPT is safe
    because the FOOT really is static during stance; a balancing humanoid's BASE
    never is, so "small base velocity" there is signal, not noise.
    """
    both_loaded = all(abs(tau[i1]) + abs(tau[i2]) >= a.zupt_load
                      for i1, i2 in load_idx)
    return (both_loaded
            and float(np.linalg.norm(dq)) < a.zupt_dq
            and float(np.linalg.norm(gyro)) < a.zupt_gyro)


def zupt_fuse_v(v, P, r):
    """Fuse the pseudo-measurement v == 0 into a 3-state velocity filter, in the
    SAME Kalman form the rw-ekf already uses for its per-foot updates. Deliberately
    a measurement, not a hard clamp: a confident P must still be able to out-vote
    the prior, so a real (but quiet) motion is attenuated rather than erased."""
    S = P + (r ** 2) * np.eye(3)
    K = P @ np.linalg.inv(S)
    return v + K @ (-v), (np.eye(3) - K) @ P


def zupt_fuse_kf12(kf, r):
    """Same pseudo-measurement against kf12's 12-state [p, v, p_feet], done with the
    proper H = [0 I3 0] so the cross-covariance to position/feet is respected (a
    ZUPT is worth most on kf12: it is the only filter here that INTEGRATES accel
    -- v += (R a + g) dt -- so it is the only one that accrues bias-driven drift a
    zero-velocity anchor can actually reset)."""
    n = kf.P.shape[0]
    Pvv = kf.P[3:6, 3:6]
    S = Pvv + (r ** 2) * np.eye(3)
    K = kf.P[:, 3:6] @ np.linalg.inv(S)          # (n,3)
    kf.x[:] = kf.x + K @ (-kf.x[3:6])
    H = np.zeros((3, n)); H[:, 3:6] = np.eye(3)
    kf.P[:, :] = (np.eye(n) - K @ H) @ kf.P


def _selftest(m, data, foot_ids, height_C, home_q, a=None):
    res = np.zeros(6)
    qj = np.array(home_q[7:7 + NJ])
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    h, v = leg_odometry(m, data, foot_ids, height_C, qj, np.zeros(NJ), ident, np.zeros(3), res)
    print(f"[selftest] static home : base_z={h:.3f} (expect ~{float(home_q[2]):.3f})  "
          f"v={np.round(v, 4)} (expect ~0)")
    dq = np.zeros(NJ); dq[3] = 0.5        # one joint moving
    _, v2 = leg_odometry(m, data, foot_ids, height_C, qj, dq, ident, np.zeros(3), res)
    print(f"[selftest] joint moving: v={np.round(v2, 4)} (expect nonzero)")
    ok = abs(h - float(home_q[2])) < 0.02 and np.linalg.norm(v) < 1e-6 and np.linalg.norm(v2) > 1e-4
    if a is not None and a.filter == "kf12":
        ok = _selftest_kf12(m, data, foot_ids, height_C, home_q, a) and ok
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _selftest_kf12(m, data, foot_ids, height_C, home_q, a):
    """Offline math checks for the reference-design filter. No DDS, no plant, no robot."""
    ok = True
    cfg = _kf12_cfg(a, m)
    imu_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    sole_off = height_C if a.sole else 0.0
    ground_ref = 0.0 if a.sole else height_C
    qj = np.array(home_q[7:7 + NJ])
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    dt = 0.005
    print(f"[selftest] --- kf12 --- mg={cfg.mg:.1f}N strike={cfg.fz_hi:.0f}N release={cfg.fz_lo:.0f}N")

    # 1. WAIST FIX: the estimate must be INVARIANT to torso_joint. That is the whole
    #    point of pelvis_from_torso -- the IMU sits on the far side of that joint.
    #    Hold the pelvis identity, rotate the waist, and let the (torso-mounted) IMU
    #    read the rotated torso. The site velocity implied by a planted foot must not move.
    dqj = np.zeros(NJ); dqj[3] = 0.5
    vs_ref = None
    for theta in (0.0, 0.6, -1.2):
        q_t = qj.copy(); q_t[TORSO_MOTOR] = theta
        quat_torso = _mat2quat(_rz(theta))          # pelvis = I, so torso reads Rz(theta)
        R_wp, om_p = pelvis_from_torso(quat_torso, np.zeros(3), theta, 0.0)
        if not np.allclose(R_wp, np.eye(3), atol=1e-9):
            ok = False
            print(f"[selftest] kf12 waist theta={theta:+.2f}: recovered pelvis R != I  FAIL")
        snap = kin_snapshot(m, data, foot_ids, imu_sid, q_t, dqj, R_wp, om_p, jacp, jacr, sole_off)
        if vs_ref is None:
            vs_ref = snap["vs"][0].copy()
        else:
            d = float(np.linalg.norm(snap["vs"][0] - vs_ref))
            good = d < 1e-6
            ok &= good
            print(f"[selftest] kf12 waist theta={theta:+.2f}: implied site-v drift={d:.2e} "
                  f"({'ok' if good else 'FAIL'})")
    # prove the defect it fixes is real: feed the raw torso quat, as the legacy paths do
    q_t = qj.copy(); q_t[TORSO_MOTOR] = 0.6
    bad = kin_snapshot(m, data, foot_ids, imu_sid, q_t, dqj, _rz(0.6), np.zeros(3),
                       jacp, jacr, sole_off)
    d_bad = float(np.linalg.norm(bad["vs"][0] - vs_ref))
    print(f"[selftest] kf12 waist UNCORRECTED (legacy) drift @theta=0.6 -> {d_bad:.4f} m/s "
          f"({'defect reproduced' if d_bad > 1e-4 else 'no effect at this pose'})")

    # 2. GRF round-trip: synthesise the torque a known foot force produces, then check
    #    grf_from_torque inverts it. Validates the J^T algebra + dof/motor indices.
    #    Sign/scale vs the REAL robot's tau_est still needs a standing log (plan V4).
    #    MUST be run at an OPERATING (bent-knee) stance: the model's 'home'/qpos0 is the
    #    SINGULAR straight-knee pose (deploy_common.cc:882 says the same), where a
    #    vertical force produces literally no joint torque and f_z is unrecoverable.
    #    Pose = unitree_rl_gym's documented H1-2 default_angles.
    q_st = np.zeros(NJ)
    for s in ("left", "right"):
        q_st[LEG_MOTOR[s][1]] = -0.16      # hip_pitch
        q_st[LEG_MOTOR[s][3]] = 0.36       # knee
        q_st[LEG_MOTOR[s][4]] = -0.20      # ankle_pitch
    R_wp, om_p = pelvis_from_torso(ident, np.zeros(3), 0.0, 0.0)
    kin_snapshot(m, data, foot_ids, imu_sid, q_st, np.zeros(NJ), R_wp, om_p, jacp, jacr, sole_off)
    for bid, nm in foot_ids:
        side = "left" if "left" in nm else "right"
        f_true = np.array([12.0, -5.0, cfg.mg / 2.0])
        mujoco.mj_jacBody(m, data, jacp, jacr, bid)
        tau_leg = data.qfrc_bias[LEG_DOF[side]] - jacp[:, LEG_DOF[side]].T @ f_true
        f_hat, fz_sig = grf_from_torque(m, data, bid, LEG_DOF[side], tau_leg, jacp, jacr,
                                        cfg.tau_sigma)
        err = float(np.linalg.norm(f_hat - f_true))
        good = err < 1e-6
        ok &= good
        print(f"[selftest] kf12 GRF {side:5s} @knee0.36: recovered {np.round(f_hat, 1)} "
              f"err={err:.2e} fz_sigma={fz_sig:.1f}N ({'ok' if good else 'FAIL'})")

    # 2b. The blind-knee limitation, measured. |J[2,:]| = d(foot_z)/d(joints) collapses
    #     as the knee straightens, so tau_est's view of f_z degrades to noise. This is
    #     why --fz-sigma-max exists, and it is why LOCKSTAND (knee 0.08) cannot use the
    #     force signal at all.
    lf = foot_ids[0][0]
    print("[selftest] kf12 blind-knee sweep (f_z observability vs knee):")
    for label, knee in (("home/qpos0 SINGULAR", 0.0), ("lockstand strat26", 0.08),
                        ("straighten kf", 0.35), ("rl_gym default", 0.36), ("stand-ish", 0.60)):
        q_k = np.zeros(NJ)
        for s in ("left", "right"):
            q_k[LEG_MOTOR[s][1]] = -knee / 2.0
            q_k[LEG_MOTOR[s][3]] = knee
            q_k[LEG_MOTOR[s][4]] = -knee / 2.0
        kin_snapshot(m, data, foot_ids, imu_sid, q_k, np.zeros(NJ), R_wp, om_p, jacp, jacr, sole_off)
        _, fz_sig = grf_from_torque(m, data, lf, LEG_DOF["left"], np.zeros(6), jacp, jacr,
                                    cfg.tau_sigma)
        mujoco.mj_jacBody(m, data, jacp, jacr, lf)
        vz = float(np.linalg.norm(jacp[2, LEG_DOF["left"]]))
        verdict = "BLIND -> hold contact on kinematics" if fz_sig > cfg.fz_sigma_max else "usable"
        print(f"[selftest]   knee={knee:4.2f} ({label:20s}) |J[2,:]|={vz:6.4f} "
              f"fz_sigma={fz_sig:8.1f}N  {verdict}")

    # 3. PROPAGATION -- the backbone rw-ekf never had.
    probe = AccelProbe("specific", a.static_dq, a.accel_probe_sec)
    kf = KF12(cfg, ground_ref); kf.x[:] = 0.0
    for _ in range(200):                        # 1s at rest: specific force reads +9.81 up
        kf.predict(probe.input(np.array([0.0, 0.0, 9.81])), dt, [1.0, 1.0], [True, True])
    rest = float(np.linalg.norm(kf.x[3:6]))
    good = rest < 1e-9
    ok &= good
    print(f"[selftest] kf12 propagate rest 1s: |v|={rest:.2e} (expect 0) ({'ok' if good else 'FAIL'})")

    kf2 = KF12(cfg, ground_ref); kf2.x[:] = 0.0
    for _ in range(100):                        # 0.5s free fall: specific force reads 0
        kf2.predict(probe.input(np.zeros(3)), dt, [0.0, 0.0], [False, False])
    vz, want = float(kf2.x[5]), -9.81 * 0.5
    good = abs(vz - want) < 1e-6
    ok &= good
    print(f"[selftest] kf12 free fall 0.5s  : v_z={vz:+.4f} (expect {want:+.4f}) "
          f"({'ok' if good else 'FAIL'})")

    # 4. ACCEL PROBE three-way classification (specific / gravity-free / dead).
    for name, sample, want in (("specific", np.array([0.0, 0.0, 9.81]), "specific"),
                               ("gravity-free", np.array([0.0, 0.0, 1e-3]), "linear"),
                               ("dead", np.zeros(3), "off")):
        p = AccelProbe("auto", a.static_dq, 0.05)
        for _ in range(20):
            p.step(sample, 0.0, dt)
        good = p.resolved == want
        ok &= good
        print(f"[selftest] kf12 probe {name:12s} -> {p.resolved} (expect {want}) "
              f"({'ok' if good else 'FAIL'})")

    # 5. CONTACT FSM: strike above fz_hi after the dwell, release below fz_lo after
    #    min-stance, and ramp trust in gradually (impact rejection).
    fsm = FootTrust("left", cfg)
    good_sig = cfg.fz_sigma_max / 5.0
    for _ in range(3):
        fsm.step(dt, cfg.mg / 2, good_sig, 0.0, 0.0, ground_ref, 0.0, ground_ref)
    struck, early = fsm.contact, fsm.trust
    for _ in range(40):
        fsm.step(dt, cfg.mg / 2, good_sig, 0.0, 0.0, ground_ref, 0.0, ground_ref)
    full = fsm.trust > 0.99
    for _ in range(40):
        fsm.step(dt, 0.0, good_sig, 0.0, 0.0, ground_ref + 0.10, 1.0, ground_ref)
    released = (not fsm.contact) and fsm.trust == 0.0
    good = struck and full and released and early < 1.0
    ok &= good
    print(f"[selftest] kf12 contact FSM     : strike={struck} td_ramp={early:.2f}<1 "
          f"full={full} release={released} ({'ok' if good else 'FAIL'})")

    # 5b. BLIND HOLD: with f_z unreliable (straight knee), noise must NOT be able to
    #     strike a swinging foot or release a planted one.
    fsm_b = FootTrust("left", cfg)
    bad_sig = cfg.fz_sigma_max * 10.0
    for _ in range(40):                     # huge bogus f_z while blind -> must NOT strike
        fsm_b.step(dt, cfg.mg, bad_sig, 0.0, 0.0, ground_ref, 0.0, ground_ref)
    no_false_strike = not fsm_b.contact
    fsm_c = FootTrust("left", cfg)
    for _ in range(60):                     # establish contact with a good signal
        fsm_c.step(dt, cfg.mg / 2, good_sig, 0.0, 0.0, ground_ref, 0.0, ground_ref)
    for _ in range(60):                     # go blind, f_z reads 0 -> must NOT release
        fsm_c.step(dt, 0.0, bad_sig, 0.0, 0.0, ground_ref, 0.0, ground_ref)
    held = fsm_c.contact and fsm_c.blind and fsm_c.trust > 0.9
    good = no_false_strike and held
    ok &= good
    print(f"[selftest] kf12 blind hold      : no_false_strike={no_false_strike} "
          f"held_planted={held} ({'ok' if good else 'FAIL'})")

    # 6. SWING REJECTION -- the rw-ekf defect (lfloor capped swing sigma at 0.08 m/s,
    #    i.e. only ~3x less trusted than a planted foot). Reference: ~100x.
    kf3 = KF12(cfg, ground_ref)
    R_sw = kf3._R([0.0, 1.0], [False, True], [False, False])
    sw = R_sw[_FVEL_ROWS[0], _FVEL_ROWS[0]][0, 0]
    st = R_sw[_FVEL_ROWS[1], _FVEL_ROWS[1]][0, 0]
    ratio = sw / max(st, 1e-9)
    good = ratio >= cfg.suspect * 0.99
    ok &= good
    print(f"[selftest] kf12 swing reject    : R_vel swing={sw:.1f} stance={st:.2f} -> {ratio:.0f}x "
          f"(rw-ekf managed ~3x) ({'ok' if good else 'FAIL'})")
    pos_sw = R_sw[_FPOS_ROWS[0], _FPOS_ROWS[0]][0, 0]
    good = abs(pos_sw - cfg.r_fpos) < 1e-12
    ok &= good
    print(f"[selftest] kf12 fpos never infl : swing R_fpos={pos_sw:.4f} == {cfg.r_fpos:.4f} "
          f"({'ok' if good else 'FAIL'})")

    # 6b. ★ THE 51-METRE BUG, reproduced offline and then guarded.
    #     Real robot 2026-07-16, suspended on the harness: feet at 13% load so contact
    #     correctly never latched -> no leg-odo corrections -> IMU-only integration ->
    #     base_z 0.975 -> 50.99 m and v -> [4.8, 10.9, 3.4] m/s in 30s. Cause: ~0.36
    #     m/s^2 of residual world accel == 2.1 deg of attitude error leaking gravity.
    #     Replay exactly that: 30s, no contact, that residual. UNGUARDED must diverge;
    #     GUARDED must stay sane.
    resid = np.array([0.12, 0.36, 0.115])          # the measured real-robot residual
    snap0 = kin_snapshot(m, data, foot_ids, imu_sid, qj, np.zeros(NJ), R_wp, om_p,
                         jacp, jacr, sole_off)
    bh0 = -min(float(snap0["pf"][k][2] + snap0["roff"][2]) for k in range(2)) + ground_ref
    z_kin = bh0 + snap0["roff"][2]

    def _hang_30s(guard):
        k = KF12(cfg, ground_ref)
        k.x[0:3] = np.array([0.0, 0.0, bh0]) + snap0["roff"]
        k.reset_feet(snap0["pf"])
        noc = 0.0
        for _ in range(6000):                      # 30 s @ 200 Hz, ZERO contacts
            noc += dt
            hang = guard and noc > a.noc_coast_sec
            k.predict(np.zeros(3) if hang else resid, dt, [0.0, 0.0], [False, False])
            k.update(snap0, [0.0, 0.0], [False, False], [False, False])
            if hang:
                k.hang_prior(z_kin, a.zupt_r)
            if guard:
                k.clamp_v(a.v_clamp)
        return float(np.linalg.norm(k.x[3:6])), float(k.x[2] - snap0["roff"][2])

    v_un, z_un = _hang_30s(guard=False)
    v_gd, z_gd = _hang_30s(guard=True)
    reproduced = v_un > 5.0 and z_un > 20.0        # the real failure, offline
    ok &= reproduced
    print(f"[selftest] kf12 hang 30s UNGUARDED: |v|={v_un:6.2f} m/s base_z={z_un:8.2f} m "
          f"({'real 51m divergence REPRODUCED' if reproduced else 'FAIL: did not reproduce'})")
    guarded_ok = v_gd < 0.05 and abs(z_gd - bh0) < 0.05
    ok &= guarded_ok
    print(f"[selftest] kf12 hang 30s GUARDED  : |v|={v_gd:6.4f} m/s base_z={z_gd:8.3f} m "
          f"(expect ~0 and ~{bh0:.3f}) ({'ok' if guarded_ok else 'FAIL'})")

    # 6c. the guard must NOT eat a REAL flight phase (0.4s < noc_coast_sec): coasting
    #     through flight is the entire point of having a propagation backbone.
    kfl = KF12(cfg, ground_ref); kfl.x[:] = 0.0
    noc = 0.0
    for _ in range(80):                            # 0.4 s of true free fall
        noc += dt
        hang = noc > a.noc_coast_sec
        kfl.predict(np.zeros(3) if hang else probe.input(np.zeros(3)), dt,
                    [0.0, 0.0], [False, False])
    vz_fl, want_fl = float(kfl.x[5]), -9.81 * 0.4
    good = abs(vz_fl - want_fl) < 1e-6
    ok &= good
    print(f"[selftest] kf12 flight 0.4s coast: v_z={vz_fl:+.4f} (expect {want_fl:+.4f}, guard must "
          f"NOT fire) ({'ok' if good else 'FAIL'})")

    # 7. FULL FILTER, planted + static: v stays ~0 and P stays bounded.
    kf4 = KF12(cfg, ground_ref)
    snap = kin_snapshot(m, data, foot_ids, imu_sid, qj, np.zeros(NJ), R_wp, om_p, jacp, jacr, sole_off)
    bh = -min(float(snap["pf"][k][2] + snap["roff"][2]) for k in range(2)) + ground_ref
    kf4.x[0:3] = np.array([0.0, 0.0, bh]) + snap["roff"]
    kf4.reset_feet(snap["pf"])
    for _ in range(400):                        # 2s planted
        kf4.predict(probe.input(np.array([0.0, 0.0, 9.81])), dt, [1.0, 1.0], [True, True])
        kf4.update(snap, [1.0, 1.0], [True, True], [False, False])
    vn = float(np.linalg.norm(kf4.x[3:6]))
    pn = float(np.max(np.diag(kf4.P)))
    good = vn < 1e-3 and np.isfinite(pn) and pn < 1e3
    ok &= good
    print(f"[selftest] kf12 planted 2s      : |v|={vn:.2e} (expect ~0) maxdiag(P)={pn:.3g} "
          f"({'ok' if good else 'FAIL'})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="H1-2 MuJoCo scene (default: handless twin scene)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None, help="pin NIC; default = auto-pin robot subnet")
    ap.add_argument("--no-auto-iface", action="store_true",
                    help="force SDK autodetermine (use for the same-host TWIN). Do NOT use "
                         "--iface lo: loopback disables multicast -> no DDS discovery.")
    ap.add_argument("--rate", type=float, default=200.0, help="publish Hz")
    ap.add_argument("--tick-dt", type=float, default=0.0,
                    help="PLANT-CLOCK the filter: seconds of plant time per rt/lowstate tick "
                         "(the plant's physics timestep; e.g. 0.002 for the RoboCasa sim). "
                         "When > 0, iterations are gated on NEW ticks and every integration "
                         "step (EKF P growth, IMU predict, complementary alphas, odometric "
                         "xy) uses tick-delta time instead of wall dt. On a slower-than-"
                         "realtime plant the wall-clocked default over-integrates by 1/RTF -- "
                         "measured: odometric xy drifted 4x faster than the robot moved, "
                         "feeding the planner a kinematically INCONSISTENT (pos, vel) pair "
                         "(the exact instability the odometric-xy note below describes) -- "
                         "and re-eats each unchanged lowstate sample several times. 0 (default) "
                         "= wall-clocked, byte-identical to the validated real/twin behavior "
                         "(their plants run at 1x, where the two clocks agree).")
    ap.add_argument("--lowstate-topic", default="rt/lowstate")
    ap.add_argument("--out-topic", default="rt/sportmodestate",
                    help="topic to publish the estimate on (use rt/sportmodestate_est for twin compare)")
    ap.add_argument("--vel-lpf-ms", type=float, default=30.0,
                    help="vertical (vz) leg-odo low-pass tau; also the leg-odo-only LPF when fusion off")
    ap.add_argument("--imu-fusion", dest="imu_fusion", action="store_true", default=True,
                    help="fuse IMU accel into the HORIZONTAL base velocity (default ON)")
    ap.add_argument("--no-imu-fusion", dest="imu_fusion", action="store_false",
                    help="disable IMU fusion -> leg odometry only (vx,vy,vz all leg-odo)")
    ap.add_argument("--imu-tau-ms", type=float, default=300.0,
                    help="complementary time constant: leg odo corrects the IMU drift over this window")
    ap.add_argument("--static-dq", type=float, default=0.3,
                    help="joint-vel-norm below which a sample is 'static' (updates the accel/gravity bias)")
    ap.add_argument("--contact", choices=["min-speed", "mean"], default="min-speed",
                    help="(complementary filter only) leg-odo foot: min-speed or mean")
    ap.add_argument("--filter", choices=["rw-ekf", "complementary", "kf12"], default="rw-ekf",
                    help="rw-ekf (default) = random-walk velocity EKF: per-foot leg-odo updates with "
                         "torque-load-scaled noise + Mahalanobis outlier gate -- caps the phantom "
                         "velocity spikes that topple the planner (offline: bal-max 1.6->0.7 m/s). "
                         "complementary = the earlier min-speed + horizontal-IMU fusion (lower RMS, "
                         "keeps the spikes). kf12 = the MIT-Cheetah/unitree_guide contact-aided "
                         "linear KF (see the module docstring) -- has a real IMU propagation "
                         "backbone, rejects swing feet ~100x, no force-accept hatch, filters base "
                         "position, and corrects the torso-mounted IMU through the waist joint.")

    g = ap.add_argument_group("kf12 (reference-design filter)")
    g.add_argument("--accel-mode", choices=["auto", "specific", "linear", "off"], default="auto",
                   help="accelerometer convention. auto (default) classifies from STATIC samples: "
                        "|mean R@a| ~9.81 => 'specific' (gravity-inclusive specific force: real IMUs "
                        "and MuJoCo's <accelerometer>, u = R@a + g); ~0 but alive => 'linear' "
                        "(gravity-free, u = R@a); identically zero => 'off' (dead sensor, no "
                        "propagation). 'off' forces the legacy no-propagation behaviour. NOTE the "
                        "old boot probe called the twin 'unusable'; the twin scene does carry a real "
                        "<accelerometer> and the bridge indexes it correctly, so that was very likely "
                        "a probe artifact (a specific-force sensor reads ~0 in free fall too).")
    g.add_argument("--accel-probe-sec", type=float, default=1.0,
                   help="seconds of STATIC samples (|dq| < --static-dq) needed to classify the accel")
    g.add_argument("--waist-fix", choices=["kf12", "all", "off"], default="kf12",
                   help="correct the TORSO-mounted IMU through torso_joint (motor 12, z-axis, "
                        "+-2.35 rad) before the leg FK. The imu site lives in torso_link, so the raw "
                        "IMU quat is the TORSO's orientation while the legs hang off the PELVIS -- "
                        "feeding it straight to the free joint rotates the whole leg FK, and hence "
                        "the base velocity, by the waist angle. kf12 (default) = fix kf12 only and "
                        "leave the legacy paths byte-identical; all = also fix rw-ekf/complementary; "
                        "off = never.")
    g.add_argument("--suspect", type=float, default=100.0,
                   help="noise inflation for an untrusted foot: scale = 1 + (1-trust)*suspect. 100 is "
                        "the value BOTH unitree_guide (_largeVariance) and MIT (high_suspect_number) ship")
    g.add_argument("--p0", type=float, default=100.0, help="initial covariance (both references: 100)")
    g.add_argument("--q-pos", type=float, default=0.02, help="MIT imu_process_noise_position")
    g.add_argument("--q-vel", type=float, default=0.02, help="MIT imu_process_noise_velocity")
    g.add_argument("--q-foot", type=float, default=0.002, help="MIT foot_process_noise_position")
    g.add_argument("--r-fpos", type=float, default=0.01,
                   help="relative-foot-POSITION meas noise (unitree_guide identified ~0.008-0.019). "
                        "These rows are never inflated -- they keep a swing foot's state fresh.")
    g.add_argument("--r-fvel", type=float, default=1.0,
                   help="relative-foot-VELOCITY meas noise (unitree_guide identified ~0.87-6.23)")
    g.add_argument("--r-fh", type=float, default=1.0, help="foot-height meas noise (unitree_guide 1.0)")
    g.add_argument("--kf-chi2", type=float, default=16.27,
                   help="per-foot Mahalanobis gate on the velocity block (3-dof chi^2; 16.27 = 99.9%%)")
    g.add_argument("--fz-hi", type=float, default=0.15,
                   help="contact STRIKE threshold as a fraction of body weight (mg read from the model)")
    g.add_argument("--fz-lo", type=float, default=0.05,
                   help="contact RELEASE threshold as a fraction of body weight")
    g.add_argument("--strike-ms", type=float, default=5.0,
                   help="f_z must exceed fz_hi this long to latch contact (Pronto's Atlas detector: "
                        "a 20-30N discontinuity lasting >5ms). NOTE at 200Hz one tick IS 5ms, so we "
                        "sit exactly at this detector's resolution limit; the references ran 500Hz.")
    g.add_argument("--min-stance-ms", type=float, default=50.0, help="minimum stance dwell before release")
    g.add_argument("--tau-sigma", type=float, default=1.0,
                   help="assumed per-joint tau_est noise (N.m). Sets the expected f_z error via "
                        "cov(f) = tau_sigma^2 * inv(J J^T)")
    g.add_argument("--fz-sigma-max", type=float, default=25.0,
                   help="hold the contact state (do not let f_z strike/release a foot) once the "
                        "expected f_z error exceeds this. A STRAIGHTENING KNEE collapses "
                        "d(foot_z)/d(joints) -- measured |J[2,:]| ~ 0.2*knee -- so f_z's error "
                        "explodes: ~8N at knee 0.6, ~12N at 0.36, ~62N at 0.08 (LOCKSTAND), blind "
                        "at 0. Against 100N/33N thresholds a locked knee makes the force signal "
                        "pure noise, so we fall back to the kinematic gates there.")
    g.add_argument("--td-ramp-ms", type=float, default=35.0,
                   help="trust ramps 0->1 over this long after a strike edge (impact rejection: "
                        "Pronto/Camurri inflate leg-odo covariance during the strike window)")
    g.add_argument("--kin-h", type=float, default=0.03,
                   help="kinematic gate: foot must be within this height of the flat-contact height")
    g.add_argument("--kin-v", type=float, default=0.15, help="kinematic gate: max foot speed (m/s)")
    g.add_argument("--kin-fail", type=float, default=0.3, help="trust multiplier when a kinematic gate fails")
    g.add_argument("--no-sole", dest="sole", action="store_false", default=True,
                   help="track the ankle_roll BODY origin instead of the SOLE contact point. The "
                        "planted-foot constraint is about the CONTACT POINT; the sole sits ~4.7cm "
                        "below the ankle body, so the two velocities differ by omega_foot x r -- "
                        "negligible standing (foot omega ~ 0), but several rad/s x 0.047m = 0.3-1.0 "
                        "m/s of BIAS through a trot's heel-strike/toe-off. On by default; this flag "
                        "exists to A/B the shipped (biased) behaviour.")
    g.add_argument("--cop", action="store_true",
                   help="enable the flat-foot CoP check (p_x = -tau_ankle_pitch/f_z): a CoP pinned at "
                        "the toe/heel edge means the contact point is MIGRATING (the foot is rolling) "
                        "-> downweight + inflate foot process noise. OFF BY DEFAULT: the H1-2 ankle is "
                        "a PARALLEL 2-DoF mechanism and mode_pr selects whether tau_est is in motor or "
                        "serial-ankle space -- verify that before trusting these signs (plan item V3).")
    g.add_argument("--cop-margin", type=float, default=0.15,
                   help="CoP must stay this fraction of the sole away from the edge to keep full trust")
    g.add_argument("--cop-fail", type=float, default=0.5, help="trust multiplier when the CoP is pinned")
    g.add_argument("--roll-infl", type=float, default=10.0,
                   help="extra foot-position noise factor while rolling (Bloesch w_p / Rotella w_z rationale)")
    g.add_argument("--sole-fwd", type=float, default=0.133,
                   help="ankle->toe (m). Default = the MEASURED real support polygon (heel -0.079, "
                        "centre +0.027, toe +0.133)")
    g.add_argument("--sole-back", type=float, default=0.079, help="ankle->heel (m), measured")
    g.add_argument("--sole-half-y", type=float, default=0.04, help="sole half-width (m)")
    g.add_argument("--noc-coast-sec", type=float, default=0.5,
                   help="how long to COAST on IMU with zero contacts before deciding we are not "
                        "flying but HANGING/held/contact-detector-failed. A real humanoid flight "
                        "phase is 0.3-0.5s, so coasting that long is correct and is the whole point "
                        "of the propagation backbone. Past it, integration is unbounded nonsense: "
                        "MEASURED on the real robot suspended -- base_z 0.975 -> 50.99m and v -> "
                        "11 m/s in 30s, from ~2 deg of tilt error leaking gravity. 0 = never fall "
                        "back (the old, divergent behaviour).")
    # ---- ZUPT: zero-velocity update (ALL filters; OFF => byte-identical) ----
    g.add_argument("--zupt", action="store_true",
                   help="ZERO-VELOCITY UPDATE: when both feet are loaded AND the joints are "
                        "static AND the base is not rotating, fuse the pseudo-measurement "
                        "base_velocity == 0. Targets the documented long-hold ceiling: the "
                        "stand is a ~90s coin-flip decided by an estimator drift spike "
                        "(held run v_err peak 0.05 m/s vs fell run 0.70) -- in a planted "
                        "double-support stand, 0 IS the truth, so a 0.70 reading is a "
                        "phantom this clamps. Worth MOST on --filter kf12 (the only filter "
                        "that integrates accel, hence the only one that accrues bias drift). "
                        "OFF by default = byte-identical.")
    g.add_argument("--zupt-load", type=float, default=15.0,
                   help="ZUPT gate: per-leg sum|tau| over (knee, ankle_pitch) [Nm] required to "
                        "count that foot as LOADED. 15.0 matches ankle_zero_snap's LOADED "
                        "convention. Both feet must pass.")
    g.add_argument("--zupt-dq", type=float, default=0.15,
                   help="ZUPT gate: |dq| (27-vector norm, rad/s) below which the joints count as "
                        "static. Half of --static-dq (0.3) on purpose -- the accel-bias probe can "
                        "afford to be loose, a velocity clamp cannot.")
    g.add_argument("--zupt-gyro", type=float, default=0.05,
                   help="ZUPT gate: |gyro| (rad/s, ~2.9 deg/s) below which the BASE counts as "
                        "non-rotating. THE SAFETY GATE -- do not remove. A rigid topple about the "
                        "ankle has |dq|~0 but a real base velocity that leg odometry sees only via "
                        "the gyro; without this term the ZUPT would erase the capture-point signal "
                        "for the exact fall it must catch.")
    g.add_argument("--zupt-vr", type=float, default=0.01,
                   help="ZUPT pseudo-measurement noise on v==0 [m/s]. Smaller = harder clamp. "
                        "(Distinct from --zupt-r, which is the kf12 no-contact HANG prior.)")
    g.add_argument("--zupt-r", type=float, default=1.0,
                   help="measurement noise of the no-contact fallback prior (v->0, p_z->kinematic "
                        "height). Sets how fast we decay to the safe estimate (~0.7s at the default). "
                        "This is what rw-ekf and Unitree's onboard estimator effectively report when "
                        "suspended -- a fail-safe, not an accuracy feature.")
    g.add_argument("--v-clamp", type=float, default=4.0,
                   help="hard backstop on |base velocity| (m/s). This estimate seeds every planner "
                        "rollout; a blown-up velocity must never reach it. 0 = off.")
    g.add_argument("--accel-clamp-g", type=float, default=4.0,
                   help="reject |world accel| above this many g as a spike (reuse last sample)")
    g.add_argument("--dt-clamp", type=float, default=1.5,
                   help="clamp the propagation dt to this multiple of nominal (a large dt in B@u is a "
                        "velocity kick; the legacy 0.5s plant-clock cap is far too loose for a "
                        "PROPAGATING filter)")
    ap.add_argument("--ekf-amax", type=float, default=3.0,
                    help="plausible body accel (m/s^2): P grows at this rate -> filter bandwidth + "
                         "how fast the gate re-accepts. 3.0 = the bench-winning high-bandwidth "
                         "setting (est-fed hold went 25s -> 68s ~= truth parity)")
    ap.add_argument("--ekf-r0", type=float, default=0.02, help="base leg-odo meas noise (m/s, 1-sigma)")
    ap.add_argument("--ekf-r1", type=float, default=0.3,
                    help="load scaling: meas sigma = r0 + r1/max(leg_load_Nm, lfloor). Gentle by "
                         "default: never goes fully deaf (engagement transients must be tracked)")
    ap.add_argument("--ekf-lfloor", type=float, default=5.0,
                    help="leg-load floor (Nm) in the noise scaling")
    ap.add_argument("--ekf-chi2", type=float, default=11.34,
                    help="Mahalanobis gate (3-dof chi^2; 11.34 = 99%%)")
    ap.add_argument("--ekf-rejcap", type=int, default=50,
                    help="max consecutive rejected updates before force-accept (divergence backstop)")
    ap.add_argument("--compare", action="store_true",
                    help="also subscribe the TRUTH sportmodestate and print estimate-vs-truth error")
    ap.add_argument("--truth-topic", default="rt/sportmodestate")
    ap.add_argument("--selftest", action="store_true", help="offline math check (no DDS), then exit")
    a = ap.parse_args()

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

    latest = {"ls": None}
    ls_sub = ChannelSubscriber(a.lowstate_topic, LowState_)
    ls_sub.Init(lambda msg: latest.__setitem__("ls", msg), 10)

    truth = {"sp": None}
    if a.compare:
        truth_sub = ChannelSubscriber(a.truth_topic, SportModeState_)
        truth_sub.Init(lambda msg: truth.__setitem__("sp", msg), 10)
        if a.out_topic == a.truth_topic:
            print("[est] WARN: --compare with out-topic==truth-topic -> you'll hear your own "
                  "publish; use --out-topic rt/sportmodestate_est for a clean comparison")

    pub = ChannelPublisher(a.out_topic, SportModeState_)
    pub.Init()
    msg = unitree_go_msg_dds__SportModeState_()

    if a.filter == "kf12":
        fdesc = (f"KF12 (MIT-Cheetah/unitree_guide contact-aided linear KF, 2-foot port: "
                 f"suspect={a.suspect:g}, chi2={a.kf_chi2:g}, accel={a.accel_mode})")
    elif a.filter == "rw-ekf":
        fdesc = (f"RW-EKF (amax={a.ekf_amax}, r0={a.ekf_r0}, r1={a.ekf_r1}, chi2={a.ekf_chi2}, "
                 f"rejcap={a.ekf_rejcap})")
    else:
        fdesc = (f"complementary (contact={a.contact}, "
                 f"imu_fusion={'on tc%.0fms' % a.imu_tau_ms if a.imu_fusion else 'OFF'}, "
                 f"vz_lpf={a.vel_lpf_ms:.0f}ms)")
    print(f"[est] publishing base sportmodestate -> '{a.out_topic}' @ {a.rate:.0f}Hz | filter={fdesc}"
          + (f" | COMPARE vs '{a.truth_topic}'" if a.compare else ""))
    print("[est] waiting for rt/lowstate ...", flush=True)

    dt = 1.0 / a.rate
    plant_clocked = a.tick_dt > 0.0
    last_tick = None   # plant-clock mode: last processed lowstate tick
    vz_tau = a.vel_lpf_ms * 1e-3
    ac_tau = a.imu_tau_ms * 1e-3
    v_filt = np.zeros(3)
    accel_bias = np.zeros(3)   # online: tracks R@accel during low-motion -> absorbs gravity (any convention)
    bias_init = False
    res = np.zeros(6)
    # --- RW-EKF state. Leg-load proxy (H1-2 has NO foot-force sensors): |tau_est| of
    #     knee + ankle-pitch per leg -- motor idx L=(3,4), R=(9,10), mapped by foot name.
    LOAD_IDX = [(3, 4) if "left" in nm else (9, 10) for _, nm in foot_ids]
    I3 = np.eye(3)
    P = I3 * 0.04
    rej = 0
    pos_xy = np.zeros(2)   # odometric base xy = integral of the velocity estimate
    zupt_n = [0]           # ticks the ZUPT fired (0 with --zupt off; also the tuning signal:
                           # ~0% during a quiet stand means the gates are too tight to ever help)

    # --- kf12 state (the reference-design filter) --------------------------------
    kcfg = _kf12_cfg(a, m)
    imu_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    sole_off = height_C if a.sole else 0.0     # sole sits height_C below the ankle body
    ground_ref = 0.0 if a.sole else height_C   # planted foot's measured height
    kf = KF12(kcfg, ground_ref)
    trust_fsm = [FootTrust("left" if "left" in nm else "right", kcfg) for _, nm in foot_ids]
    probe = AccelProbe(a.accel_mode, a.static_dq, a.accel_probe_sec)
    kf_seeded = False
    noc_t = 0.0
    hang_warned = [False]
    v_clamp_warned = [False]
    fz_dbg = [0.0, 0.0]
    fzs_dbg = [0.0, 0.0]
    a_world_last = np.zeros(3)
    waist_fix = (a.waist_fix == "all") or (a.waist_fix == "kf12" and a.filter == "kf12")
    if a.filter == "kf12":
        if imu_sid < 0:
            raise SystemExit("[est] kf12 needs an 'imu' site in the model")
        if len(foot_ids) != 2:
            raise SystemExit(f"[est] kf12 is a 2-foot port; model has {len(foot_ids)} feet")
        print(f"[est] kf12: mg={kcfg.mg:.1f}N -> strike {kcfg.fz_hi:.0f}N / release {kcfg.fz_lo:.0f}N "
              f"(expect ~{kcfg.mg/2:.0f}N per foot standing) | suspect={kcfg.suspect:g} "
              f"chi2={kcfg.chi2:g} | accel={a.accel_mode} | waist-fix={'ON' if waist_fix else 'off'} "
              f"| CoP={'on' if kcfg.cop else 'OFF (ankle tau space unverified -- plan V3)'} | foot-pt={'SOLE' if a.sole else 'ankle body (BIASED in a trot)'}")
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
        # plant-clock mode: only a NEW tick is a new plant state -- re-processing an
        # unchanged sample would re-feed the EKF the same measurement and advance the
        # integrators through plant time that never elapsed. step_dt is the plant time
        # this iteration covers (capped: a post-freeze tick jump must not integrate as
        # one giant step).
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
            acc_sum = np.zeros(3); acc_n = 0; acc_done = False

        q = np.array([ls.motor_state[i].q for i in range(NJ)])
        dq = np.array([ls.motor_state[i].dq for i in range(NJ)])
        tau = np.array([ls.motor_state[i].tau_est for i in range(NJ)])
        quat = np.array(list(ls.imu_state.quaternion))      # wxyz
        gyro = np.array(list(ls.imu_state.gyroscope))        # body frame
        accel = np.array(list(ls.imu_state.accelerometer))   # body frame (specific force)
        R = _quat2mat(quat)

        # accel HEALTH probe (first 2s): tells us whether this IMU's accel is usable for
        # full fusion (real HW should read ~9.81 gravity-reaction; the twin reads ~0 = unusable)
        # NOTE this one-shot boot probe cannot tell "gravity-free" from "in free fall" or
        # "dead" -- kf12 uses AccelProbe instead, which only classifies on STATIC samples.
        if not acc_done and a.filter != "kf12":
            acc_sum += R @ accel; acc_n += 1
            if acc_n >= int(2 * a.rate):
                gmag = float(np.linalg.norm(acc_sum / acc_n))
                if 8.5 < gmag < 11.0:
                    verdict = "HEALTHY (gravity-inclusive, ~9.81) -> full IMU fusion viable on this robot"
                elif gmag < 1.0:
                    verdict = "gravity-FREE/zero (twin convention) -> accel NOT usable for prediction"
                else:
                    verdict = "UNUSUAL convention -> inspect before trusting accel"
                print(f"[est] accel health: |mean R@accel| = {gmag:.2f} m/s^2 -> {verdict}", flush=True)
                acc_done = True

        # --- pelvis frame. The `imu` site lives in torso_link, BEHIND torso_joint (z-axis
        #     waist yaw, motor 12) -- so the raw IMU quat/gyro are the TORSO's, while the
        #     legs hang off the PELVIS. Feeding the raw quat to the free joint rotates the
        #     whole leg FK by the waist angle. See pelvis_from_torso() / --waist-fix.
        if waist_fix:
            R_wp, omega_p = pelvis_from_torso(quat, gyro, q[TORSO_MOTOR], dq[TORSO_MOTOR])
            quat_fk = _mat2quat(R_wp)
        else:
            R_wp, omega_p, quat_fk = R, np.asarray(gyro, dtype=float), quat

        if a.filter == "kf12":
            kdt = min(step_dt, dt * a.dt_clamp)      # a big dt in B@u is a velocity KICK
            snap = kin_snapshot(m, data, foot_ids, imu_sid, q, dq, R_wp, omega_p,
                                jacp, jacr, sole_off)

            # --- propagation input: the accelerometer IS the site's specific force -------
            a_world = R @ accel                      # R = TORSO orientation == the site's
            if not np.all(np.isfinite(a_world)) or np.linalg.norm(a_world) > a.accel_clamp_g * 9.81:
                a_world = a_world_last               # spike / NaN -> reuse last (DDS hiccup)
            else:
                a_world_last = a_world
            probe.step(a_world, float(np.linalg.norm(dq)), kdt)
            u = probe.input(a_world)

            # --- per-foot contact + trust (measured; we have no gait clock) --------------
            trusts, contacts, rolling = [], [], []
            for k, (bid, nm) in enumerate(foot_ids):
                side = "left" if "left" in nm else "right"
                f, fz_sig = grf_from_torque(m, data, bid, LEG_DOF[side], tau[LEG_MOTOR[side]],
                                            jacp, jacr, kcfg.tau_sigma)
                fz_dbg[k] = float(f[2])
                fzs_dbg[k] = fz_sig
                foot_z = kf.x[2] + snap["pf"][k][2]          # world foot height, current estimate
                t = trust_fsm[k].step(kdt, fz_dbg[k], fz_sig,
                                      float(tau[ANKLE_PITCH_MOTOR[side]]),
                                      float(tau[ANKLE_ROLL_MOTOR[side]]),
                                      foot_z, float(np.linalg.norm(snap["vfw"][k])), ground_ref)
                trusts.append(t)
                contacts.append(trust_fsm[k].contact)
                rolling.append(trust_fsm[k].rolling)

            if not kf_seeded:
                # seed from the standing pose: the robot is parked at bring-up, which is
                # exactly the estimator-init precondition (static double support).
                bh = -min(float(snap["pf"][k][2] + snap["roff"][2])
                          for k in range(len(foot_ids))) + ground_ref
                kf.x[0:3] = np.array([0.0, 0.0, bh]) + snap["roff"]
                kf.reset_feet(snap["pf"])
                kf_seeded = True
                print(f"[est] kf12 seeded: base_z={bh:.3f} site_z={kf.x[2]:.3f} "
                      f"fz=[{fz_dbg[0]:.0f},{fz_dbg[1]:.0f}]N (expect ~{kcfg.mg/2:.0f} each standing "
                      f"-- if these are wildly off, tau_est's sign/scale needs checking: plan V4) "
                      f"fz_sigma=[{fzs_dbg[0]:.0f},{fzs_dbg[1]:.0f}]N"
                      + ("  *** KNEES TOO STRAIGHT: f_z is noise here, holding contact on kinematics"
                         if (trust_fsm[0].blind or trust_fsm[1].blind) else ""), flush=True)

            # --- NO-CONTACT divergence guard ------------------------------------
            # Zero contacts is CORRECT to coast through for a real flight phase, and
            # unbounded nonsense past it (real robot, suspended: 0.975 -> 51m in 30s).
            noc_t = 0.0 if any(contacts) else noc_t + kdt
            hanging = a.noc_coast_sec > 0.0 and noc_t > a.noc_coast_sec
            if hanging:
                u = np.zeros(3)                     # stop integrating
            kf.predict(u, kdt, trusts, contacts)
            kf.update(snap, trusts, contacts, rolling)
            if hanging:
                bh_kin = -min(float(snap["pf"][k][2] + snap["roff"][2])
                              for k in range(len(foot_ids))) + ground_ref
                kf.hang_prior(bh_kin + snap["roff"][2], a.zupt_r)
            clamped = kf.clamp_v(a.v_clamp)
            if clamped and not v_clamp_warned[0]:
                v_clamp_warned[0] = True
                print(f"[est] WARN: |v| hit the {a.v_clamp} m/s backstop -- the filter is "
                      f"diverging (contact lost? attitude bad?). Investigate; do NOT trust "
                      f"this estimate in-loop.", flush=True)
            if hanging and not hang_warned[0]:
                hang_warned[0] = True
                print(f"[est] no contact for >{a.noc_coast_sec:g}s -> NOT flying; falling back to "
                      f"the safe estimate (v->0, z->kinematic). Feet unloaded / on the harness?",
                      flush=True)
            elif not hanging:
                hang_warned[0] = False

            # ZUPT: kf12 INTEGRATES (v += (R a + g) dt) and carries no accel-bias state,
            # so residual bias becomes unbounded drift -- exactly the failure recorded in
            # the trot attempt ("the robot never static in a trot => the bias never
            # re-learned"). A stand IS static, so here the anchor is available and this is
            # the one place a ZUPT resets real accrued drift rather than a bad measurement.
            if a.zupt and zupt_hold(tau, dq, gyro, LOAD_IDX, a):
                zupt_fuse_kf12(kf, a.zupt_vr)
                zupt_n[0] += 1

            site_p = kf.x[0:3].copy()
            site_v = kf.x[3:6].copy()
            v_filt = site_v
            base_height = site_p[2] - snap["roff"][2]
            pos_xy = site_p[0:2]
        else:
            # --- per-foot leg odometry (planted-foot constraint on the H1-2 kinematics) ---
            data.qpos[:] = 0.0; data.qpos[3:7] = quat_fk; data.qpos[7:7 + NJ] = q
            data.qvel[:] = 0.0; data.qvel[3:6] = omega_p; data.qvel[6:6 + NJ] = dq
            mujoco.mj_forward(m, data)
            base_height = -min(float(data.xpos[bid][2]) for bid, _ in foot_ids) + height_C
            vfeet = []
            for bid, _ in foot_ids:
                mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
                vfeet.append(-res[3:6].copy())
            vfeet = np.array(vfeet)

            if a.filter == "rw-ekf":
                # --- random-walk velocity EKF. P grows at a plausible-body-accel rate; each
                #     foot is a velocity measurement whose noise scales inversely with its
                #     LEG LOAD (knee+ankle |tau|: an unloaded/swinging foot is ignored), and a
                #     Mahalanobis gate rejects phantom spikes -- self-regulating because P keeps
                #     growing while rejecting, so genuine motion is re-accepted within ~0.5s.
                #     No accelerometer in the prediction (twin accel is unusable; revisit on HW). ---
                P = P + ((a.ekf_amax * step_dt) ** 2) * I3   # random-walk growth over this step's plant time
                for k in range(len(vfeet)):
                    i1, i2 = LOAD_IDX[k]
                    load_k = abs(tau[i1]) + abs(tau[i2])
                    rk = (a.ekf_r0 + a.ekf_r1 / max(load_k, a.ekf_lfloor)) ** 2
                    S = P + rk * I3
                    inn = vfeet[k] - v_filt
                    if float(inn @ np.linalg.solve(S, inn)) > a.ekf_chi2 and rej < a.ekf_rejcap:
                        rej += 1
                        continue
                    rej = 0
                    K = P @ np.linalg.inv(S)
                    v_filt = v_filt + K @ inn
                    P = (I3 - K) @ P
                # ZUPT: both feet loaded + joints static + base not rotating => v == 0.
                # Placed AFTER the per-foot updates so it corrects whatever they concluded
                # (the 0.70 m/s phantom that sag-collapses the 150s hold is a bad per-foot
                # MEASUREMENT, not accrued drift -- rw-ekf does not integrate velocity).
                if a.zupt and zupt_hold(tau, dq, gyro, LOAD_IDX, a):
                    v_filt, P = zupt_fuse_v(v_filt, P, a.zupt_vr)
                    zupt_n[0] += 1
            else:
                # --- complementary (the earlier Tier-1.5): min-speed foot + horizontal IMU
                #     fusion; vertical = leg-odo LPF. Lower RMS but keeps the motion spikes. ---
                if a.contact == "min-speed" and len(vfeet) > 1:
                    v_lo = vfeet[int(np.argmin(np.linalg.norm(vfeet, axis=1)))]
                else:
                    v_lo = vfeet.mean(axis=0)
                a_world = R @ accel
                if np.linalg.norm(dq) < a.static_dq:        # proprioceptively static -> update bias
                    accel_bias += 0.02 * (a_world - accel_bias)
                    bias_init = True
                a_lin = a_world - accel_bias
                if a.imu_fusion and bias_init:
                    v_filt[0:2] = v_filt[0:2] + a_lin[0:2] * step_dt                             # IMU predict (horizontal)
                    v_filt[0:2] += (step_dt / (ac_tau + step_dt)) * (v_lo[0:2] - v_filt[0:2])    # leg-odo correct
                    v_filt[2] += (step_dt / (vz_tau + step_dt)) * (v_lo[2] - v_filt[2])          # vertical: leg-odo LPF
                else:
                    v_filt += (step_dt / (vz_tau + step_dt)) * (v_lo - v_filt)                   # leg-odo only (warmup)

            # ODOMETRIC xy: integrate the velocity estimate so the planner gets a kinematically
            # CONSISTENT (position, velocity) pair. Publishing a FROZEN xy with nonzero velocity
            # destabilizes the model-based planner from the first tick (it believes the CoM never
            # translates while feeling it move -> mis-corrects -> the bench fell in 1.6s).
            # Absolute drift is harmless: the task costs are translation-invariant (CoM and feet
            # translate together) and the planner replans from the instantaneous state.
            pos_xy += v_filt[0:2] * step_dt

            # pelvis -> IMU site (so the node's site->pelvis recovers the estimated base).
            # Exact at ANY waist angle: torso_link.body_pos == 0 (VERIFIED), so the site
            # offset is R_torso @ IMU_OFFSET and R here IS R_torso.
            roff = R @ IMU_OFFSET
            omega_w = R @ gyro
            site_p = np.array([pos_xy[0], pos_xy[1], base_height]) + roff
            site_v = v_filt + np.cross(omega_w, roff)

        for k in range(3):
            msg.position[k] = float(site_p[k])
            msg.velocity[k] = float(site_v[k])
        pub.Write(msg)

        n += 1
        if n % int(a.rate) == 0:
            line = (f"[est] {now - t0:6.1f}s  base_z={base_height:5.3f}  "
                    f"xy=[{pos_xy[0]:+.2f},{pos_xy[1]:+.2f}]  "
                    f"v=[{v_filt[0]:+.3f},{v_filt[1]:+.3f},{v_filt[2]:+.3f}] m/s")
            if a.zupt:
                line += f"  | zupt {100.0 * zupt_n[0] / max(n, 1):4.1f}%"
            if a.filter == "kf12":
                ct = ("L" if trust_fsm[0].contact else "-") + ("R" if trust_fsm[1].contact else "-")
                blind = "".join("B" if t.blind else "." for t in trust_fsm)
                line += (f"  | {ct} trust=[{trust_fsm[0].trust:.2f},{trust_fsm[1].trust:.2f}]"
                         f" fz=[{fz_dbg[0]:4.0f},{fz_dbg[1]:4.0f}]+-[{fzs_dbg[0]:3.0f},{fzs_dbg[1]:3.0f}]N"
                         f" blind={blind} gate=[{int(kf.gated[0])}{int(kf.gated[1])}]"
                         f" acc={probe.resolved or 'probing'}"
                         + (f" HANG({noc_t:.1f}s)" if hanging else ""))
            if a.compare and truth["sp"] is not None:
                tp = np.array(list(truth["sp"].position))
                tv = np.array(list(truth["sp"].velocity))
                line += (f"  | truth_z={tp[2]:5.3f}  pos_err={np.linalg.norm(site_p - tp):.3f}m"
                         f"  vel_err={np.linalg.norm(site_v - tv):.3f}m/s")
            print(line, flush=True)


if __name__ == "__main__":
    main()
