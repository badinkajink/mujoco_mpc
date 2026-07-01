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
  so the home pose matches the model's home base height. Base XY is held at 0
  (translation-invariant for balance -- the planner re-plans from the instantaneous
  snapshot each tick). Orientation + gyro the node reads straight from rt/lowstate,
  so sportmodestate only needs position + velocity.

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


def _pick_iface(explicit):
    """Auto-pin the 192.168.123.x robot-subnet NIC (so empty binds the robot link,
    not WiFi/Tailscale), mirroring dds_topic_check/dds_live_recorder."""
    if explicit:
        return explicit, "explicit --iface"
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show"],
                             capture_output=True, text=True, timeout=4).stdout
    except Exception:
        out = ""
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 4 and p[1] != "lo" and p[3].split("/")[0].startswith(ROBOT_SUBNET):
            return p[1], f"auto-detected ({p[3].split('/')[0]} on robot subnet)"
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
        home_q = np.array(m.key_qpos[home * m.nq: home * m.nq + m.nq])
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


def _selftest(m, data, foot_ids, height_C, home_q):
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
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


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
    ap.add_argument("--filter", choices=["rw-ekf", "complementary"], default="rw-ekf",
                    help="rw-ekf (default) = random-walk velocity EKF: per-foot leg-odo updates with "
                         "torque-load-scaled noise + Mahalanobis outlier gate -- caps the phantom "
                         "velocity spikes that topple the planner (offline: bal-max 1.6->0.7 m/s). "
                         "complementary = the earlier min-speed + horizontal-IMU fusion (lower RMS, "
                         "keeps the spikes).")
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
        raise SystemExit(_selftest(m, data, foot_ids, height_C, home_q))

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

    fdesc = (f"RW-EKF (amax={a.ekf_amax}, r0={a.ekf_r0}, r1={a.ekf_r1}, chi2={a.ekf_chi2}, "
             f"rejcap={a.ekf_rejcap})" if a.filter == "rw-ekf"
             else f"complementary (contact={a.contact}, "
                  f"imu_fusion={'on tc%.0fms' % a.imu_tau_ms if a.imu_fusion else 'OFF'}, "
                  f"vz_lpf={a.vel_lpf_ms:.0f}ms)")
    print(f"[est] publishing base sportmodestate -> '{a.out_topic}' @ {a.rate:.0f}Hz | filter={fdesc}"
          + (f" | COMPARE vs '{a.truth_topic}'" if a.compare else ""))
    print("[est] waiting for rt/lowstate ...", flush=True)

    dt = 1.0 / a.rate
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
    Q_ekf = (a.ekf_amax * dt) ** 2 * I3
    rej = 0
    pos_xy = np.zeros(2)   # odometric base xy = integral of the velocity estimate
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
        if not seen:
            print("[est] rt/lowstate up -> estimating + publishing.", flush=True)
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
        if not acc_done:
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

        # --- per-foot leg odometry (planted-foot constraint on the H1-2 kinematics) ---
        data.qpos[:] = 0.0; data.qpos[3:7] = quat; data.qpos[7:7 + NJ] = q
        data.qvel[:] = 0.0; data.qvel[3:6] = gyro; data.qvel[6:6 + NJ] = dq
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
            P = P + Q_ekf
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
                v_filt[0:2] = v_filt[0:2] + a_lin[0:2] * dt                      # IMU predict (horizontal)
                v_filt[0:2] += (dt / (ac_tau + dt)) * (v_lo[0:2] - v_filt[0:2])  # leg-odo correct
                v_filt[2] += (dt / (vz_tau + dt)) * (v_lo[2] - v_filt[2])        # vertical: leg-odo LPF
            else:
                v_filt += (dt / (vz_tau + dt)) * (v_lo - v_filt)                 # leg-odo only (warmup)

        # ODOMETRIC xy: integrate the velocity estimate so the planner gets a kinematically
        # CONSISTENT (position, velocity) pair. Publishing a FROZEN xy with nonzero velocity
        # destabilizes the model-based planner from the first tick (it believes the CoM never
        # translates while feeling it move -> mis-corrects -> the bench fell in 1.6s).
        # Absolute drift is harmless: the task costs are translation-invariant (CoM and feet
        # translate together) and the planner replans from the instantaneous state.
        pos_xy += v_filt[0:2] * dt

        # pelvis -> IMU site (so the node's site->pelvis recovers the estimated base)
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
            if a.compare and truth["sp"] is not None:
                tp = np.array(list(truth["sp"].position))
                tv = np.array(list(truth["sp"].velocity))
                line += (f"  | truth_z={tp[2]:5.3f}  pos_err={np.linalg.norm(site_p - tp):.3f}m"
                         f"  vel_err={np.linalg.norm(site_v - tv):.3f}m/s")
            print(line, flush=True)


if __name__ == "__main__":
    main()
