#!/usr/bin/env python3
"""lio_bridge_node.py -- FAST-LIO /Odometry -> rt/aux_odom for the v4 estimator.

THE ONE GLUE NODE the docs called for (docs/onboard_base_velocity_estimator.md):
takes the LiDAR-inertial odometry that already runs in the HAMS stack
(livox_ros_driver2 + FAST_LIO, Livox MID-360 on the torso) and republishes it as
the v4 aux-odometry measurement, in the estimator's own conventions.

FRAME CHAIN (all constants verified against CL_Assets/mujoco_assets/
h1_2_magpie.xml:228-232 and h1_bringup's FAST-LIO config):

    camera_init ──(FAST-LIO /Odometry: pose+twist of `body`)──▶ livox frame
    livox ──E⁻¹ (fixed mount: pos LIVOX_POS, pitch LIVOX_PITCH on torso)──▶ torso
    torso ──Rz(-waist yaw, motor 12 from rt/lowstate)──▶ pelvis
    pelvis-in-camera_init ──Rz(yaw_off, low-passed vs the IMU quat)──▶ IMU-world

WHY THE YAW ALIGNMENT EXISTS: MJPC's world frame is the Unitree IMU's yaw at
boot; FAST-LIO's world is camera_init at ITS boot. Both agree on gravity, so
they differ by a yaw (plus a translation the v4 xy-anchor latches away). That
yaw is estimated continuously as yaw(IMU pelvis) - yaw(LIO pelvis), low-passed
with a long time constant (--yaw-tau, 30 s): slow enough to average noise, fast
enough to track the IMU's own yaw drift -- which is precisely how LIO's yaw
stability reaches the estimator without touching the sportmodestate seam.

VELOCITY: /Odometry twist is the LIVOX point's velocity; the pelvis velocity
differs by omega x r (r ~ 0.68 m -- at 1 rad/s that is 0.68 m/s, NOT optional).
--twist-frame declares FAST-LIO's twist convention (its stock output is the
odom/world frame; some forks emit body frame).

RUNTIME: needs BOTH ROS2 (rclpy, nav_msgs -- source the HAMS overlay) and the
twin venv's unitree_sdk2py. Run inside the ROS2 env with the venv's
site-packages on PYTHONPATH, e.g.:
  source ~/Desktop/HAMS/core_ws/install/setup.bash
  PYTHONPATH=$PYTHONPATH:~/Desktop/h12/h1_mujoco/.venv/lib/python3.10/site-packages \
      python3 lio_bridge_node.py
Offline math check (no ROS2, no DDS):
  python3 lio_bridge_node.py --selftest
"""
import argparse
import math
import time

import numpy as np

# Livox mount on torso_link (MJCF: pos + euler pitch). torso_link.body_pos == 0,
# so torso origin == pelvis origin and only the waist yaw separates their frames.
LIVOX_POS = np.array([0.04874, 0.0, 0.67980])
LIVOX_PITCH = 0.2401573
TORSO_MOTOR = 12


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _yaw(R):
    return math.atan2(R[1, 0], R[0, 0])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


E_TORSO_LIVOX = _ry(LIVOX_PITCH)          # R such that v_torso = E @ v_livox


class LioAligner:
    """Pure math: one /Odometry sample + one lowstate sample -> aux (xy, v).
    Stateless except the low-passed yaw offset. Selftestable without ROS."""

    def __init__(self, yaw_tau, twist_frame):
        self.yaw_tau = yaw_tau
        self.twist_frame = twist_frame
        self.yaw_off = None            # yaw(IMU pelvis) - yaw(LIO pelvis), LPF'd
        self.n = 0

    def step(self, p_livox, q_livox_wxyz, v_twist, w_twist,
             imu_quat_wxyz, waist_q, dt):
        """Returns (pelvis_xy, pelvis_v_world_imu_frame, yaw_off)."""
        R_cl = _quat_wxyz_to_mat(np.asarray(q_livox_wxyz, float))   # caminit<-livox
        R_ct = R_cl @ E_TORSO_LIVOX.T                               # caminit<-torso
        R_cp = R_ct @ _rz(-float(waist_q))                          # caminit<-pelvis
        r_w = R_ct @ LIVOX_POS                       # pelvis->livox lever, caminit frame
        p_pelvis = np.asarray(p_livox, float) - r_w

        v = np.asarray(v_twist, float)
        w = np.asarray(w_twist, float)
        if self.twist_frame == "body":               # rotate body-frame twist out
            v = R_cl @ v
            w = R_cl @ w
        v_pelvis = v - np.cross(w, r_w)              # omega x r: 0.68m lever, NOT optional

        # --- continuous yaw alignment against the IMU (torso) quat -------------
        R_wt_imu = _quat_wxyz_to_mat(np.asarray(imu_quat_wxyz, float))
        R_wp_imu = R_wt_imu @ _rz(-float(waist_q))
        off = _wrap(_yaw(R_wp_imu) - _yaw(R_cp))
        if self.yaw_off is None:
            self.yaw_off = off
        else:
            alpha = dt / (self.yaw_tau + dt)
            self.yaw_off = _wrap(self.yaw_off + alpha * _wrap(off - self.yaw_off))
        Rz_off = _rz(self.yaw_off)
        self.n += 1
        return (Rz_off @ p_pelvis)[:2], Rz_off @ v_pelvis, self.yaw_off


def _selftest():
    ok = True
    # Ground truth: pelvis at p, yawed 0.3 in the IMU world; LIO world rotated
    # -0.4 from IMU world (different boot yaw); waist at 0.5; pelvis moving at
    # v_true while spinning at w_true. Build the livox-frame odometry that
    # FAST-LIO would report, run the aligner, demand the truth back.
    yaw_imu, yaw_lio_off, waist = 0.3, -0.4, 0.5
    v_true = np.array([0.30, -0.10, 0.05])
    w_true = np.array([0.0, 0.8, 0.2])       # PITCH-dominant: exercises the 0.68m z-lever
    p_pelvis_imu = np.array([1.0, 2.0, 1.0])

    R_wp = _rz(yaw_imu)                              # IMU-world <- pelvis (yaw only)
    R_wt = R_wp @ _rz(waist)                         # IMU-world <- torso
    imu_quat = np.array([math.cos(yaw_imu * 0.5 + waist * 0.5), 0.0, 0.0,
                         math.sin(yaw_imu * 0.5 + waist * 0.5)])   # torso quat

    R_wc = _rz(-yaw_lio_off)                         # IMU-world <- camera_init... inverse:
    R_cw = R_wc.T                                    # camera_init <- IMU-world
    # livox pose in camera_init
    R_cl = R_cw @ R_wt @ E_TORSO_LIVOX
    p_livox_imu = p_pelvis_imu + R_wt @ LIVOX_POS
    p_livox_c = R_cw @ p_livox_imu
    q_cl = np.array([math.cos(_yaw(R_cl) * 0.5), 0.0, 0.0, math.sin(_yaw(R_cl) * 0.5)])
    # cheat: R_cl here is yaw-only by construction (all test rotations are yaw +
    # the fixed pitch); rebuild exactly to avoid quat conversion error:
    # verify R_cl is reproduced by q_cl only in yaw -- so instead pass the matrix
    # through a proper quaternion:
    from numpy.linalg import svd
    # robust mat->quat (yaw+pitch mix): use the trace method
    def mat2quat(R):
        t = np.trace(R)
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12)) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
        return q
    q_cl = mat2quat(R_cl)

    # livox point velocity in camera_init: v + w x r (w in caminit frame)
    w_c = R_cw @ w_true
    r_c = (R_cw @ R_wt) @ LIVOX_POS
    v_livox_c = R_cw @ v_true + np.cross(w_c, r_c)

    al = LioAligner(yaw_tau=0.0, twist_frame="world")   # tau 0 = trust first sample
    xy, v_out, yoff = al.step(p_livox_c, q_cl, v_livox_c, w_c,
                              imu_quat, waist, dt=0.1)
    e_v = float(np.linalg.norm(v_out - v_true))
    e_xy = float(np.linalg.norm(xy - p_pelvis_imu[:2]))
    # the aligner must recover yaw(R_wc): the caminit->IMU-world rotation, which
    # by construction above is -yaw_lio_off (R_wc = Rz(-yaw_lio_off)).
    want_off = -yaw_lio_off
    e_yaw = abs(_wrap(yoff - want_off))
    good = e_v < 1e-9 and e_xy < 1e-9 and e_yaw < 1e-9
    ok &= good
    print(f"[selftest] lio round-trip  : v_err={e_v:.2e} xy_err={e_xy:.2e} "
          f"yaw_off={yoff:+.3f} (expect {want_off:+.3f}, err {e_yaw:.2e}) "
          f"({'ok' if good else 'FAIL'})")

    # body-frame twist convention must give the same answer
    al2 = LioAligner(yaw_tau=0.0, twist_frame="body")
    xy2, v2, _ = al2.step(p_livox_c, q_cl, R_cl.T @ v_livox_c, R_cl.T @ w_c,
                          imu_quat, waist, dt=0.1)
    good = float(np.linalg.norm(v2 - v_true)) < 1e-9
    ok &= good
    print(f"[selftest] lio body twist  : v_err={np.linalg.norm(v2 - v_true):.2e} "
          f"({'ok' if good else 'FAIL'})")

    # omega x r matters: at 0.8 rad/s of PITCH the 0.68m z-lever contributes
    # ~0.55 m/s of velocity that would be silently attributed to the pelvis
    # without the correction. (A pure yaw spin only sees the 5cm x-offset.)
    lever = float(np.linalg.norm(np.cross(w_true, R_wt @ LIVOX_POS)))
    good = lever > 0.3
    ok &= good
    print(f"[selftest] lio lever check : |w x r| = {lever:.3f} m/s at |w|=0.82 rad/s "
          f"(silently wrong without the correction) ({'ok' if good else 'FAIL'})")

    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--odom-topic", default="/Odometry", help="FAST-LIO odometry (ROS2)")
    ap.add_argument("--aux-topic", default="rt/aux_odom", help="DDS output (v4 aux bus)")
    ap.add_argument("--lowstate-topic", default="rt/lowstate")
    ap.add_argument("--twist-frame", choices=["world", "body"], default="world",
                    help="frame of /Odometry twist. Stock FAST-LIO2 = world (camera_init); "
                         "some forks emit body. VERIFY on first run: stand still, rotate "
                         "the waist -- the published aux velocity must stay ~0.")
    ap.add_argument("--yaw-tau", type=float, default=30.0,
                    help="LPF time constant of the IMU-vs-LIO yaw offset (s)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--no-auto-iface", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="offline math check, then exit")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(_selftest())

    # --- DDS side (unitree sdk) ---------------------------------------------
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from base_estimator_node import _pick_iface
    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelSubscriber,
                                             ChannelPublisher)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

    if a.no_auto_iface:
        iface = None
    else:
        iface, why = _pick_iface(a.iface)
        print(f"[lio] DDS interface = {iface or 'autodetermine'} ({why})")
    if iface:
        ChannelFactoryInitialize(a.domain, iface)
    else:
        ChannelFactoryInitialize(a.domain)

    latest = {"ls": None}
    ls_sub = ChannelSubscriber(a.lowstate_topic, LowState_)
    ls_sub.Init(lambda m: latest.__setitem__("ls", m), 10)
    pub = ChannelPublisher(a.aux_topic, SportModeState_)
    pub.Init()
    out = unitree_go_msg_dds__SportModeState_()

    # --- ROS2 side ----------------------------------------------------------
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry

    aligner = LioAligner(a.yaw_tau, a.twist_frame)
    stats = {"n": 0, "t0": time.time(), "last_log": 0.0}

    class Bridge(Node):
        def __init__(self):
            super().__init__("lio_bridge")
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(Odometry, a.odom_topic, self.on_odom, qos)
            self.get_logger().info(
                f"lio_bridge: {a.odom_topic} (twist={a.twist_frame}) -> DDS "
                f"'{a.aux_topic}', yaw-tau {a.yaw_tau}s")

        def on_odom(self, msg):
            ls = latest["ls"]
            if ls is None:
                return                                  # no lowstate -> no waist/imu yet
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            tv = msg.twist.twist.linear
            tw = msg.twist.twist.angular
            imu_quat = np.array(list(ls.imu_state.quaternion))
            waist = float(ls.motor_state[TORSO_MOTOR].q)
            xy, v, yoff = aligner.step(
                np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z]),
                np.array([tv.x, tv.y, tv.z]), np.array([tw.x, tw.y, tw.z]),
                imu_quat, waist, dt=0.1)
            out.position[0] = float(xy[0])
            out.position[1] = float(xy[1])
            out.position[2] = float(p.z)                # informational; v4 ignores z
            for k in range(3):
                out.velocity[k] = float(v[k])
            out.mode = 1                                # "valid" marker
            pub.Write(out)
            stats["n"] += 1
            now = time.time()
            if now - stats["last_log"] > 5.0:
                stats["last_log"] = now
                hz = stats["n"] / max(now - stats["t0"], 1e-6)
                print(f"[lio] {stats['n']} samples ({hz:.1f} Hz avg) "
                      f"yaw_off={math.degrees(yoff):+.1f} deg "
                      f"v=[{v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}]", flush=True)

    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
