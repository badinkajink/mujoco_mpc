#!/usr/bin/env python3
"""PASSIVE yaw pre-flight: measures the IMU-vs-table yaw error using the SAME
TagCore math tag_bridge runs, and prints the exact --imu_yaw_offset_deg to pass
to h12_control_node. Subscribes only (camera + rt/lowstate); publishes nothing.

Run from the ROS2-sourced shell, robot standing at its final heading:
  python3 yaw_preflight.py [--seconds 12]
"""
import argparse, math, os, sys, time
import numpy as np

VENV = os.path.expanduser("~/Desktop/h12/h1_mujoco/.venv/lib/python3.10/site-packages")
try:
    import cv2
except Exception:
    sys.path.insert(0, VENV); import cv2

HS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HS)
import tag_bridge_node as tb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--image-topic", default="/realsense/head/color/image_raw/compressed")
    ap.add_argument("--info-topic", default="/realsense/head/color/camera_info")
    ap.add_argument("--bundle", default=os.path.expanduser(
        "~/Desktop/h12/table_tags/table_tag_bundle.yaml"))
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.11109, 0.0175, 0.68789])
    ap.add_argument("--cam-pitch-deg", type=float, default=50.760)
    a = ap.parse_args()

    # deployed defaults: mirror tag_bridge's own argparse so the math matches
    bundle = tb.load_bundle(a.bundle) if hasattr(tb, 'load_bundle') else None
    if bundle is None:
        import re
        txt = open(a.bundle).read()
        bundle = {}
        for m in re.finditer(r'\{id:\s*(\d+),\s*size:\s*([\d.]+),\s*x:\s*([\d.]+),'
                             r'\s*y:\s*([\d.]+),\s*z:\s*([\d.-]+)', txt):
            bundle[int(m.group(1))] = (float(m.group(2)),
                                       np.array([float(m.group(3)), float(m.group(4)),
                                                 float(m.group(5))]))
    core = tb.TagCore(bundle, a.cam_pos,
                      tb.optical_in_torso(math.radians(a.cam_pitch_deg)),
                      2.0, 2, 4.0)
    det = cv2.aruco.ArucoDetector(tb.DICT)

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    except ModuleNotFoundError:
        for extra in (VENV, os.path.expanduser(
                "~/Desktop/h12/h1_mujoco/submodules/unitree_sdk2_python")):
            if extra not in sys.path:
                sys.path.insert(0, extra)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    # pick the robot-subnet interface (192.168.123.x) like the deploy tools do
    iface = None
    try:
        import subprocess
        out = subprocess.check_output(['ip', '-4', '-o', 'addr'], text=True)
        for line in out.splitlines():
            if '192.168.123.' in line:
                iface = line.split()[1]
                break
    except Exception:
        pass
    print(f"[yaw] DDS interface: {iface or 'auto'}")
    ChannelFactoryInitialize(0, iface) if iface else ChannelFactoryInitialize(0)
    latest = {"ls": None, "K": None, "dist": None}
    ls = ChannelSubscriber("rt/lowstate", LowState_)
    ls.Init(lambda m: latest.__setitem__("ls", m), 10)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, CompressedImage
    yoffs = []
    diag = {"frames": 0, "det": 0, "ls": 0, "solve_fail": 0}

    class Tap(Node):
        def __init__(self):
            super().__init__("yaw_preflight")
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(CameraInfo, a.info_topic, self.oi, qos)
            self.create_subscription(CompressedImage, a.image_topic, self.oj, qos)

        def oi(self, m):
            latest["K"] = np.array(m.k, float).reshape(3, 3)
            latest["dist"] = np.array(m.d, float) if len(m.d) else np.zeros(5)

        def oj(self, m):
            diag["frames"] += 1
            if latest["ls"] is not None:
                diag["ls"] = 1
            if latest["ls"] is None or latest["K"] is None:
                return
            g = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if g is None:
                return
            corners, ids, _ = det.detectMarkers(g)
            if ids is None:
                return
            dets = {int(i): c[0] for c, i in zip(corners, ids.flatten())}
            q = np.array(list(latest["ls"].imu_state.quaternion))
            waist = float(latest["ls"].motor_state[tb.TORSO_MOTOR].q)
            r = core.step(dets, latest["K"], latest["dist"], q, waist, dt=0.033)
            diag["det"] += len(dets)
            if r is not None:
                yoffs.append(math.degrees(r[1]))
            else:
                diag["solve_fail"] += 1

    rclpy.init(args=None)
    node = Tap()
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node(); rclpy.shutdown()

    if not yoffs:
        print("[yaw] NO solves. Diagnostics: camera frames=%d  lowstate=%s  "
              "tag detections=%d  solve rejections=%d" % (
              diag["frames"], "YES" if diag["ls"] else "NO", diag["det"], diag["solve_fail"]))
        if diag["frames"] == 0:
            print("[yaw]   -> no camera: realsense up on robot PC? RMW env exported?")
        elif not diag["ls"]:
            print("[yaw]   -> no rt/lowstate: robot on + wire connected?")
        elif diag["det"] == 0:
            print("[yaw]   -> camera OK but no tags seen: robot facing the table?")
        print("YAW_PREFLIGHT_OFFSET_DEG=FAIL", flush=True)  # machine-readable (node startup)
        return
    med = float(np.median(yoffs))
    print(f"[yaw] {len(yoffs)} solves; IMU-vs-table yaw error median {med:+.2f} deg "
          f"(sd {np.std(yoffs):.2f})")
    if abs(med) < 3.0:
        print("[yaw] PASS -- heading is clean, launch the node with no offset")
    else:
        print(f"[yaw] CORRECT IT: launch the node with  --imu_yaw_offset_deg {-med:.1f}")
    # machine-readable line for h12_control_node --yaw_preflight (exact value, even when small)
    print(f"YAW_PREFLIGHT_OFFSET_DEG={-med:.2f}", flush=True)


if __name__ == "__main__":
    main()
