#!/usr/bin/env python3
"""reach_target_bridge — HAMS perception -> MJPC-MPC reach target bridge.

Architecture (chosen 2026-07-02, option A): a standalone ROS2 node that leaves the
real-robot MJPC control node (h12_control_node, pure CycloneDDS) completely
untouched. It:

  1. subscribes to a grasp/object PoseStamped (default `/mpc/reach_target`),
     published by h12_skills after graspgen (frame = `pelvis`);
  2. asks the running MJPC node for its current base pose via gRPC GetState
     (qpos[0:3] = pelvis position, qpos[3:7] = pelvis quaternion wxyz, in the
     MJPC PLANNER-WORLD frame defined by the Unitree IMU);
  3. converts the incoming target from the robot-body (`pelvis`) frame into
     MJPC-world:  target_world = base_pos + R(base_quat) . target_pelvis .
     This reconciles the two systems' DIFFERENT world frames (MJPC = Unitree-IMU
     vs HAMS = FAST-LIO `camera_init`) through the common robot-body frame — see
     memory project_hams_reach_target_wiring_2026-07-02;
  4. pushes it to the MJPC node over the EXISTING gRPC SetTaskParameters path as
     the live task params "Reach Active"=1 and "Reach X/Y/Z" (added to lean.cc
     2026-07-02). No new plumbing on the control node.

If the incoming pose is NOT already in `pelvis`, tf2 is used to bring it there
first (so camera-optical / camera_init / map targets also work), then step 3
applies. tf2 is optional; without it, non-pelvis frames are dropped with a warn.

This file has no build-time dependency on the C++ control node. It only needs the
mujoco_mpc gRPC python proto (agent_pb2 / agent_pb2_grpc) importable — pass
--ros-args -p proto_dir:=<path> if it is not on PYTHONPATH.
"""

import os
import sys
import math

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# --- optional tf2 (for non-pelvis input frames) -------------------------------
try:
    import tf2_ros
    from tf2_geometry_msgs import do_transform_pose_stamped
    _HAVE_TF2 = True
except Exception:  # pragma: no cover - tf2 not installed
    _HAVE_TF2 = False


def _import_agent_proto(proto_dir):
    """Import agent_pb2 / agent_pb2_grpc the same way the mjpc python tools do.

    Tries the package layouts mujoco_mpc.proto / proto / bare, optionally after
    prepending an explicit proto_dir to sys.path.
    """
    if proto_dir:
        proto_dir = os.path.abspath(proto_dir)
        sys.path.insert(0, proto_dir)                       # bare `import agent_pb2`
        sys.path.insert(0, os.path.dirname(proto_dir))      # `from proto import ...`
        sys.path.insert(0, os.path.dirname(os.path.dirname(proto_dir)))
    last = None
    for pkg in ("mujoco_mpc.proto", "proto", ""):
        try:
            if pkg:
                pb = __import__(f"{pkg}.agent_pb2", fromlist=["x"])
                pbg = __import__(f"{pkg}.agent_pb2_grpc", fromlist=["x"])
            else:
                pb = __import__("agent_pb2")
                pbg = __import__("agent_pb2_grpc")
            return pb, pbg
        except ImportError as e:
            last = e
    raise ImportError(
        f"could not import agent_pb2/agent_pb2_grpc (set -p proto_dir:=...): {last}")


def _quat_rot(q_wxyz, v):
    """Rotate vec v by quaternion q (w, x, y, z).  R(q) . v ."""
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


class ReachTargetBridge(Node):
    def __init__(self):
        super().__init__("reach_target_bridge")

        self.declare_parameter("target_topic", "/mpc/reach_target")
        self.declare_parameter("grpc_host", "localhost")
        # h12_control_node exposes the same gRPC service agent_server does; the
        # monitor uses port 10000 on the robot. Override if the node uses another.
        self.declare_parameter("grpc_port", 10000)
        self.declare_parameter("pelvis_frame", "pelvis")
        self.declare_parameter("proto_dir", "")
        # When true, also flip "Reach Active"=1 so the task switches from the
        # hardcoded reach_target numeric to the live target on the first message.
        self.declare_parameter("auto_activate", True)
        # Optional: also command the Strategy on each target (>=0 to enable). Off
        # by default -- the operator/skill owns the strategy; the bridge only
        # supplies WHERE to reach.
        self.declare_parameter("set_strategy", -1)
        # Drop targets whose GetState base pose looks unset (twin/debug sanity).
        self.declare_parameter("min_base_z", 0.3)

        self.topic = self.get_parameter("target_topic").value
        self.host = self.get_parameter("grpc_host").value
        self.port = int(self.get_parameter("grpc_port").value)
        self.pelvis_frame = self.get_parameter("pelvis_frame").value
        self.auto_activate = bool(self.get_parameter("auto_activate").value)
        self.set_strategy = int(self.get_parameter("set_strategy").value)
        self.min_base_z = float(self.get_parameter("min_base_z").value)

        # --- gRPC to the MJPC node -------------------------------------------
        import grpc
        self._grpc = grpc
        self.pb, self.pbg = _import_agent_proto(self.get_parameter("proto_dir").value)
        creds = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
        self.chan = grpc.secure_channel(
            f"{self.host}:{self.port}", creds,
            options=[("grpc.max_receive_message_length", 200 * 1024 * 1024)])
        self.stub = self.pbg.AgentStub(self.chan)
        self.get_logger().info(
            f"gRPC -> {self.host}:{self.port}; subscribing '{self.topic}' "
            f"(expect frame '{self.pelvis_frame}')")

        # --- tf2 (optional) ---------------------------------------------------
        if _HAVE_TF2:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        else:
            self.tf_buffer = None
            self.get_logger().warn(
                "tf2 not available; only targets already in the pelvis frame "
                "will be accepted (non-pelvis frames dropped).")

        self.sub = self.create_subscription(
            PoseStamped, self.topic, self._on_target, 10)
        self._activated = False

    # ------------------------------------------------------------------ helpers
    def _base_pose(self):
        """Return (base_pos[3], base_quat_wxyz[4]) from the MJPC node, or None."""
        try:
            st = self.stub.GetState(self.pb.GetStateRequest(), timeout=2.0).state
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"GetState failed: {e}")
            return None
        q = np.asarray(st.qpos, dtype=float)
        if q.size < 7:
            self.get_logger().error(f"GetState qpos too short ({q.size})")
            return None
        base_pos = q[0:3]
        base_quat = q[3:7]  # MuJoCo free joint quat = (w, x, y, z)
        if base_pos[2] < self.min_base_z:
            self.get_logger().warn(
                f"base z={base_pos[2]:.3f} < min_base_z; target dropped")
            return None
        return base_pos, base_quat

    def _to_pelvis(self, msg: PoseStamped):
        """Return the target position [3] expressed in the pelvis frame, or None."""
        frame = (msg.header.frame_id or "").lstrip("/")
        if frame == self.pelvis_frame or frame == "":
            p = msg.pose.position
            return np.array([p.x, p.y, p.z], dtype=float)
        if self.tf_buffer is None:
            self.get_logger().warn(
                f"target frame '{frame}' != '{self.pelvis_frame}' and tf2 "
                f"unavailable; dropped")
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                self.pelvis_frame, frame, rclpy.time.Time())
            out = do_transform_pose_stamped(msg, tf)
            p = out.pose.position
            return np.array([p.x, p.y, p.z], dtype=float)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"tf2 {frame}->{self.pelvis_frame} failed: {e}")
            return None

    def _push(self, world):
        req = self.pb.SetTaskParametersRequest()
        if self.auto_activate:
            req.parameters["Reach Active"].numeric = 1.0
        req.parameters["Reach X"].numeric = float(world[0])
        req.parameters["Reach Y"].numeric = float(world[1])
        req.parameters["Reach Z"].numeric = float(world[2])
        if self.set_strategy >= 0:
            req.parameters["Strategy"].numeric = float(self.set_strategy)
        try:
            self.stub.SetTaskParameters(req, timeout=2.0)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"SetTaskParameters failed: {e}")
            return
        if not self._activated:
            self._activated = True
            self.get_logger().info("live reach target ACTIVE (Reach Active=1)")

    # -------------------------------------------------------------------- cb
    def _on_target(self, msg: PoseStamped):
        p_pelvis = self._to_pelvis(msg)
        if p_pelvis is None:
            return
        bp = self._base_pose()
        if bp is None:
            return
        base_pos, base_quat = bp
        world = base_pos + _quat_rot(base_quat, p_pelvis)
        self.get_logger().info(
            f"target pelvis=({p_pelvis[0]:.3f},{p_pelvis[1]:.3f},{p_pelvis[2]:.3f}) "
            f"-> MJPC-world=({world[0]:.3f},{world[1]:.3f},{world[2]:.3f})")
        self._push(world)


def main(argv=None):
    rclpy.init(args=argv)
    node = ReachTargetBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
