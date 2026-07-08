#!/usr/bin/env python3
"""wss_cmd_bridge: /cmd_vel (geometry_msgs/Twist) -> MJPC deploy-node gRPC
SetTaskParameters ("Cmd Active/Vx/Vy/Seq", the live teleop seam added to the
lean task 2026-07-03).

THE one pipe shared by every velocity client: wasd_teleop.py, a gamepad
teleop_twist_joy, and (later, V4) Nav2 all just publish Twist here. The
node-side governor (lean.cc TransitionLocked) owns ALL safety -- envelope
clamp, slew, gait-arm lockout, settle-through-zero, 1 s heartbeat watchdog.
This bridge only forwards at a fixed rate and zeroes on input silence
(client-side belt; the node watchdog is the suspenders).

Usage (deploy node running with gRPC on :10000):
  python3 wss_cmd_bridge.py [--ros-args -p grpc_host:=localhost -p grpc_port:=10000 \
      -p proto_dir:=<dir with agent_pb2*.py> -p rate_hz:=20.0 -p input_timeout:=0.5]

Twist convention (BODY frame): linear.x = vx fwd(+)/back(-) m/s,
linear.y = vy left(+) m/s (governed to 0 until the V2 lateral campaign),
angular.z ignored until V3 (yaw). On shutdown sends Cmd Active=0 three times.
"""
import os
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def _import_agent_proto(proto_dir):
    """Import agent_pb2 / agent_pb2_grpc (same approach as the reach bridge)."""
    last = None
    if proto_dir:
        sys.path.insert(0, proto_dir)
        sys.path.insert(0, os.path.dirname(proto_dir))
    for pkg in ("", "proto"):
        try:
            if pkg:
                pb = __import__(f"{pkg}.agent_pb2", fromlist=["x"])
                pbg = __import__(f"{pkg}.agent_pb2_grpc", fromlist=["x"])
            else:
                pb = __import__("agent_pb2")
                pbg = __import__("agent_pb2_grpc")
            return pb, pbg
        except Exception as e:  # noqa: BLE001
            last = e
    raise ImportError(
        f"could not import agent_pb2/agent_pb2_grpc (set -p proto_dir:=...): {last}")


class WssCmdBridge(Node):
    def __init__(self):
        super().__init__("wss_cmd_bridge")
        self.declare_parameter("grpc_host", "localhost")
        self.declare_parameter("grpc_port", 10000)
        self.declare_parameter("proto_dir", "")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("input_timeout", 0.5)   # s of Twist silence -> zero

        import grpc
        self.pb, self.pbg = _import_agent_proto(self.get_parameter("proto_dir").value)
        creds = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
        host = self.get_parameter("grpc_host").value
        port = int(self.get_parameter("grpc_port").value)
        self.chan = grpc.secure_channel(f"{host}:{port}", creds)
        self.stub = self.pbg.AgentStub(self.chan)

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.last_rx = self.get_clock().now()
        self.seq = 0
        self.timeout = float(self.get_parameter("input_timeout").value)

        self.sub = self.create_subscription(
            Twist, self.get_parameter("cmd_topic").value, self._on_twist, 10)
        self.timer = self.create_timer(
            1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self.get_logger().info(
            f"bridging {self.get_parameter('cmd_topic').value} -> gRPC {host}:{port} "
            f"@ {self.get_parameter('rate_hz').value} Hz (zero after "
            f"{self.timeout}s silence; node governor owns safety)")

    def _on_twist(self, msg: Twist):
        self.vx = float(msg.linear.x)
        self.vy = float(msg.linear.y)
        self.wz = float(msg.angular.z)   # yaw-rate (V3); node clamps to drive_wz_max
        self.last_rx = self.get_clock().now()

    def _tick(self):
        vx, vy, wz = self.vx, self.vy, self.wz
        silent = (self.get_clock().now() - self.last_rx).nanoseconds * 1e-9
        if silent > self.timeout:
            vx, vy, wz = 0.0, 0.0, 0.0
        self.seq += 1
        req = self.pb.SetTaskParametersRequest()
        req.parameters["Cmd Active"].numeric = 1.0
        req.parameters["Cmd Vx"].numeric = vx
        req.parameters["Cmd Vy"].numeric = vy
        req.parameters["Cmd Wz"].numeric = wz
        req.parameters["Cmd Seq"].numeric = float(self.seq)
        try:
            self.stub.SetTaskParameters(req, timeout=0.5)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"SetTaskParameters failed: {e}", throttle_duration_sec=2.0)

    def deactivate(self):
        """Send Cmd Active=0 a few times so the node returns to legacy numerics."""
        import time
        for _ in range(3):
            try:
                req = self.pb.SetTaskParametersRequest()
                req.parameters["Cmd Active"].numeric = 0.0
                req.parameters["Cmd Vx"].numeric = 0.0
                req.parameters["Cmd Vy"].numeric = 0.0
                req.parameters["Cmd Wz"].numeric = 0.0
                self.stub.SetTaskParameters(req, timeout=0.5)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)


def main():
    rclpy.init()
    node = WssCmdBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.deactivate()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
