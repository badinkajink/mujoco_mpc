"""Launch the reach_target_bridge.

Example:
  ros2 launch reach_target_bridge reach_target_bridge.launch.py \
      target_topic:=/mpc/reach_target grpc_port:=10000 proto_dir:=/path/to/mujoco_mpc/proto
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("target_topic", default_value="/mpc/reach_target"),
        DeclareLaunchArgument("grpc_host", default_value="localhost"),
        DeclareLaunchArgument("grpc_port", default_value="10000"),
        DeclareLaunchArgument("pelvis_frame", default_value="pelvis"),
        DeclareLaunchArgument("proto_dir", default_value=""),
        DeclareLaunchArgument("auto_activate", default_value="true"),
        DeclareLaunchArgument("set_strategy", default_value="-1"),
    ]
    node = Node(
        package="reach_target_bridge",
        executable="bridge",
        name="reach_target_bridge",
        output="screen",
        parameters=[{
            "target_topic": LaunchConfiguration("target_topic"),
            "grpc_host": LaunchConfiguration("grpc_host"),
            "grpc_port": LaunchConfiguration("grpc_port"),
            "pelvis_frame": LaunchConfiguration("pelvis_frame"),
            "proto_dir": LaunchConfiguration("proto_dir"),
            "auto_activate": LaunchConfiguration("auto_activate"),
            "set_strategy": LaunchConfiguration("set_strategy"),
        }],
    )
    return LaunchDescription(args + [node])
