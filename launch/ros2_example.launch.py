from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate the launch description.

    Returns:
        The launch description.
    """
    debug = LaunchConfiguration("debug")
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "debug", default_value="False", description="Enable debug mode"
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=str(
                    Path(get_package_share_directory("object_detection_lidar"))
                    / "config"
                    / "ros_params.yaml"
                ),
                description="Path to the ROS parameters file",
            ),
            Node(
                package="object_detection_lidar",
                namespace="",  # Is also the namespace for loading the params
                executable="ros2_example_node",
                name="ros2_example_node",
                parameters=[
                    {"debug": debug},
                    params_file,
                ],
            ),
        ]
    )
