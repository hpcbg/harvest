"""Launch the HARVEST fleet bridge (and optionally the Isaac bridge).

    ros2 launch harvest_ros bridge.launch.py harvest_url:=http://harvest:8765
    ros2 launch harvest_ros bridge.launch.py isaac_bridge:=true \
        harvest_url:=http://127.0.0.1:8765     # isaac / isaac-demo modes
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "harvest_url", default_value="http://127.0.0.1:8765",
            description="Base URL of the HARVEST fleet API"),
        DeclareLaunchArgument(
            "poll_period_s", default_value="1.0",
            description="Snapshot poll period in seconds"),
        DeclareLaunchArgument(
            "isaac_bridge", default_value="false",
            description="Also start the Isaac Sim bridge node"),
        Node(
            package="harvest_ros",
            executable="isaac_bridge",
            name="harvest_isaac_bridge",
            output="screen",
            condition=IfCondition(LaunchConfiguration("isaac_bridge")),
            parameters=[{
                "harvest_url": LaunchConfiguration("harvest_url"),
            }],
        ),
        Node(
            package="harvest_ros",
            executable="fleet_bridge",
            name="harvest_fleet_bridge",
            output="screen",
            parameters=[{
                "harvest_url": LaunchConfiguration("harvest_url"),
                # LaunchConfigurations substitute to strings; force the type.
                "poll_period_s": ParameterValue(
                    LaunchConfiguration("poll_period_s"), value_type=float),
            }],
        ),
    ])
