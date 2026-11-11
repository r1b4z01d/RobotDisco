"""robot_state_publisher for Robot Disco.

    ros2 launch rd_description rsp.launch.py [sim_mode:=true] [use_sim_time:=true]

sim_mode:=true swaps the RoboClaw ros2_control hardware for mock hardware (dev stack, Isaac Sim).
"""
import os

from rd_bringup import config

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim_mode = LaunchConfiguration('sim_mode')
    xacro_file = os.path.join(get_package_share_directory('rd_description'), 'urdf', 'robot.urdf.xacro')
    # value_type=str: never YAML-parse the URDF
    robot_description = ParameterValue(Command(['xacro ', xacro_file, ' sim_mode:=', sim_mode, config.ur_kinematics_arg()]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use /clock'),
        DeclareLaunchArgument('sim_mode', default_value='false', description='mock ros2_control hardware'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description, 'use_sim_time': use_sim_time}],
            arguments=['--ros-args', '--log-level', 'WARN'],
        ),
    ])
