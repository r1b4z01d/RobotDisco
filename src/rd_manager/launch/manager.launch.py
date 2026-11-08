"""The web manager.   ros2 launch rd_manager manager.launch.py [port:=8092] [sim_mode:=true]"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rd_bringup import config


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value=str(config.get('robot.ports.manager', 8092))),
        DeclareLaunchArgument('sim_mode', default_value='false'),
        Node(package='rd_manager', executable='manager', output='screen',
             arguments=['--port', LaunchConfiguration('port'), '--sim', LaunchConfiguration('sim_mode')]),
    ])
