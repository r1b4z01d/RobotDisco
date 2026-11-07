"""move_group for the arm: the whole-robot URDF as the planning model, rd_arm's SRDF, KDL IK, OMPL
(cuMotion on the Orin: isaac.launch adds the pipeline, phase 7-8), executing on the /ur5e controllers.

    ros2 launch rd_arm moveit.launch.py [sim_mode:=true]
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_launch_description():
    share = get_package_share_directory('rd_arm')
    desc = get_package_share_directory('rd_description')
    sim_mode = LaunchConfiguration('sim_mode')
    use_sim_time = LaunchConfiguration('use_sim_time')
    cfg = os.path.join(share, 'config')
    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(desc, 'urdf', 'robot.urdf.xacro'), ' base_control:=false ur_control:=false sim_mode:=', sim_mode]),
        value_type=str)
    with open(os.path.join(cfg, 'robot.srdf')) as f:
        srdf = f.read()
    controllers = _yaml(os.path.join(cfg, 'moveit_controllers.yaml'))
    params = {
        'robot_description': robot_description,
        'robot_description_semantic': srdf,
        'robot_description_kinematics': _yaml(os.path.join(cfg, 'kinematics.yaml')),
        'robot_description_planning': _yaml(os.path.join(cfg, 'joint_limits.yaml')),
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': _yaml(os.path.join(cfg, 'ompl_planning.yaml')),
        'planning_scene_monitor': {'publish_planning_scene': True, 'publish_geometry_updates': True,
                                   'publish_state_updates': True, 'publish_transforms_updates': True},
        'publish_robot_description_semantic': True,
        'use_sim_time': use_sim_time,
    }
    params.update(controllers)
    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='moveit_ros_move_group', executable='move_group', output='screen', parameters=[params],
             arguments=['--ros-args', '--log-level', 'WARN']),
    ])
