"""move_group for the arm: the whole-robot URDF as the planning model, rd_arm's SRDF, KDL IK, and the planners:
OMPL always (the rig, the fallback), cuMotion when asked (the GPU box and the Orin, Isaac ROS 4.6 debs).

    ros2 launch rd_arm moveit.launch.py [sim_mode:=true] [cumotion:=true]

cumotion:=true makes isaac_ros_cumotion move_group's default pipeline (OMPL stays second) and starts the cuMotion
container next to it: nvidia::isaac_ros::cumotion::CumotionPlanner, fed the same whole-robot URDF written out flat
plus config/ur5e_hand.xrdf (the arm's c-space, limits and collision spheres; the body is the keepout box the Wave
server puts in the planning scene). No ESDF: nvblox is not wired in yet (read_esdf_world false).
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def _yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _setup(context):
    share = get_package_share_directory('rd_arm')
    desc = get_package_share_directory('rd_description')
    sim_mode = LaunchConfiguration('sim_mode')
    use_sim_time = LaunchConfiguration('use_sim_time')
    cumotion = LaunchConfiguration('cumotion').perform(context).lower() in ('1', 'true', 'yes')
    cfg = os.path.join(share, 'config')
    xacro_args = ['xacro ', os.path.join(desc, 'urdf', 'robot.urdf.xacro'), ' base_control:=false ur_control:=false sim_mode:=', sim_mode]
    robot_description = ParameterValue(Command(xacro_args), value_type=str)
    with open(os.path.join(cfg, 'robot.srdf')) as f:
        srdf = f.read()
    controllers = _yaml(os.path.join(cfg, 'moveit_controllers.yaml'))
    pipelines = ['isaac_ros_cumotion', 'ompl'] if cumotion else ['ompl']
    params = {
        'robot_description': robot_description,
        'robot_description_semantic': srdf,
        'robot_description_kinematics': _yaml(os.path.join(cfg, 'kinematics.yaml')),
        'robot_description_planning': _yaml(os.path.join(cfg, 'joint_limits.yaml')),
        'planning_pipelines': pipelines,
        'default_planning_pipeline': pipelines[0],
        'ompl': _yaml(os.path.join(cfg, 'ompl_planning.yaml')),
        'planning_scene_monitor': {'publish_planning_scene': True, 'publish_geometry_updates': True,
                                   'publish_state_updates': True, 'publish_transforms_updates': True},
        'publish_robot_description_semantic': True,
        'use_sim_time': use_sim_time,
    }
    params.update(controllers)
    nodes = []
    if cumotion:
        cumotion_share = get_package_share_directory('isaac_ros_cumotion_moveit')
        params['isaac_ros_cumotion'] = _yaml(os.path.join(cumotion_share, 'config', 'isaac_ros_cumotion_planning.yaml'))
        # cuMotion reads the robot from files: the same xacro, written out flat next to the temp dir
        urdf_path = os.path.join(tempfile.gettempdir(), 'rd_arm_cumotion.urdf')
        with open(urdf_path, 'w') as f:
            f.write(Command(xacro_args).perform(context))
        nodes.append(ComposableNodeContainer(
            name='cumotion_container', namespace='', package='rclcpp_components', executable='component_container_mt',
            output='screen', arguments=['--ros-args', '--log-level', 'WARN'],
            composable_node_descriptions=[
                ComposableNode(package='isaac_ros_cumotion', plugin='nvidia::isaac_ros::cumotion::StaticPlanningSceneServer',
                               name='static_planning_scene_server', parameters=[{'moveit_collision_objects_scene_file': ''}]),
                ComposableNode(package='isaac_ros_cumotion', plugin='nvidia::isaac_ros::cumotion::CumotionPlanner', name='cumotion_planner',
                               parameters=[{'urdf_file_path': urdf_path, 'xrdf_file_path': os.path.join(cfg, 'ur5e_hand.xrdf'),
                                            'read_esdf_world': False, 'update_esdf_on_request': False,
                                            'joint_states_topic': '/joint_states', 'time_dilation_factor': 0.5, 'interpolation_dt': 0.05,
                                            'publish_world_collision_spheres': True, 'publish_self_collision_spheres': False,
                                            'add_ground_plane': False, 'override_moveit_scaling_factors': False,
                                            'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}]),
            ]))
    nodes.append(Node(package='moveit_ros_move_group', executable='move_group', output='screen', parameters=[params],
                      arguments=['--ros-args', '--log-level', 'WARN']))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('cumotion', default_value='false', description='plan with Isaac ROS cuMotion (default pipeline), OMPL as the fallback'),
        OpaqueFunction(function=_setup),
    ])
