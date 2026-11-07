"""Map server + AMCL for the active map (maps/active_map, written by the manager's Maps tab).

    ros2 launch rd_nav localization.launch.py [map:=/path/to/map.yaml] [use_sim_time:=true]
                                               [set_initial_pose:=true initial_x:=0.7 initial_y:=0.9 initial_yaw:=0.0]

On the robot the Isaac occupancy-grid localizer (rd_perception, grid_localizer:=true there) gives AMCL its
first pose; the tag corrector and the manager's pose tool are the other two sources of /initialpose. The sim
(rd_sim) passes the spawn pose as AMCL's initial pose instead: Nav2's costmaps give map->base_footprint 60 s to
appear at bringup, and nothing else sets it there.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml

from rd_nav.maps_store import MapsStore


def generate_launch_description():
    share = get_package_share_directory('rd_nav')
    store = MapsStore()
    active = store.active_yaml()
    use_sim_time = LaunchConfiguration('use_sim_time')
    params = RewrittenYaml(
        source_file=os.path.join(share, 'config', 'nav2.yaml'), root_key='',
        param_rewrites={'use_sim_time': use_sim_time, 'yaml_filename': LaunchConfiguration('map')},
        convert_types=True)
    quiet = ['--ros-args', '--log-level', 'WARN']
    initial = {'set_initial_pose': ParameterValue(LaunchConfiguration('set_initial_pose'), value_type=bool),
               'initial_pose.x': ParameterValue(LaunchConfiguration('initial_x'), value_type=float),
               'initial_pose.y': ParameterValue(LaunchConfiguration('initial_y'), value_type=float),
               'initial_pose.z': 0.0,
               'initial_pose.yaw': ParameterValue(LaunchConfiguration('initial_yaw'), value_type=float)}
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        DeclareLaunchArgument('map', default_value=str(active or ''),
                              description='map yaml (default: the active map in the maps folder)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('set_initial_pose', default_value='false', description='AMCL starts from initial_x/y/yaw (the sim)'),
        DeclareLaunchArgument('initial_x', default_value='0.0'),
        DeclareLaunchArgument('initial_y', default_value='0.0'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0'),
        # nav2.yaml has no use_sim_time keys for the rewrite to hit: passed explicitly (see navigation.launch.py)
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen',
             parameters=[params, {'use_sim_time': use_sim_time}], arguments=quiet),
        Node(package='nav2_amcl', executable='amcl', name='amcl', output='screen',
             parameters=[params, {'use_sim_time': use_sim_time}, initial], arguments=quiet),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_localization',
             output='screen', arguments=quiet,
             parameters=[{'use_sim_time': use_sim_time, 'autostart': LaunchConfiguration('autostart'),
                          'node_names': ['map_server', 'amcl']}]),
    ])
