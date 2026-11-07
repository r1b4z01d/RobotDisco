"""Map server + AMCL for the active map (maps/active_map, written by the manager's Maps tab).

    ros2 launch rd_nav localization.launch.py [map:=/path/to/map.yaml] [use_sim_time:=true]

On the robot the Isaac occupancy-grid localizer (rd_perception, grid_localizer:=true there) gives AMCL its
first pose; the tag corrector and the manager's pose tool are the other two sources of /initialpose.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
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
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        DeclareLaunchArgument('map', default_value=str(active or ''),
                              description='map yaml (default: the active map in the maps folder)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen',
             parameters=[params], arguments=quiet),
        Node(package='nav2_amcl', executable='amcl', name='amcl', output='screen',
             parameters=[params], arguments=quiet),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_localization',
             output='screen', arguments=quiet,
             parameters=[{'use_sim_time': use_sim_time, 'autostart': LaunchConfiguration('autostart'),
                          'node_names': ['map_server', 'amcl']}]),
    ])
