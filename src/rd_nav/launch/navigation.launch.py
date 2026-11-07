"""Nav2 (planner, MPPI controller, behaviours, bt_navigator, waypoint follower, velocity smoother, keepout
filter servers) plus rd_nav's own servers: GoToLandmark and the AprilTag pose corrector.

    ros2 launch rd_nav navigation.launch.py [keepout_mask:=...] [use_sim_time:=true] [tag_corrector:=false]

Output: /cmd_vel_nav (TwistStamped) into rd_base's twist_mux. Needs TF map->odom->base_footprint from
localization.launch.py (or a simulator) and the two lidar scans.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

from rd_nav.maps_store import MapsStore


def generate_launch_description():
    share = get_package_share_directory('rd_nav')
    maps_dir = MapsStore().dir
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params = RewrittenYaml(
        source_file=os.path.join(share, 'config', 'nav2.yaml'), root_key='',
        param_rewrites={'use_sim_time': use_sim_time, 'autostart': autostart}, convert_types=True)
    # nav2.yaml carries no use_sim_time keys, so the rewrite above sets none: every node gets it explicitly (under
    # Isaac's clock a wall-time node stamps goals the others cannot transform; the loopback sim's clock hid this)
    sim_time = {'use_sim_time': use_sim_time}
    quiet = ['--ros-args', '--log-level', 'WARN']
    lifecycle_nodes = ['filter_mask_server', 'costmap_filter_info_server', 'controller_server',
                       'smoother_server', 'planner_server', 'behavior_server', 'bt_navigator',
                       'waypoint_follower', 'velocity_smoother']

    def nav2(package, executable, name=None, remappings=(), extra_params=()):
        return Node(package=package, executable=executable, name=name or executable, output='screen',
                    respawn=True, respawn_delay=2.0, parameters=[params, sim_time, *extra_params],
                    arguments=quiet, remappings=list(remappings))

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('keepout_mask', default_value=str(maps_dir / 'keepout_mask.yaml'),
                              description='keepout filter mask (rd_keepout builds it from keepout_zones.yaml)'),
        DeclareLaunchArgument('tag_corrector', default_value='true'),
        DeclareLaunchArgument('tag_poses_file', default_value=os.path.join(share, 'config', 'tag_poses.yaml')),
        # keepout servers first: the costmaps' keepout_filter waits for /costmap_filter_info + /filter_mask
        nav2('nav2_map_server', 'map_server', 'filter_mask_server',
             extra_params=[{'yaml_filename': LaunchConfiguration('keepout_mask')}]),
        nav2('nav2_map_server', 'costmap_filter_info_server'),
        nav2('nav2_controller', 'controller_server', remappings=[('cmd_vel', 'cmd_vel_nav_raw')]),
        nav2('nav2_smoother', 'smoother_server'),
        nav2('nav2_planner', 'planner_server'),
        nav2('nav2_behaviors', 'behavior_server', remappings=[('cmd_vel', 'cmd_vel_nav')]),
        nav2('nav2_bt_navigator', 'bt_navigator'),
        nav2('nav2_waypoint_follower', 'waypoint_follower'),
        nav2('nav2_velocity_smoother', 'velocity_smoother',
             remappings=[('cmd_vel', 'cmd_vel_nav_raw'), ('cmd_vel_smoothed', 'cmd_vel_nav')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation',
             output='screen', arguments=quiet,
             parameters=[{'use_sim_time': use_sim_time, 'autostart': autostart, 'node_names': lifecycle_nodes}]),
        Node(package='rd_nav', executable='go_to_landmark', name='go_to_landmark', output='screen',
             parameters=[{'use_sim_time': use_sim_time}]),
        Node(package='rd_nav', executable='tag_corrector', name='tag_corrector', output='screen',
             parameters=[os.path.join(share, 'config', 'tag_corrector.yaml'),
                         {'use_sim_time': use_sim_time, 'tag_poses_file': LaunchConfiguration('tag_poses_file')}],
             condition=IfCondition(LaunchConfiguration('tag_corrector'))),
    ])
