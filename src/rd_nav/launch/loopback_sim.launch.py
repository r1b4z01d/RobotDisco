"""The rig's world: Nav2's loopback simulator stands in for the base, the lidars and localization.

    ros2 launch rd_nav loopback_sim.launch.py

It drives the TF tree map->odom->base_footprint from /cmd_vel_nav (TwistStamped), fakes a scan from the
active map on both lidar topics, publishes /diff_cont/odom and /clock, and takes /initialpose. Pair it with
navigation.launch.py use_sim_time:=true (no localization). The map comes from a map_server here because
the simulator and the costmaps both subscribe to /map.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rd_nav.maps_store import MapsStore


def generate_launch_description():
    active = MapsStore().active_yaml()
    quiet = ['--ros-args', '--log-level', 'WARN']
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('rd_description'), 'launch', 'rsp.launch.py')),
        launch_arguments=[('use_sim_time', 'true'), ('sim_mode', 'true')])
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=str(active or '')),
        rsp,   # the robot's frames (lidars, camera) under base_footprint; the simulator drives map->odom->base_footprint
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen',
             parameters=[{'yaml_filename': LaunchConfiguration('map'), 'use_sim_time': True}], arguments=quiet),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_sim_map',
             parameters=[{'use_sim_time': True, 'autostart': True, 'node_names': ['map_server']}], arguments=quiet),
        Node(package='nav2_loopback_sim', executable='loopback_simulator', name='loopback_simulator', output='screen',
             parameters=[{'base_frame_id': 'base_link', 'odom_frame_id': 'odom', 'map_frame_id': 'map',   # odom->base_link like diff_cont; rsp adds base_link->base_footprint
                          'scan_frame_id': 'rp_s2e_lidar', 'enable_stamped_cmd_vel': True,
                          'publish_map_odom_tf': True, 'publish_clock': True, 'scan_use_inf': True,
                          'scan_angle_increment': 0.0174533, 'scan_range_max': 20.0, 'update_duration': 0.02}],
             remappings=[('cmd_vel', 'cmd_vel_nav'), ('odom', '/diff_cont/odom'), ('scan', '/rp_s2e_lidar/scan_filtered')]),
        # the S1 topic too, so every costmap source has data: the same scan, same frame (the costmap places
        # points by the header frame, so relabelling it rp_s1_lidar would shift every wall by the lidar spacing)
        Node(package='rd_nav', executable='scan_relay', name='s1_relay', output='screen',
             parameters=[{'use_sim_time': True, 'in_topic': '/rp_s2e_lidar/scan_filtered',
                          'out_topic': '/rp_s1_lidar/scan_filtered', 'frame_id': ''}]),
    ])
