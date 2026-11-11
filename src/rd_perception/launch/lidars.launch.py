"""The three lidars: RPLiDAR S1 (front, serial, with the range correction), RPLiDAR S2E (back, Ethernet),
Velodyne VLP-16; each 2D scan through its body filter chain.

    ros2 launch rd_perception lidars.launch.py [s1:=true] [s2e:=true] [velodyne:=true] [sim:=false]

sim:=true (rd_sim) skips the drivers and the S1 range correction: Isaac Sim publishes the raw scans and the
cloud itself, and the body filter chains run on them exactly as on the robot.

Topics: /rp_s1_lidar/scan_uncorrected -> scan_range_offset -> /rp_s1_lidar/scan -> filter -> /rp_s1_lidar/scan_filtered
        /rp_s2e_lidar/scan -> filter -> /rp_s2e_lidar/scan_filtered ;  /velodyne_points, /velodyne/scan
Addresses, ports and the S1 offset come from robot.yaml.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from rd_bringup import config


def generate_launch_description():
    share = get_package_share_directory('rd_perception')
    quiet = ['--ros-args', '--log-level', 'WARN']
    s1, s2e, velo, sim = LaunchConfiguration('s1'), LaunchConfiguration('s2e'), LaunchConfiguration('velodyne'), LaunchConfiguration('sim')
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    real = lambda on: IfCondition(PythonExpression(["'", on, "' == 'true' and '", sim, "' != 'true'"]))   # a driver: that lidar on, not the sim
    s1_cfg = config.get('sensors.rplidar_s1')
    s2e_cfg = config.get('sensors.rplidar_s2e')
    velo_cfg = config.get('sensors.velodyne')
    # the S1's range error comes and goes, so scan_range_offset measures it against the robot's own body when
    # robot.yaml names where the S1 sees it; without that it adds the fixed range_offset_m
    s1_offset = {'in_topic': 'rp_s1_lidar/scan_uncorrected', 'out_topic': 'rp_s1_lidar/scan',
                 'offset': float(s1_cfg.get('range_offset_m', 0.0))}
    body = s1_cfg.get('body_reference') or {}
    if body:
        s1_offset['reference_sectors'] = [float(edge) for sector in body['sectors_deg'] for edge in sector]
        s1_offset['reference_ranges'] = [float(r) for r in body['ranges_m']]

    s1_nodes = [
        Node(package='rplidar_ros', executable='rplidar_node', name='rplidar_s1_node', output='screen',
             respawn=True, respawn_delay=3.0, condition=real(s1),   # the CP2102 re-enumerates on a USB hiccup
             parameters=[{'channel_type': 'serial', 'serial_port': str(s1_cfg['serial_port']),
                          'serial_baudrate': int(s1_cfg.get('baudrate', 256000)), 'frame_id': s1_cfg.get('frame', 'rp_s1_lidar'),
                          'inverted': True, 'angle_compensate': True, 'scan_mode': str(s1_cfg.get('scan_mode', '')),
                          'topic_name': 'rp_s1_lidar/scan_uncorrected'}]),
        Node(package='rd_perception', executable='scan_range_offset', name='rplidar_s1_range_offset', output='screen',
             respawn=True, respawn_delay=3.0, condition=real(s1),
             parameters=[s1_offset]),
        Node(package='laser_filters', executable='scan_to_scan_filter_chain', name='rp_s1_filter', condition=IfCondition(s1),
             remappings=[('/scan', '/rp_s1_lidar/scan'), ('/scan_filtered', '/rp_s1_lidar/scan_filtered')],
             parameters=[os.path.join(share, 'config', 's1_filter.yaml'), sim_time], arguments=quiet),
    ]
    s2e_nodes = [
        Node(package='rplidar_ros', executable='rplidar_node', name='rplidar_s2e_node', output='screen',
             respawn=True, respawn_delay=3.0, condition=real(s2e),
             parameters=[{'channel_type': 'udp', 'udp_ip': str(s2e_cfg['ip']), 'udp_port': int(s2e_cfg.get('udp_port', 8089)),
                          'frame_id': s2e_cfg.get('frame', 'rp_s2e_lidar'), 'inverted': True, 'angle_compensate': True,
                          'scan_mode': str(s2e_cfg.get('scan_mode', 'DenseBoost')), 'scan_frequency': 10.0,
                          'topic_name': 'rp_s2e_lidar/scan'}]),
        Node(package='laser_filters', executable='scan_to_scan_filter_chain', name='rp_s2e_filter', condition=IfCondition(s2e),
             remappings=[('/scan', '/rp_s2e_lidar/scan'), ('/scan_filtered', '/rp_s2e_lidar/scan_filtered')],
             parameters=[os.path.join(share, 'config', 's2e_filter.yaml'), sim_time], arguments=quiet),
    ]
    convert_share = get_package_share_directory('velodyne_pointcloud')
    with open(os.path.join(convert_share, 'config', 'VLP16-velodyne_transform_node-params.yaml')) as f:
        convert_params = yaml.safe_load(f)['velodyne_transform_node']['ros__parameters']
    convert_params['calibration'] = os.path.join(convert_share, 'params', 'VLP16db.yaml')
    laserscan_share = get_package_share_directory('velodyne_laserscan')
    velo_nodes = [
        Node(package='velodyne_driver', executable='velodyne_driver_node', output='both', condition=real(velo),
             respawn=True, respawn_delay=3.0, arguments=quiet,
             parameters=[os.path.join(share, 'config', 'velodyne.yaml'),
                         {'device_ip': str(velo_cfg['ip']), 'port': int(velo_cfg.get('port', 2368))}]),
        Node(package='velodyne_pointcloud', executable='velodyne_transform_node', output='both', condition=real(velo),
             parameters=[convert_params], arguments=quiet),
        Node(package='velodyne_laserscan', executable='velodyne_laserscan_node', output='both', condition=real(velo),   # needs the driver's 'ring' field
             parameters=[os.path.join(laserscan_share, 'config', 'default-velodyne_laserscan_node-params.yaml')],
             remappings=[('scan', 'velodyne/scan')], arguments=quiet),
    ]
    return LaunchDescription([
        DeclareLaunchArgument('s1', default_value='true'),
        DeclareLaunchArgument('s2e', default_value='true'),
        DeclareLaunchArgument('velodyne', default_value='true'),
        DeclareLaunchArgument('sim', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        *s1_nodes, *s2e_nodes, *velo_nodes,
    ])
