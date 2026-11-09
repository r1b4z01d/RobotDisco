"""Isaac ROS 4.6 on the robot (NOT runnable in the rig; validated on the GPU box in phase 7, on the Orin in
phase 8). Everything is a launch argument that defaults off, so the robot drives on lidars + AMCL alone and
each GPU layer is added when it has been proven:

    ros2 launch rd_perception isaac.launch.py apriltag:=true grid_localizer:=true nvblox:=true visual_slam:=false

  apriltag        isaac_ros_apriltag on the D435 colour stream -> /tag_detections -> tag_adapter -> /tags/detections
  grid_localizer  isaac_ros_occupancy_grid_localizer: a one-shot global pose from the lidar on request
                  (/trigger_grid_search_localization), seeds AMCL through /initialpose
  nvblox          3D reconstruction from the D435 depth for the local costmap (nvblox_nav2 plugin in nav2.yaml on the robot)
  visual_slam     cuVSLAM on the stereo IR pair, fused with wheel odometry by robot_localization (phase 9)

The package / node / parameter names follow the Isaac ROS 4.6 docs (release-4.6 quickstarts); the first run on
the GPU box is expected to need adjustment, which is why nothing else depends on this file.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('rd_perception')
    apriltag = LaunchConfiguration('apriltag')
    grid = LaunchConfiguration('grid_localizer')
    nvblox = LaunchConfiguration('nvblox')
    vslam = LaunchConfiguration('visual_slam')
    # the localizer reads the nav map's own keys (image, resolution, origin, thresholds) as ROS parameters, like
    # nav2's map_server, and joins `image` to the yaml's directory: with only map_yaml_path set it opens the bare directory
    map_yaml = os.path.join(get_package_share_directory('rd_nav'), 'maps', 'zone_one.yaml')
    with open(map_yaml) as f:
        map_params = yaml.safe_load(f)

    container = ComposableNodeContainer(
        name='isaac_container', namespace='', package='rclcpp_components', executable='component_container_mt',
        output='screen',
        composable_node_descriptions=[
            # GPU image pipeline for the tag detector: rectify the D435 colour image once
            ComposableNode(package='isaac_ros_image_proc', plugin='nvidia::isaac_ros::image_proc::RectifyNode',
                           name='rectify_color', parameters=[{'output_width': 1280, 'output_height': 720}],
                           remappings=[('image_raw', '/camera/camera/color/image_raw'),
                                       ('camera_info', '/camera/camera/color/camera_info'),
                                       ('image_rect', '/camera/color/image_rect'),
                                       ('camera_info_rect', '/camera/color/camera_info_rect')],
                           condition=IfCondition(apriltag)),
            ComposableNode(package='isaac_ros_apriltag', plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
                           name='apriltag', parameters=[{'size': 0.15, 'max_tags': 16, 'tile_size': 4,
                                                         'backends': 'CUDA', 'tag_family': 'tag36h11'}],
                           remappings=[('image', '/camera/color/image_rect'), ('camera_info', '/camera/color/camera_info_rect'),
                                       ('tag_detections', '/tag_detections')],
                           condition=IfCondition(apriltag)),
            # the localizer takes isaac_ros_pointcloud_interfaces/FlatScan, not LaserScan: convert the lidar scan first
            ComposableNode(package='isaac_ros_pointcloud_utils',
                           plugin='nvidia::isaac_ros::pointcloud_utils::LaserScantoFlatScanNode', name='laserscan_to_flatscan',
                           remappings=[('scan', '/rp_s2e_lidar/scan_filtered')],
                           condition=IfCondition(grid)),
            ComposableNode(package='isaac_ros_occupancy_grid_localizer',
                           plugin='nvidia::isaac_ros::occupancy_grid_localizer::OccupancyGridLocalizerNode',
                           name='occupancy_grid_localizer',
                           parameters=[{**map_params, 'loc_result_frame': 'map', 'map_yaml_path': map_yaml,
                                        'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}],
                           remappings=[('localization_result', '/initialpose')],   # NOT flatscan_localization: every message on that topic triggers a localization
                           condition=IfCondition(grid)),
            ComposableNode(package='isaac_ros_visual_slam', plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
                           name='visual_slam',
                           parameters=[{'enable_rectified_pose': True, 'rectified_images': True, 'enable_imu_fusion': False,
                                        'num_cameras': 2, 'base_frame': 'base_link', 'enable_localization_n_mapping': False,
                                        'publish_map_to_odom_tf': False, 'publish_odom_to_base_tf': False}],
                           remappings=[('visual_slam/image_0', '/camera/camera/infra1/image_rect_raw'),
                                       ('visual_slam/camera_info_0', '/camera/camera/infra1/camera_info'),
                                       ('visual_slam/image_1', '/camera/camera/infra2/image_rect_raw'),
                                       ('visual_slam/camera_info_1', '/camera/camera/infra2/camera_info')],
                           condition=IfCondition(vslam)),
        ])
    nvblox_node = Node(package='nvblox_ros', executable='nvblox_node', name='nvblox_node', output='screen',
                       parameters=[os.path.join(share, 'config', 'nvblox.yaml')],
                       remappings=[('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
                                   ('depth/camera_info', '/camera/camera/aligned_depth_to_color/camera_info'),
                                   ('color/image', '/camera/camera/color/image_raw'),
                                   ('color/camera_info', '/camera/camera/color/camera_info')],
                       condition=IfCondition(nvblox))
    adapter = Node(package='rd_perception', executable='tag_adapter', name='tag_adapter', output='screen',
                   parameters=[{'source': 'isaac', 'in_topic': '/tag_detections', 'out_topic': '/tags/detections'}],
                   condition=IfCondition(apriltag))
    return LaunchDescription([
        DeclareLaunchArgument('apriltag', default_value='false'),
        DeclareLaunchArgument('grid_localizer', default_value='false'),
        DeclareLaunchArgument('nvblox', default_value='false'),
        DeclareLaunchArgument('visual_slam', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        container, nvblox_node, adapter,
    ])
