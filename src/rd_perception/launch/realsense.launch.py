"""RealSense D435: colour 1280x720 at 15 fps (v1's choice for the face and the tags), depth + pointcloud for
the local costmap / nvblox, the stereo IR pair for visual SLAM when enabled.

    ros2 launch rd_perception realsense.launch.py [infra:=true] [pointcloud:=true]

Topics (realsense2_camera, namespace camera): /camera/camera/color/image_raw, .../color/camera_info,
/camera/camera/depth/image_rect_raw, /camera/camera/depth/color/points
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from rd_bringup import config


def generate_launch_description():
    rs = os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
    serial = str(config.get('sensors.realsense_d435.serial', '') or '')
    return LaunchDescription([
        DeclareLaunchArgument('infra', default_value='false', description='stereo IR pair (visual SLAM)'),
        DeclareLaunchArgument('pointcloud', default_value='true'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(rs), launch_arguments=[
            ('camera_name', 'camera'), ('camera_namespace', 'camera'),
            ('serial_no', f"'{serial}'"), ('device_type', 'd435'),
            ('rgb_camera.color_profile', '1280,720,15'),
            ('depth_module.depth_profile', '848,480,15'),
            ('depth_module.infra_profile', '848,480,15'),
            ('enable_infra1', LaunchConfiguration('infra')), ('enable_infra2', LaunchConfiguration('infra')),
            ('pointcloud.enable', LaunchConfiguration('pointcloud')),
            ('align_depth.enable', 'true'),
            ('publish_tf', 'false'),     # the URDF carries the camera frames (realsense_d435.xacro)
            ('log_level', 'warn'),
        ]),
    ])
