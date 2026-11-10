"""The face.   ros2 launch rd_face face.launch.py [sim_mode:=true]   (sim: headless renderer on /face/image,
no face id, no WLED panels, the fake scene camera as the image)"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('rd_face')
    sim = LaunchConfiguration('sim_mode')
    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false'),
        DeclareLaunchArgument('image_topic', default_value=PythonExpression(["'/cameras/scene/image' if '", sim, "' == 'true' else '/camera/camera/color/image_raw'"])),
        Node(package='rd_face', executable='face', name='face', output='screen',
             parameters=[os.path.join(share, 'config', 'face.yaml'),
                         {'image_topic': LaunchConfiguration('image_topic'),
                          'renderer': PythonExpression(["'headless' if '", sim, "' == 'true' else 'gl'"]),
                          'face_id_enabled': PythonExpression(["'", sim, "' != 'true'"]),
                          'wled_enabled': PythonExpression(["'", sim, "' != 'true'"]),      # no panels in the rig
                          'fullscreen': PythonExpression(["'", sim, "' != 'true'"])}]),
    ])
