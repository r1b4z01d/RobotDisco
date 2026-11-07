"""The three USB webcams (scene, wrist, side) as ROS 2 image topics via usb_cam, addressed by USB port
(robot.yaml sensors.usb_cameras). Replaces v1's multicast camera service: one publisher per camera,
every consumer (recorder, manager pages, face) subscribes.

    ros2 launch rd_perception cameras.launch.py [cameras:=scene,wrist,side]

Topics: /cameras/<name>/image_raw (rgb8, rotated upright per the mount), /cameras/<name>/image_raw/compressed,
        /cameras/<name>/camera_info
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rd_bringup import config


def _nodes(context):
    wanted = [c.strip() for c in LaunchConfiguration('cameras').perform(context).split(',') if c.strip()]
    cams = config.get('sensors.usb_cameras')
    by_path = str(cams['by_path'])
    out = []
    for name in wanted:
        c = cams[name]
        out.append(Node(
            package='usb_cam', executable='usb_cam_node_exe', name=f'{name}_camera', namespace=f'cameras/{name}',
            output='screen', respawn=True, respawn_delay=3.0,
            parameters=[{'video_device': by_path.format(port=c['port']),
                         'image_width': int(cams.get('width', 1280)), 'image_height': int(cams.get('height', 720)),
                         'framerate': float(cams.get('fps', 30)), 'pixel_format': 'mjpeg2rgb',
                         'camera_name': name, 'frame_id': f'{name}_camera', 'io_method': 'mmap'}],
            arguments=['--ros-args', '--log-level', 'WARN']))
        rot = int(c.get('rotate', 0))
        if rot:
            out.append(Node(package='rd_perception', executable='image_rotate', name=f'{name}_rotate',
                            namespace=f'cameras/{name}', parameters=[{'degrees': rot}],
                            remappings=[('in', 'image_raw'), ('out', 'image')]))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('cameras', default_value='scene,wrist,side'),
        OpaqueFunction(function=_nodes),
    ])
