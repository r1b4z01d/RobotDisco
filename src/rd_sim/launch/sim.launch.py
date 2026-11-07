"""The dev stack against Isaac Sim on the GPU box: Isaac Sim (its own python) with rd_sim's scene, then the
v2 stack in sim_mode with use_sim_time. Isaac Sim itself is launched by hand (or by the `isaac` arg):

    ros2 run rd_sim export_urdf /tmp/robot_disco.urdf
    ~/isaacsim/python.sh $(ros2 pkg prefix rd_sim)/share/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf
    ros2 launch rd_sim sim.launch.py            # nav + arm (mock UR mirrored into the sim) + vla (fake) + manager

Not runnable in the Docker rig (no GPU): phase 7.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rd_bringup.launch_util import share


def inc(pkg, f, **args):
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(share(pkg, 'launch', f)), launch_arguments=list(args.items()))


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('manager', default_value='true'),
        DeclareLaunchArgument('arm', default_value='true'),
        DeclareLaunchArgument('vla', default_value='true'),
        inc('rd_description', 'rsp.launch.py', sim_mode='true', use_sim_time='true'),
        inc('rd_nav', 'nav.launch.py', use_sim_time='true'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(share('rd_arm', 'launch', 'arm.launch.py')),
                                 launch_arguments=[('sim_mode', 'true'), ('use_sim_time', 'true')], condition=IfCondition(LaunchConfiguration('arm'))),
        Node(package='rd_sim', executable='arm_mirror', name='arm_mirror', output='screen', parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('arm'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(share('rd_vla', 'launch', 'vla.launch.py')),
                                 launch_arguments=[('backend', 'fake'), ('python', 'python3'), ('device', 'cpu'), ('use_sim_time', 'true')],
                                 condition=IfCondition(LaunchConfiguration('vla'))),
        Node(package='rd_manager', executable='manager', name='rd_manager', output='screen', arguments=['--sim', 'true'],
             condition=IfCondition(LaunchConfiguration('manager'))),
    ])
