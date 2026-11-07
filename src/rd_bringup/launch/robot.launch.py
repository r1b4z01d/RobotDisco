"""Robot Disco, everything on the robot PC in one launch (phase 2 scope: the base).

    ros2 launch rd_bringup robot.launch.py                      # the real robot
    ros2 launch rd_bringup robot.launch.py sim_mode:=true       # the rig: mock base hardware, no devices

Each subsystem is a launch argument (true/false) so a unit can run one of them, and sim_mode flows down
to every include. Later phases add nav, perception, arm, manager, face and voice here.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from rd_bringup.launch_util import share


def _include(package, launch_file, gate, **args):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(share(package, 'launch', launch_file)),
        launch_arguments=list(args.items()),
        condition=IfCondition(LaunchConfiguration(gate)),
    )


def generate_launch_description():
    sim_mode = LaunchConfiguration('sim_mode')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false', description='mock hardware everywhere'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('base', default_value='true', description='rd_base: controllers, twist_mux, collision monitor, joystick'),
        DeclareLaunchArgument('joystick', default_value='true'),
        _include('rd_base', 'base.launch.py', 'base',
                 sim_mode=sim_mode, use_sim_time=use_sim_time, joystick=LaunchConfiguration('joystick')),
    ])
