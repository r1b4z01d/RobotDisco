"""Localization + navigation together (what the robot's supervisor unit runs).

    ros2 launch rd_nav nav.launch.py [use_sim_time:=true] [localization:=false]
                                     [set_initial_pose:=true initial_x:=.. initial_y:=.. initial_yaw:=..]   # the sim's spawn pose for AMCL
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from rd_bringup.launch_util import share


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('localization', default_value='true', description='map server + AMCL'),
        DeclareLaunchArgument('tag_corrector', default_value='true'),
        DeclareLaunchArgument('set_initial_pose', default_value='false'),
        DeclareLaunchArgument('initial_x', default_value='0.0'),
        DeclareLaunchArgument('initial_y', default_value='0.0'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(share('rd_nav', 'launch', 'localization.launch.py')),
                                 launch_arguments=[('use_sim_time', use_sim_time), ('set_initial_pose', LaunchConfiguration('set_initial_pose')),
                                                   ('initial_x', LaunchConfiguration('initial_x')), ('initial_y', LaunchConfiguration('initial_y')),
                                                   ('initial_yaw', LaunchConfiguration('initial_yaw'))],
                                 condition=IfCondition(LaunchConfiguration('localization'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(share('rd_nav', 'launch', 'navigation.launch.py')),
                                 launch_arguments=[('use_sim_time', use_sim_time),
                                                   ('tag_corrector', LaunchConfiguration('tag_corrector'))]),
    ])
