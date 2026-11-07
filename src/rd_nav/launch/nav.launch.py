"""Localization + navigation together (what the robot's supervisor unit runs).

    ros2 launch rd_nav nav.launch.py [use_sim_time:=true] [localization:=false]
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
        IncludeLaunchDescription(PythonLaunchDescriptionSource(share('rd_nav', 'launch', 'localization.launch.py')),
                                 launch_arguments=[('use_sim_time', use_sim_time)],
                                 condition=IfCondition(LaunchConfiguration('localization'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(share('rd_nav', 'launch', 'navigation.launch.py')),
                                 launch_arguments=[('use_sim_time', use_sim_time),
                                                   ('tag_corrector', LaunchConfiguration('tag_corrector'))]),
    ])
