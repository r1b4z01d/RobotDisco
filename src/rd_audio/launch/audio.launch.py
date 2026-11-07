"""The Say server.   ros2 launch rd_audio audio.launch.py [sim_mode:=true]   (sim: fake voice, no player)"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    sim = LaunchConfiguration('sim_mode')
    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false'),
        DeclareLaunchArgument('engine', default_value='auto'),
        Node(package='rd_audio', executable='say_server', name='say_server', output='screen',
             parameters=[{'engine': PythonExpression(["'fake' if '", sim, "' == 'true' else '", LaunchConfiguration('engine'), "'"]),
                          'player': PythonExpression(["'none' if '", sim, "' == 'true' else 'aplay'"]),
                          'sink_watchdog': PythonExpression(["'", sim, "' != 'true'"])}]),
    ])
