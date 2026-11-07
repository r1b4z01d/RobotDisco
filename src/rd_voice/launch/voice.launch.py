"""The voice assistant.   ros2 launch rd_voice voice.launch.py [sim_mode:=true]   (sim: no ears, no LLM: rules only)"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    sim = LaunchConfiguration('sim_mode')
    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false'),
        DeclareLaunchArgument('llm_backend', default_value='auto'),
        Node(package='rd_voice', executable='voice', name='voice', output='screen',
             parameters=[{'ears_enabled': PythonExpression(["'", sim, "' != 'true'"]),
                          'llm_backend': PythonExpression(["'none' if '", sim, "' == 'true' else '", LaunchConfiguration('llm_backend'), "'"]),
                          'llm_enabled': PythonExpression(["'", sim, "' != 'true'"])}]),
    ])
