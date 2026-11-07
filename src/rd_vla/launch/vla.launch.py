"""The VLA stack: the inference server (its own process; the LeRobot venv's python on the robot), the
RunPolicy runner, the gamepad teleop.

    ros2 launch rd_vla vla.launch.py                                        # robot: smolvla from the serve link
    ros2 launch rd_vla vla.launch.py backend:=fake                          # rig
    ros2 launch rd_vla vla.launch.py backend:=smolvla ckpt:=tigger_1_merged_smolvla_1 device:=cuda
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rd_bringup import config


def generate_launch_description():
    share = get_package_share_directory('rd_vla')
    backend = LaunchConfiguration('backend')
    port = str(config.get('robot.ports.infer_server', 8777))
    server = ExecuteProcess(
        cmd=[LaunchConfiguration('python'), '-m', 'rd_vla.infer_server', '--backend', backend, '--port', port,
             '--device', LaunchConfiguration('device'), '--ckpt', LaunchConfiguration('ckpt'),
             '--fake-close-after', LaunchConfiguration('fake_close_after')],
        output='screen', condition=IfCondition(LaunchConfiguration('infer_server')))
    runner = Node(package='rd_vla', executable='policy_runner', name='policy_runner', output='screen',
                  parameters=[os.path.join(share, 'config', 'runner.yaml'),
                              {'use_sim_time': LaunchConfiguration('use_sim_time'), 'infer_port': int(port)}])
    teleop = Node(package='rd_vla', executable='teleop', name='vla_teleop', output='screen',
                  parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
                  condition=IfCondition(LaunchConfiguration('teleop')))
    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='smolvla'),
        DeclareLaunchArgument('ckpt', default_value='', description='run[@step] or a path; empty = the server starts without a model'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('python', default_value='/opt/vla/bin/python', description='the interpreter with lerobot'),
        DeclareLaunchArgument('infer_server', default_value='true'),
        DeclareLaunchArgument('teleop', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('fake_close_after', default_value='10'),
        server, runner, teleop,
    ])
