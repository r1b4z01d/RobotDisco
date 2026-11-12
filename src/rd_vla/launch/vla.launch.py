"""The VLA stack: the inference server and the RunPolicy runner (both under `python`: the LeRobot venv's on the
robot and the GPU box, so the policy loads and the recorder writes LeRobot datasets), the SpaceMouse teleop and its episode recording.

    ros2 launch rd_vla vla.launch.py                                        # robot: smolvla from the serve link
    ros2 launch rd_vla vla.launch.py backend:=fake                          # rig
    ros2 launch rd_vla vla.launch.py backend:=smolvla ckpt:=tigger_1_merged_smolvla_1 device:=cuda n_action_steps:=8
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
             '--fake-close-after', LaunchConfiguration('fake_close_after'), '--n-action-steps', LaunchConfiguration('n_action_steps')],
        output='screen', condition=IfCondition(LaunchConfiguration('infer_server')))
    # the runner under the same interpreter as the server: with the LeRobot venv its recorder writes LeRobot datasets
    # (under the system python, where lerobot is absent, it falls back to npz episodes: the rig)
    runner = ExecuteProcess(
        cmd=[LaunchConfiguration('python'), '-m', 'rd_vla.policy_runner', '--ros-args', '-r', '__node:=policy_runner',
             '--params-file', os.path.join(share, 'config', 'runner.yaml'),
             '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')], '-p', f'infer_port:={port}', '-p', ['server_unit:=', LaunchConfiguration('server_unit')]],
        output='screen')
    # the teleop under the same interpreter too: it records the demonstrations
    teleop = ExecuteProcess(
        cmd=[LaunchConfiguration('python'), '-m', 'rd_vla.teleop', '--ros-args', '-r', '__node:=vla_teleop',
             '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')], '-p', ['debug_input:=', LaunchConfiguration('teleop_debug_input')]],
        output='screen', condition=IfCondition(LaunchConfiguration('teleop')))
    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='smolvla'),
        DeclareLaunchArgument('ckpt', default_value='', description='run[@step] or a path; empty = the server starts without a model'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('python', default_value='/opt/vla/bin/python', description='the interpreter with lerobot'),
        DeclareLaunchArgument('infer_server', default_value='true'),
        DeclareLaunchArgument('teleop_debug_input', default_value='false', description='the rig: the puck is the topic /teleop/debug_input'),
        DeclareLaunchArgument('teleop', default_value='true', description='the SpaceMouse teleop and recorder (idle until /teleop/enable)'),
        DeclareLaunchArgument('server_unit', default_value='none', description='the supervisor unit that serves the policy, started for a run and stopped when idle (the robot: vla_infer_server); none = leave the server alone'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('fake_close_after', default_value='10'),
        DeclareLaunchArgument('n_action_steps', default_value='0', description='smolvla: execute only the first N actions of each 50-step chunk before re-observing (v1 drove with 4..8); 0 = the whole chunk, open loop'),
        server, runner, teleop,
    ])
