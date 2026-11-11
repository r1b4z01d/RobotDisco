"""The mobile base: robot_state_publisher, ros2_control (RoboClaw diff drive or mock hardware), twist_mux,
the Nav2 collision monitor and the gamepad.

    ros2 launch rd_base base.launch.py                                   # the robot
    ros2 launch rd_base base.launch.py sim_mode:=true joystick:=false    # the rig
    ros2 launch rd_base base.launch.py sim_mode:=true joystick:=false controllers:=false   # rd_sim: Isaac drives the wheels,
                                                                          # only rsp + twist_mux + collision monitor run

Velocity path:  cmd_vel_joy (100) / cmd_vel_web (90) / cmd_vel_nav (10)  -> twist_mux -> cmd_vel_raw
                -> collision_monitor -> /diff_cont/cmd_vel -> diff_cont -> wheels, /odom, odom->base_link
Every cmd_vel topic carries geometry_msgs/TwistStamped (Jazzy's twist_mux speaks nothing else).
Lidar scans for the collision monitor come from rd_perception (/rp_s1_lidar/scan_filtered, /rp_s2e_lidar/scan_filtered).

The URDF is generated here from the xacro (not read back from robot_state_publisher): v1 learned that the
parameter read raced rsp and silently dropped the controller manager (RD, 2026-10-08).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, Shutdown, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from rd_bringup import config


def generate_launch_description():
    share = get_package_share_directory('rd_base')
    desc_share = get_package_share_directory('rd_description')
    sim_mode = LaunchConfiguration('sim_mode')
    use_sim_time = LaunchConfiguration('use_sim_time')
    joystick = LaunchConfiguration('joystick')
    collision_monitor = LaunchConfiguration('collision_monitor')
    controllers = LaunchConfiguration('controllers')
    quiet = ['--ros-args', '--log-level', 'WARN']

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(desc_share, 'launch', 'rsp.launch.py')),
        launch_arguments=[('use_sim_time', use_sim_time), ('sim_mode', sim_mode)])

    xacro_file = os.path.join(desc_share, 'urdf', 'robot.urdf.xacro')
    # the RoboClaw's serial link (header UART ttyTHS1 at 115200 since 2026-11-08) from robot.yaml; ignored in sim_mode
    roboclaw = [' roboclaw_port:=', str(config.get('sensors.roboclaw.port', '/dev/ttyTHS1')),
                ' roboclaw_baud:=', str(config.get('sensors.roboclaw.baud', 115200))]
    robot_description = ParameterValue(Command(['xacro ', xacro_file, ' sim_mode:=', sim_mode] + roboclaw + [config.ur_kinematics_arg()]), value_type=str)

    # No respawn: v1 saw the OnProcessStart spawners re-run on a respawn and take the whole launch down.
    controller_manager = Node(
        package='controller_manager', executable='ros2_control_node',
        parameters=[{'robot_description': robot_description, 'use_sim_time': use_sim_time},
                    os.path.join(share, 'config', 'controllers.yaml')],
        output='screen', arguments=quiet, condition=IfCondition(controllers))
    spawners = [
        Node(package='controller_manager', executable='spawner',
             arguments=[name, '--controller-manager-timeout', '30'] + quiet, output='screen')
        for name in ('joint_broad', 'diff_cont')]

    # ros2_control never reactivates a hardware component that reported an error, and the controller manager stays
    # up without it: this node exits once the wheel hardware has been down for 5 s, that ends the launch, and the
    # unit's autorestart starts a fresh one (rd_base/hardware_watchdog.py)
    watchdog = Node(package='rd_base', executable='hardware_watchdog', name='hardware_watchdog', output='screen',
                    parameters=[{'use_sim_time': use_sim_time}], condition=IfCondition(controllers),
                    on_exit=Shutdown(reason='the wheel hardware went down'))

    twist_mux = Node(
        package='twist_mux', executable='twist_mux',
        parameters=[os.path.join(share, 'config', 'twist_mux.yaml'), {'use_sim_time': use_sim_time}],
        remappings=[('/cmd_vel_out', '/cmd_vel_raw')], arguments=quiet)

    monitor = Node(
        package='nav2_collision_monitor', executable='collision_monitor', name='collision_monitor',
        parameters=[os.path.join(share, 'config', 'collision_monitor.yaml'), {'use_sim_time': use_sim_time}],
        output='screen', condition=IfCondition(collision_monitor))
    monitor_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_collision_monitor',
        parameters=[{'use_sim_time': use_sim_time, 'autostart': True, 'node_names': ['collision_monitor']}],
        output='screen', arguments=quiet, condition=IfCondition(collision_monitor))
    joy_params = os.path.join(share, 'config', 'joystick.yaml')
    joy = Node(
        package='joy', executable='joy_node',
        # joy 3.x (SDL2) segfaults when the pad re-enumerates while running (the 2.4 GHz receiver does that
        # on every sleep/wake): come back by itself.
        respawn=True, respawn_delay=1.0,
        parameters=[joy_params, {'use_sim_time': use_sim_time,
                                 'device_name': config.get('gamepad.device_name', '8BitDo Ultimate 2C Wireless Controller')}],
        arguments=quiet, condition=IfCondition(joystick))
    teleop = Node(
        package='teleop_twist_joy', executable='teleop_node', name='teleop_node',
        # the deadman and turbo buttons have other indices on Bluetooth than on the 2.4 GHz receiver, so they sit
        # in robot.yaml beside the device name they belong to (joystick.yaml holds the receiver's as the default)
        parameters=[joy_params, {'use_sim_time': use_sim_time},
                    {key: int(config.get(f'gamepad.{key}')) for key in ('enable_button', 'enable_turbo_button')
                     if config.get(f'gamepad.{key}', None) is not None}],
        # teleop reads the guarded pad: joy republishes a pad's last state after the pad is gone (rd_base/joy_guard.py)
        remappings=[('/cmd_vel', '/cmd_vel_joy'), ('joy', '/joy_guarded'), ('/joy', '/joy_guarded')], arguments=quiet,
        condition=IfCondition(joystick))
    joy_guard = Node(package='rd_base', executable='joy_guard', name='joy_guard', output='screen',
                     parameters=[{'use_sim_time': use_sim_time}], condition=IfCondition(joystick))

    # the pack's voltage and current from the monitor board over UDP -> /battery, and the low-battery warnings
    battery = Node(package='rd_base', executable='battery_monitor', name='battery_monitor', output='screen',
                   parameters=[{'use_sim_time': use_sim_time}], condition=IfCondition(LaunchConfiguration('battery')))

    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false', description='mock ros2_control hardware'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('joystick', default_value='true', description='joy + teleop_twist_joy'),
        DeclareLaunchArgument('collision_monitor', default_value='true',
                              description='gate cmd_vel through the Nav2 collision monitor (needs the lidar scans)'),
        DeclareLaunchArgument('controllers', default_value='true',
                              description='ros2_control (controller manager + diff_cont): false when a simulator moves the wheels'),
        DeclareLaunchArgument('battery', default_value='true', description='listen for the battery monitor board (UDP) and publish /battery'),
        rsp,
        TimerAction(period=2.0, actions=[controller_manager]),
        RegisterEventHandler(OnProcessStart(target_action=controller_manager, on_start=spawners)),
        twist_mux, watchdog,
        monitor, monitor_lifecycle,
        joy, joy_guard, teleop, battery,
    ])
