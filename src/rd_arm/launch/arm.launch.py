"""The arm and the hand: the UR driver's controller manager under /ur5e (always on; mock hardware in
sim_mode), its helper nodes, MoveIt, and rd_arm's servers (ArmPower, Wave, the streamer, the hand node).

    ros2 launch rd_arm arm.launch.py                       # robot
    ros2 launch rd_arm arm.launch.py sim_mode:=true        # rig: mock arm, fake dashboard, fake hand

In sim_mode the fake dashboard (:29999) and the fake hand (udp :8766) run on loopback and ArmPower / the
hand node point at them.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from rd_bringup import config


def generate_launch_description():
    share = get_package_share_directory('rd_arm')
    desc = get_package_share_directory('rd_description')
    sim_mode = LaunchConfiguration('sim_mode')
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_ip = LaunchConfiguration('robot_ip')
    quiet = ['--ros-args', '--log-level', 'WARN']
    sim = IfCondition(sim_mode)
    real = UnlessCondition(sim_mode)

    robot_description = ParameterValue(Command([
        'xacro ', os.path.join(desc, 'urdf', 'robot.urdf.xacro'),
        ' base_control:=false ur_control:=true sim_mode:=', sim_mode, ' robot_ip:=', robot_ip,
        ' tool_voltage:=', str(config.get('arm.tool_voltage', 24)),
        ' initial_positions_file:=', os.path.join(share, 'config', 'initial_positions.yaml')]), value_type=str)

    # Jazzy's controller_manager reads the URDF from ~/robot_description (topic), so the arm-control variant
    # of the description is latched on /ur5e/robot_description for it (rsp publishes the whole-robot one on /).
    description = Node(package='rd_arm', executable='description_publisher', name='description_publisher', namespace='ur5e',
                       parameters=[{'robot_description': robot_description}], output='screen')
    cm = Node(package='controller_manager', executable='ros2_control_node', namespace='ur5e', output='screen',
              parameters=[{'use_sim_time': use_sim_time}, os.path.join(share, 'config', 'ur_controllers.yaml')],
              remappings=[('/ur5e/joint_states', '/joint_states')], arguments=quiet)

    def spawner(name, condition=None, inactive=False):
        args = [name, '-c', '/ur5e/controller_manager', '--controller-manager-timeout', '30'] + (['--inactive'] if inactive else [])
        return Node(package='controller_manager', executable='spawner', namespace='ur5e', arguments=args + quiet,
                    output='screen', condition=condition)
    spawners = [
        spawner('joint_state_broadcaster'),
        spawner('io_and_status_controller'),
        spawner('speed_scaling_state_broadcaster'),
        spawner('force_torque_sensor_broadcaster'),
        spawner('scaled_joint_trajectory_controller', real),
        spawner('joint_trajectory_controller', sim),
        spawner('joint_trajectory_controller', real, inactive=True),
        spawner('forward_position_controller', inactive=True),
        spawner('ur_configuration_controller', real),
    ]
    # the driver's helpers (robot only)
    dashboard = Node(package='ur_robot_driver', executable='dashboard_client', name='dashboard_client', namespace='ur5e',
                     parameters=[{'robot_ip': robot_ip}], output='screen', condition=real)
    stopper = Node(package='ur_robot_driver', executable='controller_stopper_node', name='controller_stopper', namespace='ur5e',
                   parameters=[{'headless_mode': True, 'joint_controller_active': True,
                                'consistent_controllers': ['io_and_status_controller', 'force_torque_sensor_broadcaster',
                                                           'joint_state_broadcaster', 'speed_scaling_state_broadcaster',
                                                           'ur_configuration_controller']}],
                   output='screen', condition=real)
    urscript = Node(package='ur_robot_driver', executable='urscript_interface', name='urscript_interface', namespace='ur5e',
                    parameters=[{'robot_ip': robot_ip}], output='screen', condition=real)

    # the rig's fakes
    fake_dashboard = Node(package='rd_arm', executable='fake_dashboard', name='fake_dashboard', output='screen', condition=sim)
    fake_hand = Node(package='rd_arm', executable='fake_hand', name='fake_hand', output='screen',
                     arguments=['8766', '--state'], condition=sim)

    arm_power = Node(package='rd_arm', executable='arm_power', name='arm_power', output='screen',
                     parameters=[{'use_sim_time': use_sim_time, 'sim_mode': sim_mode,
                                  'ip': PythonExpression(["'127.0.0.1' if '", sim_mode, "' == 'true' else '", robot_ip, "'"])}])
    hand = Node(package='rd_arm', executable='hand', name='hand', output='screen', condition=IfCondition(LaunchConfiguration('hand')),
                parameters=[{'use_sim_time': use_sim_time,
                             'host': PythonExpression(["'127.0.0.1' if '", sim_mode, "' == 'true' else '", str(config.get('hand.ip')), "'"])}])
    wave = Node(package='rd_arm', executable='wave_server', name='wave_server', output='screen',
                parameters=[{'use_sim_time': use_sim_time,
                             'controller': PythonExpression(["'/ur5e/joint_trajectory_controller' if '", sim_mode,
                                                             "' == 'true' else '/ur5e/scaled_joint_trajectory_controller'"])}])
    stream = Node(package='rd_arm', executable='stream', name='arm_stream', output='screen',
                  parameters=[{'use_sim_time': use_sim_time}], condition=IfCondition(LaunchConfiguration('stream')))
    moveit = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'moveit.launch.py')),
                                      launch_arguments=[('sim_mode', sim_mode), ('use_sim_time', use_sim_time)],
                                      condition=IfCondition(LaunchConfiguration('moveit')))

    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('robot_ip', default_value=str(config.get('arm.ip'))),
        DeclareLaunchArgument('hand', default_value='true'),
        DeclareLaunchArgument('moveit', default_value='true'),
        DeclareLaunchArgument('stream', default_value='true'),
        fake_dashboard, fake_hand, description,
        TimerAction(period=1.0, actions=[cm]),
        RegisterEventHandler(OnProcessStart(target_action=cm, on_start=spawners)),
        dashboard, stopper, urscript,
        TimerAction(period=2.0, actions=[arm_power, hand, moveit]),
        TimerAction(period=4.0, actions=[wave, stream]),
    ])
