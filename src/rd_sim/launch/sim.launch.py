"""The dev stack against Isaac Sim on the GPU box: Isaac Sim (its own python) with rd_sim's scene, then the
5.0 stack in sim_mode with use_sim_time. Isaac Sim itself is launched by hand (or by the `isaac` arg):

    ros2 run rd_sim export_urdf /tmp/robot_disco.urdf
    ~/isaacsim-venv/bin/python $(ros2 pkg prefix rd_sim)/share/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf --headless
    ros2 launch rd_sim sim.launch.py            # base command path + scan filters + nav + arm (mock UR mirrored) + vla (fake) + voice + manager
    ros2 launch rd_sim sim.launch.py vla_backend:=smolvla vla_ckpt:=tigger_1_merged_smolvla_1   # the real policy on the GPU

Not runnable in the Docker rig (no GPU): phase 7.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rd_bringup.launch_util import share


def inc(pkg, f, **args):
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(share(pkg, 'launch', f)), launch_arguments=list(args.items()))


def _vla(context):
    # fake: the rig's scripted policy under the system python. smolvla: the LeRobot venv's python on the GPU, serving
    # vla_ckpt (run[@step] under <data>/outputs/train, or a path) with vla_n_action_steps of each chunk per observation
    backend = LaunchConfiguration('vla_backend').perform(context)
    real = backend != 'fake'
    python = '/opt/vla/bin/python' if os.path.exists('/opt/vla/bin/python') else 'python3'   # the venv: the policy, LeRobot recordings
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(share('rd_vla', 'launch', 'vla.launch.py')),
                                     launch_arguments=[('backend', backend), ('ckpt', LaunchConfiguration('vla_ckpt')),
                                                       ('python', python), ('device', 'cuda' if real else 'cpu'),
                                                       ('n_action_steps', LaunchConfiguration('vla_n_action_steps')), ('use_sim_time', 'true')],
                                     condition=IfCondition(LaunchConfiguration('vla')))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('manager', default_value='true'),
        DeclareLaunchArgument('arm', default_value='true'),
        DeclareLaunchArgument('vla', default_value='true'),
        DeclareLaunchArgument('vla_backend', default_value='fake', description='fake | smolvla (the LeRobot venv on the GPU, vla_ckpt loaded at start)'),
        DeclareLaunchArgument('vla_ckpt', default_value='', description='smolvla: run[@step] under the data dir, e.g. tigger_1_merged_smolvla_1; empty = select per RunPolicy goal'),
        DeclareLaunchArgument('vla_n_action_steps', default_value='8', description='smolvla: actions executed per observation (v1 drove with 4..8 of the 50-step chunk); 0 = open loop'),
        DeclareLaunchArgument('voice', default_value='true', description='the voice assistant (ears and LLM off: the rules and the landmark tasks) and a silent Say server, so the manager\'s text box and /api/say work'),
        # rsp + twist_mux + collision monitor: the robot's own command path (cmd_vel_nav / web / joy -> mux -> monitor ->
        # /diff_cont/cmd_vel), which the sim's bridge then integrates; no ros2_control, Isaac moves the wheels
        inc('rd_base', 'base.launch.py', sim_mode='true', use_sim_time='true', joystick='false', controllers='false'),
        inc('rd_perception', 'lidars.launch.py', sim='true', use_sim_time='true'),   # the body filter chains on the sim's raw scans, as on the robot
        # AMCL starts at the spawn pose (isaac_scene.py's defaults = the Charger landmark): Nav2's costmaps give the
        # map frame 60 s to appear at bringup, and in the sim nothing else sets the initial pose
        inc('rd_nav', 'nav.launch.py', use_sim_time='true', set_initial_pose='true', initial_x='0.6957', initial_y='0.9208', initial_yaw='0.0548',
            odom_alpha='0.05'),   # the sim's odometry is exact: AMCL's default 0.2 motion noise random-walks it in the corridor
        IncludeLaunchDescription(PythonLaunchDescriptionSource(share('rd_arm', 'launch', 'arm.launch.py')),
                                 launch_arguments=[('sim_mode', 'true'), ('use_sim_time', 'true'), ('cumotion', 'true')],   # the GPU box has the Isaac debs
                                 condition=IfCondition(LaunchConfiguration('arm'))),
        # the RealSense IR frames: on the robot realsense2_camera publishes them; the URDF has only camera_link(_optical), so the
        # sim gives the two IR imagers (the right one 50 mm along optical x) their frames for the visual SLAM camera_infos
        Node(package='tf2_ros', executable='static_transform_publisher', name='infra1_tf', parameters=[{'use_sim_time': True}],
             arguments=['--frame-id', 'camera_link_optical', '--child-frame-id', 'camera_infra1_optical_frame']),
        Node(package='tf2_ros', executable='static_transform_publisher', name='infra2_tf', parameters=[{'use_sim_time': True}],
             arguments=['--x', '0.05', '--frame-id', 'camera_link_optical', '--child-frame-id', 'camera_infra2_optical_frame']),
        Node(package='rd_sim', executable='arm_mirror', name='arm_mirror', output='screen', parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('arm'))),
        OpaqueFunction(function=_vla),
        Node(package='rd_audio', executable='say_server', name='say_server', output='screen',
             parameters=[{'engine': 'fake', 'player': 'none', 'sink_watchdog': False, 'sink': '', 'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('voice'))),
        Node(package='rd_voice', executable='voice', name='voice', output='screen',
             parameters=[{'ears_enabled': False, 'llm_enabled': False, 'llm_backend': 'none', 'volume_sink': '', 'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('voice'))),
        Node(package='rd_manager', executable='manager', output='screen', arguments=['--sim', 'true'],
             parameters=[{'use_sim_time': True}],       # its stamps and TF ages follow Isaac's /clock
             condition=IfCondition(LaunchConfiguration('manager'))),
    ])
