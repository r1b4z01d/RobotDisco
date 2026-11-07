"""Every sensor on the robot (what the supervisor unit runs). The rig runs none of the hardware drivers
(sim_mode:=true turns them all off and starts the fake tag detector instead).

    ros2 launch rd_perception perception.launch.py [sim_mode:=false] [lidars:=true] [realsense:=true]
                                                   [cameras:=true] [isaac:=false] [apriltag:=false] ...
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from rd_bringup.launch_util import share


def _real(name):
    """true when <name>:=true and sim_mode:=false"""
    return IfCondition(PythonExpression(["'", LaunchConfiguration(name), "' == 'true' and '",
                                         LaunchConfiguration('sim_mode'), "' != 'true'"]))


def generate_launch_description():
    inc = lambda pkg, f, cond, **a: IncludeLaunchDescription(  # noqa: E731
        PythonLaunchDescriptionSource(share(pkg, 'launch', f)), launch_arguments=list(a.items()), condition=cond)
    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false', description='no hardware drivers; fake tags'),
        DeclareLaunchArgument('lidars', default_value='true'),
        DeclareLaunchArgument('realsense', default_value='true'),
        DeclareLaunchArgument('cameras', default_value='true'),
        DeclareLaunchArgument('isaac', default_value='false', description='the Isaac ROS container (robot only)'),
        DeclareLaunchArgument('apriltag', default_value='false'),
        DeclareLaunchArgument('grid_localizer', default_value='false'),
        DeclareLaunchArgument('nvblox', default_value='false'),
        DeclareLaunchArgument('visual_slam', default_value='false'),
        DeclareLaunchArgument('fake_tags', default_value='[7.0, 5.0, 1.0, 0.5, 0.0]',
                              description='sim_mode: tags in the map as id, x, y, z, yaw'),
        inc('rd_perception', 'lidars.launch.py', _real('lidars')),
        inc('rd_perception', 'realsense.launch.py', _real('realsense'), infra=LaunchConfiguration('visual_slam')),
        inc('rd_perception', 'cameras.launch.py', _real('cameras')),
        inc('rd_perception', 'isaac.launch.py', _real('isaac'),
            apriltag=LaunchConfiguration('apriltag'), grid_localizer=LaunchConfiguration('grid_localizer'),
            nvblox=LaunchConfiguration('nvblox'), visual_slam=LaunchConfiguration('visual_slam')),
        Node(package='rd_perception', executable='fake_tags', name='fake_tags', output='screen',
             parameters=[{'use_sim_time': True, 'tags': PythonExpression([LaunchConfiguration('fake_tags')])}],
             condition=IfCondition(LaunchConfiguration('sim_mode'))),
    ])
