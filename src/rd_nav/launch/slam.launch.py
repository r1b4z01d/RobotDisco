"""slam_toolbox (async) for a new map or to continue one, as the manager's Maps tab requested
(maps/slam_request.yaml). Run INSTEAD of localization.launch.py; save with slam_toolbox's services.

    ros2 launch rd_nav slam.launch.py [use_sim_time:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rd_nav.maps_store import MapsStore


def generate_launch_description():
    share = get_package_share_directory('rd_nav')
    use_sim_time = LaunchConfiguration('use_sim_time')
    overrides = MapsStore().slam_overrides()
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox', output='screen',
             parameters=[os.path.join(share, 'config', 'slam.yaml'), {'use_sim_time': use_sim_time}, overrides]),
        # slam_toolbox 2.8 is a lifecycle node: started alone it sits unconfigured, takes no scan and publishes no map
        # (the robot's first "new map", 2026-11-10: the unit ran for minutes and logged one line). This configures and
        # activates it; no bond, the node only makes one when told to.
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_slam', output='screen',
             parameters=[{'use_sim_time': use_sim_time, 'autostart': True, 'node_names': ['slam_toolbox'], 'bond_timeout': 0.0}]),
    ])
