"""Launch test in the loopback simulator: Nav2 comes up, GoToLandmark drives the robot to a landmark,
rejects an unknown one, and the tag corrector moves AMCL's pose when a taught tag says so.
"""
import math
import os
import tempfile
import time
import unittest

import launch
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from rclpy.action import ActionClient
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from rd_interfaces.action import GoToLandmark
from rd_interfaces.msg import TagDetection, TagDetectionArray
from rd_nav.tag_corrector import load_tag_poses, mat_to_quat, transform_to_mat, xy_yaw_to_mat
import numpy as np

TAG_POSES = os.path.join(tempfile.mkdtemp(), 'tag_poses.yaml')
# tag 7 taught at map (5.0, 1.0), facing -x (its z axis points at the robot standing at smaller x)
with open(TAG_POSES, 'w') as f:
    yaml.safe_dump({'tags': {7: {'position': {'x': 5.0, 'y': 1.0, 'z': 0.5},
                                 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}}}, f)


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    share = get_package_share_directory('rd_nav')
    sim = IncludeLaunchDescription(PythonLaunchDescriptionSource(share + '/launch/loopback_sim.launch.py'))
    nav = IncludeLaunchDescription(PythonLaunchDescriptionSource(share + '/launch/navigation.launch.py'),
                                   launch_arguments=[('use_sim_time', 'true'), ('tag_poses_file', TAG_POSES)])
    return launch.LaunchDescription([sim, nav, launch_testing.actions.ReadyToTest()]), {}


def _initialpose(node, x, y, yaw):
    m = PoseWithCovarianceStamped()
    m.header.frame_id = 'map'
    m.header.stamp = node.get_clock().now().to_msg()
    m.pose.pose.position.x = x
    m.pose.pose.position.y = y
    m.pose.pose.orientation.z = math.sin(yaw / 2)
    m.pose.pose.orientation.w = math.cos(yaw / 2)
    m.pose.covariance[0] = m.pose.covariance[7] = 0.01
    m.pose.covariance[35] = 0.01
    return m


class TestNav(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('nav_test', parameter_overrides=[rclpy.Parameter('use_sim_time', value=True)])
        cls.tf = Buffer()
        cls.tfl = TransformListener(cls.tf, cls.node)
        cls.initialpose = cls.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        cls.tags = cls.node.create_publisher(TagDetectionArray, '/tags/detections', 10)
        cls.client = ActionClient(cls.node, GoToLandmark, '/go_to_landmark')
        cls.is_active = cls.node.create_client(Trigger, '/lifecycle_manager_navigation/is_active')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def pose(self):
        t = self.tf.lookup_transform('map', 'base_footprint', rclpy.time.Time()).transform
        q = t.rotation
        return t.translation.x, t.translation.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))

    def wait_for(self, pred, seconds, what):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            try:
                if pred():
                    return
            except Exception:  # noqa: BLE001 - TF not there yet
                pass
        self.fail('timed out waiting for ' + what)

    def go(self, landmark, timeout=150.0, **kw):
        goal = GoToLandmark.Goal(landmark=landmark, **kw)
        fut = self.client.send_goal_async(goal)
        self.wait_for(fut.done, 10, 'goal response')
        handle = fut.result()
        self.assertTrue(handle.accepted)
        res = handle.get_result_async()
        self.wait_for(res.done, timeout, f'result for {landmark}')
        return res.result().status, res.result().result

    def test_1_comes_up_and_localizes(self):
        self.assertTrue(self.client.wait_for_server(timeout_sec=90), 'go_to_landmark server')
        self.assertTrue(self.is_active.wait_for_service(timeout_sec=60), 'nav2 lifecycle manager')
        # the simulator publishes no TF until it has an initial pose, and Nav2 cannot activate without TF
        self.spin(2.0)

        def placed():
            self.initialpose.publish(_initialpose(self.node, 0.6957, 0.9208, 0.05))   # at the charger
            self.spin(0.5)
            return abs(self.pose()[0] - 0.6957) < 0.1
        self.wait_for(placed, 30, 'the simulator to take the initial pose')

        def nav2_active():
            fut = self.is_active.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self.node, fut, timeout_sec=2.0)
            return fut.done() and fut.result().success
        self.wait_for(nav2_active, 120, 'Nav2 to be active')
        self.spin(1.0)

    def test_2_unknown_landmark(self):
        status, result = self.go('the moon', timeout=20)
        self.assertEqual(status, GoalStatus.STATUS_ABORTED)
        self.assertEqual(result.message, 'unknown_landmark')

    def test_3_drives_to_the_hallway(self):
        t0 = time.monotonic()
        status, result = self.go('hallway')
        self.assertEqual(status, GoalStatus.STATUS_SUCCEEDED, result.message)
        self.assertEqual(result.message, 'arrived')
        x, y, yaw = self.pose()
        self.assertLess(math.hypot(x - 2.7181, y - 1.9097), 0.4, f'ended at ({x:.2f}, {y:.2f})')
        self.assertLess(abs(yaw - 0.0903), 0.4)
        print(f'hallway reached in {time.monotonic() - t0:.0f} s')

    def test_4_tag_corrector_moves_the_pose(self):
        # the robot thinks it is at the hallway (2.72, 1.91). A detection of tag 7 (taught at map 5.0, 1.0)
        # computed for a robot really standing at (3.5, 1.0) facing +x must make the corrector publish
        # /initialpose, and the simulator jump there.
        before = self.pose()
        truth = (3.5, 1.0, 0.0)                 # where the robot really stands, says the tag
        T_map_tag = load_tag_poses(TAG_POSES)[7]
        T_base_cam = transform_to_mat(self.tf.lookup_transform('base_footprint', 'camera_link_optical', rclpy.time.Time()))
        T_cam_tag = np.linalg.inv(xy_yaw_to_mat(*truth) @ T_base_cam) @ T_map_tag
        qx, qy, qz, qw = mat_to_quat(T_cam_tag[:3, :3])
        for _ in range(6):
            m = TagDetectionArray()
            m.header.frame_id = 'camera_link_optical'
            m.header.stamp = self.node.get_clock().now().to_msg()
            d = TagDetection(id=7, family='tag36h11', size=0.15)
            d.pose.header = m.header
            d.pose.pose.position.x, d.pose.pose.position.y, d.pose.pose.position.z = (float(v) for v in T_cam_tag[:3, 3])
            d.pose.pose.orientation.x, d.pose.pose.orientation.y = float(qx), float(qy)
            d.pose.pose.orientation.z, d.pose.pose.orientation.w = float(qz), float(qw)
            m.detections.append(d)
            self.tags.publish(m)
            self.spin(0.2)
        self.wait_for(lambda: math.hypot(self.pose()[0] - before[0], self.pose()[1] - before[1]) > 0.5, 15,
                      'the corrector to move the robot')
        x, y, yaw = self.pose()
        self.assertLess(math.hypot(x - truth[0], y - truth[1]), 0.15, f'corrected to ({x:.2f}, {y:.2f})')
        self.assertLess(abs(yaw - truth[2]), 0.1)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -15])
