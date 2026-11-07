"""Launch test: the S1 range-offset node corrects a scan end to end, and the fake tag detector sees a tag
from the robot's TF pose (and not one behind it)."""
import math
import time
import unittest

import launch
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from launch_ros.actions import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster

from rd_interfaces.msg import TagDetectionArray


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    return launch.LaunchDescription([
        Node(package='rd_perception', executable='scan_range_offset', name='offset',
             parameters=[{'in_topic': '/s1/uncorrected', 'out_topic': '/s1/scan', 'offset': -0.33}]),
        # tag 7 at map (2, 0, 0.5) facing the robot, tag 8 behind the robot
        Node(package='rd_perception', executable='fake_tags', name='fake_tags',
             parameters=[{'tags': [7.0, 2.0, 0.0, 0.5, 3.14159, 8.0, -2.0, 0.0, 0.5, 0.0], 'rate_hz': 20.0}]),
        launch_testing.actions.ReadyToTest(),
    ]), {}


class TestNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('perception_test')
        cls.scans = []
        cls.tags = []
        cls.node.create_subscription(LaserScan, '/s1/scan', cls.scans.append, 10)
        cls.node.create_subscription(TagDetectionArray, '/tags/detections', cls.tags.append, 10)
        cls.pub = cls.node.create_publisher(LaserScan, '/s1/uncorrected', QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        # a static map -> camera_link_optical: the camera at the origin looking along +x, optical convention
        cls.tf = StaticTransformBroadcaster(cls.node)
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = 'map', 'camera_link_optical'
        t.header.stamp = cls.node.get_clock().now().to_msg()
        # optical (z fwd, x right, y down) from map: rpy (-90, 0, -90) deg
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = -0.5, 0.5, -0.5, 0.5
        cls.tf.sendTransform(t)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def test_scan_offset(self):
        for _ in range(20):
            m = LaserScan()
            m.header.frame_id = 'rp_s1_lidar'
            m.range_min, m.range_max = 0.1, 20.0
            m.ranges = [0.83, 1.32, 0.2, float('inf')]
            self.pub.publish(m)
            self.spin(0.1)
            if self.scans:
                break
        self.assertTrue(self.scans, 'no corrected scan')
        r = self.scans[-1].ranges
        self.assertAlmostEqual(r[0], 0.50, places=5)
        self.assertAlmostEqual(r[1], 0.99, places=5)
        self.assertAlmostEqual(r[2], 0.10, places=5)
        self.assertEqual(r[3], float('inf'))

    def test_fake_tags_see_what_is_in_front(self):
        end = time.monotonic() + 10.0
        seen = None
        while time.monotonic() < end and seen is None:
            self.spin(0.2)
            seen = next((m for m in self.tags if m.detections), None)
        self.assertIsNotNone(seen, f'no detections in {len(self.tags)} messages')
        ids = {d.id for d in seen.detections}
        self.assertEqual(ids, {7})
        d = seen.detections[0]
        self.assertAlmostEqual(d.pose.pose.position.z, 2.0, delta=0.05)     # 2 m ahead = optical z
        self.assertAlmostEqual(d.pose.pose.position.y, -0.5, delta=0.05)    # 0.5 m up = optical -y
        self.assertEqual(d.pose.header.frame_id, 'camera_link_optical')


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -15])
