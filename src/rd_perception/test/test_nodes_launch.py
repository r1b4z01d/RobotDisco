"""Launch test: the S1 range-offset node corrects a scan end to end, the fake tag detector sees a tag
from the robot's TF pose (and not one behind it), and lidars.launch.py sim:=true runs the body filter chain
without drivers (a point on the robot's body is dropped, a far one kept)."""
import math
import time
import unittest

import launch
import launch_testing
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
        IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory('rd_perception') + '/launch/lidars.launch.py'),
                                 launch_arguments=[('sim', 'true')]),       # filter chains only (rd_sim's mode)
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
        cls.filtered = []
        cls.node.create_subscription(LaserScan, '/rp_s2e_lidar/scan_filtered', cls.filtered.append, QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        cls.raw_s2e = cls.node.create_publisher(LaserScan, '/rp_s2e_lidar/scan', QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        cls.pub = cls.node.create_publisher(LaserScan, '/s1/uncorrected', QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        # a static map -> camera_link_optical: the camera at the origin looking along +x, optical convention
        cls.tf = StaticTransformBroadcaster(cls.node)
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = 'map', 'camera_link_optical'
        t.header.stamp = cls.node.get_clock().now().to_msg()
        # optical (z fwd, x right, y down) from map: rpy (-90, 0, -90) deg
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = -0.5, 0.5, -0.5, 0.5
        # base_link -> rp_s2e_lidar as the URDF has it (the rear lidar, 0.52 m behind the axle)
        t2 = TransformStamped()
        t2.header.frame_id, t2.child_frame_id = 'base_link', 'rp_s2e_lidar'
        t2.header.stamp = t.header.stamp
        t2.transform.translation.x = -0.52
        t2.transform.rotation.w = 1.0
        cls.tf.sendTransform([t, t2])

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


    def test_body_filter_chain_in_sim_mode(self):
        # a 360-ray scan from the rear lidar: straight ahead (towards the robot) 0.3 m = on the body, dropped;
        # straight back 3.0 m = the room, kept
        scan = LaserScan()
        scan.header.frame_id = 'rp_s2e_lidar'
        scan.angle_min, scan.angle_max, scan.angle_increment = -math.pi, math.pi - 2 * math.pi / 360, 2 * math.pi / 360
        scan.range_min, scan.range_max = 0.05, 20.0
        scan.ranges = [3.0] * 360
        i_fwd, i_back = 180, 0
        scan.ranges[i_fwd] = 0.3
        end = time.monotonic() + 15
        while time.monotonic() < end and not self.filtered:
            scan.header.stamp = self.node.get_clock().now().to_msg()
            self.raw_s2e.publish(scan)
            self.spin(0.25)
        self.assertTrue(self.filtered, 'no filtered S2E scan (lidars.launch.py sim:=true filter chain)')
        out = self.filtered[-1]
        self.assertEqual(len(out.ranges), 360)
        self.assertFalse(math.isfinite(out.ranges[i_fwd]), f'the body point survived: {out.ranges[i_fwd]}')
        self.assertAlmostEqual(out.ranges[i_back], 3.0, places=3)

@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -15])
