"""Launch test: the base comes up on mock hardware and the whole velocity path works.

cmd_vel_nav (TwistStamped) -> twist_mux -> cmd_vel_raw -> collision_monitor -> diff_cont -> /odom moves,
and a scan with points inside the stop polygon brings the robot to a halt.
"""
import math
import time
import unittest

import launch
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TwistStamped
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, LaserScan
from tf2_msgs.msg import TFMessage


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(get_package_share_directory('rd_base') + '/launch/base.launch.py'),
        launch_arguments=[('sim_mode', 'true'), ('joystick', 'false'), ('collision_monitor', 'true')])
    return launch.LaunchDescription([base, launch_testing.actions.ReadyToTest()]), {}


def _scan(frame, obstacle_ahead=None):
    """A 360-degree scan with nothing in range, or one object `obstacle_ahead` metres dead ahead."""
    m = LaserScan()
    m.header.frame_id = frame
    m.angle_min, m.angle_max, m.angle_increment = -math.pi, math.pi, math.radians(1.0)
    m.range_min, m.range_max = 0.1, 20.0
    n = int(round((m.angle_max - m.angle_min) / m.angle_increment)) + 1
    m.ranges = [float('inf')] * n
    if obstacle_ahead is not None:
        for deg in range(-4, 5):   # nine beams -> well past min_points
            m.ranges[(n // 2) + deg] = obstacle_ahead
    return m


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('base_test')
        cls.odom = []
        cls.joints = []
        cls.tf_frames = set()
        cls.node.create_subscription(Odometry, '/diff_cont/odom', cls.odom.append, 10)
        cls.node.create_subscription(JointState, '/joint_states', cls.joints.append, 10)
        cls.node.create_subscription(TFMessage, '/tf', lambda m: cls.tf_frames.update(
            (t.header.frame_id, t.child_frame_id) for t in m.transforms), 10)
        cls.cmd = cls.node.create_publisher(TwistStamped, '/cmd_vel_nav', 10)
        scan_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        cls.s1 = cls.node.create_publisher(LaserScan, '/rp_s1_lidar/scan_filtered', scan_qos)
        cls.s2 = cls.node.create_publisher(LaserScan, '/rp_s2e_lidar/scan_filtered', scan_qos)
        cls.obstacle = None

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _drive(self, seconds, vx=0.3, obstacle=None):
        """Publish scans + a forward command at 10 Hz for `seconds`, spinning in between."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            now = self.node.get_clock().now().to_msg()
            for pub, frame in ((self.s1, 'rp_s1_lidar'), (self.s2, 'rp_s2e_lidar')):
                m = _scan(frame, obstacle if frame == 'rp_s1_lidar' else None)
                m.header.stamp = now
                pub.publish(m)
            t = TwistStamped()
            t.header.stamp = now
            t.header.frame_id = 'base_link'
            t.twist.linear.x = vx
            self.cmd.publish(t)
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def _wait(self, pred, seconds, what):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if pred():
                return
        self.fail(f'timed out waiting for {what}')

    def test_1_comes_up(self):
        self._wait(lambda: self.joints, 60, '/joint_states from the joint broadcaster')
        self._wait(lambda: self.odom, 30, '/diff_cont/odom')
        names = set(self.joints[-1].name)
        self.assertIn('left_wheel_joint', names)
        self.assertIn('right_wheel_joint', names)

    def test_2_drives_on_command(self):
        self._drive(3.0)          # let the monitor see fresh scans and the mux pick the source
        x0 = self.odom[-1].pose.pose.position.x
        self._drive(3.0)
        x1 = self.odom[-1].pose.pose.position.x
        self.assertGreater(x1 - x0, 0.3, f'odom did not move ({x0:.3f} -> {x1:.3f}); the velocity path is broken')
        self.assertIn(('odom', 'base_link'), self.tf_frames)

    def test_3_stops_for_an_obstacle(self):
        self._drive(3.0, obstacle=0.35)    # inside the stop polygon nose (0.5 m)
        v = [m.twist.twist.linear.x for m in self.odom[-10:]]
        self.assertLess(max(v), 0.02, f'robot still moving with an obstacle 0.35 m ahead: {v}')
        self._drive(4.0)                   # obstacle gone: moves again
        self.assertGreater(self.odom[-1].twist.twist.linear.x, 0.1)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # spawners exit 0 on success; everything else is killed by the harness
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -15])
