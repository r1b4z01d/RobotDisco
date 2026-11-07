"""Launch test: the Say server with the fake voice and no player speaks a sentence: /speaking goes true then
false, /mouth/audio chunks arrive at 25 Hz for the sentence's duration, the result reports the seconds,
and an interrupting goal cuts the first one short."""
import time
import unittest

import launch
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from action_msgs.msg import GoalStatus
from launch_ros.actions import Node
from rclpy.action import ActionClient
from std_msgs.msg import Bool, Float32MultiArray, String

from rd_interfaces.action import Say


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    return launch.LaunchDescription([
        Node(package='rd_audio', executable='say_server', name='say_server', output='screen',
             parameters=[{'engine': 'fake', 'player': 'none', 'sink_watchdog': False, 'sink': ''}]),
        launch_testing.actions.ReadyToTest()]), {}


class TestSay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('say_test')
        cls.speaking = []
        cls.chunks = []
        cls.node.create_subscription(Bool, '/speaking', lambda m: cls.speaking.append((time.monotonic(), m.data)), 10)
        cls.node.create_subscription(Float32MultiArray, '/mouth/audio', lambda m: cls.chunks.append(time.monotonic()), 50)
        cls.text = cls.node.create_publisher(String, '/say_text', 10)
        cls.client = ActionClient(cls.node, Say, '/say')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def say(self, text, interrupt=False, wait=True):
        fut = self.client.send_goal_async(Say.Goal(text=text, interrupt=interrupt))
        while not fut.done():
            rclpy.spin_once(self.node, timeout_sec=0.1)
        h = fut.result()
        self.assertTrue(h.accepted)
        res = h.get_result_async()
        if not wait:
            return res
        end = time.monotonic() + 30
        while not res.done() and time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return res.result()

    def test_1_speaks_with_mouth_feed(self):
        self.assertTrue(self.client.wait_for_server(timeout_sec=30))
        self.spin(1.0)
        text = 'hello there, I am robot disco'      # 29 chars -> ~1.7 s of fake speech
        r = self.say(text)
        self.assertEqual(r.status, GoalStatus.STATUS_SUCCEEDED, r.result.message)
        self.assertAlmostEqual(r.result.seconds, 29 * 0.06, delta=0.4)
        states = [s for _, s in self.speaking]
        self.assertEqual(states[:2], [True, False])
        self.assertGreater(len(self.chunks), 30)                 # 25 chunks a second
        span = self.chunks[-1] - self.chunks[0]
        self.assertAlmostEqual(span, 29 * 0.06, delta=0.4)

    def test_2_interrupt(self):
        self.speaking.clear()
        first = self.say('a very long sentence that keeps going on and on and on and on and on', wait=False)
        self.spin(0.5)
        r2 = self.say('stop', interrupt=True)
        self.assertEqual(r2.status, GoalStatus.STATUS_SUCCEEDED)
        end = time.monotonic() + 10
        while not first.done() and time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        r1 = first.result()
        self.assertEqual(r1.status, GoalStatus.STATUS_ABORTED)
        self.assertEqual(r1.result.message, 'interrupted')
        self.assertLess(r1.result.seconds, 2.0)

    def test_3_text_topic(self):
        self.speaking.clear()
        self.text.publish(String(data='hi'))
        end = time.monotonic() + 5
        while time.monotonic() < end and not any(s for _, s in self.speaking):
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertTrue(any(s for _, s in self.speaking))


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -15])
