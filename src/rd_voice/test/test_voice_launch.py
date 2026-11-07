"""Launch test: the voice node with no ears and no LLM (rules only) against rd_audio's Say server (fake
voice) and stub GoToLandmark / RunPolicy action servers in the test. "robot disco, pick up tigger" must
drive to the dining room, run the tigger policy with the landmark's task string, and say so."""
import time
import unittest

import launch
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from launch_ros.actions import Node
from rclpy.action import ActionServer
from std_msgs.msg import Bool, String

from rd_interfaces.action import GoToLandmark, RunPolicy


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    return launch.LaunchDescription([
        Node(package='rd_audio', executable='say_server', name='say_server', output='screen',
             parameters=[{'engine': 'fake', 'player': 'none', 'sink_watchdog': False, 'sink': ''}]),
        Node(package='rd_voice', executable='voice', name='voice', output='screen',
             parameters=[{'ears_enabled': False, 'llm_enabled': False, 'llm_backend': 'none', 'volume_sink': ''}]),
        launch_testing.actions.ReadyToTest()]), {}


class TestVoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('voice_test')
        cls.replies, cls.states, cls.nav_goals, cls.policy_goals, cls.speaking = [], [], [], [], []
        cls.node.create_subscription(String, '/voice/reply', lambda m: cls.replies.append(m.data), 10)
        cls.node.create_subscription(String, '/voice/state', lambda m: cls.states.append(m.data), 10)
        cls.node.create_subscription(Bool, '/speaking', lambda m: cls.speaking.append(m.data), 10)
        cls.text = cls.node.create_publisher(String, '/voice/text_in', 10)

        def nav_cb(handle):
            cls.nav_goals.append(handle.request.landmark)
            time.sleep(0.5)
            handle.succeed()
            return GoToLandmark.Result(success=True, message='arrived')

        def policy_cb(handle):
            cls.policy_goals.append(handle.request)
            time.sleep(0.5)
            handle.succeed()
            return RunPolicy.Result(success=True, message='done', steps=30, seconds=2.5)
        cls.nav_srv = ActionServer(cls.node, GoToLandmark, 'go_to_landmark', nav_cb)
        cls.policy_srv = ActionServer(cls.node, RunPolicy, 'run_policy', policy_cb)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def wait_for(self, pred, seconds, what):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if pred():
                return
        self.fail('timed out waiting for ' + what)

    def test_1_time_question(self):
        self.wait_for(lambda: self.states, 30, 'the voice node')
        self.text.publish(String(data='robot disco what time is it'))
        self.wait_for(lambda: self.replies, 15, 'a reply')
        self.assertRegex(self.replies[-1], r'\d')
        self.wait_for(lambda: True in self.speaking, 10, 'speech')

    def test_2_landmark_task(self):
        self.replies.clear()
        self.text.publish(String(data='robot disco, could you pick up tigger please'))
        self.wait_for(lambda: self.replies, 15, 'the task reply')
        self.assertEqual(self.replies[-1], 'Heading to the Dining Room to pick up tigger.')
        self.wait_for(lambda: self.nav_goals, 15, 'the navigation goal')
        self.assertEqual(self.nav_goals[-1], 'dining-room')
        self.wait_for(lambda: self.policy_goals, 15, 'the policy goal')
        g = self.policy_goals[-1]
        self.assertEqual(g.task, 'Grab and hold tigger')
        self.assertEqual(g.checkpoint, 'tigger_1_merged_smolvla_1')
        self.assertEqual(g.done_when, 'closed')
        self.assertEqual(g.landmark, 'dining-room')

    def test_3_ignores_talk_without_the_wake_phrase(self):
        self.replies.clear()
        self.text.publish(String(data='what time is it'))
        end = time.monotonic() + 3
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertEqual(self.replies, [])


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -15])
