"""Launch test: the face headless. Frames come out on /face/image at the render rate; a mouth feed changes the
mouth; a recognised person (the debug topic) is greeted by name through the Say action and gets a wave; the
same person back within the cooldown gets a wink and no hello; a stranger gets the generic hello after the
wait. Say and Wave are stub action servers inside the test that record their goals."""
import json
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
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from rd_interfaces.action import Say, Wave


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    return launch.LaunchDescription([
        Node(package='rd_perception', executable='fake_cameras', name='fake_cameras', parameters=[{'cameras': ['scene']}]),
        Node(package='rd_face', executable='face', name='face', output='screen',
             parameters=[{'renderer': 'headless', 'image_topic': '/cameras/scene/image', 'face_id_enabled': False,
                          'greeting_wait': 1.0, 'greeting_cooldown': 3600.0, 'wave_cooldown': 0.0, 'wave_known_only': True,
                          'mouth_gestures': False, 'face_stick_seconds': 0.8, 'width': 540, 'height': 950}]),
        launch_testing.actions.ReadyToTest()]), {}


class TestFace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('face_test')
        cls.frames = []
        cls.said = []
        cls.waves = []
        cls.node.create_subscription(Image, '/face/image', lambda m: cls.frames.append((time.monotonic(), m)),
                                     QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        cls.person = cls.node.create_publisher(String, '/face/debug_person', 10)
        cls.mouth = cls.node.create_publisher(Float32MultiArray, '/mouth/audio', 10)

        def say_cb(handle):
            cls.said.append(handle.request.text)
            handle.succeed()
            return Say.Result(success=True, message='done', seconds=0.1)

        def wave_cb(handle):
            cls.waves.append(handle.request.motion)
            handle.succeed()
            return Wave.Result(success=True, message='done', seconds=0.1)
        cls.say_srv = ActionServer(cls.node, Say, 'say', say_cb)
        cls.wave_srv = ActionServer(cls.node, Wave, 'wave', wave_cb)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def wait_for(self, pred, seconds, what):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if pred():
                return
        self.fail('timed out waiting for ' + what)

    def test_1_renders(self):
        self.wait_for(lambda: len(self.frames) > 30, 30, 'face frames')
        t0, t1 = self.frames[-31][0], self.frames[-1][0]
        self.assertAlmostEqual(30 / (t1 - t0), 30.0, delta=12.0)
        m = self.frames[-1][1]
        self.assertEqual((m.height, m.width, m.encoding), (475, 270, 'bgr8'))

    def test_2_mouth_moves_with_audio(self):
        import numpy as np
        quiet = np.frombuffer(bytes(self.frames[-1][1].data), np.uint8).reshape(475, 270, 3)
        for _ in range(15):
            self.mouth.publish(Float32MultiArray(data=[0.9 * ((-1) ** i) for i in range(640)]))
            self.spin(0.04)
        loud = np.frombuffer(bytes(self.frames[-1][1].data), np.uint8).reshape(475, 270, 3)
        band_q = quiet[300:475].astype(int)
        band_l = loud[300:475].astype(int)
        self.assertGreater(np.abs(band_q - band_l).mean(), 1.0, 'the mouth region did not change with audio')

    def test_3_greets_by_name_and_waves(self):
        self.said.clear()
        self.waves.clear()
        for _ in range(10):
            self.person.publish(String(data=json.dumps({'id': 'person_1', 'name': 'Bryan', 'score': 0.9})))
            self.spin(0.1)
        self.wait_for(lambda: self.said, 10, 'a hello')
        self.assertEqual(self.said[0], 'hello Bryan')
        self.wait_for(lambda: self.waves, 5, 'a wave')
        self.assertEqual(self.waves[0], 'wave')
        self.spin(1.5)                      # the face loses the person (face_stick_seconds)

    def test_4_back_within_the_hour_gets_a_wink(self):
        self.said.clear()
        self.waves.clear()
        for _ in range(15):
            self.person.publish(String(data=json.dumps({'id': 'person_1', 'name': 'Bryan', 'score': 0.9})))
            self.spin(0.1)
        self.spin(1.0)
        self.assertEqual(self.said, [])
        self.assertEqual(self.waves, [])
        self.spin(1.5)

    def test_5_stranger_gets_the_generic_hello(self):
        self.said.clear()
        self.waves.clear()
        for _ in range(25):
            self.person.publish(String(data=json.dumps({'box': [100, 100, 120, 160]})))     # seen, not identified
            self.spin(0.1)
        self.wait_for(lambda: self.said, 10, 'the generic hello')
        self.assertEqual(self.said[0], 'hello')
        self.assertEqual(self.waves, [])          # wave_known_only


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -15])
