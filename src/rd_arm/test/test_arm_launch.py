"""Launch test on the mock arm: controllers up, ArmPower through the fake dashboard, the hand through the
fake hand, a collision-aware Wave, and 50 Hz streaming with IK."""
import math
import time
import unittest

import launch
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import PoseStamped
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener

from rd_interfaces.action import Wave
from rd_interfaces.srv import ArmPower

UR = ['ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint', 'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    arm = IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory('rd_arm') + '/launch/arm.launch.py'),
                                   launch_arguments=[('sim_mode', 'true')])
    rsp = IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory('rd_description') + '/launch/rsp.launch.py'),
                                   launch_arguments=[('sim_mode', 'true')])
    return launch.LaunchDescription([rsp, arm, launch_testing.actions.ReadyToTest()]), {}


class TestArm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('arm_test')
        cls.q = {}
        cls.cmds = []
        cls.hand_state = []
        cls.node.create_subscription(JointState, '/joint_states', lambda m: cls.q.update(zip(m.name, m.position)), 10)
        cls.node.create_subscription(Float64MultiArray, '/ur5e/forward_position_controller/commands', lambda m: cls.cmds.append(time.monotonic()), 10)
        cls.node.create_subscription(Float32MultiArray, '/hand/state', cls.hand_state.append, 10)
        cls.list_ctl = cls.node.create_client(ListControllers, '/ur5e/controller_manager/list_controllers')
        cls.power = cls.node.create_client(ArmPower, '/arm/power')
        cls.hand_close = cls.node.create_client(Trigger, '/hand/close')
        cls.hand_open = cls.node.create_client(Trigger, '/hand/open')
        cls.stream = cls.node.create_client(SetBool, '/arm/stream')
        cls.wave = ActionClient(cls.node, Wave, '/wave')
        cls.joint_target = cls.node.create_publisher(Float64MultiArray, '/arm/target_joints', 10)
        cls.pose_target = cls.node.create_publisher(PoseStamped, '/arm/target_pose', 10)
        cls.tf = Buffer()
        cls.tfl = TransformListener(cls.tf, cls.node)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def call(self, client, req, timeout=30.0):
        self.assertTrue(client.wait_for_service(timeout_sec=30), f'service {client.srv_name}')
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        self.assertTrue(fut.done(), f'{client.srv_name} timed out')
        return fut.result()

    def wait_for(self, pred, seconds, what):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            try:
                if pred():
                    return
            except Exception:  # noqa: BLE001
                pass
        self.fail('timed out waiting for ' + what)

    def joints(self):
        return [self.q[n] for n in UR]

    def active_controllers(self):
        r = self.call(self.list_ctl, ListControllers.Request())
        return {c.name for c in r.controller if c.state == 'active'}

    def test_1_controllers_and_joint_states(self):
        self.wait_for(lambda: all(n in self.q for n in UR), 60, 'UR joint states')
        self.wait_for(lambda: {'joint_state_broadcaster', 'joint_trajectory_controller'} <= self.active_controllers(), 60, 'controllers active')
        self.assertNotIn('forward_position_controller', self.active_controllers())
        self.assertAlmostEqual(self.q['ur_shoulder_pan_joint'], 4.010228, places=3)    # the mock starts at vla_home

    def test_2_arm_power(self):
        st = self.call(self.power, ArmPower.Request(command=ArmPower.Request.STATUS))
        self.assertTrue(st.success and st.remote_control)
        self.assertEqual(st.robot_mode, 'POWER_OFF')
        on = self.call(self.power, ArmPower.Request(command=ArmPower.Request.POWER_ON, timeout_s=20.0), timeout=60)
        self.assertTrue(on.success, on.message)
        self.assertEqual(on.robot_mode, 'RUNNING')
        self.assertIn('powered up', on.message)

    def test_3_hand(self):
        r = self.call(self.hand_close, Trigger.Request())
        self.assertTrue(r.success, r.message)
        self.wait_for(lambda: self.hand_state and list(self.hand_state[-1].data[:4]) == [1.0] * 4
                      and max(self.hand_state[-1].data[4:]) > 0.99, 5, 'closed hand reported back by the fake')
        r = self.call(self.hand_open, Trigger.Request())
        self.assertTrue(r.success)
        self.wait_for(lambda: self.hand_state and max(self.hand_state[-1].data) < 0.01, 5, 'open hand')

    def _wave(self, return_to, before):
        phases = []
        goal = Wave.Goal(motion='wave', return_to=return_to, speed_scale=2.0, wiggle_fingers=True)
        fut = self.wave.send_goal_async(goal, feedback_callback=lambda fb: phases.append(fb.feedback.phase))
        self.wait_for(fut.done, 20, 'wave goal response')
        handle = fut.result()
        self.assertTrue(handle.accepted)
        moved = [False]

        def track():
            if max(abs(a - b) for a, b in zip(self.joints(), before)) > 0.3:
                moved[0] = True
            return res.done()
        res = handle.get_result_async()
        self.wait_for(track, 120, 'the wave to finish')
        r = res.result()
        self.assertEqual(r.status, GoalStatus.STATUS_SUCCEEDED, r.result.message)
        self.assertIn('moving', phases)
        self.assertTrue(moved[0], 'the arm never left the start pose during the wave')
        return r.result.message

    def test_4_wave(self):
        self.assertTrue(self.wave.wait_for_server(timeout_sec=60))
        import yaml
        poses = yaml.safe_load(open(get_package_share_directory('rd_arm') + '/config/poses.yaml'))['poses']
        home = self.joints()
        # 1. the wave, then MoveIt plans the return to the transport pose (a real plan through the keepout scene)
        msg = self._wave('transport', home)
        self.assertIn('returned to transport', msg)
        self.assertLess(max(abs(a - b) for a, b in zip(self.joints(), poses['transport'])), 0.05, 'not at transport')
        # 2. from transport: MoveIt plans the approach to the recording's start, then the return home
        msg = self._wave('home', self.joints())
        self.assertIn('returned to home', msg)
        self.assertLess(max(abs(a - b) for a, b in zip(self.joints(), home)), 0.05, 'did not return home')

    def test_5_stream(self):
        r = self.call(self.stream, SetBool.Request(data=True))
        self.assertTrue(r.success, r.message)
        self.wait_for(lambda: 'forward_position_controller' in self.active_controllers(), 10, 'forward position controller')
        q0 = self.joints()
        target = list(q0)
        target[5] += 0.3
        self.cmds.clear()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            self.joint_target.publish(Float64MultiArray(data=target))
            self.spin(0.1)
        self.assertAlmostEqual(self.q['ur_wrist_3_joint'], target[5], delta=0.02)
        rate = len(self.cmds) / 2.0
        self.assertGreater(rate, 40.0, f'command rate {rate:.0f} Hz')
        # a Cartesian target: 5 cm lower than where the tool is now, through IK
        t = self.tf.lookup_transform('ur_base_link', 'ur_tool0', rclpy.time.Time())
        p = PoseStamped()
        p.header.frame_id = 'ur_base_link'
        p.pose.position.x, p.pose.position.y, p.pose.position.z = t.transform.translation.x, t.transform.translation.y, t.transform.translation.z - 0.05
        p.pose.orientation = t.transform.rotation
        z0 = t.transform.translation.z
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            p.header.stamp = self.node.get_clock().now().to_msg()
            self.pose_target.publish(p)
            self.spin(0.2)
        z1 = self.tf.lookup_transform('ur_base_link', 'ur_tool0', rclpy.time.Time()).transform.translation.z
        self.assertAlmostEqual(z1, z0 - 0.05, delta=0.015, msg=f'tool z {z0:.3f} -> {z1:.3f}')
        r = self.call(self.stream, SetBool.Request(data=False))
        self.assertTrue(r.success)
        self.wait_for(lambda: 'joint_trajectory_controller' in self.active_controllers(), 10, 'trajectory controller back')


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # -11: move_group's known segfault on shutdown (MoveIt 2.x); the tests above prove it was alive
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -11, -15])
