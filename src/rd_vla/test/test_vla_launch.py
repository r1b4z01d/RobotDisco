"""Launch test: the fake inference server + the runner + the mock arm + fake cameras. RunPolicy with
done_when=closed runs end to end: the arm streams toward the policy's drifting target, the hand closes
after the fake backend's N steps, the run ends 'done', the eval.json and the recorded episode exist."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import launch
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from rd_interfaces.action import RunPolicy

TMP = Path(tempfile.mkdtemp())
os.environ['RD_EVALS_DIR'] = str(TMP / 'evals')       # the launched nodes inherit the test's environment
os.environ['RD_RECORD_ROOT'] = str(TMP / 'datasets')
os.environ['RD_MODELS_DIR'] = str(TMP / 'train')


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    rsp = IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory('rd_description') + '/launch/rsp.launch.py'),
                                   launch_arguments=[('sim_mode', 'true')])
    arm = IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory('rd_arm') + '/launch/arm.launch.py'),
                                   launch_arguments=[('sim_mode', 'true')])
    vla = IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory('rd_vla') + '/launch/vla.launch.py'),
                                   launch_arguments=[('backend', 'fake'), ('python', 'python3'), ('device', 'cpu'),
                                                     ('teleop', 'false'), ('fake_close_after', '8')])
    cams = Node(package='rd_perception', executable='fake_cameras', name='fake_cameras')
    return launch.LaunchDescription([rsp, arm, vla, cams, launch_testing.actions.ReadyToTest()]), {}


class TestVla(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('vla_test')
        cls.q = {}
        cls.hand = []
        cls.node.create_subscription(JointState, '/joint_states', lambda m: cls.q.update(zip(m.name, m.position)), 10)
        cls.node.create_subscription(Float32MultiArray, '/hand/state', cls.hand.append, 10)
        cls.client = ActionClient(cls.node, RunPolicy, '/run_policy')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

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

    def test_1_run_policy_until_the_grasp(self):
        self.assertTrue(self.client.wait_for_server(timeout_sec=90), 'run_policy server')
        self.wait_for(lambda: 'ur_wrist_3_joint' in self.q, 60, 'joint states')
        time.sleep(3.0)      # the arm's spawners, move_group and the fake server settle
        goal = RunPolicy.Goal(task='pick up tigger', checkpoint='demo_run', backend='fake', max_seconds=60.0,
                              home_first=True, record=True, landmark='dining-room', done_when='closed', still_seconds=1.0)
        feedback = []
        fut = self.client.send_goal_async(goal, feedback_callback=lambda fb: feedback.append(fb.feedback))
        self.wait_for(fut.done, 20, 'goal response')
        handle = fut.result()
        self.assertTrue(handle.accepted)
        res = handle.get_result_async()
        self.wait_for(res.done, 90, 'the run to end')
        r = res.result()
        self.assertEqual(r.status, GoalStatus.STATUS_SUCCEEDED, r.result.message)
        self.assertEqual(r.result.message, 'done')
        self.assertGreaterEqual(r.result.steps, 8 + 12)            # closed after 8 steps, held 1 s at 12 Hz
        self.assertIn('running', {f.phase for f in feedback})
        # the hand ended closed, the run has its eval.json and the recorded episode
        self.wait_for(lambda: self.hand and min(self.hand[-1].data[:4]) > 0.99, 5, 'closed hand')
        run_dir = Path(r.result.run_dir)
        self.assertEqual(run_dir.parent, TMP / 'evals')
        meta = json.loads((run_dir / 'eval.json').read_text())
        self.assertEqual(meta['ended'], 'done')
        self.assertEqual(meta['landmark'], 'dining-room')
        self.assertEqual(meta['successes'], 1)
        self.assertTrue((run_dir / 'steps.csv').is_file())
        episodes = list((TMP / 'datasets').glob('**/episode_000000.npz')) + list((TMP / 'datasets').glob('**/meta/info.json'))
        self.assertTrue(episodes, f'no recorded episode under {TMP / "datasets"}')

    def test_2_second_goal_times_out_cleanly(self):
        goal = RunPolicy.Goal(task='hold', checkpoint='', backend='fake', max_seconds=2.0, home_first=False, record=False)
        fut = self.client.send_goal_async(goal)
        self.wait_for(fut.done, 20, 'goal response')
        res = fut.result().get_result_async()
        self.wait_for(res.done, 30, 'the run to time out')
        self.assertEqual(res.result().result.message, 'timeout')


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -11, -15])
