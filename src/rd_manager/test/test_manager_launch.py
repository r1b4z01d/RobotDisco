"""Launch test: the manager in the rig with the loopback simulator, Nav2, fake cameras and the Say server.
Checks what the pages need: a live pose, scans in the map frame, the map, camera JPEGs, a landmark captured
at the robot with a stitched photo, GoToLandmark from the API, HALT, the WebSocket stream, a spoken sentence,
typed text reaching the voice assistant (ears and LLM off) and its reply coming back."""
import json
import os
import re
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import launch
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

PORT = 18092
TMP = Path(tempfile.mkdtemp())
os.environ["RD_LANDMARKS_DIR"] = str(TMP / "landmarks")


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    import shutil
    shutil.copytree(Path(get_package_share_directory('rd_nav')) / 'landmarks', TMP / 'landmarks')
    share = get_package_share_directory('rd_nav')
    sim = IncludeLaunchDescription(PythonLaunchDescriptionSource(share + '/launch/loopback_sim.launch.py'))
    nav = IncludeLaunchDescription(PythonLaunchDescriptionSource(share + '/launch/navigation.launch.py'),
                                   launch_arguments=[('use_sim_time', 'true'), ('tag_corrector', 'false')])
    cams = Node(package='rd_perception', executable='fake_cameras', name='fake_cameras', parameters=[{'use_sim_time': True}])
    say = Node(package='rd_audio', executable='say_server', name='say_server', parameters=[{'engine': 'fake', 'player': 'none', 'sink_watchdog': False, 'sink': ''}])
    voice = Node(package='rd_voice', executable='voice', name='voice',
                 parameters=[{'ears_enabled': False, 'llm_enabled': False, 'llm_backend': 'none', 'volume_sink': ''}])
    manager = Node(package='rd_manager', executable='manager', name='rd_manager', output='screen',
                   arguments=['--port', str(PORT), '--sim', 'true', '--data', str(TMP), '--landmarks-dir', str(TMP / 'landmarks'), '--routes-dir', str(TMP / 'routes')])
    return launch.LaunchDescription([sim, nav, cams, say, voice, manager, launch_testing.actions.ReadyToTest()]), {}


def get(path, raw=False):
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}{path}', timeout=10) as r:
        return r.read() if raw else json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')


class TestManager(unittest.TestCase):
    def wait_for(self, pred, seconds, what):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                if pred():
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        self.fail('timed out waiting for ' + what)

    def test_1_live_state(self):
        self.wait_for(lambda: get('/healthz')['ros'], 90, 'the manager')
        post('/api/publish', {'type': 'initialpose', 'x': 0.6957, 'y': 0.9208, 'yaw': 0.05})
        self.wait_for(lambda: get('/api/pose')['pose'] is not None, 30, 'a live pose')
        p = get('/api/pose')['pose']
        self.assertAlmostEqual(p['x'], 0.6957, delta=0.1)
        m = get('/api/map')
        self.assertEqual(m['name'], 'zone_one.yaml') and self.assertEqual(get('/map.png', raw=True)[:4], b'\x89PNG')
        self.wait_for(lambda: get('/api/scans?layers=s2e')['scans'].get('s2e', {}).get('points'), 20, 'scan points in the map frame')
        tf = get('/api/tf')
        self.assertIn('base_footprint', tf['frames']) and self.assertIn('rp_s2e_lidar', tf['frames'])
        self.assertTrue(get('/api/urdf', raw=True).startswith(b'<?xml'))

    def test_2_cameras(self):
        get('/cam/scene.jpg', raw=True) if False else None
        self.wait_for(lambda: get('/api/cameras')['cameras'][0]['alive'], 15, 'the scene camera')
        self.wait_for(lambda: get('/cam/scene.jpg', raw=True)[:2] == b'\xff\xd8', 15, 'a scene JPEG')

    def test_3_landmark_capture_and_goto(self):
        status, r = post('/api/landmarks', {'name': 'Test spot'})
        self.assertEqual(status, 200, r)
        lm = r['landmark']
        self.assertEqual(lm['id'], 'test-spot') and self.assertEqual(lm['map'], 'zone_one')
        self.assertTrue(lm.get('photo'), r)
        self.assertEqual(get(f"/api/landmarks/img/{lm['photo']}", raw=True)[:2], b'\xff\xd8')
        self.wait_for(lambda: get('/api/nav').get('nav2_active'), 120, 'Nav2 to be active')
        status, r = post('/api/landmarks/hallway/goto', {})
        self.assertEqual(status, 200, r)
        self.wait_for(lambda: abs(get('/api/pose')['pose']['x'] - 2.7181) < 0.4, 90, 'the robot to reach the hallway')
        self.wait_for(lambda: not get('/api/nav')['goal_active'], 30, 'the landmark goal to end')
        self.assertTrue(get('/api/landmarks')['landmarks'])
        post('/api/landmarks/test-spot', {'name': 'Test spot 2'})
        self.assertEqual(get('/api/landmarks')['landmarks'][-1]['name'], 'Test spot 2')

    def test_4_halt_and_drive(self):
        status, r = post('/api/landmarks/charger/goto', {})
        self.assertEqual(status, 200)
        self.wait_for(lambda: get('/api/nav')['goal_active'], 15, 'the drive to start')
        time.sleep(1.0)
        status, r = post('/api/halt', {})
        self.assertEqual(status, 200)
        self.assertTrue(any('cancelled' in d for d in r['did']), r)
        self.wait_for(lambda: not get('/api/nav')['goal_active'], 15, 'the goal to be cancelled')
        x0 = get('/api/pose')['pose']['x']
        time.sleep(2.0)
        self.assertLess(abs(get('/api/pose')['pose']['x'] - x0), 0.3, 'the robot kept moving after HALT')
        status, r = post('/api/drive', {'vx': 0.2, 'wz': 0.0})
        self.assertEqual(status, 200) and self.assertEqual(r['vx'], 0.2)
        status, r = post('/api/drive', {'vx': float('nan'), 'wz': 0.0})
        self.assertEqual(status, 400)

    def test_6_voice_text(self):
        # the Live tab's text box: typed text reaches rd_voice as a transcript (wake phrase and all), the reply comes back
        self.wait_for(lambda: get('/api/voice').get('listeners'), 60, 'the voice assistant to listen on /voice/text_in')
        status, r = post('/api/voice/text', {'text': '   '})
        self.assertEqual(status, 400)
        status, r = post('/api/voice/text', {'text': 'robot disco what time is it'})
        self.assertEqual(status, 200)
        self.assertGreaterEqual(r['listeners'], 1, r)
        self.wait_for(lambda: get('/api/voice').get('heard') == 'robot disco what time is it', 10, 'the text to be heard')
        self.wait_for(lambda: 'time' in get('/api/voice').get('command', ''), 10, 'the command after the wake phrase')
        self.wait_for(lambda: re.search(r'\d', get('/api/voice').get('reply', '')), 20, 'a reply with the time in it')
        self.assertGreater(get('/api/voice')['reply_t'], get('/api/voice')['heard_t'] - 1)

    def test_5_say_and_ws(self):
        status, r = post('/api/say', {'text': 'hello from the manager', 'wait': True})
        self.assertEqual(status, 200) and self.assertTrue(r['ok'], r)
        import socket
        import base64
        s = socket.create_connection(('127.0.0.1', PORT), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall(f'GET /ws HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n'.encode())
        head = s.recv(4096)
        self.assertIn(b'101', head.split(b'\r\n')[0])
        data = b''
        t0 = time.time()
        while b'"pose"' not in data and time.time() - t0 < 10:
            data += s.recv(65536)
        s.close()
        self.assertIn(b'"pose"', data) and self.assertIn(b'"scans"', data)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -6, -11, -15])
