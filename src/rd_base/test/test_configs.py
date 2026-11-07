"""The YAML files parse and carry the parameters the launch relies on."""
from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parents[1] / 'config'


def load(name):
    return yaml.safe_load((CFG / name).read_text())


def test_controllers():
    c = load('controllers.yaml')
    cm = c['controller_manager']['ros__parameters']
    assert cm['diff_cont']['type'] == 'diff_drive_controller/DiffDriveController'
    d = c['diff_cont']['ros__parameters']
    assert d['use_stamped_vel'] is True
    assert abs(d['wheel_separation'] - 0.563) < 1e-6 and abs(d['wheel_radius'] - 0.1592) < 1e-6


def test_twist_mux_priorities():
    t = load('twist_mux.yaml')['twist_mux']['ros__parameters']['topics']
    assert t['joystick']['priority'] > t['web']['priority'] > t['navigation']['priority']
    assert {t[k]['topic'] for k in t} == {'cmd_vel_nav', 'cmd_vel_web', 'cmd_vel_joy'}


def test_collision_monitor_jazzy_names():
    m = load('collision_monitor.yaml')['collision_monitor']['ros__parameters']
    assert m['cmd_vel_in_topic'] == 'cmd_vel_raw'
    assert m['cmd_vel_out_topic'] == '/diff_cont/cmd_vel'
    assert m['enable_stamped_cmd_vel'] is True
    for p in m['polygons']:
        assert 'min_points' in m[p] and 'max_points' not in m[p], p
        pts = yaml.safe_load(m[p]['points'])
        assert len(pts) >= 3 and all(len(v) == 2 for v in pts)


def test_joystick_deadman():
    j = load('joystick.yaml')['teleop_node']['ros__parameters']
    assert j['require_enable_button'] is True and j['enable_button'] >= 0
    assert j['publish_stamped_twist'] is True
