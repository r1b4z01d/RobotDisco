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


def test_gamepad_buttons_in_robot_yaml_belong_to_its_link():
    """The deadman's number depends on the link (Bluetooth 7, receiver 5): robot.yaml must name both or neither."""
    g = yaml.safe_load((CFG.parents[2] / 'config' / 'robot.yaml').read_text())['gamepad']
    known = {'8BitDo Ultimate 2C Wireless': (7, 6), '8BitDo Ultimate 2C Wireless Controller': (5, 4)}
    assert g['device_name'] in known
    if 'enable_button' in g or 'enable_turbo_button' in g:
        assert (g['enable_button'], g['enable_turbo_button']) == known[g['device_name']]
    else:
        assert g['device_name'] == '8BitDo Ultimate 2C Wireless Controller'    # joystick.yaml's defaults are the receiver's


def test_hardware_watchdog_gives_up_only_when_the_hardware_stays_down():
    from rd_base.hardware_watchdog import Watch
    w = Watch(grace_s=45.0, down_s=5.0)
    assert not w.update(0.0, None) and not w.update(30.0, "unconfigured")     # coming up: inside the grace period
    assert not w.update(31.0, "active")
    assert not w.update(40.0, "unconfigured") and not w.update(44.0, None)    # down, but not yet for 5 s
    assert not w.update(44.5, "active")                                        # came back: the clock starts again
    assert not w.update(50.0, "unconfigured") and not w.update(54.9, "unconfigured")
    assert w.update(55.0, "unconfigured")                                      # down for 5 s: give up
    never = Watch(grace_s=45.0, down_s=5.0)
    assert not never.update(0.0, None) and not never.update(44.0, None)
    assert not never.update(46.0, None) and never.update(51.0, None)           # never came up: grace, then the same 5 s


def test_joy_guard_lets_no_stale_pad_state_through():
    from rd_base.joy_guard import Guard, device_present
    g = Guard(rest=0.1)
    rest = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert g.update(True, rest, [0] * 8) == ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0] * 8)      # triggers came up at 1.0: zero until they move
    assert g.update(True, [0.0, 0.5, 0, 0, 1.0, 1.0], [0] * 7 + [1])[0][1] == 0.5           # a centred stick is live at once
    # the pad drops with the stick forward and the deadman held: joy goes on publishing exactly that
    stale_axes, stale_buttons = [0.0, 0.8, 0, 0, 1.0, 1.0], [0] * 7 + [1]
    assert g.update(False, stale_axes, stale_buttons) == ([0.0] * 6, [0] * 8)                # gone: neutral
    # it is back, joy still holds the old state: nothing of it passes
    assert g.update(True, stale_axes, stale_buttons) == ([0.0] * 6, [0] * 8)
    # the deadman is released and pressed again, the stick has not moved: still no drive
    assert g.update(True, stale_axes, [0] * 8) == ([0.0] * 6, [0] * 8)
    assert g.update(True, stale_axes, stale_buttons) == ([0.0] * 6, [0] * 7 + [1])
    # the stick moves: from now on it is the real stick
    assert g.update(True, [0.0, 0.3, 0, 0, 1.0, 1.0], stale_buttons)[0][1] == 0.3
    listing = 'I: Bus=0005\nN: Name="8BitDo Ultimate 2C Wireless"\nH: Handlers=js1 event13\n'
    assert device_present("8BitDo Ultimate 2C Wireless", listing) and not device_present("8BitDo Ultimate 2C Wireless Controller", listing)


def test_battery_monitor_parses_the_boards_line():
    from rd_base.battery_monitor import parse, percentage
    s = parse(b'{"Voltage":25.31,"Current":-3120.5,"Temperature":31.2}')
    assert abs(s["voltage"] - 25.31) < 1e-6 and abs(s["current_a"] + 3.1205) < 1e-6 and abs(s["temperature"] - 31.2) < 1e-6
    assert parse(b'{"Voltage":25.3}') == {"voltage": 25.3, "current_a": 0.0, "temperature": None}
    for junk in (b"", b"hello", b'{"Current":1}', b'{"Voltage":"x"}', b'{"Voltage":-5}', b'{"Voltage":NaN}', b"\xff\xfe"):
        assert parse(junk) is None
    assert percentage(25.0, 20.0, 30.0) == 0.5 and percentage(35.0, 20.0, 30.0) == 1.0 and percentage(10.0, 20.0, 30.0) == 0.0
    assert percentage(25.0, None, None) is None and percentage(25.0, 0.0, 0.0) is None


def test_battery_alerts_hold_then_speak_once_and_repeat_only_when_critical():
    from rd_base.battery_monitor import Alerts, sentence
    a = Alerts(low_v=23.0, critical_v=21.0, hold_s=10.0, hysteresis_v=0.3, repeat_s=300.0)
    assert a.update(25.0, 0) is None
    assert a.update(22.5, 1) is None and a.update(25.0, 5) is None          # a sag of four seconds: nothing
    assert a.update(22.5, 20) is None and a.update(22.6, 29) is None
    assert a.update(22.5, 30) == "low"                                       # held ten seconds: said once
    assert all(a.update(22.5, t) is None for t in range(31, 400, 10))        # and not again
    assert a.update(23.2, 400) is None and a.update(23.2, 420) is None and a.level == "low"     # inside the hysteresis
    assert a.update(20.5, 430) is None and a.update(20.5, 440) == "critical"
    assert a.update(20.5, 500) is None and a.update(20.5, 740) == "critical"  # every five minutes while it lasts
    assert a.update(25.0, 750) is None and a.update(25.0, 760) is None and a.level == "ok"      # charged: quiet
    assert a.update(22.0, 800) is None and a.update(22.0, 810) == "low"      # and it warns again next time
    quiet = Alerts(None, None)
    assert all(quiet.update(1.0, t) is None for t in range(0, 100, 5))       # no levels configured: never
    assert sentence("low", 22.5, None) == "My battery is low, 22.5 volts."
    assert "20 percent" in sentence("critical", 21.0, 0.2) and "charge me" in sentence("critical", 21.0, 0.2)
