"""ROS-free tests: the hand protocol, the hand link against the fake hand, the dashboard client against the
fake dashboard (incl. protective-stop recovery), the motion library."""
import math
import time
from pathlib import Path

import pytest

from rd_arm import motions
from rd_arm.dashboard import UrDashboard
from rd_arm.fake_dashboard import FakeDashboard
from rd_arm.fake_hand import FakeHand
from rd_arm.hand_link import HandLink
from rd_arm.hand_protocol import (JOINT_CLOSE_OFFSETS, JOINT_OPEN_OFFSETS, curls_to_offsets, format_joint_command,
                                  offsets_to_curls, parse_joint_command, parse_state_line)

PKG = Path(__file__).resolve().parents[1]


def test_protocol_roundtrip():
    assert curls_to_offsets([0, 0, 0, 0]) == list(JOINT_OPEN_OFFSETS)
    assert curls_to_offsets({"index": 1, "middle": 1, "ring": 1, "thumb": 1}) == list(JOINT_CLOSE_OFFSETS)
    half = curls_to_offsets([0.5, 0, 0, 0])
    assert half[0] == pytest.approx(27.5) and half[1] == pytest.approx(-27.5) and half[2:] == list(JOINT_OPEN_OFFSETS[2:])
    assert offsets_to_curls(half) == pytest.approx([0.5, 0, 0, 0])
    line = format_joint_command(half, 9999)
    assert line.startswith("J:27.5,-27.5,") and line.endswith(",2000\n")      # speed clamped to the firmware range
    assert parse_joint_command(line) == ([27.5, -27.5, -35.0, 35.0, -35.0, 35.0, -35.0, 35.0], 2000)
    assert parse_joint_command("K:123") is None and parse_joint_command("J:1,2") is None
    assert parse_state_line("S:0.1,0.2,0.3,0.4,0,0,0,0") == ([0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0])
    with pytest.raises(ValueError):
        curls_to_offsets([1, 2, 3])


def test_hand_link_against_fake_hand():
    hand = FakeHand(stream_state=True).start()
    try:
        link = HandLink("127.0.0.1", hand.port, probe_timeout=3.0).connect()
        assert link.firmware_uptime_ms is not None
        link.send_curls([1, 1, 1, 1])
        link.send_curls([0, 0.5, 0, 0], speed=500)
        time.sleep(0.2)
        assert hand.commands == 2 and hand.speed == 500
        assert hand.curls == pytest.approx([0, 0.5, 0, 0])
        state = link.read_state()
        assert state is not None and state[0] == pytest.approx([0, 0.5, 0, 0], abs=1e-3)
        assert link.send_errors == 0 and link.sent == 2
        link.close()
    finally:
        hand.stop()


def test_hand_link_probe_survives_loss():
    hand = FakeHand(loss=0.5).start()
    try:
        link = HandLink("127.0.0.1", hand.port, timeout=0.1, probe_timeout=8.0).connect()
        assert link.connected
        link.close()
    finally:
        hand.stop()


def test_hand_link_refuses_a_closed_port():
    with pytest.raises(RuntimeError, match="refused|did not answer"):
        HandLink("127.0.0.1", 1, timeout=0.1, probe_timeout=1.0).connect()


def test_dashboard_power_up_from_off():
    d = FakeDashboard(brake_delay=0.05).start()
    try:
        dash = UrDashboard("127.0.0.1", d.port)
        st = dash.status()
        assert st["remote"] and st["robotmode"] == "POWER_OFF" and st["safety"] == "NORMAL" and not st["running"]
        assert "powered up" in dash.power_up(timeout=5)
        assert dash.status()["robotmode"] == "RUNNING"
        assert dash.power_up() == "UR already RUNNING"
        assert dash.power_off() == "powered off"
        assert "power on" in d.commands and "brake release" in d.commands and "power off" in d.commands
    finally:
        d.stop()


def test_dashboard_clears_a_protective_stop_and_a_fault():
    d = FakeDashboard(safety="PROTECTIVE_STOP", robotmode="IDLE", brake_delay=0.05).start()
    try:
        dash = UrDashboard("127.0.0.1", d.port)
        msg = dash.power_up(timeout=5)
        assert "cleared protective stop" in msg and dash.status()["robotmode"] == "RUNNING"
        d.safety, d.robotmode = "FAULT", "POWER_OFF"
        msg = dash.power_up(timeout=5)
        assert "restarted the safety" in msg and dash.status()["robotmode"] == "RUNNING"
    finally:
        d.stop()


def test_dashboard_needs_remote_control():
    d = FakeDashboard(remote=False).start()
    try:
        with pytest.raises(RuntimeError, match="Remote Control"):
            UrDashboard("127.0.0.1", d.port).power_up()
    finally:
        d.stop()


def test_dashboard_unreachable():
    with pytest.raises(OSError):
        UrDashboard("127.0.0.1", 1, timeout=0.2).status()


def test_motion_library(monkeypatch):
    monkeypatch.delenv("RD_MOTIONS_DIR", raising=False)
    assert "wave" in motions.list_motions()
    m = motions.load_motion("wave")
    assert m["rate"] == 50.0 and len(m["q"]) > 100 and motions.duration(m) > 2.0
    assert motions.max_joint_step(m) < 0.05                       # a 50 Hz recording never jumps
    capped, n = motions.cap_steps([[0] * 6, [1] * 6, [1.1] * 6], 0.5)
    assert n == 2 and capped[1] == [0.5] * 6 and capped[2] == [1.0] * 6     # each step bounded by 0.5 from the previous OUTPUT
    rs = motions.resample(m["q"], m["rate"], 2.0)
    assert rs[1][0] == pytest.approx(0.01) and rs[-1][0] == pytest.approx((len(m["q"]) - 1) * 0.01)
    poses = motions.load_poses()
    assert set(poses) >= {"vla_home", "transport"} and len(poses["vla_home"]) == 6
    assert poses["vla_home"] == pytest.approx([4.010228, -1.322734, -0.822906, -1.143980, 2.198752, -0.787094], abs=1e-6)  # the robot's recorded home
    assert motions.joint_distance(poses["vla_home"], m["q"][0]) > 1.0      # the wave starts elsewhere: a Wave from home begins with a MoveIt approach
    with pytest.raises(FileNotFoundError):
        motions.load_motion("nope")
