"""The Hand tab's command builder: curl and abduction to the J: line (ROS-free)."""
import pytest

from rd_arm.hand_protocol import curls_to_offsets, flex_abduct_to_offsets
from rd_manager.hand_api import build_command


def test_no_abduction_is_the_plain_curl_line():
    cmd = build_command([0.0, 0.5, 1.0, 0.25], [0, 0, 0, 0], 200)
    assert cmd["offsets"] == [round(o, 2) for o in curls_to_offsets([0.0, 0.5, 1.0, 0.25])]
    assert cmd["line"].startswith("J:") and cmd["line"].endswith(",200")


def test_abduction_moves_both_servos_of_a_finger_the_same_way():
    flat = build_command([0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0])["offsets"]
    spread = build_command([0.5, 0.5, 0.5, 0.5], [1.0, -1.0, 0.5, 0.0])["offsets"]
    assert [round(s - f, 2) for s, f in zip(spread, flat)] == [20.0, 20.0, -20.0, -20.0, 10.0, 10.0, 0.0, 0.0]
    # flex (A - B) / 2 is untouched, abduct (A + B) / 2 is what was asked
    assert all(abs((spread[i] - spread[i + 1]) - (flat[i] - flat[i + 1])) < 1e-6 for i in (0, 2, 4, 6))
    assert flex_abduct_to_offsets([0.3] * 4, [0.0] * 4) == pytest.approx(curls_to_offsets([0.3] * 4))


def test_clamps_and_refuses_wrong_counts():
    cmd = build_command([2.0, -1.0, 0.5, 0.5], [3.0, -3.0, 0, 0], 99999)
    assert cmd["curls"][:2] == [1.0, 0.0] and cmd["abduct"][:2] == [1.0, -1.0] and cmd["speed"] == 2000
    with pytest.raises(ValueError):
        build_command([0.0] * 3)
    with pytest.raises(ValueError):
        build_command([0.0] * 4, [0.0] * 3)
