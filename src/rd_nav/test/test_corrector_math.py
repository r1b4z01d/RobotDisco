"""The tag corrector's maths and consensus window, without ROS."""
import math

import numpy as np

from rd_nav.tag_corrector import Consensus, mat_to_quat, mat_to_xy_yaw, quat_to_mat, xy_yaw_to_mat


def test_quat_roundtrip():
    for yaw in (0.0, 0.7, -2.5, 3.1):
        M = xy_yaw_to_mat(1.0, -2.0, yaw)
        q = mat_to_quat(M[:3, :3])
        assert np.allclose(quat_to_mat(*q), M[:3, :3], atol=1e-9)
        x, y, y2 = mat_to_xy_yaw(M)
        assert (x, y) == (1.0, -2.0) and abs(y2 - yaw) < 1e-9


def test_teach_then_use_recovers_the_base_pose():
    T_map_base = xy_yaw_to_mat(3.0, 1.0, 0.5)
    T_base_tag = xy_yaw_to_mat(1.2, 0.3, -1.3)
    T_map_tag = T_map_base @ T_base_tag                       # teach
    back = T_map_tag @ np.linalg.inv(T_base_tag)              # use
    assert np.allclose(back, T_map_base, atol=1e-12)


def test_consensus_needs_agreement():
    c = Consensus(min_detections=3, window_sec=1.0, spread_xy=0.15, spread_yaw=0.1)
    c.add(0.0, 1, 1.0, 1.0, 0.0)
    c.add(0.1, 1, 1.05, 1.0, 0.02)
    assert c.estimate() is None                              # too few
    c.add(0.2, 2, 1.0, 1.05, -0.02)
    x, y, yaw = c.estimate()
    assert abs(x - 1.0167) < 1e-3 and abs(y - 1.0167) < 1e-3 and abs(yaw) < 0.01
    assert c.tags() == [1, 2]
    c.add(0.3, 1, 2.0, 1.0, 0.0)                              # an outlier breaks agreement
    assert c.estimate() is None
    c.add(2.0, 1, 5.0, 5.0, 1.0)                              # the window expires the old ones
    assert len(c.window) == 1 and c.estimate() is None


def test_consensus_wraps_yaw():
    c = Consensus(min_detections=2, window_sec=5.0, spread_xy=1.0, spread_yaw=0.2)
    c.add(0.0, 1, 0, 0, math.pi - 0.05)
    c.add(0.1, 1, 0, 0, -math.pi + 0.05)
    _, _, yaw = c.estimate()
    assert abs(abs(yaw) - math.pi) < 1e-6
