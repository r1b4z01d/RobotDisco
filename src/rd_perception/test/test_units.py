"""ROS-free tests: the S1 range correction, the Isaac tag conversion, the image rotation, the configs."""
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import yaml

from rd_perception.image_rotate import rotate, scale_to_height, square_crop
from rd_perception.scan_range_offset import OffsetTracker, apply_offset, reference_offset
from rd_perception.tag_adapter import convert

CFG = Path(__file__).resolve().parents[1] / 'config'


def test_apply_offset_shortens_and_clamps():
    inf = float('inf')
    out = apply_offset([0.83, 1.32, 0.40, inf, float('nan'), 25.0], -0.33, 0.1, 20.0)
    assert abs(out[0] - 0.50) < 1e-6 and abs(out[1] - 0.99) < 1e-6
    assert abs(out[2] - 0.1) < 1e-6          # 0.40 - 0.33 < range_min -> clamped, not dropped
    assert out[3] == inf and math.isnan(out[4]) and out[5] == 25.0    # invalid / out of range untouched
    assert apply_offset([1.0, 2.0], 0.0, 0.1, 20.0) == [1.0, 2.0]


SECTORS, EXPECTED = [(-72.0, -57.0), (57.0, 72.0)], [0.367, 0.371]


def _s1_scan(error=0.0, left=0.367, right=0.371, room=2.0):
    """A 1080-beam scan like the S1's: the room everywhere, the wheel housings in the two reference sectors."""
    deg = -179.0 + np.arange(1080) / 3.0
    r = np.full(1080, room, dtype=np.float32)
    r[(deg >= -72) & (deg < -57)] = left
    r[(deg >= 57) & (deg < 72)] = right
    return (r + error).tolist(), math.radians(-179.0), math.radians(1 / 3.0)


def test_reference_offset_measures_the_error_on_the_body():
    for error in (0.0, 0.10, 0.33):
        ranges, angle_min, increment = _s1_scan(error)
        assert abs(reference_offset(ranges, angle_min, increment, 0.15, 40.0, SECTORS, EXPECTED) + error) < 1e-3
    # the correction it gives puts the room back where it is
    ranges, angle_min, increment = _s1_scan(0.33)
    offset = reference_offset(ranges, angle_min, increment, 0.15, 40.0, SECTORS, EXPECTED)
    assert abs(apply_offset(ranges, offset, 0.15, 40.0)[0] - 2.0) < 1e-3


def test_reference_offset_refuses_a_scan_that_cannot_say():
    ranges, angle_min, increment = _s1_scan(left=0.20)            # something in front of one housing: sectors disagree
    assert reference_offset(ranges, angle_min, increment, 0.15, 40.0, SECTORS, EXPECTED) is None
    ranges, angle_min, increment = _s1_scan()
    deg = -179.0 + np.arange(1080) / 3.0
    blind = np.array(ranges)
    blind[(deg >= 57) & (deg < 68)] = float('inf')                # most of a sector without a return
    assert reference_offset(blind.tolist(), angle_min, increment, 0.15, 40.0, SECTORS, EXPECTED) is None
    assert reference_offset(ranges, angle_min, increment, 0.15, 40.0, [(200.0, 210.0)], [0.4]) is None   # no such beams


def test_offset_tracker():
    t = OffsetTracker(-0.33)
    for _ in range(4):
        assert t.update(-0.005) == -0.33 and not t.measured      # the fallback until five scans agree
    assert t.update(-0.005) == 0.0 and t.measured                 # a healthy unit: inside the deadband, untouched
    for _ in range(20):
        t.update(-0.33)
    assert abs(t.value + 0.33) < 1e-9                             # the fault back: followed within the window
    assert t.update(None) == t.value and t.update(0.40) == t.value and t.update(-0.9) == t.value
    assert t.missed == 3 and abs(t.value + 0.33) < 1e-9           # no estimate, or not this fault: held
    t.update(-0.33)
    assert t.missed == 0


def test_convert_isaac_like_message():
    pose = NS(position=NS(x=0.1, y=0.2, z=1.5), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))
    det = NS(id=7, family='tag36h11', pose=NS(header=NS(frame_id='camera_link_optical', stamp=NS(sec=1, nanosec=0)),
                                                 pose=NS(pose=pose, covariance=[0.0] * 36)))
    msg = NS(header=NS(frame_id='camera_link_optical', stamp=NS(sec=1, nanosec=0)), detections=[det])
    out = convert(msg, default_size=0.15)
    assert len(out.detections) == 1
    d = out.detections[0]
    assert d.id == 7 and d.family == 'tag36h11' and abs(d.size - 0.15) < 1e-6
    assert d.pose.header.frame_id == 'camera_link_optical' and abs(d.pose.pose.position.z - 1.5) < 1e-6


def test_rotate():
    a = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)      # 2 rows, 3 cols, rgb
    data, h, w = rotate(a.tobytes(), 2, 3, 3, 90)
    r = np.frombuffer(data, np.uint8).reshape(h, w, 3)
    assert (h, w) == (3, 2)
    assert (r[0, 0] == a[1, 0]).all()          # 90 deg clockwise: bottom-left becomes top-left
    data, h, w = rotate(a.tobytes(), 2, 3, 3, 180)
    assert (np.frombuffer(data, np.uint8).reshape(h, w, 3)[0, 0] == a[1, 2]).all()
    data, h, w = rotate(a.tobytes(), 2, 3, 3, 0)
    assert data == a.tobytes()


def test_square_crop():
    a = np.arange(5 * 3 * 3, dtype=np.uint8).reshape(5, 3, 3)      # portrait: 5 rows, 3 cols (the scene camera after 270 deg)
    data, h, w = square_crop(a.tobytes(), 5, 3, 3, 0.0)
    assert (h, w) == (3, 3) and (np.frombuffer(data, np.uint8).reshape(3, 3, 3) == a[0:3]).all()      # 0 = the top
    data, h, w = square_crop(a.tobytes(), 5, 3, 3, 1.0)
    assert (np.frombuffer(data, np.uint8).reshape(3, 3, 3) == a[2:5]).all()                           # 1 = the bottom
    data, h, w = square_crop(a.tobytes(), 5, 3, 3, 0.5)
    assert (np.frombuffer(data, np.uint8).reshape(3, 3, 3) == a[1:4]).all()                           # the middle
    b = np.ascontiguousarray(a.transpose(1, 0, 2))                                                    # landscape: the cut slides along the width
    data, h, w = square_crop(b.tobytes(), 3, 5, 3, 1.0)
    assert (h, w) == (3, 3) and (np.frombuffer(data, np.uint8).reshape(3, 3, 3) == b[:, 2:5]).all()
    data, h, w = square_crop(a[:3].tobytes(), 3, 3, 3, 0.4)                                           # already square: untouched
    assert data == a[:3].tobytes()


def test_scale_to_height():
    a = np.zeros((720, 1280, 3), np.uint8)
    a[:, 640:] = 200                                                   # right half bright
    data, h, w = scale_to_height(a.tobytes(), 720, 1280, 3, 540)      # the wrist: 1280x720 -> 960x540
    r = np.frombuffer(data, np.uint8).reshape(h, w, 3)
    assert (h, w) == (540, 960) and r[270, 100, 0] == 0 and r[270, 900, 0] == 200
    data, h, w = scale_to_height(np.zeros((720, 720, 3), np.uint8).tobytes(), 720, 720, 3, 540)   # a square stays square
    assert (h, w) == (540, 540)
    same = a.tobytes()
    assert scale_to_height(same, 720, 1280, 3, 0)[0] is same and scale_to_height(same, 720, 1280, 3, 720)[0] is same


def test_filter_configs_are_laser_filters_chains():
    for name in ('s1_filter.yaml', 's2e_filter.yaml'):
        c = yaml.safe_load((CFG / name).read_text())['/**']['ros__parameters']      # any node name (the launch names the chains)
        assert c['filter1']['type'] == 'laser_filters/LaserScanPolygonFilter'
        poly = yaml.safe_load(c['filter1']['params']['polygon'].replace('] [', '], ['))
        assert len(poly) >= 4
        assert c['filter2']['type'] == 'laser_filters/LaserScanSpeckleFilter'


def test_robot_yaml_has_every_sensor():
    from rd_bringup import config
    s = config.get('sensors')
    assert s['rplidar_s1']['serial_port'].startswith('/dev/serial/by-id/')
    assert s['rplidar_s2e']['ip'] and s['velodyne']['ip']
    cams = s['usb_cameras']
    assert '{port}' in cams['by_path'] and {'scene', 'wrist', 'side'} <= set(cams)
    assert {cams[c]['rotate'] for c in ('scene', 'wrist', 'side')} <= {0, 90, 180, 270}
