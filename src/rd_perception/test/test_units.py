"""ROS-free tests: the S1 range correction, the Isaac tag conversion, the image rotation, the configs."""
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import yaml

from rd_perception.image_rotate import rotate
from rd_perception.scan_range_offset import apply_offset
from rd_perception.tag_adapter import convert

CFG = Path(__file__).resolve().parents[1] / 'config'


def test_apply_offset_shortens_and_clamps():
    inf = float('inf')
    out = apply_offset([0.83, 1.32, 0.40, inf, float('nan'), 25.0], -0.33, 0.1, 20.0)
    assert abs(out[0] - 0.50) < 1e-6 and abs(out[1] - 0.99) < 1e-6
    assert abs(out[2] - 0.1) < 1e-6          # 0.40 - 0.33 < range_min -> clamped, not dropped
    assert out[3] == inf and math.isnan(out[4]) and out[5] == 25.0    # invalid / out of range untouched
    assert apply_offset([1.0, 2.0], 0.0, 0.1, 20.0) == [1.0, 2.0]


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
