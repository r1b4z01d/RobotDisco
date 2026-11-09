#!/usr/bin/env python3
"""Ground-truth check of the simulated robot: where AMCL puts it, and how well each lidar matches the map.

Needs the sim and the stack running. Truth = the sim's odometry (/diff_cont/odom, a displacement from the spawn
pose) composed with the spawn pose. Each lidar beam is ray-cast against the map file from the TRUE lidar pose
(truth base pose + the static base_footprint -> lidar TF) and compared with the measured range. Why the truth
pose and not AMCL's: a localization error looks exactly like a lidar error (2026-11-08: the same scans scored 23-41%
within 10 cm of the map against AMCL's pose and 91-94% within 5 cm against the truth).

    python3 sim_truth_check.py [--spawn X Y YAW] [--map zone_one]
"""
import argparse
import math
import re
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def load_map(name):
    from ament_index_python.packages import get_package_share_directory
    d = Path(get_package_share_directory("rd_nav")) / "maps"
    meta = yaml.safe_load((d / f"{name}.yaml").read_text())
    b = (d / meta["image"]).read_bytes()
    m = re.match(rb"P5\s+(?:#.*\s+)*(\d+)\s+(\d+)\s+(\d+)\s", b)
    w, h = int(m.group(1)), int(m.group(2))
    return np.frombuffer(b[m.end():], np.uint8).reshape(h, w) < 89, meta["resolution"], meta["origin"][0], meta["origin"][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spawn", nargs=3, type=float, default=[0.6957, 0.9208, 0.0548])
    ap.add_argument("--map", default="zone_one")
    a = ap.parse_args()
    occ, res, ox, oy = load_map(a.map)
    H, W = occ.shape

    def cast(x, y, ang, rmax=3.5):
        dx, dy = math.cos(ang), math.sin(ang)
        for k in range(int(rmax / 0.01)):
            c, r = int((x + dx * k * 0.01 - ox) / res), H - 1 - int((y + dy * k * 0.01 - oy) / res)
            if not (0 <= c < W and 0 <= r < H):
                return None
            if occ[r, c]:
                return k * 0.01
        return None

    rclpy.init()
    n = rclpy.create_node("sim_truth_check")
    buf = Buffer()
    TransformListener(buf, n)
    got, odom, amcl = {}, {}, {}
    n.create_subscription(Odometry, "/diff_cont/odom", lambda m: odom.update(p=(m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))), 10)
    n.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", lambda m: amcl.update(p=(m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))),
                          QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
    for name in ("rp_s1_lidar", "rp_s2e_lidar"):
        n.create_subscription(LaserScan, f"/{name}/scan", (lambda k: lambda m: got.__setitem__(k, m))(name), 10)
    t0 = time.time()
    while time.time() - t0 < 7:
        rclpy.spin_once(n, timeout_sec=0.1)
    if "p" not in odom:
        raise SystemExit("no /diff_cont/odom: is the sim running?")
    sx, sy, sa = a.spawn
    o = odom["p"]
    bx, by, ba = sx + math.cos(sa) * o[0] - math.sin(sa) * o[1], sy + math.sin(sa) * o[0] + math.cos(sa) * o[1], sa + o[2]
    print(f"truth : ({bx:.3f}, {by:.3f}) yaw {math.degrees(ba):.1f} deg")
    if "p" in amcl:
        p = amcl["p"]
        dyaw = (p[2] - ba + math.pi) % (2 * math.pi) - math.pi
        print(f"AMCL  : ({p[0]:.3f}, {p[1]:.3f}) yaw {math.degrees(p[2]):.1f} deg   error {math.hypot(p[0] - bx, p[1] - by):.3f} m, {math.degrees(dyaw):.1f} deg")
    for name, m in sorted(got.items()):
        tl = buf.lookup_transform("base_footprint", m.header.frame_id, rclpy.time.Time())
        t, ly = tl.transform.translation, yaw_of(tl.transform.rotation)
        lx, lyy, lyaw = bx + math.cos(ba) * t.x - math.sin(ba) * t.y, by + math.sin(ba) * t.x + math.cos(ba) * t.y, ba + ly
        errs = []
        for i in range(0, len(m.ranges), 2):
            r = m.ranges[i]
            if not (m.range_min < r < 2.8):
                continue
            e = cast(lx, lyy, m.angle_min + i * m.angle_increment + lyaw)
            if e is not None and e < 2.8 and abs(r - e) < 0.8:
                errs.append(r - e)
        e = np.array(errs)
        if len(e):
            print(f"{name:12s}: {len(e):4d} beams vs the map from the true pose: median {np.median(e):+.3f} m, within 5 cm {100 * np.mean(np.abs(e) < 0.05):.0f}%, within 10 cm {100 * np.mean(np.abs(e) < 0.10):.0f}%")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
