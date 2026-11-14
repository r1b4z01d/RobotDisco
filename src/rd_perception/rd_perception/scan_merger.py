"""One 360 degree scan from all the lidars, for localisation.

AMCL takes a single LaserScan. v1 gave it the back lidar alone, so localisation saw half the room and froze the day
that lidar dropped off the network (the robot, 2026-11-13). This node merges the front and back RPLidars' filtered
scans and the Velodyne's scan into one, in the base frame:

  sub  the `scans` topics (sensor_msgs/LaserScan, each in its own frame)
  pub  /scan_merged      sensor_msgs/LaserScan in `frame`, `rate_hz`, the nearest return in each `angle_increment` bin

Each scan is carried to the output's moment through the odometry frame (where the base was when that scan was taken
against where it is now), so three lidars that fire at different instants do not smear into each other while the robot
turns. A lidar that goes quiet is left out after `max_age` and the others carry on. Returns inside `body_box` are the
robot itself (the Velodyne sees the arm and the column; the RPLidars' own body filters have already run).

Only localisation reads this. The costmaps and the collision monitor keep the separate scans, and mapping keeps the
back lidar: lidars at three heights do not draw one clean map.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import math

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


def scan_points(ranges, angle_min: float, angle_increment: float, range_min: float, range_max: float) -> np.ndarray:
    """(n, 3) points in the scan's own frame for its valid returns."""
    r = np.asarray(ranges, dtype=np.float32)
    ok = np.isfinite(r) & (r >= max(range_min, 0.0)) & (r <= range_max) & (r > 0.0)
    a = angle_min + angle_increment * np.nonzero(ok)[0].astype(np.float32)
    r = r[ok]
    return np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], axis=1)


def transform_matrix(t) -> np.ndarray:
    """4x4 from a geometry_msgs Transform."""
    q, p = t.rotation, t.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), p.x],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), p.y],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), p.z],
                     [0.0, 0.0, 0.0, 1.0]])


def merge(clouds, bins: int, range_min: float, range_max: float, body_box=None) -> np.ndarray:
    """`bins` ranges over -pi..pi (inf where nothing was seen) from point sets already in the output frame: the
    nearest return in each bin, without the points inside body_box = (xmin, xmax, ymin, ymax)."""
    out = np.full(bins, np.inf, dtype=np.float32)
    for pts in clouds:
        if pts is None or not len(pts):
            continue
        x, y = pts[:, 0], pts[:, 1]
        keep = np.ones(len(pts), dtype=bool)
        if body_box is not None and len(body_box) == 4:
            keep &= ~((x >= body_box[0]) & (x <= body_box[1]) & (y >= body_box[2]) & (y <= body_box[3]))
        r = np.hypot(x, y)
        keep &= (r >= range_min) & (r <= range_max)
        if not keep.any():
            continue
        idx = np.floor((np.arctan2(y[keep], x[keep]) + math.pi) / (2 * math.pi) * bins).astype(np.int64) % bins
        np.minimum.at(out, idx, r[keep].astype(np.float32))
    return out


class ScanMerger(Node):
    def __init__(self):
        super().__init__("scan_merger")
        dp = self.declare_parameter
        dp("scans", ["/rp_s1_lidar/scan_filtered", "/rp_s2e_lidar/scan_filtered", "/velodyne/scan"])
        dp("output_topic", "/scan_merged")
        dp("frame", "base_footprint")
        dp("odom_frame", "odom")
        dp("rate_hz", 10.0)
        dp("angle_increment_deg", 0.5)
        dp("range_min", 0.15)
        dp("range_max", 30.0)
        dp("max_age", 0.5)                                  # s: a lidar silent for longer is left out
        dp("body_box", [-0.70, 0.60, -0.45, 0.45])           # m, base frame: the robot's own outline with the arm tucked
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.frame, self.odom_frame = str(p("frame")), str(p("odom_frame"))
        self.bins = max(8, int(round(360.0 / float(p("angle_increment_deg")))))
        self.range_min, self.range_max, self.max_age = float(p("range_min")), float(p("range_max")), float(p("max_age"))
        self.body_box = [float(v) for v in p("body_box")]
        self.tf = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf, self)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.latest: dict = {}                               # topic -> (stamp msg, frame, points in the scan's frame)
        for topic in p("scans"):
            self.create_subscription(LaserScan, topic, lambda m, t=topic: self.on_scan(t, m), qos)
        self.pub = self.create_publisher(LaserScan, str(p("output_topic")), 10)
        self.missing: set = set()
        self.create_timer(1.0 / float(p("rate_hz")), self.tick)

    def on_scan(self, topic: str, m: LaserScan):
        pts = scan_points(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max if m.range_max > 0 else self.range_max)
        self.latest[topic] = (m.header.stamp, m.header.frame_id, pts)

    def to_base(self, frame: str, stamp, out_stamp):
        """4x4 that carries points taken in `frame` at `stamp` to the base at `out_stamp`: through the odometry frame
        when it can be had, else as if the robot had not moved in between."""
        try:
            t = self.tf.lookup_transform_full(self.frame, Time.from_msg(out_stamp), frame, Time.from_msg(stamp), self.odom_frame)
        except TransformException:
            t = self.tf.lookup_transform(self.frame, frame, Time())
        return transform_matrix(t.transform)

    def tick(self):
        now = self.get_clock().now()
        fresh = {}
        for topic, (stamp, frame, pts) in self.latest.items():
            age = (now - Time.from_msg(stamp)).nanoseconds * 1e-9
            if -1.0 < age <= self.max_age:
                fresh[topic] = (stamp, frame, pts)
        gone = set(self.latest) - set(fresh)
        if gone != self.missing:
            for topic in gone - self.missing:
                self.get_logger().warn(f"{topic} is silent: left out of the merged scan")
            for topic in self.missing - gone:
                self.get_logger().info(f"{topic} is back in the merged scan")
            self.missing = gone
        if not fresh:
            return
        out_stamp = max((s for s, _, _ in fresh.values()), key=lambda s: (s.sec, s.nanosec))
        clouds = []
        for topic, (stamp, frame, pts) in fresh.items():
            try:
                T = self.to_base(frame, stamp, out_stamp)
            except TransformException as e:
                self.get_logger().warn(f"no transform {self.frame} <- {frame}: {e}", throttle_duration_sec=5.0)
                continue
            clouds.append(pts @ T[:3, :3].T + T[:3, 3])
        if not clouds:
            return
        m = LaserScan()
        m.header.stamp, m.header.frame_id = out_stamp, self.frame
        m.angle_increment = 2 * math.pi / self.bins
        m.angle_min = -math.pi + m.angle_increment / 2            # each range sits at its bin's centre
        m.angle_max = m.angle_min + m.angle_increment * (self.bins - 1)
        m.range_min, m.range_max = self.range_min, self.range_max
        m.scan_time = 0.1
        m.ranges = merge(clouds, self.bins, self.range_min, self.range_max, self.body_box).tolist()
        self.pub.publish(m)


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = ScanMerger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
