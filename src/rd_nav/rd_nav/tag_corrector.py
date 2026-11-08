"""AprilTag pose corrector (v1 apriltag_relocalizer, same math).

Watches rd_interfaces/TagDetectionArray. For every detected tag with a taught map-frame pose it works out
where the robot must be on the map and, when that disagrees with AMCL by more than a threshold, publishes
the corrected pose on /initialpose.

Services (std_srvs/Trigger):
  ~/teach_landmarks   record map->tag for every visible tag from the current AMCL pose, save to tag_poses_file
  ~/relocalize_now    publish /initialpose from the current detections, ignoring threshold and rate limit

Math (T_a_b = pose of b in a, 4x4):  teach  T_map_tag = T_map_base . T_base_tag
                                     use    T_map_base = T_map_tag . inv(T_base_tag)
T_base_tag = TF(base <- camera) . detection pose. The camera->tag pose comes from the detection message,
never from the tag_N TF frames a detector may broadcast (they arrive after the message; v1 2026-11-01).
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import math
import os
import tempfile
import time
from collections import deque

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from rd_interfaces.msg import TagDetectionArray


# --- homogeneous-transform helpers (no tf_transformations dependency) ---

def quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def transform_to_mat(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    M = np.eye(4)
    M[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
    M[:3, 3] = [t.x, t.y, t.z]
    return M


def pose_to_mat(pose):
    p, q = pose.position, pose.orientation
    M = np.eye(4)
    M[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
    M[:3, 3] = [p.x, p.y, p.z]
    return M


def xy_yaw_to_mat(x, y, yaw):
    M = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    M[:2, :2] = [[c, -s], [s, c]]
    M[:2, 3] = [x, y]
    return M


def mat_to_xy_yaw(M):
    return float(M[0, 3]), float(M[1, 3]), math.atan2(M[1, 0], M[0, 0])


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def circular_mean(angles):
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))


class Consensus:
    """A window of (time, tag, x, y, yaw) estimates; the mean when enough of them agree. ROS-free."""

    def __init__(self, min_detections=3, window_sec=1.0, spread_xy=0.15, spread_yaw=0.10):
        self.min_detections, self.window_sec = min_detections, window_sec
        self.spread_xy, self.spread_yaw = spread_xy, spread_yaw
        self.window: deque = deque()

    def add(self, now, tag_id, x, y, yaw):
        self.window.append((now, tag_id, x, y, yaw))
        while self.window and now - self.window[0][0] > self.window_sec:
            self.window.popleft()

    def tags(self):
        return sorted({w[1] for w in self.window})

    def estimate(self):
        if len(self.window) < self.min_detections:
            return None
        xs = [w[2] for w in self.window]
        ys = [w[3] for w in self.window]
        yaws = [w[4] for w in self.window]
        mx, my, myaw = sum(xs) / len(xs), sum(ys) / len(ys), circular_mean(yaws)
        if max(math.hypot(x - mx, y - my) for x, y in zip(xs, ys)) > self.spread_xy:
            return None
        if max(abs(wrap(a - myaw)) for a in yaws) > self.spread_yaw:
            return None
        return mx, my, myaw


def load_tag_poses(path: str) -> dict:
    """tag id -> 4x4 T_map_tag from the YAML the teach service writes."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    data = yaml.safe_load(open(path)) or {}
    for tag_id, lm in (data.get("landmarks") or data.get("tags") or {}).items():
        pos, q = lm["position"], lm["orientation"]
        M = np.eye(4)
        M[:3, :3] = quat_to_mat(q["x"], q["y"], q["z"], q["w"])
        M[:3, 3] = [pos["x"], pos["y"], pos["z"]]
        out[int(tag_id)] = M
    return out


def save_tag_poses(path: str, poses: dict) -> None:
    out = {"tags": {}}
    for tag_id, M in sorted(poses.items()):
        qx, qy, qz, qw = mat_to_quat(M[:3, :3])
        out["tags"][int(tag_id)] = {
            "position": {"x": float(M[0, 3]), "y": float(M[1, 3]), "z": float(M[2, 3])},
            "orientation": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
        }
    header = "# Map-frame poses of AprilTag landmarks, written by tag_corrector ~/teach_landmarks.\n# Re-teach after re-mapping or moving a tag.\n"
    # write through to the real file behind a symlink install (v1 lost taught poses to a replaced symlink)
    target = os.path.realpath(path)
    d = os.path.dirname(target) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(header)
        yaml.safe_dump(out, f, sort_keys=True)
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)


class TagCorrector(Node):
    def __init__(self):
        super().__init__("tag_corrector")
        dp = self.declare_parameter
        dp("detections_topic", "/tags/detections")
        dp("tag_poses_file", "")
        dp("map_frame", "map")
        dp("base_frame", "base_footprint")
        dp("max_range", 3.0)
        dp("min_detections", 3)
        dp("window_sec", 1.0)
        dp("spread_xy", 0.15)
        dp("spread_yaw", 0.10)
        dp("auto_correct", True)
        dp("position_threshold", 0.30)
        dp("yaw_threshold", 0.20)
        dp("min_interval", 5.0)
        dp("initial_pose_std_xy", 0.10)
        dp("initial_pose_std_yaw", 0.10)
        dp("tilt_warn", 0.25)
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.tag_poses_file = p("tag_poses_file")
        self.map_frame, self.base_frame = p("map_frame"), p("base_frame")
        self.max_range = float(p("max_range"))
        self.auto_correct = bool(p("auto_correct"))
        self.position_threshold, self.yaw_threshold = float(p("position_threshold")), float(p("yaw_threshold"))
        self.min_interval = float(p("min_interval"))
        self.std_xy, self.std_yaw = float(p("initial_pose_std_xy")), float(p("initial_pose_std_yaw"))
        self.tilt_warn = float(p("tilt_warn"))
        self.consensus = Consensus(int(p("min_detections")), float(p("window_sec")), float(p("spread_xy")), float(p("spread_yaw")))
        self.tag_poses = load_tag_poses(self.tag_poses_file)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.last_detections = None
        self.last_correction = 0.0
        self.corrections = 0
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.estimate_pub = self.create_publisher(PoseStamped, "~/estimated_pose", 10)
        self.create_subscription(TagDetectionArray, p("detections_topic"), self.on_detections, 10)
        self.create_service(Trigger, "~/teach_landmarks", self.on_teach)
        self.create_service(Trigger, "~/relocalize_now", self.on_relocalize_now)
        self.get_logger().info(f"{len(self.tag_poses)} taught tag(s) from {self.tag_poses_file or '<none>'}; "
                               f"auto_correct={self.auto_correct}, thresholds {self.position_threshold} m / "
                               f"{math.degrees(self.yaw_threshold):.0f} deg")

    # ---- TF
    def lookup(self, target, source, stamp):
        return transform_to_mat(self.tf.lookup_transform(target, source, stamp))

    def current_map_base(self):
        try:
            return self.lookup(self.map_frame, self.base_frame, Time())
        except TransformException:
            return None

    def base_from_camera(self, msg):
        cam = msg.header.frame_id
        err = None
        for stamp in (Time.from_msg(msg.header.stamp), Time()):
            try:
                return self.lookup(self.base_frame, cam, stamp)
            except TransformException as e:
                err = e
        self.get_logger().warn(f"TF {self.base_frame} <- {cam}: {err}", throttle_duration_sec=5.0)
        return None

    def visible_tags(self, msg):
        if not msg.detections:
            return
        T_base_cam = self.base_from_camera(msg)
        if T_base_cam is None:
            return
        for det in msg.detections:
            pos = det.pose.pose.position
            if math.sqrt(pos.x ** 2 + pos.y ** 2 + pos.z ** 2) > self.max_range:
                continue
            yield det.id, T_base_cam @ pose_to_mat(det.pose.pose)

    # ---- detections
    def on_detections(self, msg):
        self.last_detections = msg
        now = time.monotonic()
        for tag_id, T_base_tag in self.visible_tags(msg):
            T_map_tag = self.tag_poses.get(tag_id)
            if T_map_tag is None:
                self.get_logger().info(f"tag {tag_id} seen but not taught", throttle_duration_sec=10.0)
                continue
            T_map_base = T_map_tag @ np.linalg.inv(T_base_tag)
            self.check_tilt(T_map_base, tag_id)
            x, y, yaw = mat_to_xy_yaw(T_map_base)
            self.consensus.add(now, tag_id, x, y, yaw)
        est = self.consensus.estimate()
        if est is None:
            return
        x, y, yaw = est
        self.publish_estimate(x, y, yaw)
        if not self.auto_correct or now - self.last_correction < self.min_interval:
            return
        cur = self.current_map_base()
        if cur is None:
            self.get_logger().warn("no map->base TF (AMCL down?); correcting anyway", throttle_duration_sec=10.0)
            self.correct(x, y, yaw, "no AMCL pose")
            return
        cx, cy, cyaw = mat_to_xy_yaw(cur)
        dpos, dyaw = math.hypot(x - cx, y - cy), abs(wrap(yaw - cyaw))
        if dpos > self.position_threshold or dyaw > self.yaw_threshold:
            self.correct(x, y, yaw, f"off by {dpos:.2f} m / {math.degrees(dyaw):.1f} deg")

    def check_tilt(self, T, tag_id):
        R = T[:3, :3]
        pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
        roll = math.atan2(R[2, 1], R[2, 2])
        if abs(roll) > self.tilt_warn or abs(pitch) > self.tilt_warn:
            self.get_logger().warn(f"tag {tag_id}: estimated base roll/pitch {math.degrees(roll):.0f}/"
                                   f"{math.degrees(pitch):.0f} deg; check the camera mount or re-teach",
                                   throttle_duration_sec=10.0)

    # ---- publishing
    def pose_msg(self, x, y, yaw):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose.position.x = x
        m.pose.pose.position.y = y
        m.pose.pose.orientation.z = math.sin(yaw / 2)
        m.pose.pose.orientation.w = math.cos(yaw / 2)
        cov = [0.0] * 36
        cov[0] = cov[7] = self.std_xy ** 2
        cov[35] = self.std_yaw ** 2
        m.pose.covariance = cov
        return m

    def publish_estimate(self, x, y, yaw):
        m = self.pose_msg(x, y, yaw)
        ps = PoseStamped()
        ps.header = m.header
        ps.pose = m.pose.pose
        self.estimate_pub.publish(ps)

    def correct(self, x, y, yaw, why):
        self.initialpose_pub.publish(self.pose_msg(x, y, yaw))
        self.last_correction = time.monotonic()
        self.corrections += 1
        self.get_logger().info(f"relocalized to ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg) "
                               f"from tag(s) {self.consensus.tags()}: {why}")

    # ---- services
    def on_teach(self, _req, res):
        msg = self.last_detections
        if msg is None or (self.get_clock().now() - Time.from_msg(msg.header.stamp)) > Duration(seconds=1.0):
            res.message = "no recent detections"
            return res
        T_map_base = self.current_map_base()
        if T_map_base is None:
            res.message = f"no {self.map_frame}->{self.base_frame} TF; is AMCL running and localized?"
            return res
        taught = []
        for tag_id, T_base_tag in self.visible_tags(msg):
            self.tag_poses[tag_id] = T_map_base @ T_base_tag
            taught.append(tag_id)
        if not taught:
            res.message = "no tags in view within max_range"
            return res
        if not self.tag_poses_file:
            res.success = True
            res.message = f"taught {taught} in memory only; tag_poses_file is not set"
            return res
        try:
            save_tag_poses(self.tag_poses_file, self.tag_poses)
        except OSError as e:
            res.message = f"taught {taught} but could not write {self.tag_poses_file}: {e}"
            return res
        res.success = True
        res.message = f"taught tag(s) {taught} -> {self.tag_poses_file}"
        self.get_logger().info(res.message)
        return res

    def on_relocalize_now(self, _req, res):
        est = self.consensus.estimate()
        if est is None:
            res.message = (f"need {self.consensus.min_detections} agreeing detections of a taught tag in the "
                           f"last {self.consensus.window_sec}s; have {len(self.consensus.window)}")
            return res
        self.correct(*est, "requested")
        res.success = True
        res.message = "published /initialpose"
        return res


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = TagCorrector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
