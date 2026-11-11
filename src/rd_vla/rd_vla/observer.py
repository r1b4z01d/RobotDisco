"""The robot as an observation: cameras (rgb8 Image topics), the arm's joints (/joint_states), the end
effector pose (TF ur_base -> ur_tool0 as xyz + rotation vector) and the commanded hand curls
(/hand/state). Produces the 16-D state + images of the schema at the policy's rate. rclpy; no cv_bridge
(the v1 two-OpenCV trap), images are decoded from the message bytes directly.

The pose frame is `ur_base`, the UR controller's base frame that v1's RTDE poses (and so every v1 dataset and
checkpoint) are in; the URDF's `ur_base_link` is that frame turned 180 deg about Z (v1's home TCP (-0.189, -0.107,
0.991) read (0.205, 0.124, 0.98) in ur_base_link, 2026-11-09), and a policy fed the wrong one steers 0.3 m off.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformException, TransformListener

from .geometry import quat_to_rotvec
from .schema import CAMERA_KEYS, make_state

UR_JOINTS = ["ur_shoulder_pan_joint", "ur_shoulder_lift_joint", "ur_elbow_joint", "ur_wrist_1_joint", "ur_wrist_2_joint", "ur_wrist_3_joint"]


def image_to_rgb(m: Image) -> np.ndarray:
    """sensor_msgs/Image (rgb8 / bgr8 / mono8) -> HxWx3 uint8 RGB."""
    data = np.frombuffer(bytes(m.data), dtype=np.uint8)
    if m.encoding in ("rgb8", "bgr8"):
        a = data.reshape(m.height, m.step // 3, 3)[:, :m.width, :]
        return np.ascontiguousarray(a[..., ::-1] if m.encoding == "bgr8" else a)
    if m.encoding == "mono8":
        a = data.reshape(m.height, m.step)[:, :m.width]
        return np.ascontiguousarray(np.repeat(a[..., None], 3, axis=2))
    raise ValueError(f"unsupported encoding {m.encoding}")


class Observer:
    def __init__(self, node, cameras=CAMERA_KEYS, camera_topic="/cameras/{name}/image", base_frame="ur_base",
                 tool_frame="ur_tool0", callback_group=None):
        self.node = node
        self.cameras = tuple(cameras)
        self.base_frame, self.tool_frame = base_frame, tool_frame
        self._lock = threading.Lock()
        self._images: dict[str, tuple[float, np.ndarray]] = {}
        self._joints = None
        self._curls = [0.0, 0.0, 0.0, 0.0]
        self.tf = Buffer()
        self.tfl = None
        # Nothing is subscribed until cameras_on() and everything is dropped at cameras_off(): idle, the policy runner
        # spent half a core on three 30 fps camera streams it never looked at, and an eighth of one on the transforms
        # (88 Hz through a Python listener) and the joint states of an arm nobody was running a policy on.
        self._camera_topic, self._callback_group = camera_topic, callback_group
        self._subs: list = []

    def cameras_on(self) -> None:
        """Start observing: the cameras, the joints, the hand and the transforms."""
        if self._subs:
            return
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        cg = self._callback_group
        self._subs = [self.node.create_subscription(Image, self._camera_topic.format(name=cam), self._on_image(cam), qos, callback_group=cg)
                      for cam in self.cameras]
        self._subs.append(self.node.create_subscription(JointState, "/joint_states", self._on_joints, 10, callback_group=cg))
        self._subs.append(self.node.create_subscription(Float32MultiArray, "/hand/state", self._on_hand, 10, callback_group=cg))
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self.node)

    def cameras_off(self) -> None:
        for s in self._subs:
            self.node.destroy_subscription(s)
        self._subs = []
        if self.tfl is not None:
            self.tfl.unregister()
            self.tfl = None
        with self._lock:
            self._images.clear()               # the next run must not start on this run's last frames
            self._joints = None

    def wait_ready(self, max_age: float = 1.0, timeout: float = 6.0, cancelled=lambda: False) -> tuple[bool, str]:
        """ready(), given the time the cameras need to start flowing after cameras_on()."""
        end = time.monotonic() + timeout
        ok, why = self.ready(max_age)
        while not ok and time.monotonic() < end and not cancelled():
            time.sleep(0.1)
            ok, why = self.ready(max_age)
        return ok, why

    def _on_image(self, cam):
        def cb(m):
            try:
                img = image_to_rgb(m)
            except ValueError:
                return
            with self._lock:
                self._images[cam] = (time.monotonic(), img)
        return cb

    def _on_joints(self, m: JointState):
        pos = dict(zip(m.name, m.position))
        if all(j in pos for j in UR_JOINTS):
            with self._lock:
                self._joints = [float(pos[j]) for j in UR_JOINTS]

    def _on_hand(self, m: Float32MultiArray):
        if len(m.data) >= 4:
            with self._lock:
                self._curls = [float(v) for v in m.data[:4]]

    def ee_pose(self):
        """[x, y, z, rx, ry, rz] of the tool in the base frame, or None."""
        try:
            t = self.tf.lookup_transform(self.base_frame, self.tool_frame, self.node.get_clock().now(),
                                         timeout=__import__("rclpy").duration.Duration(seconds=0.0))
        except TransformException:
            try:
                t = self.tf.lookup_transform(self.base_frame, self.tool_frame, __import__("rclpy").time.Time())
            except TransformException:
                return None
        tr, q = t.transform.translation, t.transform.rotation
        return [tr.x, tr.y, tr.z] + quat_to_rotvec([q.x, q.y, q.z, q.w])

    def ready(self, max_age: float = 1.0) -> tuple[bool, str]:
        now = time.monotonic()
        with self._lock:
            missing = [c for c in self.cameras if c not in self._images or now - self._images[c][0] > max_age]
            joints = self._joints
        if missing:
            return False, f"no fresh image from {missing}"
        if joints is None:
            return False, "no UR joint states"
        if self.ee_pose() is None:
            return False, f"no TF {self.base_frame} -> {self.tool_frame}"
        return True, ""

    def observe(self):
        """(state[16], images{cam: rgb}, age_s) or None."""
        pose = self.ee_pose()
        with self._lock:
            if pose is None or self._joints is None or any(c not in self._images for c in self.cameras):
                return None
            images = {c: self._images[c][1] for c in self.cameras}
            age = max(time.monotonic() - self._images[c][0] for c in self.cameras)
            state = make_state(pose, self._joints, self._curls)
        return state, images, age
