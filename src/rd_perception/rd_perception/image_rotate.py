"""Rotate an Image topic by 90/180/270 degrees (the webcams are mounted sideways; v1 rotated in every consumer), and
optionally cut a square out of the upright frame (v1's SCENE_SQUARE / SIDE_SQUARE in its camera service).

The square matters for the policies: v1's datasets store scene and side as 540x540 squares taken from the upright
portrait frame (scene at 0.40, side at 0.30 of the way down its long axis), so a checkpoint trained on them must
see the same cut, not the whole frame letterboxed into its 512x512. `square` is that fraction; -1 leaves the frame.
`height` scales the result to that many rows (v1 stored 540: 540x540 squares, a 960x540 wrist); 0 keeps the size.
Every camera gets this node, also one with nothing to rotate or cut: /cameras/<name>/image is the topic the
recorder, the policies and the manager read, and the wrist camera had none on the first run on the robot.
"""
import numpy as np
from rd_bringup.process import exit_with_parent
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def rotate(data: bytes, height: int, width: int, channels: int, degrees: int):
    """(bytes, h, w) of the rotated image; `degrees` clockwise in {0, 90, 180, 270}."""
    a = np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels)
    k = {0: 0, 90: 3, 180: 2, 270: 1}[degrees % 360]      # np.rot90 is counter-clockwise
    r = np.ascontiguousarray(np.rot90(a, k))
    return r.tobytes(), r.shape[0], r.shape[1]


def square_crop(data: bytes, height: int, width: int, channels: int, at: float):
    """(bytes, h, w) of the largest square of the frame, placed by `at` in [0, 1] along the long axis (0 = top / left)."""
    if height == width:
        return data, height, width
    a = np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels)
    side = min(height, width)
    off = int(round((max(height, width) - side) * min(max(float(at), 0.0), 1.0)))
    r = np.ascontiguousarray(a[off:off + side, :] if height > width else a[:, off:off + side])
    return r.tobytes(), r.shape[0], r.shape[1]


def scale_to_height(data: bytes, height: int, width: int, channels: int, out_height: int):
    """(bytes, h, w) scaled to `out_height` rows, the aspect ratio kept; unchanged when out_height is 0 or equal."""
    if out_height <= 0 or out_height == height:
        return data, height, width
    import cv2
    a = np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels)
    out_w = max(1, int(round(width * out_height / height)))
    r = cv2.resize(a, (out_w, out_height), interpolation=cv2.INTER_AREA if out_height < height else cv2.INTER_LINEAR)
    r = np.ascontiguousarray(r.reshape(out_height, out_w, channels))
    return r.tobytes(), out_height, out_w


class ImageRotate(Node):
    def __init__(self):
        super().__init__("image_rotate")
        self.declare_parameter("degrees", 90)
        self.declare_parameter("square", -1.0)     # fraction along the upright frame's long axis; < 0 = no crop
        self.declare_parameter("height", 0)        # rows of the published image; 0 = as it comes
        self.degrees = int(self.get_parameter("degrees").value)
        self.square = float(self.get_parameter("square").value)
        self.height = int(self.get_parameter("height").value)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(Image, "out", qos)
        # Take the camera's frames only while somebody takes ours. Three of these at 30 fps cost most of three of
        # the Orin's cores for images nobody was looking at (2026-11-10); a subscriber now waits half a second at most.
        self.qos = qos
        self.sub = None
        self.create_timer(0.5, self.follow_demand)
        self.follow_demand()

    def follow_demand(self):
        wanted = self.pub.get_subscription_count() > 0
        if wanted and self.sub is None:
            self.sub = self.create_subscription(Image, "in", self.cb, self.qos)
        elif not wanted and self.sub is not None:
            self.destroy_subscription(self.sub)
            self.sub = None

    def cb(self, m: Image):
        ch = max(1, m.step // max(1, m.width))
        data, h, w = rotate(bytes(m.data), m.height, m.width, ch, self.degrees)
        if self.square >= 0.0:
            data, h, w = square_crop(data, h, w, ch, self.square)
        data, h, w = scale_to_height(data, h, w, ch, self.height)
        out = Image(header=m.header, height=h, width=w, encoding=m.encoding, is_bigendian=m.is_bigendian, step=w * ch, data=data)
        self.pub.publish(out)


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = ImageRotate()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
