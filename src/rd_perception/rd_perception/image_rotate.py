"""Rotate an Image topic by 90/180/270 degrees (the webcams are mounted sideways; v1 rotated in every consumer)."""
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


class ImageRotate(Node):
    def __init__(self):
        super().__init__("image_rotate")
        self.declare_parameter("degrees", 90)
        self.degrees = int(self.get_parameter("degrees").value)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(Image, "out", qos)
        self.create_subscription(Image, "in", self.cb, qos)

    def cb(self, m: Image):
        ch = max(1, m.step // max(1, m.width))
        data, h, w = rotate(bytes(m.data), m.height, m.width, ch, self.degrees)
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
