"""The rig's webcams: synthetic rgb8 images on /cameras/<name>/image (a moving pattern with the camera
name and the frame number drawn in, so a recording can be told apart frame by frame)."""
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class FakeCameras(Node):
    def __init__(self):
        super().__init__("fake_cameras")
        self.declare_parameter("cameras", ["scene", "wrist", "side"])
        self.declare_parameter("width", 320)
        self.declare_parameter("height", 180)
        self.declare_parameter("fps", 15.0)
        self.w, self.h = int(self.get_parameter("width").value), int(self.get_parameter("height").value)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pubs = {c: self.create_publisher(Image, f"/cameras/{c}/image", qos) for c in self.get_parameter("cameras").value}
        self.n = 0
        self.create_timer(1.0 / float(self.get_parameter("fps").value), self.tick)

    def tick(self):
        self.n += 1
        stamp = self.get_clock().now().to_msg()
        for i, (cam, pub) in enumerate(self.pubs.items()):
            img = np.zeros((self.h, self.w, 3), np.uint8)
            img[:, :, i % 3] = 60
            x = int((self.n * 3 + i * 50) % self.w)
            cv2.rectangle(img, (x, 20), (x + 30, 50), (255, 255, 255), -1)
            cv2.putText(img, f"{cam} {self.n}", (5, self.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            m = Image(height=self.h, width=self.w, encoding="rgb8", is_bigendian=0, step=self.w * 3, data=img.tobytes())
            m.header.stamp = stamp
            m.header.frame_id = f"{cam}_camera"
            pub.publish(m)


def main():
    rclpy.init()
    node = FakeCameras()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
