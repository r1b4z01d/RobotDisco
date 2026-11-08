"""Publish a rendered AprilTag (tag36h11) on the RealSense colour topics, with a CameraInfo, so the Isaac ROS
rectify + AprilTag pipeline (rd_perception isaac.launch.py apriltag:=true) has a tag to detect on a box
without a camera. Validated on the GPU box 2026-11-08: tag 3 detected at the image centre, 0.571 m (render
geometry: 0.57 m), 10 Hz at every stage. 1280x720 rgb8 at 10 Hz; the tag's edge covers `tag_px` pixels at image centre, which at
the focal length below puts a 0.15 m tag about `dist` metres in front of the camera.

    ros2 run rd_perception fake_tag_image [--id 3] [--tag-px 240] [--hz 10]
"""
import argparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rd_bringup.process import exit_with_parent
from sensor_msgs.msg import CameraInfo, Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=3)
    ap.add_argument("--tag-px", type=int, default=240)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    a = ap.parse_args()

    W, H = 1280, 720
    fx = fy = 910.0  # a D435 colour stream at 1280x720 is about 910 px focal length
    cx, cy = W / 2, H / 2
    aruco = cv2.aruco
    d = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    gen = getattr(aruco, "generateImageMarker", None) or aruco.drawMarker
    # the 36h11 marker image includes its black border; AprilTag's "size" is that black square's edge
    tag = gen(d, a.id, a.tag_px)
    canvas = np.full((H, W), 200, np.uint8)  # light grey wall
    y0, x0 = int(cy - a.tag_px / 2), int(cx - a.tag_px / 2)
    pad = a.tag_px // 6  # white quiet zone around the tag
    canvas[y0 - pad:y0 + a.tag_px + pad, x0 - pad:x0 + a.tag_px + pad] = 255
    canvas[y0:y0 + a.tag_px, x0:x0 + a.tag_px] = tag
    rgb = np.ascontiguousarray(np.dstack([canvas] * 3))
    dist = 0.15 * fx / a.tag_px
    print(f"tag36h11 id {a.id}, {a.tag_px} px edge -> a 0.15 m tag ~{dist:.2f} m ahead; {W}x{H} rgb8 at {a.hz:g} Hz", flush=True)

    rclpy.init()
    exit_with_parent()
    n = Node("tag_image_pub")
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    pub_img = n.create_publisher(Image, a.image_topic, qos)
    pub_info = n.create_publisher(CameraInfo, a.info_topic, qos)
    img = Image(height=H, width=W, encoding="rgb8", is_bigendian=0, step=W * 3, data=rgb.tobytes())
    img.header.frame_id = "camera_color_optical_frame"
    info = CameraInfo(height=H, width=W, distortion_model="plumb_bob", d=[0.0] * 5,
                      k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                      p=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0])
    info.header.frame_id = img.header.frame_id

    def tick():
        stamp = n.get_clock().now().to_msg()
        img.header.stamp = stamp
        info.header.stamp = stamp
        pub_img.publish(img)
        pub_info.publish(info)

    n.create_timer(1.0 / a.hz, tick)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
