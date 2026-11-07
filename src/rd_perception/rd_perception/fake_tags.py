"""The rig's AprilTag detector: publishes rd_interfaces/TagDetectionArray for tags placed in the map,
as the camera would see them from the robot's current TF pose (within max_range and a field of view).

    ros2 run rd_perception fake_tags --ros-args -p tags:="[7, 5.0, 1.0, 0.5, 0.0]"   # id, x, y, z, yaw (map)

Pairs with rd_nav's tag corrector: with the robot localized the detections agree with AMCL; teleport it
(/initialpose) and the corrector pulls it back. Also what a scripted perception scene looks like without Isaac.
"""
import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

from rd_interfaces.msg import TagDetection, TagDetectionArray
from rd_nav.tag_corrector import mat_to_quat, transform_to_mat, xy_yaw_to_mat


class FakeTags(Node):
    def __init__(self):
        super().__init__("fake_tags")
        self.declare_parameter("tags", [7.0, 5.0, 1.0, 0.5, 0.0])      # flat list of id, x, y, z, yaw per tag
        self.declare_parameter("camera_frame", "camera_link_optical")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("max_range", 3.0)
        self.declare_parameter("half_fov_deg", 35.0)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("out_topic", "/tags/detections")
        flat = [float(v) for v in self.get_parameter("tags").value]
        self.tags = {}
        for i in range(0, len(flat) - 4, 5):
            tid, x, y, z, yaw = flat[i:i + 5]
            M = xy_yaw_to_mat(x, y, yaw)
            M[2, 3] = z
            self.tags[int(tid)] = M
        self.camera_frame = self.get_parameter("camera_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.max_range = float(self.get_parameter("max_range").value)
        self.half_fov = math.radians(float(self.get_parameter("half_fov_deg").value))
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.pub = self.create_publisher(TagDetectionArray, self.get_parameter("out_topic").value, 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.tick)

    def tick(self):
        try:
            T_map_cam = transform_to_mat(self.tf.lookup_transform(self.map_frame, self.camera_frame, rclpy.time.Time()))
        except TransformException:
            return
        T_cam_map = np.linalg.inv(T_map_cam)
        msg = TagDetectionArray()
        msg.header.frame_id = self.camera_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for tid, T_map_tag in self.tags.items():
            T_cam_tag = T_cam_map @ T_map_tag
            x, y, z = T_cam_tag[:3, 3]
            rng = math.sqrt(x * x + y * y + z * z)
            if z <= 0.05 or rng > self.max_range or abs(math.atan2(x, z)) > self.half_fov:
                continue   # behind the camera, too far, or outside the field of view (optical frame: z forward)
            d = TagDetection(id=tid, family="tag36h11", size=0.15, decision_margin=50.0)
            d.pose.header = msg.header
            d.pose.pose.position.x, d.pose.pose.position.y, d.pose.pose.position.z = float(x), float(y), float(z)
            qx, qy, qz, qw = mat_to_quat(T_cam_tag[:3, :3])
            d.pose.pose.orientation.x, d.pose.pose.orientation.y = float(qx), float(qy)
            d.pose.pose.orientation.z, d.pose.pose.orientation.w = float(qz), float(qw)
            msg.detections.append(d)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FakeTags()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
