"""Republish the detector's own AprilTag message as rd_interfaces/TagDetectionArray on /tags/detections.

Sources (parameter `source`):
  isaac     isaac_ros_apriltag_interfaces/AprilTagDetectionArray (the robot: isaac_ros_apriltag), on
            `in_topic` (default /tag_detections). Imported lazily, so the rig (no Isaac) still builds.
The conversion is a plain function over duck-typed detections (id, family, pose.pose.pose, size) so it is
unit-tested without the Isaac package.
"""
import rclpy
from rd_bringup.process import exit_with_parent
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from rd_interfaces.msg import TagDetection, TagDetectionArray


# The tag's own axes. isaac_ros_apriltag reports the AprilTag library's frame: x to the tag's right, y down, z into
# the tag. Everything downstream (rd_nav's tag corrector, the poses taught on the robot with v1's apriltag_ros and its
# z_up option) uses x out of the tag towards the viewer, y to the tag's right, z up. Found on the robot 2026-11-10:
# unconverted, a taught tag put the robot rolled over by 88 deg and the corrector refused every detection; of the 24
# axis conventions this one alone put it level, on the floor, where AMCL had it and where the laser scan fits the map.
# The rotation from the library's frame to ours, as a quaternion (x, y, z, w): new x = -z, new y = +x, new z = -y.
LIBRARY_TO_Z_UP = (0.5, 0.5, -0.5, 0.5)


def quat_multiply(a, b):
    """Hamilton product of two (x, y, z, w) quaternions: the rotation b, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def convert(msg, default_size: float = 0.15, z_up: bool = True) -> TagDetectionArray:
    """Isaac's array (header; detections[] with id, family, pose: PoseWithCovarianceStamped) -> ours, the tag's
    orientation turned from the AprilTag library's axes to the z-up ones (z_up=False leaves it as it comes)."""
    out = TagDetectionArray()
    out.header = msg.header
    for d in msg.detections:
        t = TagDetection()
        t.id = int(d.id)
        t.family = str(getattr(d, "family", "") or "tag36h11")
        t.size = float(getattr(d, "size", 0.0) or default_size)
        t.decision_margin = float(getattr(d, "decision_margin", 0.0) or 0.0)
        pose = d.pose.pose.pose if hasattr(d.pose, "pose") and hasattr(d.pose.pose, "pose") else d.pose.pose
        t.pose.header = getattr(d.pose, "header", msg.header)
        if not t.pose.header.frame_id:
            t.pose.header = msg.header
        t.pose.pose.position.x, t.pose.pose.position.y, t.pose.pose.position.z = float(pose.position.x), float(pose.position.y), float(pose.position.z)
        o = pose.orientation
        q = (float(o.x), float(o.y), float(o.z), float(o.w))
        if z_up:
            q = quat_multiply(q, LIBRARY_TO_Z_UP)
        t.pose.pose.orientation.x, t.pose.pose.orientation.y, t.pose.pose.orientation.z, t.pose.pose.orientation.w = q
        out.detections.append(t)
    return out


class TagAdapter(Node):
    def __init__(self):
        super().__init__("tag_adapter")
        self.declare_parameter("source", "isaac")
        self.declare_parameter("in_topic", "/tag_detections")
        self.declare_parameter("out_topic", "/tags/detections")
        self.declare_parameter("tag_size", 0.15)
        self.declare_parameter("z_up", True)          # turn the detector's tag axes to the ones the taught poses use
        self.size = float(self.get_parameter("tag_size").value)
        self.z_up = bool(self.get_parameter("z_up").value)
        source = self.get_parameter("source").value
        if source == "isaac":
            from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray  # noqa: PLC0415 - Isaac only on the robot
            msg_type = AprilTagDetectionArray
        else:
            raise ValueError(f"unknown tag source {source!r}")
        self.pub = self.create_publisher(TagDetectionArray, self.get_parameter("out_topic").value, 10)
        self.create_subscription(msg_type, self.get_parameter("in_topic").value, self.cb, 10)

    def cb(self, msg):
        self.pub.publish(convert(msg, self.size, self.z_up))


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = TagAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
