"""Republish a LaserScan under another frame (the rig: one simulated scan feeds both lidar topics)."""
import rclpy
from rd_bringup.process import exit_with_parent
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class ScanRelay(Node):
    def __init__(self):
        super().__init__("scan_relay")
        self.declare_parameter("in_topic", "scan")
        self.declare_parameter("out_topic", "scan_copy")
        self.declare_parameter("frame_id", "")
        self.frame = self.get_parameter("frame_id").value
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(LaserScan, self.get_parameter("out_topic").value, qos)
        self.create_subscription(LaserScan, self.get_parameter("in_topic").value, self.cb, qos)

    def cb(self, m):
        if self.frame:
            m.header.frame_id = self.frame
        self.pub.publish(m)


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    n = ScanRelay()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
