"""Latch a URDF string on <namespace>/robot_description (std_msgs/String, transient local).

Jazzy's controller_manager takes the robot description from the `robot_description` topic in its own
namespace (the parameter is deprecated), and robot_state_publisher only publishes the whole-robot one on
/robot_description. The arm's controller manager under /ur5e needs the arm-control variant of the URDF
(ur_control:=true base_control:=false), so this node publishes it there.
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


class DescriptionPublisher(Node):
    def __init__(self):
        super().__init__("description_publisher")
        self.declare_parameter("robot_description", "")
        urdf = self.get_parameter("robot_description").value
        if not urdf.strip():
            raise SystemExit("robot_description parameter is empty")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, "robot_description", qos)   # relative: /<namespace>/robot_description
        self.pub.publish(String(data=urdf))
        self.get_logger().info(f"published {len(urdf)} bytes of URDF on {self.pub.topic_name}")


def main():
    rclpy.init()
    node = DescriptionPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
