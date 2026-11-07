"""Mirror the mock UR's commanded joints into Isaac Sim: /joint_states (the ur_ joints from rd_arm's mock
controller manager) -> /ur5e/sim_joint_command (sensor_msgs/JointState with the UR's own joint names, which
the bridge's articulation controller drives). The sim arm then follows whatever MoveIt / the streamer command."""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

UR = ["ur_shoulder_pan_joint", "ur_shoulder_lift_joint", "ur_elbow_joint", "ur_wrist_1_joint", "ur_wrist_2_joint", "ur_wrist_3_joint"]


class ArmMirror(Node):
    def __init__(self):
        super().__init__("arm_mirror")
        self.pub = self.create_publisher(JointState, "/ur5e/sim_joint_command", 10)
        self.create_subscription(JointState, "/joint_states", self.cb, 10)

    def cb(self, m: JointState):
        pos = dict(zip(m.name, m.position))
        if not all(j in pos for j in UR):
            return
        out = JointState()
        out.header = m.header
        out.name = list(UR)
        out.position = [float(pos[j]) for j in UR]
        self.pub.publish(out)


def main():
    rclpy.init()
    n = ArmMirror()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
