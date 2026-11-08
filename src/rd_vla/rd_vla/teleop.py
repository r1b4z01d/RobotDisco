"""Gamepad teleop of the arm and the hand through rd_arm's streamer (v1 gamepad.py's mapping on /joy):

  left stick -> x, y      right stick -> roll, pitch      triggers -> z (RT up, LT down)
  LB / RB -> open / close the hand     START -> streaming on/off     BACK -> home
Publishes /arm/target_pose at `rate_hz` (the pose integrates the sticks; clamped per tick like the policy
runner) and /hand/curls. Button / axis indices as rd_base's joystick.yaml (the 8BitDo on its receiver).
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, Trigger

from .geometry import clamp_step, quat_to_rotvec, rotvec_to_quat
from .observer import Observer


class Teleop(Node):
    def __init__(self):
        super().__init__("vla_teleop")
        dp = self.declare_parameter
        dp("rate_hz", 12.0)
        dp("linear_speed", 0.10)      # m/s at full stick
        dp("angular_speed", 0.6)      # rad/s
        dp("max_step_m", 0.03)
        dp("axis_x", 1); dp("axis_y", 0); dp("axis_roll", 3); dp("axis_pitch", 4); dp("axis_lt", 2); dp("axis_rt", 5)
        dp("button_open", 4); dp("button_close", 5); dp("button_stream", 7); dp("button_home", 6)
        dp("deadzone", 0.12)
        self.observer = Observer(self, cameras=())
        self.pose_pub = self.create_publisher(PoseStamped, "/arm/target_pose", 10)
        self.hand_pub = self.create_publisher(Float32MultiArray, "/hand/curls", 10)
        self.stream = self.create_client(SetBool, "/arm/stream")
        self.home = self.create_client(Trigger, "/arm/go_home")
        self.create_subscription(Joy, "/joy", self.on_joy, 10)
        self.joy = None
        self.prev_buttons = []
        self.target = None
        self.curls = [0.0] * 4
        self.streaming = False
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.tick)

    def _axis(self, m: Joy, name: str) -> float:
        i = int(self.get_parameter(name).value)
        v = m.axes[i] if 0 <= i < len(m.axes) else 0.0
        return 0.0 if abs(v) < float(self.get_parameter("deadzone").value) else float(v)

    def _pressed(self, m: Joy, name: str) -> bool:
        i = int(self.get_parameter(name).value)
        now = bool(m.buttons[i]) if 0 <= i < len(m.buttons) else False
        before = bool(self.prev_buttons[i]) if i < len(self.prev_buttons) else False
        return now and not before

    def on_joy(self, m: Joy):
        if self._pressed(m, "button_stream"):
            self.streaming = not self.streaming
            self.target = None
            if self.stream.wait_for_service(timeout_sec=0.5):
                self.stream.call_async(SetBool.Request(data=self.streaming))
            self.get_logger().info("streaming " + ("on" if self.streaming else "off"))
        if self._pressed(m, "button_home") and self.home.wait_for_service(timeout_sec=0.5):
            self.home.call_async(Trigger.Request())
        if self._pressed(m, "button_open"):
            self.curls = [0.0] * 4
            self.hand_pub.publish(Float32MultiArray(data=self.curls))
        if self._pressed(m, "button_close"):
            self.curls = [1.0] * 4
            self.hand_pub.publish(Float32MultiArray(data=self.curls))
        self.prev_buttons = list(m.buttons)
        self.joy = m

    def tick(self):
        if not self.streaming or self.joy is None:
            return
        pose = self.observer.ee_pose()
        if pose is None:
            return
        if self.target is None:
            self.target = list(pose)
        dt = 1.0 / float(self.get_parameter("rate_hz").value)
        lin, ang = float(self.get_parameter("linear_speed").value), float(self.get_parameter("angular_speed").value)
        m = self.joy
        vx, vy = self._axis(m, "axis_x"), self._axis(m, "axis_y")
        vz = (1 - self._axis(m, "axis_rt")) / 2 - (1 - self._axis(m, "axis_lt")) / 2 if len(m.axes) > 5 else 0.0
        self.target[0] += vx * lin * dt
        self.target[1] += vy * lin * dt
        self.target[2] += vz * lin * dt
        self.target[3] += self._axis(m, "axis_roll") * ang * dt
        self.target[4] += self._axis(m, "axis_pitch") * ang * dt
        xyz = clamp_step(self.target[:3], pose[:3], float(self.get_parameter("max_step_m").value))
        msg = PoseStamped()
        msg.header.frame_id = self.observer.base_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = xyz
        qx, qy, qz, qw = rotvec_to_quat(self.target[3:])
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = qx, qy, qz, qw
        self.pose_pub.publish(msg)


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = Teleop()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
