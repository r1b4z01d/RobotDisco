"""The AmazingHand as ROS 2 topics and services (the only client of the hand on the robot).

  sub  /hand/curls    std_msgs/Float32MultiArray [index, middle, ring, thumb] in 0..1   (policies, teleop)
  sub  /hand/offsets  std_msgs/Float32MultiArray 8 per-servo degrees                   (the manager's Hand tab)
  pub  /hand/state    std_msgs/Float32MultiArray [commanded curls x4, reported flex x4 (or the commanded again)]
  srv  /hand/open, /hand/close   std_srvs/Trigger
  srv  /hand/wiggle              std_srvs/SetBool   the greeting's finger wiggle (v1 hand_wiggle.py) on/off
  srv  /hand/reconnect           std_srvs/Trigger   re-probe the firmware

The link is UDP with a liveness probe; a dead hand is an error at connect and counted send errors afterwards
(diagnostic in /hand/state's layout label). Address and port from robot.yaml (RD__hand__ip in the rig).
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool, Trigger

from rd_bringup import config

from .hand_link import HandLink
from .hand_protocol import DEFAULT_SPEED, curls_to_offsets, offsets_to_curls


class HandNode(Node):
    def __init__(self):
        super().__init__("hand")
        self.declare_parameter("host", str(config.get("hand.ip")))
        self.declare_parameter("port", int(config.get("hand.ports.udp", 8766)))
        self.declare_parameter("speed", DEFAULT_SPEED)
        self.declare_parameter("state_rate_hz", 10.0)
        self.declare_parameter("probe_timeout", 12.0)
        self.declare_parameter("wiggle_hz", 1.5)
        self.declare_parameter("wiggle_curl", 0.6)
        self.declare_parameter("wiggle_phase", 0.25)
        self.speed = int(self.get_parameter("speed").value)
        self.link: HandLink | None = None
        self.curls = [0.0, 0.0, 0.0, 0.0]
        self.reported = None
        self.wiggle_t0 = None
        self.create_subscription(Float32MultiArray, "/hand/curls", self.on_curls, 10)
        self.create_subscription(Float32MultiArray, "/hand/offsets", self.on_offsets, 10)
        self.state_pub = self.create_publisher(Float32MultiArray, "/hand/state", 10)
        self.create_service(Trigger, "/hand/open", lambda req, res: self._set_all(0.0, res))
        self.create_service(Trigger, "/hand/close", lambda req, res: self._set_all(1.0, res))
        self.create_service(SetBool, "/hand/wiggle", self.on_wiggle)
        self.create_service(Trigger, "/hand/reconnect", self.on_reconnect)
        self.create_timer(1.0 / float(self.get_parameter("state_rate_hz").value), self.tick_state)
        self.create_timer(1.0 / 20.0, self.tick_wiggle)
        self.connect()

    # ---- link
    def connect(self) -> bool:
        host, port = self.get_parameter("host").value, int(self.get_parameter("port").value)
        try:
            self.link = HandLink(host, port, speed=self.speed, probe_timeout=float(self.get_parameter("probe_timeout").value)).connect()
            self.get_logger().info(f"hand at {host}:{port}/udp answered (firmware up {self.link.firmware_uptime_ms} ms)")
            return True
        except RuntimeError as e:
            self.link = None
            self.get_logger().error(str(e))
            return False

    def send(self, offsets) -> bool:
        if self.link is None:
            return False
        self.link.send_offsets(offsets, self.speed)
        return True

    # ---- callbacks
    def on_curls(self, m: Float32MultiArray):
        if len(m.data) != 4:
            self.get_logger().warn(f"/hand/curls wants 4 values, got {len(m.data)}", throttle_duration_sec=5.0)
            return
        self.wiggle_t0 = None
        self.curls = [max(0.0, min(1.0, float(v))) for v in m.data]
        self.send(curls_to_offsets(self.curls))

    def on_offsets(self, m: Float32MultiArray):
        if len(m.data) != 8:
            self.get_logger().warn(f"/hand/offsets wants 8 values, got {len(m.data)}", throttle_duration_sec=5.0)
            return
        self.wiggle_t0 = None
        self.curls = offsets_to_curls(m.data)
        self.send([float(v) for v in m.data])

    def _set_all(self, curl: float, res):
        self.wiggle_t0 = None
        self.curls = [curl] * 4
        ok = self.send(curls_to_offsets(self.curls))
        if ok:
            self.send(curls_to_offsets(self.curls))      # twice: UDP
        res.success = ok
        res.message = ("hand " + ("closed" if curl else "open")) if ok else "hand not connected"
        return res

    def on_wiggle(self, req, res):
        if req.data:
            if self.link is None and not self.connect():
                res.success, res.message = False, "hand not connected"
                return res
            self.wiggle_t0 = time.monotonic()
            res.message = "wiggling"
        else:
            self.wiggle_t0 = None
            self.curls = [0.0, 0.0, 0.0, self.curls[3]]
            for _ in range(2):
                self.send(curls_to_offsets(self.curls))
            res.message = "wiggle off"
        res.success = True
        return res

    def on_reconnect(self, req, res):
        if self.link is not None:
            self.link.close()
        res.success = self.connect()
        res.message = "connected" if res.success else "no answer from the hand"
        return res

    def tick_wiggle(self):
        if self.wiggle_t0 is None:
            return
        hz = float(self.get_parameter("wiggle_hz").value)
        curl = float(self.get_parameter("wiggle_curl").value)
        lag = float(self.get_parameter("wiggle_phase").value) / max(0.05, hz)
        t = time.monotonic() - self.wiggle_t0

        def c(delay):
            x = max(0.0, t - delay)
            return curl * (0.5 - 0.5 * math.cos(2.0 * math.pi * hz * x))
        self.curls = [c(0.0), c(lag), c(2 * lag), self.curls[3]]
        self.send(curls_to_offsets(self.curls))

    def tick_state(self):
        if self.link is not None:
            st = self.link.read_state()
            if st is not None:
                self.reported = st[0]
        m = Float32MultiArray()
        m.layout.dim = [MultiArrayDimension(label=("ok" if self.link is not None else "disconnected")
                                                  + f" sent={self.link.sent if self.link else 0} errors={self.link.send_errors if self.link else 0}",
                                            size=8, stride=8)]
        m.data = [float(v) for v in self.curls] + [float(v) for v in (self.reported or self.curls)]
        self.state_pub.publish(m)


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = HandNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.link is not None:
            node.link.close()
        node.destroy_node()
        rclpy.try_shutdown()
