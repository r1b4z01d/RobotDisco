"""ArmPower service: the one implementation of UR power-up / power-off / protective-stop recovery
(v1 had it in ur_power.py, eval_hw.py and the face node). Talks to the dashboard directly (rd_arm.dashboard);
on the robot it then asks the driver to resend the control program (headless mode). It does NOT set the tool
voltage: PolyScope's installation keeps the 12 V the hand runs on, and the `tool_voltage` parameter is only the value
a future power-cycle of the hand would use (v1's eval_hw toggled it to reboot the ESP32).

    ros2 service call /arm/power rd_interfaces/srv/ArmPower "{command: 1}"     # POWER_ON
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from rd_bringup import config
from rd_interfaces.srv import ArmPower

from .dashboard import UrDashboard


class ArmPowerServer(Node):
    def __init__(self):
        super().__init__("arm_power")
        self.declare_parameter("ip", str(config.get("arm.ip")))
        self.declare_parameter("dashboard_port", int(config.get("arm.ports.dashboard", 29999)))
        self.declare_parameter("tool_voltage", int(config.get("arm.tool_voltage", 12)))
        self.declare_parameter("resend_program", True)     # the driver's io_and_status_controller/resend_robot_program
        self.declare_parameter("sim_mode", False)
        self.dash = UrDashboard(self.get_parameter("ip").value, int(self.get_parameter("dashboard_port").value),
                                log=lambda s: self.get_logger().info(s))
        cb = ReentrantCallbackGroup()
        self.resend = self.create_client(Trigger, "/ur5e/io_and_status_controller/resend_robot_program", callback_group=cb)
        self.create_service(ArmPower, "/arm/power", self.on_request, callback_group=cb)
        # The control program's watchdog. The driver's program ends when its stream to the UR drops ("Lost connection
        # to stream", seen once or twice a day on the loaded Orin) and nothing restarted it: the arm then ignored
        # every target with no fault shown (a policy run on 2026-11-14, teleop on 2026-11-15). Resent here when it
        # has been stopped for a few seconds with the UR powered, safety normal and in remote control; a UR that is
        # off, stopped or under the pendant is left alone.
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool
        self.declare_parameter("program_watchdog", True)
        self._running, self._stopped_since, self._last_resend = None, None, 0.0
        self.create_subscription(Bool, "/ur5e/io_and_status_controller/robot_program_running", self.on_running,
                                 QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE),
                                 callback_group=cb)
        self.create_timer(2.0, self.watchdog, callback_group=cb)
        self.get_logger().info(f"dashboard {self.dash.ip}:{self.dash.port}")

    def on_running(self, m):
        import time
        self._running = bool(m.data)
        self._stopped_since = None if m.data else (self._stopped_since or time.monotonic())

    def watchdog(self):
        import time
        now = time.monotonic()
        if (self._running is not False or self._stopped_since is None or now - self._stopped_since < 3.0
                or now - self._last_resend < 10.0 or not self.get_parameter("program_watchdog").value
                or self.get_parameter("sim_mode").value or not self.resend.service_is_ready()):
            return
        try:
            st = self.dash.status()
        except (RuntimeError, OSError):
            return
        if st["robotmode"] != "RUNNING" or st["safety"] != "NORMAL" or not st["remote"]:
            return
        self._last_resend = now
        r = self.resend.call(Trigger.Request())
        self.get_logger().warn("the UR's control program had stopped: " + ("resent" if r.success else f"resend failed: {r.message}"))

    def fill(self, res, st: dict):
        res.robot_mode = st["robotmode"]
        res.safety_status = st["safety"]
        res.remote_control = bool(st["remote"])
        res.program_running = bool(st["running"])

    def on_request(self, req, res):
        try:
            if req.command == ArmPower.Request.STATUS:
                st = self.dash.status()
                self.fill(res, st)
                res.success = True
                res.message = f"{st['robotmode']} / {st['safety']}" + ("" if st["remote"] else " (local control)")
                return res
            if req.command == ArmPower.Request.POWER_ON:
                msg = self.dash.power_up(timeout=float(req.timeout_s or 60.0))
                if self.get_parameter("resend_program").value and not self.get_parameter("sim_mode").value:
                    if self.resend.wait_for_service(timeout_sec=2.0):
                        r = self.resend.call(Trigger.Request())
                        msg += "; control program " + ("resent" if r.success else f"NOT resent: {r.message}")
                    else:
                        msg += "; driver not running (no resend_robot_program service)"
            elif req.command == ArmPower.Request.POWER_OFF:
                msg = self.dash.power_off(shutdown=False, timeout=float(req.timeout_s or 30.0))
            elif req.command == ArmPower.Request.SHUTDOWN:
                msg = self.dash.power_off(shutdown=True, timeout=float(req.timeout_s or 30.0))
            elif req.command == ArmPower.Request.UNLOCK_PROTECTIVE_STOP:
                msg = self.dash.unlock_protective_stop()
            else:
                res.success = False
                res.message = f"unknown command {req.command}"
                return res
            st = self.dash.status()
            self.fill(res, st)
            res.success = True
            res.message = msg
        except (RuntimeError, OSError) as e:
            res.success = False
            res.message = str(e)
            try:
                self.fill(res, self.dash.status())
            except OSError:
                pass
        self.get_logger().info(f"command {req.command}: {res.message}")
        return res


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = ArmPowerServer()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
