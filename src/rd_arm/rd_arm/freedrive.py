"""Freedrive (hand-guiding the arm) and the hand's power-cycle, both through the UR driver.

  /arm/freedrive         std_srvs/SetBool   on: the motion controllers stop, the driver's freedrive_mode_controller
                                            starts and is kept alive (it drops out by itself one second after the last
                                            enable message, so a dead node never leaves the arm limp); off: back to the
                                            trajectory controller
  /arm/freedrive_state   std_msgs/Bool      once a second
  /hand/power_cycle      std_srvs/Trigger   the tool connector's supply off for two seconds and on again: the hand's
                                            ESP32 and servos reboot (v1 eval_hw / the panel's hand recovery). The
                                            voltage comes back at robot.yaml arm.tool_voltage, which must be 12.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import time

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from rd_bringup import config

HAND_VOLTAGE = 12          # the AmazingHand's supply. 24 V on the tool connector would destroy it.


def tool_voltage() -> int:
    """robot.yaml arm.tool_voltage, refused unless it is the hand's 12 V."""
    v = int(config.get("arm.tool_voltage", HAND_VOLTAGE))
    if v != HAND_VOLTAGE:
        raise ValueError(f"robot.yaml arm.tool_voltage is {v}: the hand runs on {HAND_VOLTAGE} V, nothing else is ever set")
    return v


class Freedrive(Node):
    def __init__(self):
        super().__init__("freedrive")
        dp = self.declare_parameter
        dp("controller_manager", "/ur5e/controller_manager")
        dp("freedrive_controller", "freedrive_mode_controller")
        dp("motion_controllers", ["scaled_joint_trajectory_controller", "joint_trajectory_controller", "forward_position_controller"])
        dp("trajectory_controllers", ["scaled_joint_trajectory_controller", "joint_trajectory_controller"])
        dp("set_io", "/ur5e/io_and_status_controller/set_io")
        dp("off_seconds", 2.0)
        self.cb = ReentrantCallbackGroup()
        cm = str(self.get_parameter("controller_manager").value)
        self.switch = self.create_client(SwitchController, cm + "/switch_controller", callback_group=self.cb)
        self.list_controllers = self.create_client(ListControllers, cm + "/list_controllers", callback_group=self.cb)
        ctl = str(self.get_parameter("freedrive_controller").value)
        self.enable_pub = self.create_publisher(Bool, f"/ur5e/{ctl}/enable_freedrive_mode", 10)
        self.state_pub = self.create_publisher(Bool, "/arm/freedrive_state", 10)
        self.create_service(SetBool, "/arm/freedrive", self.on_freedrive, callback_group=self.cb)
        self.create_service(Trigger, "/hand/power_cycle", self.on_power_cycle, callback_group=self.cb)
        self.on = False
        self.create_timer(0.2, self.keep_alive, callback_group=self.cb)
        self.create_timer(1.0, lambda: self.state_pub.publish(Bool(data=self.on)), callback_group=self.cb)

    def keep_alive(self):
        if self.on:
            self.enable_pub.publish(Bool(data=True))

    def on_freedrive(self, req, res):
        if not self.switch.wait_for_service(timeout_sec=3.0) or not self.list_controllers.wait_for_service(timeout_sec=3.0):
            res.success, res.message = False, "no controller manager"
            return res
        loaded = {c.name: c.state for c in self.list_controllers.call(ListControllers.Request()).controller}
        ctl = str(self.get_parameter("freedrive_controller").value)
        if ctl not in loaded:
            res.success, res.message = False, f"{ctl} is not loaded"
            return res
        if req.data:
            stop = [c for c in self.get_parameter("motion_controllers").value if loaded.get(c) == "active"]
            start = [ctl] if loaded[ctl] != "active" else []
        else:
            self.on = False
            for _ in range(3):                        # tell the arm before the controller goes
                self.enable_pub.publish(Bool(data=False))
                time.sleep(0.05)
            stop = [ctl] if loaded[ctl] == "active" else []
            start = [c for c in self.get_parameter("trajectory_controllers").value if c in loaded][:1]
            start = [c for c in start if loaded[c] != "active"]
        ok = True
        if stop or start:
            r = self.switch.call(SwitchController.Request(activate_controllers=start, deactivate_controllers=stop,
                                                          strictness=SwitchController.Request.STRICT, activate_asap=True))
            ok = bool(r.ok)
        self.on = bool(req.data) and ok
        res.success = ok
        res.message = ("freedrive on: the arm can be moved by hand" if self.on else "freedrive off") if ok else "controller switch failed"
        self.get_logger().info(res.message)
        return res

    def on_power_cycle(self, _req, res):
        from ur_msgs.srv import SetIO
        try:
            volts = tool_voltage()
        except ValueError as e:
            res.success, res.message = False, str(e)
            return res
        client = self.create_client(SetIO, str(self.get_parameter("set_io").value), callback_group=self.cb)
        if not client.wait_for_service(timeout_sec=3.0):
            res.success, res.message = False, "the driver's set_io service is not there (is the arm up?)"
            return res
        off = client.call(SetIO.Request(fun=SetIO.Request.FUN_SET_TOOL_VOLTAGE, pin=0, state=float(SetIO.Request.STATE_TOOL_VOLTAGE_0V)))
        time.sleep(float(self.get_parameter("off_seconds").value))
        on = client.call(SetIO.Request(fun=SetIO.Request.FUN_SET_TOOL_VOLTAGE, pin=0, state=float(volts)))
        res.success = bool(off.success and on.success)
        res.message = f"tool supply cycled, back at {volts} V; the hand needs a few seconds to rejoin" if res.success else "the driver refused the tool voltage"
        self.get_logger().info(res.message)
        return res


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = Freedrive()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
