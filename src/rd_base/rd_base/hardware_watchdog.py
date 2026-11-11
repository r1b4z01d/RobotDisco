"""End the base launch when the wheel hardware is no longer active, so its supervisor unit starts it afresh.

ros2_control deactivates a hardware component whose read or write reports an error and never brings it back:
the controller manager stays up, RUNNING as far as supervisor can tell, with no odometry and no wheels (the
robot, 2026-11-10). The patched RoboClaw interface now rides out half a second of failed exchanges; past that
the link is really down, and the one recovery that is known to work is a fresh launch. This node watches the
controller manager's hardware list and exits when the component has been down for `down_s`; the launch file
shuts the launch down on that exit and supervisor (autorestart) brings the unit back.

Nothing here commands the wheels. While the hardware is down the RoboClaw's own serial timeout holds them still.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import sys
import time

import rclpy
from controller_manager_msgs.srv import ListHardwareComponents
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class Watch:
    """When to give up (pure, tested). `label` is the component's lifecycle state, None when nobody answered."""

    def __init__(self, grace_s: float = 45.0, down_s: float = 5.0):
        self.grace_s, self.down_s = grace_s, down_s
        self.started = None
        self.was_active = False
        self.down_since = None

    def update(self, now: float, label: str | None) -> bool:
        if self.started is None:
            self.started = now
        if label == "active":
            self.was_active, self.down_since = True, None
            return False
        if not self.was_active and now - self.started < self.grace_s:
            return False                       # still coming up
        if self.down_since is None:
            self.down_since = now
        return now - self.down_since >= self.down_s


class HardwareWatchdog(Node):
    def __init__(self):
        super().__init__("hardware_watchdog")
        self.declare_parameter("controller_manager", "/controller_manager")
        self.declare_parameter("grace_s", 45.0)        # from the launch's start until the hardware must be active
        self.declare_parameter("down_s", 5.0)          # how long it may be down after that
        self.watch = Watch(float(self.get_parameter("grace_s").value), float(self.get_parameter("down_s").value))
        self.client = self.create_client(ListHardwareComponents, self.get_parameter("controller_manager").value + "/list_hardware_components")
        self.label = None
        self.pending = None
        self.asked_at = 0.0
        self.give_up = False
        self.create_timer(1.0, self.tick)

    def tick(self):
        now = time.monotonic()
        if self.pending is not None and self.pending.done():
            res = self.pending.result()
            self.label = res.component[0].state.label if res is not None and res.component else None
            self.pending = None
        elif self.pending is not None and now - self.asked_at > 3.0:
            self.pending.cancel()                      # the controller manager is not answering
            self.pending, self.label = None, None
        if self.pending is None:
            if self.client.service_is_ready():
                self.pending, self.asked_at = self.client.call_async(ListHardwareComponents.Request()), now
            else:
                self.label = None
        if self.watch.update(now, self.label):
            self.get_logger().error(f"the wheel hardware is '{self.label or 'not answering'}' and has been for {self.watch.down_s:.0f} s: "
                                    "ending the launch so that its unit restarts it")
            self.give_up = True


def main():
    rclpy.init()
    exit_with_parent()
    node = HardwareWatchdog()
    try:
        while rclpy.ok() and not node.give_up:
            rclpy.spin_once(node, timeout_sec=0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        failed = node.give_up
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
