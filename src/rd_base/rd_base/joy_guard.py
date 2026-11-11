"""Stand between the joystick reader and the base's teleop: no stale pad state reaches the wheels.

joy's SDL node goes on publishing a pad's LAST axes and buttons at its autorepeat rate after the pad is gone
(Bluetooth drop, sleep, flat battery), and when the pad is back the old values stand until each control next
moves. On the robot (2026-11-10) that was a stick position "saved" from before a drop: pressing the deadman drove
the base forward with the stick at rest, until the stick was pulled. Had the deadman itself been held when the link
dropped, the base would have kept driving.

  sub  /joy           sensor_msgs/Joy   the reader's output
  pub  /joy_guarded   sensor_msgs/Joy   the same, except:
        - while no input device with the pad's name exists, everything is zero;
        - after the pad (re)appears, an axis that came back off centre reads zero until it moves, and a button
          that came back pressed reads released until it has been seen released.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy

from rd_bringup import config


def device_present(name: str, listing: str | None = None) -> bool:
    """Is an input device with exactly this name connected (the kernel's list, /proc/bus/input/devices)?"""
    if listing is None:
        try:
            with open("/proc/bus/input/devices", encoding="utf-8", errors="replace") as f:
                listing = f.read()
        except OSError:
            return True          # cannot tell (not Linux, no /proc): do not block the pad on a guess
    return f'N: Name="{name}"' in listing


class Guard:
    """What may pass (pure, tested). `rest` is how far off centre an axis may come back and still be taken as live."""

    def __init__(self, rest: float = 0.1):
        self.rest = rest
        self.present = False
        self.axes_from: list | None = None       # each axis's value when the pad came back; None once it has moved
        self.buttons_held: list | None = None    # buttons that came back pressed and have not been released yet

    def update(self, present: bool, axes, buttons):
        axes, buttons = [float(a) for a in axes], [int(b) for b in buttons]
        if not present:
            self.present = False
            return [0.0] * len(axes), [0] * len(buttons)
        if not self.present or self.axes_from is None or len(self.axes_from) != len(axes) or len(self.buttons_held) != len(buttons):
            self.present = True
            self.axes_from = [a if abs(a) > self.rest else None for a in axes]
            self.buttons_held = [bool(b) for b in buttons]
        out_axes, out_buttons = [], []
        for i, a in enumerate(axes):
            if self.axes_from[i] is not None and a != self.axes_from[i]:
                self.axes_from[i] = None                      # it moved: the pad is really reporting this axis
            out_axes.append(a if self.axes_from[i] is None else 0.0)
        for i, b in enumerate(buttons):
            if self.buttons_held[i] and not b:
                self.buttons_held[i] = False
            out_buttons.append(0 if self.buttons_held[i] else b)
        return out_axes, out_buttons


class JoyGuard(Node):
    def __init__(self):
        super().__init__("joy_guard")
        self.declare_parameter("device_name", str(config.get("gamepad.device_name", "")))
        self.declare_parameter("rest", 0.1)
        self.name = str(self.get_parameter("device_name").value)
        self.guard = Guard(float(self.get_parameter("rest").value))
        self.present = device_present(self.name) if self.name else True
        self.checked = 0.0
        self.pub = self.create_publisher(Joy, "/joy_guarded", 10)
        self.create_subscription(Joy, "/joy", self.on_joy, 10)
        self.get_logger().info(f"guarding /joy for '{self.name}' ({'connected' if self.present else 'not connected'})")

    def on_joy(self, m: Joy):
        now = time.monotonic()
        if self.name and now - self.checked > 0.25:
            self.checked = now
            present = device_present(self.name)
            if present != self.present:
                self.get_logger().info(f"'{self.name}' {'connected' if present else 'gone: sending a neutral pad'}")
            self.present = present
        m.axes, m.buttons = self.guard.update(self.present, m.axes, m.buttons)
        self.pub.publish(m)


def main():
    rclpy.init()
    exit_with_parent()
    node = JoyGuard()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
