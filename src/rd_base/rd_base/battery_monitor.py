"""The battery, from the monitor board (firmware/RD_Battery_Monitor: an ESP32 with an INA228 on the pack).

The board sends one JSON datagram four times a second, {"Voltage": V, "Current": mA, "Temperature": degC}, to this
PC's UDP port (robot.yaml battery.ports.udp). This node averages them and publishes sensor_msgs/BatteryState on
/battery once a second, and says so (the Say action, and /battery/alert) when the pack is low or critical: the robot
died of a flat battery four times on 2026-11-10 with nothing to tell anyone.

The levels are the pack's own and live in robot.yaml battery: full_v and empty_v give the percentage (linear in
voltage: a rough gauge, voltage sags under load), low_v and critical_v the warnings. Without them the node reports
volts and amps and warns about nothing.
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Optional


def parse(datagram: bytes) -> Optional[dict]:
    """The board's line as {voltage (V), current_a (A), temperature (degC or None)}; None for anything else."""
    try:
        d = json.loads(datagram.decode("utf-8", "replace"))
        v, ma = float(d["Voltage"]), float(d.get("Current", 0.0))
        t = d.get("Temperature")
        t = float(t) if t is not None else None
    except (ValueError, TypeError, KeyError):
        return None
    if not (math.isfinite(v) and math.isfinite(ma)) or not 0.0 <= v < 200.0:
        return None
    return {"voltage": v, "current_a": ma / 1000.0, "temperature": t if t is not None and math.isfinite(t) else None}


def percentage(voltage: float, empty_v: Optional[float], full_v: Optional[float]) -> Optional[float]:
    """0..1 between the pack's empty and full voltage; None when they are not configured."""
    if not empty_v or not full_v or full_v <= empty_v:
        return None
    return max(0.0, min(1.0, (voltage - empty_v) / (full_v - empty_v)))


class Alerts:
    """ok / low / critical with a hold time (a sag while the motors start is not a flat pack) and hysteresis.
    update() returns the level to announce now, or None: once on entering low, on entering critical and then every
    `repeat_s` while critical."""

    def __init__(self, low_v: Optional[float], critical_v: Optional[float], hold_s: float = 10.0,
                 hysteresis_v: float = 0.3, repeat_s: float = 300.0):
        self.low_v, self.critical_v = low_v or None, critical_v or None
        self.hold_s, self.hysteresis_v, self.repeat_s = hold_s, hysteresis_v, repeat_s
        self.level, self._candidate, self._since, self._said = "ok", "ok", 0.0, -1e18

    def _raw(self, v: float) -> str:
        up = self.hysteresis_v                        # leaving a level needs this much more than entering it
        if self.critical_v and (v < self.critical_v or (self.level == "critical" and v < self.critical_v + up)):
            return "critical"
        if self.low_v and (v < self.low_v or (self.level != "ok" and v < self.low_v + up)):
            return "low"
        return "ok"

    def update(self, voltage: float, now: float) -> Optional[str]:
        raw = self._raw(voltage)
        if raw != self._candidate:
            self._candidate, self._since = raw, now
        if raw != self.level and now - self._since >= self.hold_s:
            worse = ("ok", "low", "critical").index(raw) > ("ok", "low", "critical").index(self.level)
            self.level = raw
            if worse:
                self._said = now
                return raw
        if self.level == "critical" and now - self._said >= self.repeat_s:
            self._said = now
            return "critical"
        return None


def sentence(level: str, voltage: float, pct: Optional[float]) -> str:
    amount = f"{round(pct * 100)} percent" if pct is not None else f"{voltage:.1f} volts"
    if level == "critical":
        return f"My battery is almost empty, {amount}. Please charge me now."
    return f"My battery is low, {amount}."


def main(args=None):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from sensor_msgs.msg import BatteryState
    from std_msgs.msg import String

    from rd_bringup import config
    from rd_bringup.process import exit_with_parent
    from rd_interfaces.action import Say

    exit_with_parent()
    rclpy.init(args=args)
    node = Node("battery_monitor")
    dp = node.declare_parameter
    cfg = lambda key: config.get(f"battery.{key}", None)  # noqa: E731
    port = int(dp("port", int(config.get("battery.ports.udp", 8889))).value)
    full_v, empty_v = float(dp("full_v", float(cfg("full_v") or 0.0)).value), float(dp("empty_v", float(cfg("empty_v") or 0.0)).value)
    low_v, critical_v = float(dp("low_v", float(cfg("low_v") or 0.0)).value), float(dp("critical_v", float(cfg("critical_v") or 0.0)).value)
    sign = float(dp("current_sign", float(cfg("current_sign") or 1.0)).value)      # -1 if the shunt reads discharge as positive
    stale_s = float(dp("stale_s", 5.0).value)
    speak = bool(dp("speak", True).value)
    alerts = Alerts(low_v, critical_v, float(dp("hold_s", 10.0).value))
    pub = node.create_publisher(BatteryState, "/battery", 10)
    alert_pub = node.create_publisher(String, "/battery/alert", 10)
    say = ActionClient(node, Say, "say")

    lock = threading.Lock()
    samples: list = []
    state = {"last": 0.0, "seen": False, "silent_logged": False}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)

    def listen():
        while rclpy.ok():
            try:
                data, _ = sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                return
            s = parse(data)
            if s is not None:
                with lock:
                    samples.append(s)
                    state["last"] = time.monotonic()

    threading.Thread(target=listen, daemon=True).start()
    if not (low_v or critical_v):
        node.get_logger().warn("robot.yaml battery.low_v / critical_v are not set: reporting only, no warnings")
    node.get_logger().info(f"listening for the battery monitor on UDP {port}")

    def tick():
        now = time.monotonic()
        with lock:
            batch, samples[:] = list(samples), []
            last = state["last"]
        m = BatteryState()
        m.header.stamp = node.get_clock().now().to_msg()
        m.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_UNKNOWN
        m.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN
        m.charge = m.capacity = m.design_capacity = float("nan")
        if not batch:
            if state["seen"] and now - last > stale_s and not state["silent_logged"]:
                state["silent_logged"] = True
                node.get_logger().warn(f"no word from the battery monitor for {stale_s:.0f} s")
            if now - last > stale_s:
                m.present = False
                m.voltage = m.current = m.percentage = m.temperature = float("nan")
                m.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
                pub.publish(m)
            return
        if not state["seen"] or state["silent_logged"]:
            node.get_logger().info(f"battery monitor heard: {batch[-1]['voltage']:.2f} V")
        state["seen"], state["silent_logged"] = True, False
        v = sum(s["voltage"] for s in batch) / len(batch)
        a = sign * sum(s["current_a"] for s in batch) / len(batch)
        temps = [s["temperature"] for s in batch if s["temperature"] is not None]
        pct = percentage(v, empty_v, full_v)
        m.present = True
        m.voltage, m.current = v, a
        m.temperature = sum(temps) / len(temps) if temps else float("nan")
        m.percentage = pct if pct is not None else float("nan")
        m.power_supply_status = (BatteryState.POWER_SUPPLY_STATUS_CHARGING if a > 0.2 else
                                 BatteryState.POWER_SUPPLY_STATUS_DISCHARGING if a < -0.05 else BatteryState.POWER_SUPPLY_STATUS_UNKNOWN)
        pub.publish(m)
        level = alerts.update(v, now)
        if level:
            text = sentence(level, v, pct)
            node.get_logger().warn(f"battery {level}: {v:.2f} V")
            alert_pub.publish(String(data=f"{level}: {text}"))
            if speak and say.server_is_ready():
                say.send_goal_async(Say.Goal(text=text, interrupt=False))

    node.create_timer(1.0, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        sock.close()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
