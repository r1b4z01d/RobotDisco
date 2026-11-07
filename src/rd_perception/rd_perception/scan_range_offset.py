"""Add a constant to every range of a LaserScan and republish it (v1 scan_range_offset.py).

The front RPLidar S1 returns every range a constant ~0.33 m too LONG (box at a taped 0.50 m -> 0.83 m, same
in both scan modes, same after a power cycle; 2026-11-02). The error is additive, so it is corrected here,
between the driver (<topic>_uncorrected) and the filter chain. Ranges that would fall below range_min
after the correction are clamped to range_min (something IS that close); inf / nan pass through.
Re-measure with a box at a taped distance after any change to the unit: sensors.rplidar_s1.range_offset_m.
"""
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


def apply_offset(ranges, offset: float, range_min: float, range_max: float):
    """The correction on a list of ranges (pure function, tested)."""
    r = np.asarray(ranges, dtype=np.float32)
    if offset:
        ok = np.isfinite(r) & (r >= range_min) & (r <= range_max)
        r[ok] = np.maximum(r[ok] + offset, range_min)
    return r.tolist()


class ScanRangeOffset(Node):
    def __init__(self):
        super().__init__("scan_range_offset")
        self.declare_parameter("in_topic", "scan_uncorrected")
        self.declare_parameter("out_topic", "scan")
        self.declare_parameter("offset", 0.0)
        self.offset = float(self.get_parameter("offset").value)
        in_topic = self.get_parameter("in_topic").value
        out_topic = self.get_parameter("out_topic").value
        self.pub = self.create_publisher(LaserScan, out_topic, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
        self.sub = self.create_subscription(LaserScan, in_topic, self.cb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        self.add_on_set_parameters_callback(self._on_params)
        self.get_logger().info(f"{in_topic} -> {out_topic}, range offset {self.offset:+.3f} m")

    def _on_params(self, params):
        for p in params:
            if p.name == "offset":
                self.offset = float(p.value)
                self.get_logger().info(f"range offset now {self.offset:+.3f} m")
        return SetParametersResult(successful=True)

    def cb(self, m: LaserScan):
        m.ranges = apply_offset(m.ranges, self.offset, m.range_min, m.range_max)
        self.pub.publish(m)


def main():
    rclpy.init()
    node = ScanRangeOffset()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
