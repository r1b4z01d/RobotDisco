"""Correct the front RPLidar S1's range error and republish the scan (v1 scan_range_offset.py, plus the body reference).

The S1 returns every range too LONG by an amount that is the same for every beam but not the same from day to day:
v1 watched the robot's own wheels creep outwards in the scan for a week, then measured +0.33 m on 2026-11-02 (box at
a taped 0.50 m -> 0.83 m, same in both scan modes, same after a power cycle). On 2026-11-10, after the robot had been
off for the reflash, the error was gone: uncorrected ranges matched the Velodyne, and -0.33 would have put the wheel
housings inside the sensor. A fixed number is therefore wrong in one direction or the other, and too long is the
dangerous one on the lidar that looks where the robot drives.

So the error is measured on every scan. The S1 always sees the robot's own drive-wheel housings in two fixed sectors at
a known range (sensors.rplidar_s1.body_reference); the difference between that range and the one measured is the
offset, and it is ADDED to every range between the driver (<topic>_uncorrected) and the filter chain. Without a
reference the node adds the fixed `offset`, as v1 did.

Ranges that would fall below range_min after the correction are clamped to range_min (something IS that close);
inf / nan pass through.
"""
from collections import deque

import numpy as np
from rd_bringup.process import exit_with_parent
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

HOLD_SCANS = 20        # with a reference: scans to hold back at start while it measures (2 s), before falling back
MISSING_SCANS = 50     # scans without a usable reference before saying so (5 s)


def apply_offset(ranges, offset: float, range_min: float, range_max: float):
    """The correction on a list of ranges (pure function, tested)."""
    r = np.asarray(ranges, dtype=np.float32)
    if offset:
        ok = np.isfinite(r) & (r >= range_min) & (r <= range_max)
        r[ok] = np.maximum(r[ok] + offset, range_min)
    return r.tolist()


def reference_offset(ranges, angle_min: float, angle_increment: float, range_min: float, range_max: float,
                     sectors, expected, max_disagreement: float = 0.05, min_fill: float = 0.5):
    """The offset one scan implies (pure function, tested): for each sector (scan-frame degrees, [low, high)) the known
    range to the robot's own body minus the median range measured there, averaged over the sectors.

    None when the scan cannot say: a sector with returns on fewer than min_fill of its beams, or sectors that disagree
    by more than max_disagreement (the error is the same on every beam, so a disagreement means something is in front
    of one of them).
    """
    r = np.asarray(ranges, dtype=np.float32)
    deg = np.degrees(angle_min + np.arange(len(r)) * angle_increment)
    found = []
    for (low, high), want in zip(sectors, expected):
        beams = r[(deg >= low) & (deg < high)]
        if not len(beams):
            return None
        hits = beams[np.isfinite(beams) & (beams >= range_min) & (beams <= range_max)]
        if len(hits) < min_fill * len(beams):
            return None
        found.append(float(want) - float(np.median(hits)))
    if not found or max(found) - min(found) > max_disagreement:
        return None
    return sum(found) / len(found)


class OffsetTracker:
    """The offset in use, from the per-scan estimates (pure, tested): the median of the last `window` usable ones, zero
    inside the deadband (a healthy unit gets its ranges untouched), the fallback until `need` of them have arrived,
    and held at the last value while estimates are missing. An estimate outside [lowest, highest] is not the fault this
    corrects (it only ever read long), so it counts as missing."""

    def __init__(self, fallback: float, window: int = 20, need: int = 5, deadband: float = 0.03,
                 lowest: float = -0.50, highest: float = 0.05):
        self.value = float(fallback)
        self.recent = deque(maxlen=window)
        self.need, self.deadband, self.lowest, self.highest = need, deadband, lowest, highest
        self.measured = False
        self.missed = 0

    def update(self, estimate) -> float:
        if estimate is None or not self.lowest <= estimate <= self.highest:
            self.missed += 1
            return self.value
        self.missed = 0
        self.recent.append(estimate)
        if len(self.recent) >= self.need:
            median = float(np.median(self.recent))
            self.value = 0.0 if abs(median) < self.deadband else median
            self.measured = True
        return self.value


class ScanRangeOffset(Node):
    def __init__(self):
        super().__init__("scan_range_offset")
        self.declare_parameter("in_topic", "scan_uncorrected")
        self.declare_parameter("out_topic", "scan")
        self.declare_parameter("offset", 0.0)            # the fixed offset; with a reference, only until it has measured
        flat = self._doubles("reference_sectors")        # [low, high, low, high, ...] scan-frame degrees; empty = fixed
        self.expected = self._doubles("reference_ranges")   # the true range in each sector, metres
        self.sectors = list(zip(flat[0::2], flat[1::2]))
        if len(flat) % 2 or len(self.sectors) != len(self.expected):
            raise ValueError(f"reference_sectors {flat} needs one [low, high] pair per entry of reference_ranges {self.expected}")
        self.offset = float(self.get_parameter("offset").value)
        self.tracker = OffsetTracker(self.offset)
        self.reported = None      # the measured offset last written to the log
        self.seen = 0
        in_topic = self.get_parameter("in_topic").value
        out_topic = self.get_parameter("out_topic").value
        self.pub = self.create_publisher(LaserScan, out_topic, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
        self.sub = self.create_subscription(LaserScan, in_topic, self.cb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        self.add_on_set_parameters_callback(self._on_params)
        how = f"measured against the body in {self.sectors} (until then {self.offset:+.3f} m)" if self.sectors else f"fixed {self.offset:+.3f} m"
        self.get_logger().info(f"{in_topic} -> {out_topic}, range offset {how}")

    def _doubles(self, name):
        """A list-of-doubles parameter that may be left unset (rclpy cannot infer a type from an empty list default)."""
        self.declare_parameter(name, Parameter.Type.DOUBLE_ARRAY)
        try:
            return [float(v) for v in self.get_parameter(name).value or []]
        except ParameterUninitializedException:
            return []

    def _on_params(self, params):
        for p in params:
            if p.name == "offset":
                self.offset = float(p.value)
                if not self.tracker.measured:
                    self.tracker.value = self.offset
                self.get_logger().info(f"range offset now {self.offset:+.3f} m" + (" (until the body reference measures)" if self.sectors else ""))
        return SetParametersResult(successful=True)

    def _measured(self, m: LaserScan):
        """The offset for this scan from the body reference; None while the first scans are held back for it."""
        self.seen += 1
        estimate = reference_offset(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max, self.sectors, self.expected)
        offset = self.tracker.update(estimate)
        if self.tracker.measured and (self.reported is None or abs(offset - self.reported) > 0.02):
            self.get_logger().info(f"range offset {offset:+.3f} m, measured against the body")
            self.reported = offset
        if self.tracker.missed == MISSING_SCANS:
            self.get_logger().warning(f"no body reference for {MISSING_SCANS} scans (sectors blocked, empty or out of range); holding the range offset at {offset:+.3f} m")
        if not self.tracker.measured and self.seen <= HOLD_SCANS:
            return None
        return offset

    def cb(self, m: LaserScan):
        offset = self._measured(m) if self.sectors else self.offset
        if offset is None:
            return
        m.ranges = apply_offset(m.ranges, offset, m.range_min, m.range_max)
        self.pub.publish(m)


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
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
