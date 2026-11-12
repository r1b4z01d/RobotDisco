"""The 3Dconnexion SpaceMouse, read from Linux hidraw with nothing but os.read (v1's teleop/spacemouse.py used the
`hid` pip module; the robot has no internet to install it and the reports are the same).

The wireless receiver (256f:c652) shows up as several hidraw nodes; the 6-DOF data comes on one of them, so all are
opened and whichever speaks is parsed. Access comes from provision/udev/99-robot-disco.rules.

    python3 -m rd_vla.spacemouse        # move the puck: prints axes and buttons
"""
from __future__ import annotations

import glob
import os
import struct
import time

VENDOR_3DCONNEXION = "0000256F"
# the old Logitech-branded units share Logitech's vendor id with mice and webcams: only these product ids
LOGITECH_PIDS = {"C603", "C605", "C606", "C621", "C623", "C625", "C626", "C627", "C628", "C629", "C62B"}


def find_nodes(sys_class: str = "/sys/class/hidraw") -> list[str]:
    """The /dev/hidrawN nodes that belong to a SpaceMouse."""
    out = []
    for h in sorted(glob.glob(os.path.join(sys_class, "hidraw*"))):
        try:
            uevent = open(os.path.join(h, "device", "uevent")).read()
        except OSError:
            continue
        hid_id = next((line.split("=", 1)[1] for line in uevent.splitlines() if line.startswith("HID_ID=")), "")
        parts = hid_id.upper().split(":")
        if len(parts) != 3:
            continue
        vendor, product = parts[1], parts[2][-4:]
        if vendor == VENDOR_3DCONNEXION or (vendor == "0000046D" and product in LOGITECH_PIDS):
            out.append("/dev/" + os.path.basename(h))
    return out


def normalise(raw: int, scale: float, deadband: float) -> float:
    """A raw axis count as -1..1, zero inside the deadband and ramping from its edge."""
    v = max(-1.0, min(1.0, raw / scale))
    if abs(v) <= deadband:
        return 0.0
    return (v - deadband * (1.0 if v > 0 else -1.0)) / (1.0 - deadband)


class SpaceMouseState:
    """Parses the device's reports into the latest axes and buttons. No I/O: fed bytes, so it is testable."""
    AXES = ("x", "y", "z", "roll", "pitch", "yaw")

    def __init__(self, scale: float = 350.0, deadband: float = 0.12):
        self.scale, self.deadband = scale, deadband
        self.axes = dict.fromkeys(self.AXES, 0.0)
        self.buttons = [0] * 8
        self.reports = 0

    def _set(self, names, data: bytes) -> None:
        for name, raw in zip(names, struct.unpack("<" + "h" * len(names), data)):
            self.axes[name] = normalise(raw, self.scale, self.deadband)

    def feed(self, report: bytes) -> None:
        if not report:
            return
        rid = report[0]
        if rid == 1 and len(report) >= 13:            # translation and rotation in one report (newer firmware)
            self._set(self.AXES, report[1:13])
        elif rid == 1 and len(report) >= 7:
            self._set(self.AXES[:3], report[1:7])
        elif rid == 2 and len(report) >= 7:
            self._set(self.AXES[3:], report[1:7])
        elif rid == 3 and len(report) >= 2:
            self.buttons = [(report[1] >> i) & 1 for i in range(8)]
        else:
            return
        self.reports += 1

    def zero(self) -> None:
        self.axes = dict.fromkeys(self.AXES, 0.0)
        self.buttons = [0] * 8


class SpaceMouse(SpaceMouseState):
    """The device: open() the nodes, read() drains what has arrived. A puck that goes quiet (asleep, receiver
    pulled) reads as centred after `timeout` s, so a lost link never leaves a deflection standing."""

    def __init__(self, scale: float = 350.0, deadband: float = 0.12, timeout: float = 0.5):
        super().__init__(scale, deadband)
        self.timeout = timeout
        self.fds: list[int] = []
        self.nodes: list[str] = []
        self.last = 0.0

    def open(self) -> bool:
        self.close()
        for node in find_nodes():
            try:
                self.fds.append(os.open(node, os.O_RDONLY | os.O_NONBLOCK))
                self.nodes.append(node)
            except OSError:
                continue
        return bool(self.fds)

    def read(self) -> bool:
        """Drain pending reports. False when the device is gone (the caller opens it again)."""
        for fd in list(self.fds):
            try:
                while True:
                    data = os.read(fd, 64)
                    if not data:
                        break
                    self.feed(data)
                    self.last = time.monotonic()
            except BlockingIOError:
                pass
            except OSError:                               # unplugged
                self.close()
                self.zero()
                return False
        # the puck reports only while it moves, and sends zeros as it comes to rest; silence WITH a deflection standing means the link dropped
        if any(self.axes.values()) and time.monotonic() - self.last > self.timeout:
            self.zero()
        return bool(self.fds)

    def close(self) -> None:
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds, self.nodes = [], []


def main() -> None:
    sm = SpaceMouse()
    if not sm.open():
        raise SystemExit("no SpaceMouse found (lsusb shows 256f:...? udev rule installed?)")
    print("opened", sm.nodes, "- move the puck, Ctrl-C to stop")
    try:
        while True:
            sm.read()
            if any(sm.axes.values()) or any(sm.buttons):
                print(" ".join(f"{k}={v:+.2f}" for k, v in sm.axes.items()), "buttons", sm.buttons[:2], flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        sm.close()


if __name__ == "__main__":
    main()
