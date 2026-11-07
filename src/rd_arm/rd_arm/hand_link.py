"""UDP link to the AmazingHand (v1 amazing_hand_client.py's transport, ROS-free): a connected datagram
socket, a PATIENT liveness probe ("Q" -> "K:<uptime>", retried through the link's loss bursts; ECONNREFUSED
three times = old firmware without the UDP listener), fire-and-forget sends that count errors instead of
raising, and a non-blocking reader for streamed "S:" state lines.
"""
from __future__ import annotations

import errno
import select
import socket
import time

from .hand_protocol import DEFAULT_SPEED, curls_to_offsets, format_joint_command, parse_state_line


class HandLink:
    def __init__(self, host: str, port: int = 8766, *, speed: int = DEFAULT_SPEED, timeout: float = 0.5,
                 probe_timeout: float = 12.0):
        self.host, self.port = host, int(port)
        self.speed = int(speed)
        self.timeout = timeout
        self.probe_timeout = probe_timeout
        self._sock: socket.socket | None = None
        self._rx = b""
        self.send_errors = 0
        self.sent = 0
        self.last_error: Exception | None = None
        self.firmware_uptime_ms: int | None = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> "HandLink":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        last: Exception | None = None
        refused = 0
        deadline = time.monotonic() + max(self.timeout, float(self.probe_timeout))
        while time.monotonic() < deadline:
            try:
                sock.send(b"Q\n")
                data = sock.recv(256)
            except socket.timeout:
                continue
            except OSError as e:
                last = e
                if getattr(e, "errno", None) == errno.ECONNREFUSED:
                    refused += 1
                    if refused >= 3:
                        break
                time.sleep(0.25)
                continue
            if data.startswith(b"K:"):
                try:
                    self.firmware_uptime_ms = int(data[2:].strip() or 0)
                except ValueError:
                    self.firmware_uptime_ms = None
                self._sock = sock
                return self
        sock.close()
        if refused >= 3:
            raise RuntimeError(f"AmazingHand at {self.host}:{self.port}/udp refused the liveness probe (port closed): "
                               f"flash the UDP-capable firmware (firmware/AmazingHand)")
        raise RuntimeError(f"AmazingHand at {self.host}:{self.port}/udp did not answer the liveness probe within "
                           f"{self.probe_timeout:.0f}s{f' ({last})' if last else ''}: hand off, mid-reboot, or the Wi-Fi "
                           f"link is dropping everything")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send_line(self, line: str) -> None:
        if self._sock is None:
            raise RuntimeError("not connected")
        try:
            self._sock.send(line.encode("ascii"))
            self.sent += 1
        except OSError as e:
            self.send_errors += 1
            self.last_error = e

    def send_offsets(self, offsets, speed: int | None = None) -> None:
        self.send_line(format_joint_command(offsets, self.speed if speed is None else speed))

    def send_curls(self, curls, speed: int | None = None) -> None:
        self.send_offsets(curls_to_offsets(curls), speed)

    def read_state(self):
        """Latest streamed (flex[4], abduct[4]), or None. Non-blocking."""
        if self._sock is None:
            raise RuntimeError("not connected")
        try:
            while select.select([self._sock], [], [], 0)[0]:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                self._rx += chunk
        except (BlockingIOError, OSError):
            pass
        if b"\n" not in self._rx:
            return None
        *lines, self._rx = self._rx.split(b"\n")
        latest = None
        for ln in lines:
            parsed = parse_state_line(ln.decode("ascii", "ignore"))
            if parsed is not None:
                latest = parsed
        return latest
