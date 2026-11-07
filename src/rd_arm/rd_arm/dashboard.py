"""UR dashboard client (port :29999), ROS-free: v1's ur_dashboard.py / ur_power.py, the power-up logic that
eval_hw, the teleop panel and the face all ran, now in one place behind rd_arm's ArmPower service.

Each command opens its own short connection (the dashboard accepts one client at a time and PolyScope
drops idle ones). power_up(): Remote Control required; FAULT/VIOLATION -> close safety popup + restart
safety; PROTECTIVE_STOP -> close popup + unlock (C161A0 "large movement while powered off" after a hard
power-off is the usual one; it repeating after clean shutdowns means an encoder/safety fault); then power
on, brake release. It releases the brakes without knowing whether the arm is clear, exactly as v1 did.
"""
from __future__ import annotations

import socket
import threading
import time
from typing import Callable, Optional


class UrDashboard:
    def __init__(self, ip: str, port: int = 29999, timeout: float = 3.0, log: Optional[Callable[[str], None]] = None):
        self.ip, self.port, self.timeout = ip, int(port), timeout
        self._log = log or (lambda s: None)
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._buf = b""

    # -- transport
    def _connect(self) -> None:
        self._sock = socket.create_connection((self.ip, self.port), timeout=self.timeout)
        self._buf = b""
        self._readline()                       # "Connected: Universal Robots Dashboard Server"

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _readline(self) -> str:
        assert self._sock is not None
        while b"\n" not in self._buf:
            chunk = self._sock.recv(1024)
            if not chunk:
                raise OSError("dashboard closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("utf-8", "replace").strip()

    def _cmd(self, cmd: str) -> str:
        assert self._sock is not None
        self._sock.sendall((cmd + "\n").encode())
        return self._readline()

    def one(self, cmd: str) -> str:
        """One command on its own connection."""
        with self._lock:
            try:
                self._connect()
                return self._cmd(cmd)
            finally:
                self._close()

    @staticmethod
    def value(reply: str) -> str:
        return reply.split(":", 1)[-1].strip()

    # -- queries
    def status(self) -> dict:
        with self._lock:
            try:
                self._connect()
                st = {
                    "remote": self._cmd("is in remote control").lower() == "true",
                    "robotmode": self.value(self._cmd("robotmode")),
                    "safety": self.value(self._cmd("safetystatus")),
                    "running": self._cmd("running").lower().endswith("true"),
                    "loaded": self.value(self._cmd("get loaded program")),
                    "state": self._cmd("programState"),
                }
            finally:
                self._close()
        return st

    def _wait(self, cmd: str, want: set, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.value(self.one(cmd)) in want:
                return True
            time.sleep(1.0)
        return False

    # -- actions
    def power_up(self, timeout: float = 60.0) -> str:
        if self.one("is in remote control").lower() != "true":
            raise RuntimeError("pendant is not in Remote Control mode; switch it on the teach pendant")
        if self.value(self.one("robotmode")) == "RUNNING" and self.value(self.one("safetystatus")) == "NORMAL":
            return "UR already RUNNING"
        note = ""
        ss = self.value(self.one("safetystatus"))
        if ss in ("FAULT", "VIOLATION"):
            note = f"restarted the safety system after {ss}; "
            self._log(f"UR safety status {ss}: restarting safety via the dashboard")
            self.one("close safety popup")
            reply = self.one("restart safety")
            if not self._wait("safetystatus", {"NORMAL"}, 45):
                raise RuntimeError(f"safety did not return to NORMAL after 'restart safety' ({reply}); "
                                   f"now {self.one('safetystatus')}; restart it on the pendant")
            ss = "NORMAL"
        if ss == "PROTECTIVE_STOP":
            note = "cleared protective stop; "
            self._log("UR in PROTECTIVE_STOP: clearing it via the dashboard (repeats after clean exits => encoder/safety fault)")
            self.one("close safety popup")
            self.one("unlock protective stop")
            if not self._wait("safetystatus", {"NORMAL"}, 15):
                raise RuntimeError("safety status did not return to NORMAL: " + self.one("safetystatus"))
        if self.value(self.one("robotmode")) != "RUNNING":
            self.one("power on")
            if not self._wait("robotmode", {"IDLE", "RUNNING"}, timeout):
                raise RuntimeError("timed out waiting for IDLE: " + self.one("robotmode"))
            self.one("brake release")
            if not self._wait("robotmode", {"RUNNING"}, timeout):
                raise RuntimeError("timed out waiting for RUNNING: " + self.one("robotmode"))
            note += "powered up (brakes released)"
        return note.rstrip("; ") or "UR already RUNNING"

    def power_off(self, shutdown: bool = False, timeout: float = 30.0) -> str:
        if self.value(self.one("robotmode")) not in ("POWER_OFF", "IDLE"):
            self.one("power off")
            if not self._wait("robotmode", {"POWER_OFF", "IDLE"}, timeout):
                raise RuntimeError("timed out waiting for POWER_OFF: " + self.one("robotmode"))
        if shutdown:
            self.one("shutdown")
            return "powered off, controller shutting down"
        return "powered off"

    def unlock_protective_stop(self) -> str:
        self.one("close safety popup")
        reply = self.one("unlock protective stop")
        if not self._wait("safetystatus", {"NORMAL"}, 15):
            raise RuntimeError("safety status did not return to NORMAL: " + self.one("safetystatus"))
        return reply
