"""The rig's UR dashboard server: a TCP server on :29999 speaking enough of the dashboard protocol for
ArmPower (remote control flag, robotmode / safetystatus state machine, power on / brake release / power
off / unlock protective stop / restart safety / close safety popup, running / programState / get loaded
program). ROS-free. Start it in a protective stop to test the recovery path.

    python3 -m rd_arm.fake_dashboard [port] [--protective-stop] [--local]
"""
from __future__ import annotations

import socket
import sys
import threading
import time


class FakeDashboard:
    def __init__(self, port: int = 0, host: str = "127.0.0.1", remote: bool = True,
                 robotmode: str = "POWER_OFF", safety: str = "NORMAL", brake_delay: float = 0.3):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(4)
        self.srv.settimeout(0.2)
        self.port = self.srv.getsockname()[1]
        self.remote, self.robotmode, self.safety = remote, robotmode, safety
        self.brake_delay = brake_delay
        self.running = False
        self.loaded = "external_control.urp"
        self.commands: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="fake_dashboard")

    def start(self) -> "FakeDashboard":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.srv.close()

    def reply(self, cmd: str) -> str:
        self.commands.append(cmd)
        c = cmd.strip().lower()
        if c == "is in remote control":
            return "true" if self.remote else "false"
        if c == "robotmode":
            return f"Robotmode: {self.robotmode}"
        if c == "safetystatus":
            return f"Safetystatus: {self.safety}"
        if c == "running":
            return f"Program running: {str(self.running).lower()}"
        if c == "programstate":
            return ("PLAYING " if self.running else "STOPPED ") + self.loaded
        if c == "get loaded program":
            return f"Loaded program: /programs/{self.loaded}"
        if not self.remote:
            return "Command not allowed in local control"
        if c == "power on":
            if self.safety != "NORMAL":
                return "Powering on is not possible in current safety state"
            self.robotmode = "IDLE"
            return "Powering on"
        if c == "brake release":
            if self.robotmode != "IDLE":
                return "Brake release is only possible when the robot is idle"
            threading.Timer(self.brake_delay, lambda: setattr(self, "robotmode", "RUNNING")).start()
            return "Brake releasing"
        if c == "power off":
            self.robotmode = "POWER_OFF"
            self.running = False
            return "Powering off"
        if c == "close safety popup":
            return "closing safety popup"
        if c == "unlock protective stop":
            if self.safety == "PROTECTIVE_STOP":
                self.safety = "NORMAL"
                self.robotmode = "IDLE" if self.robotmode == "RUNNING" else self.robotmode
                return "Protective stop releasing"
            return "Cannot unlock protective stop until 5s after occurrence"
        if c == "restart safety":
            if self.safety in ("FAULT", "VIOLATION"):
                self.safety = "NORMAL"
                self.robotmode = "POWER_OFF"
                return "Restarting safety"
            return "Safety already running"
        if c == "shutdown":
            self.robotmode = "POWER_OFF"
            return "Shutting down"
        if c.startswith("load "):
            self.loaded = c[5:].split("/")[-1]
            return f"Loading program: /programs/{self.loaded}"
        if c == "play":
            self.running = True
            return "Starting program"
        if c == "stop":
            self.running = False
            return "Stopped"
        return f"could not understand: '{cmd}'"

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(2.0)
            try:
                conn.sendall(b"Connected: Universal Robots Dashboard Server\n")
                buf = b""
                while not self._stop.is_set():
                    chunk = conn.recv(1024)
                    if not chunk:
                        return
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        conn.sendall((self.reply(line.decode("utf-8", "ignore")) + "\n").encode())
            except OSError:
                return

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    port = int(argv[0]) if argv and argv[0].isdigit() else 29999
    d = FakeDashboard(port, remote="--local" not in argv,
                      safety="PROTECTIVE_STOP" if "--protective-stop" in argv else "NORMAL").start()
    print(f"fake UR dashboard on tcp :{d.port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        d.stop()


if __name__ == "__main__":
    main()
