"""The rig's AmazingHand: a UDP server speaking the firmware's line protocol on loopback.

    python3 -m rd_arm.fake_hand [port] [--loss 0.2] [--state]

Answers Q with K:<uptime>, applies J: setpoints (kept as the current offsets), optionally streams S: state
lines back, and can drop a fraction of datagrams to mimic the Wi-Fi link. ROS-free; used by the hand node's
tests and by anything that wants a hand without a hand.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import random
import socket
import sys
import threading
import time

from .hand_protocol import JOINT_OPEN_OFFSETS, offsets_to_curls, parse_joint_command


class FakeHand:
    def __init__(self, port: int = 0, host: str = "127.0.0.1", loss: float = 0.0, stream_state: bool = False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.loss = loss
        self.stream_state = stream_state
        self.offsets = list(JOINT_OPEN_OFFSETS)
        self.speed = 0
        self.commands = 0
        self.probes = 0
        self.t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="fake_hand")

    @property
    def curls(self):
        return offsets_to_curls(self.offsets)

    def start(self) -> "FakeHand":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.sock.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            if self.loss and random.random() < self.loss:
                continue
            for raw in data.split(b"\n"):
                line = raw.decode("ascii", "ignore").strip()
                if not line:
                    continue
                if line == "Q":
                    self.probes += 1
                    if not (self.loss and random.random() < self.loss):
                        self.sock.sendto(f"K:{int((time.monotonic() - self.t0) * 1000)}\n".encode(), addr)
                    continue
                parsed = parse_joint_command(line)
                if parsed is not None:
                    self.offsets, self.speed = parsed
                    self.commands += 1
                    if self.stream_state:
                        f = self.curls
                        self.sock.sendto(("S:" + ",".join(f"{v:.3f}" for v in f) + ",0,0,0,0\n").encode(), addr)


def main(argv=None):
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    argv = sys.argv[1:] if argv is None else argv
    port = int(argv[0]) if argv and argv[0].isdigit() else 8766
    loss = float(argv[argv.index("--loss") + 1]) if "--loss" in argv else 0.0
    h = FakeHand(port, loss=loss, stream_state="--state" in argv).start()
    print(f"fake hand on udp :{h.port} (loss {loss:.0%})", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        h.stop()


if __name__ == "__main__":
    main()
