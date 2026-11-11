"""The Hand tab's server side (v1 webui/server.py's hand section): per-finger curl + abduction -> one J: line, the
firmware's gestures, a raw line, and the liveness probe.

Commands go out through rd_arm's hand node, the hand's one owner (/hand/offsets with the speed as a ninth value,
/hand/raw for a gesture character or a typed line), so the manager and a running policy cannot interleave lines
from two sockets. Only the probe ("Q" -> "K:<uptime ms>") is asked from here directly: it changes nothing on the
hand and the page wants its answer now.
"""
from __future__ import annotations

import socket
import threading
import time

from rd_arm.hand_protocol import SPEED_MAX, SPEED_MIN, curls_to_offsets, flex_abduct_to_offsets, format_joint_command

GESTURES = {"0": "open", "1": "close", "2": "open, progressive", "3": "spread", "4": "clench",
            "5": "index pointing", "6": "no-no-no", "7": "perfect (OK)", "8": "victory", "9": "scissors",
            "a": "pinch", "b": "middle finger"}   # parseHandData() in the hand's firmware (HAND_CTRL.h)


def build_command(curls, abduct=None, speed=200) -> dict:
    """Four curls (0 open .. 1 closed) and four abductions (-1 .. 1, the sideways spread) -> the servo offsets and
    the J: line they make. ValueError on a wrong count. Pure, tested."""
    if len(curls) != 4:
        raise ValueError("curls must have 4 values (the channels index, middle, ring, thumb)")
    if abduct is not None and len(abduct) != 4:
        raise ValueError("abduct must have 4 values")
    curls = [min(1.0, max(0.0, float(c))) for c in curls]
    abduct = [min(1.0, max(-1.0, float(a))) for a in (abduct or [0.0] * 4)]
    speed = min(SPEED_MAX, max(SPEED_MIN, int(speed)))
    offsets = flex_abduct_to_offsets(curls, abduct) if any(abs(a) > 1e-6 for a in abduct) else curls_to_offsets(curls)
    return {"line": format_joint_command(offsets, speed).strip(), "offsets": [round(o, 2) for o in offsets],
            "curls": curls, "abduct": abduct, "speed": speed}


def probe(host: str, port: int, tries: int = 4, wait: float = 0.35) -> dict:
    """Liveness: "Q" -> "K:<uptime ms>". Several tries: the hand's Wi-Fi link drops packets in bursts."""
    last = None
    for _ in range(tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(wait)
                sock.connect((host, port))
                sock.send(b"Q\n")
                reply = sock.recv(256).decode("ascii", "ignore").strip()
        except socket.timeout:
            continue
        except OSError as e:
            last = f"{type(e).__name__}: {e}"
            continue
        if reply.startswith("K:"):
            try:
                return {"alive": True, "uptime_ms": int(reply[2:].strip()), "error": None}
            except ValueError:
                return {"alive": True, "uptime_ms": None, "error": None}
    return {"alive": False, "uptime_ms": None,
            "error": last or "no answer to the liveness probe (hand off, rebooting, or the Wi-Fi link is dropping everything)"}


class HandState:
    """What this server last sent, for the page's sliders and its status line."""

    def __init__(self):
        self.lock = threading.Lock()
        self.d = {"curls": [0.0] * 4, "abduct": [0.0] * 4, "speed": 200, "sent": 0, "errors": 0, "last": None}
        self.alive_t = None

    def sent(self, line: str, error: str | None = None, **fields):
        with self.lock:
            self.d["errors" if error else "sent"] += 1
            self.d["last"] = {"line": line, "t": time.time(), "error": error}
            self.d.update(fields)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.d)
