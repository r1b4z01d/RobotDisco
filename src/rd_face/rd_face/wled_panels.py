"""The WLED matrix panels: a beating heart on each matrix when a known person is recognised.

The frames go out as WLED realtime UDP (DNRGB, port 21324). Realtime data only overrides what the
panels show while it keeps arriving: once the stream stops, WLED goes back to the effect it was
running by itself, so nothing has to be saved or restored and a lost packet cannot leave the heart
up. The layout (matrix size, one heart per segment) is read from the WLED's JSON API.

v1's robot_face/wled_panels.py (2026-11-09) with the two ports as arguments, so the tests can stand a fake WLED on
loopback. Standalone test (no ROS):  python3 wled_panels.py <panels.ip from robot.yaml> [seconds]
"""
import json
import math
import socket
import sys
import threading
import time
import urllib.request

UDP_PORT = 21324          # WLED realtime UDP
HTTP_PORT = 80            # WLED JSON API
DNRGB_MAX_LEDS = 489      # LEDs per DNRGB packet
FPS = 15.0


def heart_coverage(u: float, v: float) -> bool:
    """Inside the heart curve (x^2 + y^2 - 1)^3 - x^2 y^3 <= 0; u right, v up, about -1.2 .. 1.2."""
    a = u * u + v * v - 1.0
    return a * a * a - u * u * v * v * v <= 0.0


def heart_frame(width: int, height: int, boxes, scale: float, color=(255, 0, 40), ss: int = 3) -> bytes:
    """RGB bytes (row-major, width * height) with one heart centred in each box (x0, y0, x1, y1).

    scale is the heart's size relative to the largest that fits the box; edges are supersampled."""
    buf = bytearray(width * height * 3)
    for (x0, y0, x1, y1) in boxes:
        bw, bh = x1 - x0, y1 - y0
        cx, cy = x0 + bw / 2.0, y0 + bh / 2.0
        # the curve spans about 2.3 units wide and 2.2 tall; keep a one-pixel margin
        unit = max((min(bw, bh) - 2) / 2.4, 1.0) * scale
        for y in range(y0, y1):
            for x in range(x0, x1):
                hits = 0
                for sy in range(ss):
                    for sx in range(ss):
                        u = (x + (sx + 0.5) / ss - cx) / unit
                        v = -(y + (sy + 0.5) / ss - cy) / unit + 0.1
                        hits += heart_coverage(u, v)
                if hits:
                    k = hits / float(ss * ss)
                    i = (y * width + x) * 3
                    buf[i:i + 3] = bytes(int(c * k) for c in color)
    return bytes(buf)


def beat(t: float, period: float = 1.0) -> float:
    """Heartbeat size over time: two quick swells per period, resting at 0.82."""
    p = (t % period) / period
    swell = math.exp(-((p - 0.12) / 0.07) ** 2) + 0.6 * math.exp(-((p - 0.36) / 0.07) ** 2)
    return 0.82 + 0.18 * min(swell, 1.0)


class WledPanels:
    """Shows the heart on a WLED matrix; show_heart() returns at once, the frames go out on a thread."""

    def __init__(self, host: str, seconds: float = 3.0, logger=None, http_port: int = HTTP_PORT, udp_port: int = UDP_PORT):
        self.host = host
        self.seconds = seconds
        self.http_port, self.udp_port = int(http_port), int(udp_port)
        self.log = logger
        self._layout = None           # (width, height, boxes), read from the WLED once
        self._busy = threading.Lock()

    def layout(self):
        if self._layout is None:
            with urllib.request.urlopen(f"http://{self.host}:{self.http_port}/json", timeout=2.0) as r:
                d = json.loads(r.read())
            m = d["info"]["leds"].get("matrix")
            if not m:
                raise RuntimeError("the WLED is not set up as a 2D matrix")
            w, h = int(m["w"]), int(m["h"])
            boxes = [(s["start"], s.get("startY", 0), s["stop"], s.get("stopY", h))
                     for s in d["state"]["seg"] if s.get("stop", 0) > s.get("start", 0)]
            self._layout = (w, h, boxes or [(0, 0, w, h)])
        return self._layout

    def show_heart(self, seconds: float = None) -> bool:
        """Start the heart (False when one is already showing)."""
        if not self.host or not self._busy.acquire(blocking=False):
            return False
        threading.Thread(target=self._run, args=(seconds or self.seconds,), daemon=True).start()
        return True

    def _run(self, seconds: float):
        try:
            self.play(seconds)
        except Exception as e:  # noqa: BLE001 — the panels are decoration; never take the face node down
            if self.log:
                self.log.warning(f"WLED panels at {self.host}: {e}")
        finally:
            self._busy.release()

    def play(self, seconds: float):
        w, h, boxes = self.layout()
        frames = {}                   # the beat repeats: render each size once
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            t0 = time.monotonic()
            n = 0
            while True:
                t = time.monotonic() - t0
                if t >= seconds:
                    break
                scale = round(beat(t), 2)
                if scale not in frames:
                    frames[scale] = heart_frame(w, h, boxes, scale)
                self._send(sock, frames[scale], timeout_s=2)
                n += 1
                time.sleep(max(0.0, t0 + n / FPS - time.monotonic()))
            # a zero timeout ends realtime mode now: the WLED resumes its own effect
            sock.sendto(bytes([4, 0, 0, 0, 0, 0, 0]), (self.host, self.udp_port))
        finally:
            sock.close()

    def _send(self, sock, rgb: bytes, timeout_s: int):
        total = len(rgb) // 3
        for start in range(0, total, DNRGB_MAX_LEDS):
            chunk = rgb[start * 3:(start + DNRGB_MAX_LEDS) * 3]
            sock.sendto(bytes([4, timeout_s, start >> 8, start & 0xFF]) + chunk, (self.host, self.udp_port))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    panels = WledPanels(sys.argv[1])
    print("layout:", panels.layout())
    panels.play(float(sys.argv[2]) if len(sys.argv) > 2 else 3.0)
