"""The WLED panels module without a WLED: the heart's geometry and beat, and against a fake WLED on loopback
(its JSON layout over HTTP, a UDP socket for the realtime frames) the layout it reads and the DNRGB packets it sends."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from rd_face.wled_panels import DNRGB_MAX_LEDS, WledPanels, beat, heart_frame

W, H = 32, 16
WLED_JSON = {"info": {"leds": {"matrix": {"w": W, "h": H}}},
             "state": {"seg": [{"id": 0, "start": 0, "stop": 16, "startY": 0, "stopY": 16, "n": "Right"},
                               {"id": 1, "start": 16, "stop": 32, "startY": 0, "stopY": 16, "n": "Left"},
                               {"id": 2, "start": 0, "stop": 0}]}}       # an unused segment: skipped


class FakeWled:
    def __init__(self, body=WLED_JSON):
        fake = self
        self.packets = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass
        self.http = HTTPServer(("127.0.0.1", 0), Handler)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(("127.0.0.1", 0))
        self.udp.settimeout(0.2)
        self.http_port, self.udp_port = self.http.server_address[1], self.udp.getsockname()[1]
        self._stop = threading.Event()
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        threading.Thread(target=self._recv, daemon=True).start()
        del fake

    def _recv(self):
        while not self._stop.is_set():
            try:
                self.packets.append(self.udp.recv(4096))
            except socket.timeout:
                continue
            except OSError:
                return

    def stop(self):
        self._stop.set()
        self.http.shutdown()
        self.udp.close()


def lit(buf, w, x0, y0, x1, y1):
    return sum(1 for y in range(y0, y1) for x in range(x0, x1) if any(buf[(y * w + x) * 3:(y * w + x) * 3 + 3]))


def test_heart_frame_and_beat():
    boxes = [(0, 0, 16, 16), (16, 0, 32, 16)]
    f = heart_frame(W, H, boxes, 1.0)
    assert len(f) == W * H * 3
    left, right = lit(f, W, 0, 0, 16, 16), lit(f, W, 16, 0, 32, 16)
    assert left == right and 60 < left < 16 * 16                      # one heart per box, the same in each
    assert not any(f[(0 * W + x) * 3 + c] for x in range(W) for c in range(3))      # the top row stays dark (margin)
    assert f[(8 * W + 8) * 3] == 255 and f[(8 * W + 8) * 3 + 1] == 0   # the middle of a heart is full red
    # the two lobes: the row near the top is lit left and right of the centre notch, the bottom row only at the tip
    small = heart_frame(W, H, boxes, 0.82)
    assert lit(small, W, 0, 0, 16, 16) < left                          # the resting size is smaller
    vals = [beat(t / 100.0) for t in range(100)]
    assert min(vals) >= 0.82 and max(vals) <= 1.0 and max(vals) > 0.99 and abs(beat(0.7) - 0.82) < 0.01
    assert beat(0.12) == beat(1.12)                                    # periodic


def test_layout_and_realtime_packets():
    wled = FakeWled()
    try:
        p = WledPanels("127.0.0.1", seconds=0.6, http_port=wled.http_port, udp_port=wled.udp_port)
        assert p.layout() == (W, H, [(0, 0, 16, 16), (16, 0, 32, 16)])
        assert p.show_heart() is True
        assert p.show_heart() is False                                  # one at a time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any(pk[:2] == bytes([4, 0]) for pk in wled.packets):
            time.sleep(0.05)
        pk = list(wled.packets)
        assert pk[-1] == bytes([4, 0, 0, 0, 0, 0, 0])                    # realtime mode ended: the WLED resumes its effect
        frames = [x for x in pk[:-1] if x[2:4] == bytes([0, 0])]
        tails = [x for x in pk[:-1] if x[2:4] != bytes([0, 0])]
        assert len(frames) >= 5 and len(frames) == len(tails)            # ~15 fps for 0.6 s, two packets per 512-LED frame
        assert all(x[0] == 4 and x[1] == 2 for x in pk[:-1])             # DNRGB, 2 s timeout
        assert len(frames[0]) == 4 + DNRGB_MAX_LEDS * 3
        start = (tails[0][2] << 8) | tails[0][3]
        assert start == DNRGB_MAX_LEDS and len(tails[0]) == 4 + (W * H - DNRGB_MAX_LEDS) * 3
        assert any(frames[0][4:])                                        # there is a heart in it
        time.sleep(0.2)
        assert p.show_heart() is True                                    # free again after the first one ended
    finally:
        wled.stop()


def test_off_and_unreachable_are_quiet():
    assert WledPanels("").show_heart() is False                          # "" = no panels

    class Log:
        lines = []

        def warning(self, m):
            self.lines.append(m)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                                            # nothing listens here
    p = WledPanels("127.0.0.1", seconds=0.5, logger=Log(), http_port=port)
    assert p.show_heart() is True                                        # returns at once; the failure stays on its thread
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not Log.lines:
        time.sleep(0.05)
    assert Log.lines and "WLED panels" in Log.lines[0]
    time.sleep(0.1)
    assert p.show_heart() is True                                        # and it is not left busy
