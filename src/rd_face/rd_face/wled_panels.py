"""The WLED matrix panels: a steady heart on each matrix when a known person is recognised.

The frames go out as WLED realtime UDP (DNRGB, port 21324). Realtime data only overrides what the
panels show while it keeps arriving: once the stream stops, WLED goes back to the effect it was
running by itself, so nothing has to be saved or restored and a lost packet cannot leave the heart
up. The layout (matrix size, one heart per segment) is read from the WLED's JSON API.

v1's robot_face/wled_panels.py (2026-11-09) with the two ports as arguments, so the tests can stand a fake WLED on
loopback. Standalone test (no ROS):  python3 wled_panels.py <panels.ip from robot.yaml> [seconds]
"""
import json
import socket
import sys
import threading
import time
import urllib.request

UDP_PORT = 21324          # WLED realtime UDP
HTTP_PORT = 80            # WLED JSON API
# LEDs per DNRGB packet. The protocol allows 489 (a 1471-byte datagram), but this WLED takes in only the first 1460
# bytes of one: the last four LEDs of every packet stayed dark, four holes in a row wherever a packet ended inside
# the heart (seen in its own live view, 2026-11-19). 480 keeps well inside that.
DNRGB_MAX_LEDS = 480
DDP_PORT = 4048


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


TREE_GREEN, TREE_DARK, TREE_TRUNK, TREE_STAR = (0, 150, 30), (0, 95, 20), (110, 55, 15), (255, 200, 0)
TREE_LIGHTS = ((255, 0, 0), (255, 190, 0), (0, 90, 255), (255, 255, 255), (255, 0, 160))


def tree_parts(width: int, height: int, boxes):
    """A Christmas tree in each box, the full height of it: (still, lights, star). `still` is the RGB frame of what
    never changes (the green tiers and the trunk); lights is [(pixel index, colour, group)], group 0..2 by colour, so
    each group can follow a different part of the music; star is the star's pixel indices."""
    buf = bytearray(width * height * 3)
    lights, star = [], []

    def put(x, y, c):
        if 0 <= x < width and 0 <= y < height:
            buf[(y * width + x) * 3:(y * width + x) * 3 + 3] = bytes(c)

    for (x0, y0, x1, y1) in boxes:
        bw, bh = x1 - x0, y1 - y0
        cx = x0 + bw // 2                       # the tree's axis is between columns cx - 1 and cx
        sy = y0 + 2                             # the star's centre: it takes rows y0 .. y0 + 4
        top, base = y0 + 5, y1 - max(4, bh // 10)       # the green runs from top to base, the trunk from there to the last row
        half_max = bw // 2 - 1
        tiers = 3
        tier_h = (base - top) / tiers
        inside = []
        for y in range(top, base):
            t = int((y - top) / tier_h)                           # which tier, 0 at the top
            f = ((y - top) - t * tier_h) / tier_h                 # 0..1 down this tier
            # each tier widens from a little over half the one above's base to its own base
            lo, hi = half_max * (t + 0.25) / (tiers + 0.25) * 0.55, half_max * (t + 1.25) / (tiers + 0.25)
            half = max(1, int(round(lo + (hi - lo) * f)))
            for x in range(cx - half, cx + half):
                put(x, y, TREE_DARK if (x + y) % 5 == 0 else TREE_GREEN)
                inside.append((x, y, half))
        # the lights: a fixed scatter, the same every time (no random: the two panels must match)
        n = 0
        for (x, y, half) in inside:
            if half >= 3 and abs(x - cx) < half - 1 and (x * 7 + y * 13) % 23 == 0:
                lights.append((y * width + x, TREE_LIGHTS[n % len(TREE_LIGHTS)], n % 3))
                n += 1
        tw = max(2, bw // 8)
        for y in range(base, y1):
            for x in range(cx - tw // 2, cx - tw // 2 + tw):
                put(x, y, TREE_TRUNK)
        for dx in range(-3, 3):                                   # the star: a small diamond
            for dy in range(-2, 3):
                if dx in (-1, 0) or dy == 0 or (abs(dx + 0.5) < 2 and abs(dy) < 2):
                    x, y = cx + dx, sy + dy
                    if x0 <= x < x1 and y0 <= y < y1:
                        star.append(y * width + x)
    return bytes(buf), lights, star


def tree_frame(width: int, height: int, boxes, levels=(1.0, 1.0, 1.0), star_level: float = 1.0) -> bytes:
    """The tree's RGB frame with each group of lights at its level (0..1) and the star at star_level."""
    still, lights, star = tree_parts(width, height, boxes)
    return _tree_lit(still, lights, star, levels, star_level)


def _tree_lit(still: bytes, lights, star, levels, star_level: float) -> bytes:
    buf = bytearray(still)
    for i, colour, group in lights:
        k = max(0.0, min(1.0, levels[group]))
        buf[i * 3:i * 3 + 3] = bytes(int(c * k) for c in colour)
    k = max(0.0, min(1.0, star_level))
    lit = bytes(int(c * k) for c in TREE_STAR)
    for i in star:
        buf[i * 3:i * 3 + 3] = lit
    return bytes(buf)


SOUND_SYNC_GROUP, SOUND_SYNC_PORT = "239.0.0.1", 11988     # the WLED's "UDP Sound Sync": what its microphone hears, 20 a second


class MicLevels:
    """How loud the bass, the middle and the treble are, each 0..1 against its own recent peak, from the WLED's sound
    sync packets (its microphone's 16 frequency channels). With no sound for a while every level rests at `rest`."""

    BANDS = ((0, 4), (4, 10), (10, 16))        # which of the 16 channels: bass, middle, treble

    def __init__(self, rest: float = 0.6):
        self.rest = rest
        self.peaks = [8.0, 8.0, 8.0]           # a floor: room noise must not be stretched to full brightness
        self.levels = [rest, rest, rest]
        self.quiet = 0

    def feed(self, packet: bytes):
        if len(packet) < 34 or packet[:5] != b"00002":
            return self.levels
        bins = packet[18:34]
        if max(bins) < 6:                                        # silence: glow steadily
            self.quiet += 1
            if self.quiet > 20:
                self.levels = [lv + (self.rest - lv) * 0.15 for lv in self.levels]
                return self.levels
        else:
            self.quiet = 0
        for b, (lo, hi) in enumerate(self.BANDS):
            e = sum(bins[lo:hi]) / float(hi - lo)
            self.peaks[b] = max(e, self.peaks[b] * 0.995, 8.0)   # the band's own loudest lately: an automatic gain
            v = (e / self.peaks[b]) ** 1.5                       # the exponent keeps the lights dark between beats
            self.levels[b] = v if v > self.levels[b] else self.levels[b] * 0.6 + v * 0.4     # up at once, down slower
        return self.levels


class WledMic:
    """The WLED's microphone as the robot hears of it: its sound sync packets (multicast). read() gives the newest
    one, or b"" when none has come since the last call."""

    def __init__(self, host: str):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect((host, SOUND_SYNC_PORT))               # which of this machine's addresses faces the WLED
            mine = probe.getsockname()[0]
        finally:
            probe.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", SOUND_SYNC_PORT))
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(SOUND_SYNC_GROUP) + socket.inet_aton(mine))
        self.sock.setblocking(False)

    def read(self) -> bytes:
        newest = b""
        while True:
            try:
                newest = self.sock.recv(256)
            except (BlockingIOError, OSError):
                return newest

    def close(self) -> None:
        self.sock.close()


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
        frame = heart_frame(w, h, boxes, 1.0)         # one steady, full-size heart (it used to beat: Bryan, 2026-11-13)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            t0 = time.monotonic()
            while time.monotonic() - t0 < seconds:
                self._send(sock, frame, timeout_s=2)      # resent well inside the WLED's realtime timeout
                time.sleep(min(0.25, max(0.0, t0 + seconds - time.monotonic())))
            # a zero timeout ends realtime mode now: the WLED resumes its own effect
            sock.sendto(bytes([4, 0, 0, 0, 0, 0, 0]), (self.host, self.udp_port))
        finally:
            sock.close()

    def hold(self, frame_of) -> None:
        """Put a still picture up and leave it: frame_of(width, height, boxes) -> RGB bytes. A realtime timeout of 255
        is the WLED's "until told otherwise", so nothing has to keep sending. release() gives the panels back."""
        w, h, boxes = self.layout()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            frame = frame_of(w, h, boxes)
            for _ in range(3):                    # UDP: three times, a lost packet would leave a band of the old picture
                self._send(sock, frame, timeout_s=255)
                time.sleep(0.05)
        finally:
            sock.close()

    def sound_sync(self, on: bool = True) -> None:
        """The WLED sends what its microphone hears (UDP Sound Sync, send mode) or stops. A saved setting of its."""
        want = 1 if on else 0
        with urllib.request.urlopen(f"http://{self.host}:{self.http_port}/json/cfg", timeout=3.0) as r:
            now = (json.loads(r.read()).get("um", {}).get("AudioReactive", {}).get("sync", {}) or {}).get("mode")
        if now != want:
            req = urllib.request.Request(f"http://{self.host}:{self.http_port}/json/cfg", method="POST", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"um": {"AudioReactive": {"sync": {"mode": want}}}}).encode())
            urllib.request.urlopen(req, timeout=4.0).read()

    def _send_ddp(self, sock, rgb: bytes) -> None:
        """One frame by DDP (:4048). Unlike DNRGB, the WLED keeps its microphone running while DDP pixels arrive
        (it suspends sound processing in its "UDP" realtime mode: seen in its info page, 2026-11-20)."""
        n, off, seq = len(rgb), 0, 1
        while off < n:
            chunk = rgb[off:off + 1440]
            last = off + len(chunk) >= n
            sock.sendto(bytes([0x41 if last else 0x40, seq & 15, 0x01, 1]) + off.to_bytes(4, "big") + len(chunk).to_bytes(2, "big") + chunk,
                        (self.host, DDP_PORT))
            off += len(chunk)
            seq += 1

    def tree_show(self, stop, audio=None, fps: float = 20.0) -> None:
        """The tree until stop (a threading.Event) is set, its lights and star following the music: `audio` is a
        callable returning the WLED's newest sound sync packet (or b"" for none: WledMic.read); without one the
        lights are steady. Blocks; run it on a thread. The panels are given back at the end."""
        w, h, boxes = self.layout()
        still, lights, star = tree_parts(w, h, boxes)
        music = MicLevels()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            last, last_sent = None, 0.0
            while not stop.is_set():
                t0 = time.monotonic()
                pcm = audio() if audio else b""
                bass, mid, treble = music.feed(pcm) if pcm else music.levels
                frame = _tree_lit(still, lights, star, (0.08 + 0.92 * bass, 0.08 + 0.92 * mid, 0.08 + 0.92 * treble), 0.35 + 0.65 * bass)
                if frame != last or t0 - last_sent > 0.5:        # an unchanged picture is only kept alive
                    self._send_ddp(sock, frame)
                    last, last_sent = frame, t0
                stop.wait(max(0.0, 1.0 / fps - (time.monotonic() - t0)))
            sock.sendto(bytes([4, 0, 0, 0, 0, 0, 0]), (self.host, self.udp_port))
        finally:
            sock.close()

    def release(self) -> None:
        """End realtime mode now: the WLED resumes its own effect."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for _ in range(2):
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
