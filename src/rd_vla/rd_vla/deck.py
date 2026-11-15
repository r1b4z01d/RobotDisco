"""The Elgato Stream Deck (15 keys) as the teleop's control surface: v1 teleop/streamdeck.py and the key logic of
manual_panel.py. Needs the `streamdeck` pip package and Pillow in the interpreter the teleop runs under, and the
udev rule that opens the deck's hidraw node. Without a deck, or without the library, the teleop runs as before.

While teleop is off (not enabled, not in freedrive, nothing being recorded or rated) the deck is the robot's remote:

    row 1   0 TELEOP on     1 speaker mute     2 volume -     3 volume +     4 HALT
    row 2   5 microphone mute                  6..9   landmarks 1..4: press, then press again within 3 s to go
    row 3  10..14  landmarks 5..9

Those keys go through the manager's HTTP API on this machine (halt, volume, landmarks, the microphone), so they do
what the pages' buttons do. With teleop on:

    row 1   0 ● REC  (recording: + flag a re-grasp, END closes the flagged stretch)
            1 ⌂ HOME (recording: 💾 SAVE)      2 🆓 FREEDRIVE (recording: 🗑 DISCARD)
            3 ⟳ recover the arm                4 ✖ teleop off
    row 2   5 🖐 open the hand     6..9   open finger 1..4     (held)
    row 3  10 ✊ close the hand   11..14  close finger 1..4    (held)
    after a save, for 30 s: row 3 is 1..5 stars for that episode

Fingers are numbered as they sit on the hand, thumb first: 1 thumb, 2 index, 3 middle, 4 ring. The curl channels
(the dataset's action columns: index, middle, ring, thumb) do not drive the fingers their names suggest on this
hand (v1, measured 2026-10-26: channel 0 moves the middle finger, 1 the ring, 2 the index, 3 the thumb), so the
keys map a finger number to its channel.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request

FINGER_CHANNEL = {1: 3, 2: 2, 3: 0, 4: 1}
# key -> (curl channel or -1 for the whole hand, +1 close / -1 open)
FINGER_KEYS = {5: (-1, -1), **{5 + n: (FINGER_CHANNEL[n], -1) for n in (1, 2, 3, 4)},
               10: (-1, +1), **{10 + n: (FINGER_CHANNEL[n], +1) for n in (1, 2, 3, 4)}}
FINGER_LABELS = {5: "🖐", 6: "1", 7: "2", 8: "3", 9: "4", 10: "✊", 11: "1", 12: "2", 13: "3", 14: "4"}
RATING_KEYS = (10, 11, 12, 13, 14)
RATING_SECONDS = 30.0
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
EMOJI_EXTRA = {"✊", "✋", "👍", "👎", "✅"}
RED, WHITE, DARK = (225, 45, 45), (240, 240, 240), (25, 25, 25)


class StreamDeckPad:
    """The device: open it, draw a label on a key, report key presses. flip = mounted upside down."""

    def __init__(self, flip: bool = False):
        self.deck = None
        self.key_count = 0
        self.flip = flip
        self._images: dict = {}                 # (text, bg, fg) -> the key image in the deck's own format, drawn once
        self._io = threading.Lock()             # one image on the wire at a time: two interleaved arrive as garbage

    def _phys(self, key: int) -> int:
        return (self.key_count - 1 - key) if self.flip else key

    def open(self) -> "StreamDeckPad":
        from StreamDeck.DeviceManager import DeviceManager
        decks = DeviceManager().enumerate()
        if not decks:
            raise RuntimeError("no Stream Deck found")
        self.deck = decks[0]
        self.deck.open()
        self.deck.reset()
        self.deck.set_brightness(60)
        self.key_count = self.deck.key_count()
        return self

    def set_label(self, key: int, text: str, bg=DARK, fg=WHITE) -> None:
        if self.deck is None or not 0 <= key < self.key_count:
            return
        cached = self._images.get((text, bg, fg))
        if cached is not None:                    # a blinking key is the same two pictures over and over
            with self._io:
                self.deck.set_key_image(self._phys(key), cached)
            return
        self._drawing = (text, bg, fg)
        from PIL import Image, ImageDraw, ImageFont
        from StreamDeck.ImageHelpers import PILHelper
        img = PILHelper.create_image(self.deck, background=bg)
        draw = ImageDraw.Draw(img)
        lines = text.split("\n")
        w, h = img.size
        if len(lines) == 1 and text and all(ord(c) >= 0x1F000 or c in EMOJI_EXTRA or c == "️" for c in text):
            try:                                  # a lone emoji: the colour font exists only as 109 px bitmaps
                big = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
                ImageDraw.Draw(big).text((20, 10), text, font=ImageFont.truetype(EMOJI_FONT, 109), embedded_color=True)
                bb = big.getbbox()
                if bb:
                    glyph = big.crop(bb)
                    scale = min(w, h) * 0.78 / max(glyph.width, glyph.height)
                    glyph = glyph.resize((max(1, int(glyph.width * scale)), max(1, int(glyph.height * scale))), Image.LANCZOS)
                    img = img.convert("RGBA")
                    img.alpha_composite(glyph, ((w - glyph.width) // 2, (h - glyph.height) // 2))
                    img = img.convert("RGB")
                    self._show(key, img)
                    return
            except Exception:  # noqa: BLE001  no emoji font here: draw the text instead
                pass
        font = None
        for size in (44, 36, 30, 26, 22, 20, 18, 16, 14, 12):     # as large as the widest line allows
            try:
                cand = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
            except Exception:  # noqa: BLE001
                break
            font = cand
            widest = max((cand.getbbox(ln)[2] - cand.getbbox(ln)[0]) for ln in lines) if lines else 0
            if widest <= w - 8 and (cand.getbbox("Ag")[3] - cand.getbbox("Ag")[1] + 4) * len(lines) <= h - 6:
                break
        font = font or ImageFont.load_default()
        line_h = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + 3
        y = (h - line_h * len(lines)) // 2
        for ln in lines:
            bb = draw.textbbox((0, 0), ln, font=font)
            draw.text(((w - (bb[2] - bb[0])) // 2, y), ln, font=font, fill=fg)
            y += line_h
        self._show(key, img)

    def _show(self, key: int, img) -> None:
        from PIL import Image
        from StreamDeck.ImageHelpers import PILHelper
        if self.flip:
            img = img.transpose(Image.ROTATE_180)
        native = PILHelper.to_native_format(self.deck, img)
        what = getattr(self, "_drawing", None)
        if what is not None:
            self._images[what] = native
        with self._io:
            self.deck.set_key_image(self._phys(key), native)

    def on_key(self, callback) -> None:
        self.deck.set_key_callback(lambda _deck, key, state: callback(self._phys(key), bool(state)))

    def close(self) -> None:
        if self.deck is not None:
            try:
                self.deck.reset()
                self.deck.close()
            except Exception:  # noqa: BLE001
                pass
            self.deck = None


LANDMARK_KEYS = (6, 7, 8, 9, 10, 11, 12, 13, 14)
ARM_SECONDS = 3.0             # a landmark key drives the robot: the second press must follow the first this soon
VOLUME_STEP = 10


class Remote:
    """The idle keys' line to the manager (http://127.0.0.1:<port>): its state fetched every two seconds on a thread,
    its commands sent on threads, so no key ever waits on the network. `fetch` is injectable for the tests."""

    def __init__(self, base: str, log=print, fetch=None):
        self.base, self.log = base.rstrip("/"), log
        self.fetch = fetch or self._http
        self.volume, self.muted, self.mic_muted, self.landmarks = None, False, False, []
        self._stop = threading.Event()
        self._thread = None

    def _http(self, path: str, body: dict | None = None, timeout: float = 4.0) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="deck-remote")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def refresh(self) -> None:
        try:
            v = self.fetch("/api/volume")
            self.volume, self.muted = v.get("volume"), bool(v.get("muted"))
            self.mic_muted = bool(self.fetch("/api/voice").get("mic_muted"))
            d = self.fetch("/api/landmarks")
            here = d.get("map")
            self.landmarks = [(lm["id"], lm.get("name") or lm["id"]) for lm in d.get("landmarks", []) if lm.get("map") == here]
        except Exception:  # noqa: BLE001  the manager is away: keep what was known
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(2.0)

    def send(self, what: str, path: str, body: dict) -> None:
        def work():
            try:
                self.fetch(path, body)
                self.log(f"deck: {what}")
            except Exception as e:  # noqa: BLE001
                self.log(f"deck: {what} failed ({e})")
            self.refresh()
        threading.Thread(target=work, daemon=True, name="deck-send").start()


class Deck:
    """The keys' behaviour. `teleop` is the node (command(), nudge(), rate(), recording, freedrive, ...); `pad` may
    be handed in (the tests' fake), else the device is opened here and looked for again every few seconds."""

    def __init__(self, teleop, flip: bool = True, pad=None, clock=time.monotonic, remote=None):
        self.t, self.flip, self.clock = teleop, flip, clock
        self.remote = remote                # the idle keys' line to the manager; None = teleop keys only (the old deck)
        self._armed = (None, 0.0)           # (landmark key, until): pressed once, waiting for the second press
        self._was_idle = None
        self.pad = pad
        self.events: queue.Queue = queue.Queue()
        self.held: dict = {}
        self.rating_until = 0.0
        self.save_flash_until = 0.0
        self._shown: dict = {}
        self._tried = -1e9
        self._want_rating = False
        self._unavailable = False
        self._polling = threading.Lock()
        if pad is not None:
            self._attach()

    # ---- the device
    def _attach(self) -> None:
        self._shown.clear()
        self._was_idle = None
        self.pad.on_key(lambda key, down: self.events.put((key, down)))
        if self.remote is not None:
            self.remote.start()

    def idle(self) -> bool:
        """Teleop off and nothing of it pending: the keys are the robot's remote."""
        t = self.t
        return self.remote is not None and not (t.enabled or t.freedrive or t.recording or getattr(t, "saving", False)
                                                or self.rating_until or self._want_rating)

    def _open(self) -> None:
        now = self.clock()
        if self._unavailable or now - self._tried < 5.0:
            return
        self._tried = now
        try:
            self.pad = StreamDeckPad(flip=self.flip).open()
            self._attach()
            self.t.log(f"Stream Deck ready ({self.pad.key_count} keys)")
        except ImportError as e:
            self._unavailable = True                    # no library in this interpreter: do not keep asking
            self.t.log(f"Stream Deck: python package missing ({e}); provision.sh venvs installs it")
        except Exception:  # noqa: BLE001  not plugged in, or busy: look again later
            self.pad = None

    def _label(self, key: int, text: str, bg=DARK, fg=WHITE) -> None:
        if self._shown.get(key) != (text, bg, fg):
            self._shown[key] = (text, bg, fg)
            try:
                self.pad.set_label(key, text, bg=bg, fg=fg)
            except Exception:  # noqa: BLE001  unplugged while drawing
                self.pad = None

    def close(self) -> None:
        if self.pad is not None:
            self.pad.close()
            self.pad = None

    # ---- the keys
    def enter_rating(self) -> None:
        """Called by the node (its save thread) when an episode has been written: the stars come up on the next poll."""
        self._want_rating = True

    def _press_idle(self, key: int, down: bool) -> None:
        if not down:
            return
        r, now = self.remote, self.clock()
        armed_key, until = self._armed
        self._armed = (None, 0.0)                       # any other key disarms a landmark
        if key == 0:
            self.t.command("enable", {"on": True})
        elif key == 1:
            r.muted = not r.muted
            r.send("speaker muted" if r.muted else "speaker on", "/api/volume", {"muted": r.muted})
        elif key in (2, 3):
            if r.volume is not None:
                r.volume = max(0, min(100, int(r.volume) + (VOLUME_STEP if key == 3 else -VOLUME_STEP)))
                r.send(f"volume {r.volume} %", "/api/volume", {"volume": r.volume})
        elif key == 4:
            r.send("HALT", "/api/halt", {})
        elif key == 5:
            r.mic_muted = not r.mic_muted
            r.send("microphone muted" if r.mic_muted else "microphone on", "/api/voice/mic", {"muted": r.mic_muted})
        elif key in LANDMARK_KEYS:
            i = LANDMARK_KEYS.index(key)
            if i < len(r.landmarks):
                lid, name = r.landmarks[i]
                if armed_key == key and now < until:
                    r.send(f"going to {name}", f"/api/landmarks/{lid}/goto", {})
                else:
                    self._armed = (key, now + ARM_SECONDS)

    def _render_idle(self) -> None:
        r, now = self.remote, self.clock()
        self._label(0, "TELEOP\nON", bg=(25, 90, 40))
        self._label(1, "🔇" if r.muted else "🔊", bg=(120, 30, 30) if r.muted else DARK)
        vol = "—" if r.volume is None else f"{int(r.volume)}"
        self._label(2, f"VOL −\n{vol}")
        self._label(3, f"VOL +\n{vol}")
        self._label(4, "HALT", bg=(190, 25, 25))
        self._label(5, "MIC\nOFF" if r.mic_muted else "MIC\nON", bg=(120, 30, 30) if r.mic_muted else DARK)
        armed_key, until = self._armed
        if armed_key is not None and now >= until:
            self._armed, armed_key = (None, 0.0), None
        for i, key in enumerate(LANDMARK_KEYS):
            if i < len(r.landmarks):
                name = r.landmarks[i][1]
                if key == armed_key:
                    self._label(key, "GO?\n" + name.split()[0][:8], bg=(200, 130, 0))
                else:
                    self._label(key, "\n".join(w[:9] for w in name.split()[:3]), bg=(30, 50, 90))
            else:
                self._label(key, "")

    def _press(self, key: int, down: bool) -> None:
        t = self.t
        if self.idle():
            self.held.clear()
            self._press_idle(key, down)
            return
        if self.rating_until and key in RATING_KEYS:
            if down:
                t.log(t.rate(RATING_KEYS.index(key) + 1))
                self.rating_until = 0.0
            return
        if key in FINGER_KEYS:
            self.held[key] = down
            return
        if not down:
            return
        if key == 0:
            t.command("grasp_mark" if t.recording else "rec")
        elif key == 1:
            if t.recording:
                self.save_flash_until = self.clock() + 1.5
                t.command("save")
            else:
                t.command("home")
        elif key == 2:
            t.command("discard" if t.recording else "freedrive")
        elif key == 3:
            t.command("reconnect")
        elif key == 4:
            t.command("quit")

    def poll(self) -> None:
        # One poll at a time. The node's timer may fire again on another executor thread while this one is still at
        # work (a key that starts a recording waits for the first frames): two at once drew over each other's key
        # images, which arrived distorted, and started the same recording twice (2026-11-15).
        if not self._polling.acquire(blocking=False):
            return
        try:
            self._poll()
        finally:
            self._polling.release()

    def _poll(self) -> None:
        if self.pad is None:
            self._open()
            if self.pad is None:
                return
        while True:
            try:
                key, down = self.events.get_nowait()
            except queue.Empty:
                break
            self._press(key, down)
        t = self.t
        close_rate, open_rate = float(t.p("grasp_close_rate")), float(t.p("grasp_open_rate"))
        scale = 0.1 * float(t.p("fps"))                 # the rates are per teleop tick; this runs every 0.1 s
        for key, down in self.held.items():
            if down and not (self.rating_until and key in RATING_KEYS):
                finger, sign = FINGER_KEYS[key]
                t.nudge((close_rate if sign > 0 else -open_rate) * scale, finger)
        self._render()

    def _render(self) -> None:
        t, now = self.t, self.clock()
        if self._want_rating:
            self._want_rating, self.rating_until = False, now + RATING_SECONDS
        if self.rating_until and now > self.rating_until:
            self.rating_until = 0.0
        idle = self.idle()
        if idle != self._was_idle:                      # the other set of keys: nothing stays held or armed across it
            self._was_idle, self._armed = idle, (None, 0.0)
            self.held.clear()
        if idle:
            self._render_idle()
            return
        for key in (5, 6, 7, 8, 9):
            self._label(key, FINGER_LABELS[key])
        self._label(3, "⟳")
        blink = int(time.time() * 1.5) % 2 == 0
        rec = bool(t.recording)
        if rec:
            open_range = t._regrasp_open is not None
            grasps = len(t.regrasp_ranges) + 1
            bg = ((200, 130, 0) if blink else (60, 38, 0)) if open_range else ((200, 30, 30) if blink else (45, 0, 0))
            self._label(0, f"END\n{grasps + 1}" if open_range else f"+\n{grasps}", bg=bg)
            self._label(1, "💾", bg=((20, 170, 20) if blink else (0, 55, 0)) if now < self.save_flash_until else DARK)
            self._label(2, "🗑")
        else:
            self._label(0, "●", fg=RED)
            self._label(1, "⌂", bg=(150, 112, 22))
            self._label(2, "🆓", bg=(30, 90, 175) if t.freedrive else (60, 60, 64))
        self._label(4, "✖", bg=(90, 20, 20) if (t.enabled or t.freedrive) else DARK, fg=WHITE if (t.enabled or t.freedrive) else RED)
        for i, key in enumerate(RATING_KEYS):
            if self.rating_until:
                self._label(key, f"{i + 1}\n★", bg=(70, 55, 10), fg=(245, 210, 60))
            else:
                self._label(key, FINGER_LABELS[key])
