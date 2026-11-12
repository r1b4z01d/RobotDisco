"""The Elgato Stream Deck (15 keys) as the teleop's control surface: v1 teleop/streamdeck.py and the key logic of
manual_panel.py. Needs the `streamdeck` pip package and Pillow in the interpreter the teleop runs under, and the
udev rule that opens the deck's hidraw node. Without a deck, or without the library, the teleop runs as before.

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

import queue
import time

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
        self.deck.set_key_image(self._phys(key), PILHelper.to_native_format(self.deck, img))

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


class Deck:
    """The keys' behaviour. `teleop` is the node (command(), nudge(), rate(), recording, freedrive, ...); `pad` may
    be handed in (the tests' fake), else the device is opened here and looked for again every few seconds."""

    def __init__(self, teleop, flip: bool = True, pad=None, clock=time.monotonic):
        self.t, self.flip, self.clock = teleop, flip, clock
        self.pad = pad
        self.events: queue.Queue = queue.Queue()
        self.held: dict = {}
        self.rating_until = 0.0
        self.save_flash_until = 0.0
        self._shown: dict = {}
        self._tried = -1e9
        self._want_rating = False
        self._unavailable = False
        if pad is not None:
            self._attach()

    # ---- the device
    def _attach(self) -> None:
        self._shown.clear()
        self.pad.on_key(lambda key, down: self.events.put((key, down)))
        for key, label in FINGER_LABELS.items():
            self._label(key, label)
        self._label(3, "⟳")
        self._label(4, "✖", fg=RED)

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

    def _press(self, key: int, down: bool) -> None:
        t = self.t
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
