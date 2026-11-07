"""A renderer with the GlEyeRenderer's call shape that draws to a numpy canvas (eye_renderer.draw_eye + a
mouth polyline), for the rig and for a face preview image topic: no window, no GL. Same mood / blink / gaze /
wink / mouth inputs as the GL renderer."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .eye_renderer import draw_eye


class HeadlessRenderer:
    def __init__(self, width=540, height=950, eye_hw=75.0, eye_hh=75.0, eye_side_margin=40.0, eye_top_margin=40.0,
                 mouth_enabled=True, mouth_gap=275.0, mouth_hw=None, mouth_amp=35.0, **_):
        self.width, self.height = int(width), int(height)
        self.eye_hw, self.eye_hh = float(eye_hw), float(eye_hh)
        self.eye_side_margin, self.eye_top_margin = float(eye_side_margin), float(eye_top_margin)
        self.mouth_enabled, self.mouth_gap, self.mouth_amp = bool(mouth_enabled), float(mouth_gap), float(mouth_amp)
        self.mouth_hw = mouth_hw
        self.frame = np.zeros((self.height, self.width, 3), np.uint8)
        self.frames = 0
        self.last = {}

    def should_close(self) -> bool:
        return False

    def set_fullscreen(self, on: bool) -> None:
        pass

    def destroy(self) -> None:
        pass

    def render(self, blink, gaze_offset, mood, dt=0.033, mouth_level=0.0, mouth_samples=None, mouth_smile=0.0,
               mouth_skew=0.0, mouth_shape=0.0, mouth_path=None, wink=0.0, wink_eye=1) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), np.uint8)
        cy = self.eye_top_margin + self.eye_hh
        centers = [(self.eye_side_margin + self.eye_hw, cy), (self.width - self.eye_side_margin - self.eye_hw, cy)]
        for i, (cx, cy_) in enumerate(centers):
            b = max(blink, wink if i == wink_eye else 0.0)
            draw_eye(canvas, (cx + gaze_offset[0], cy_ + gaze_offset[1]), int(self.eye_hw), int(self.eye_hh * 0.8),
                     (0.0, 0.0), b, (0, 0, 0), mood == 3, (255, 220, 100), 5, 12, 8, 2)
        if self.mouth_enabled:
            mcx = self.width / 2.0
            mcy = cy + self.eye_hh + self.mouth_gap
            hw = float(self.mouth_hw) if self.mouth_hw else (centers[1][0] - centers[0][0]) / 2.0
            n = 128
            xs = np.linspace(-1.0, 1.0, n)
            if mouth_samples is not None and len(mouth_samples) > 1:
                src = np.asarray(mouth_samples, dtype=np.float32)
                ys = np.interp(np.linspace(0, len(src) - 1, n), np.arange(len(src)), src) * mouth_level
            else:
                ys = np.sin(xs * math.pi * 3.0) * mouth_level * 0.6
            ys = ys - (1.0 - xs ** 2) * (mouth_smile / max(self.mouth_amp, 1e-6)) * 0.5 + xs * (mouth_skew / max(self.mouth_amp, 1e-6)) * 0.5
            pts = np.stack([mcx + xs * hw, mcy + ys * self.mouth_amp], axis=1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, (255, 185, 30), 4, cv2.LINE_AA)
        self.frame = canvas
        self.frames += 1
        self.last = {"blink": float(blink), "wink": float(wink), "mood": int(mood), "mouth_level": float(mouth_level),
                     "gaze": (float(gaze_offset[0]), float(gaze_offset[1]))}
        return canvas
