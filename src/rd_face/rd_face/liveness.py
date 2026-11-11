"""Is a detected face a head in the room, or a face on a screen or in a picture? From the depth image alone.

A head stands in front of whatever is behind it: left and right of it the depth jumps to the background (or has
no return). A face on a television is part of one flat surface: the depth beside it continues the plane, also
when the screen is seen at an angle (then one side is nearer and the other farther by the same amount). The
second check is size: box width x depth / focal length is the face's real width, and a close-up on a 55 inch
screen is half a metre wide.

Used only to decide whether a stranger is ENROLLED in the gallery (v1 learned the cast of whatever was on).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

WIDTH_M = (0.09, 0.30)       # a detector box around a real face, child to adult, loose
FLAT_M = 0.06                # left and right of the face lie this close to the face's own plane: a flat surface
MIN_VALID = 0.2              # a patch needs this share of valid depth pixels to count


def _median(depth_m: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Optional[float]:
    h, w = depth_m.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.1)]
    if valid.size < MIN_VALID * patch.size:
        return None
    return float(np.median(valid))


def measure(depth_m: np.ndarray, box: Tuple[int, int, int, int], fx: float) -> Optional[dict]:
    """{z, width_m, left, right}: the face's depth, its real width, and how much farther (m) the scene is on
    either side of it (None where the depth has no return). None when the face itself has no depth."""
    x, y, w, h = (int(round(v)) for v in box)
    z = _median(depth_m, x + w // 4, y + h // 4, x + 3 * w // 4, y + 3 * h // 4)
    if z is None or fx <= 0:
        return None
    side = max(4, w // 2)
    gap = max(2, w // 8)                                     # clear of the ears and the hair
    left = _median(depth_m, x - gap - side, y + h // 4, x - gap, y + 3 * h // 4)
    right = _median(depth_m, x + w + gap, y + h // 4, x + w + gap + side, y + 3 * h // 4)
    return {"z": z, "width_m": w * z / fx,
            "left": None if left is None else left - z, "right": None if right is None else right - z}


def refusal(depth_m: np.ndarray, box: Tuple[int, int, int, int], fx: float) -> Tuple[Optional[str], Optional[dict]]:
    """(why this face must not be enrolled, the measurement). (None, m) for a head; (None, None) without depth."""
    m = measure(depth_m, box, fx)
    if m is None:
        return None, None
    if not WIDTH_M[0] <= m["width_m"] <= WIDTH_M[1]:
        return f"a face {m['width_m'] * 100:.0f} cm wide at {m['z']:.1f} m is not a person's", m
    if m["left"] is not None and m["right"] is not None and abs((m["left"] + m["right"]) / 2.0) < FLAT_M:
        return f"a flat face at {m['z']:.1f} m (a screen or a picture)", m
    return None, m
