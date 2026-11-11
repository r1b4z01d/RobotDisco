"""The AmazingHand's ASCII line protocol (firmware/AmazingHand ESP32), ROS-free.

    J:<a0>,<a1>,<a2>,<a3>,<a4>,<a5>,<a6>,<a7>,<speed>\\n   per-servo angle offsets in degrees (4 fingers x 2)
    Q\\n  ->  K:<uptime_ms>\\n                               liveness probe
    S:<f0>,<f1>,<f2>,<f3>,<a0>,<a1>,<a2>,<a3>\\n            (firmware with state streaming) flex + abduct

UDP :8766, one line per datagram (the hand is an ESP32 on 2.4 GHz Wi-Fi with 12-25 % loss; TCP :8765 is
the legacy transport and stalls seconds late on that link). Per-finger "curl" in [0, 1] maps to the two
servos of that finger through the open / close tables (verbatim from v1).
"""
from __future__ import annotations

from collections.abc import Mapping

FINGER_ORDER = ("index", "middle", "ring", "thumb")
FINGER_TO_SERVO = {"index": (0, 1), "middle": (2, 3), "ring": (4, 5), "thumb": (6, 7)}
JOINT_OPEN_OFFSETS = (-35.0, 35.0, -35.0, 35.0, -35.0, 35.0, -35.0, 35.0)
JOINT_CLOSE_OFFSETS = (90.0, -90.0, 90.0, -90.0, 90.0, -90.0, 70.0, -70.0)
DEFAULT_SPEED = 200          # firmware units 50..2000
SPEED_MIN, SPEED_MAX = 50, 2000


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def curls_to_offsets(curls) -> list[float]:
    """Per-finger curl in [0,1] (mapping, or [index, middle, ring, thumb]) -> 8 per-servo offsets (deg)."""
    if isinstance(curls, Mapping):
        vals = [float(curls.get(f, 0.0)) for f in FINGER_ORDER]
    else:
        vals = [float(v) for v in curls]
        if len(vals) != 4:
            raise ValueError(f"need 4 curls, got {len(vals)}")
    out = [0.0] * 8
    for f, c in zip(FINGER_ORDER, vals):
        c = clamp01(c)
        for s in FINGER_TO_SERVO[f]:
            out[s] = JOINT_OPEN_OFFSETS[s] + (JOINT_CLOSE_OFFSETS[s] - JOINT_OPEN_OFFSETS[s]) * c
    return out


ABDUCT_MAX_DEG = 20.0      # the sideways spread per finger, +-20 deg (the mechanism's limit)


def flex_abduct_to_offsets(flex, abduct) -> list[float]:
    """Per-finger flex [0,1] and abduction [-1,1] -> 8 per-servo offsets (deg). v1's amazing_hand_client.

    A finger's two servos are A (even) and B (odd): flex = (A - B) / 2 and abduct = (A + B) / 2, so
    A = flex_angle + abd and B = -flex_angle + abd, where flex_angle runs servo A's open..close range and
    abd = abduct * ABDUCT_MAX_DEG. With abduct 0 this is curls_to_offsets.
    """
    flex, abduct = [float(v) for v in flex], [float(v) for v in abduct]
    if len(flex) != 4 or len(abduct) != 4:
        raise ValueError(f"need 4 flex + 4 abduct, got {len(flex)} + {len(abduct)}")
    out = [0.0] * 8
    for i, f in enumerate(FINGER_ORDER):
        a, b = FINGER_TO_SERVO[f]
        flex_angle = JOINT_OPEN_OFFSETS[a] + (JOINT_CLOSE_OFFSETS[a] - JOINT_OPEN_OFFSETS[a]) * clamp01(flex[i])
        abd = max(-1.0, min(1.0, abduct[i])) * ABDUCT_MAX_DEG
        out[a], out[b] = flex_angle + abd, -flex_angle + abd
    return out


def offsets_to_curls(offsets) -> list[float]:
    """Inverse of curls_to_offsets (mean over a finger's two servos)."""
    out = []
    for f in FINGER_ORDER:
        acc = 0.0
        for s in FINGER_TO_SERVO[f]:
            span = JOINT_CLOSE_OFFSETS[s] - JOINT_OPEN_OFFSETS[s]
            acc += (float(offsets[s]) - JOINT_OPEN_OFFSETS[s]) / span
        out.append(clamp01(acc / 2.0))
    return out


def format_joint_command(offsets, speed: int = DEFAULT_SPEED) -> str:
    vals = [float(v) for v in offsets]
    if len(vals) != 8:
        raise ValueError(f"need 8 offsets, got {len(vals)}")
    speed = int(max(SPEED_MIN, min(SPEED_MAX, int(speed))))
    return "J:" + ",".join(f"{v:.1f}" for v in vals) + f",{speed}\n"


def parse_joint_command(line: str):
    """'J:...' -> (offsets[8], speed) or None (what the fake hand and the recorder parse)."""
    line = line.strip()
    if not line.startswith("J:"):
        return None
    parts = line[2:].split(",")
    if len(parts) != 9:
        return None
    try:
        return [float(v) for v in parts[:8]], int(float(parts[8]))
    except ValueError:
        return None


def parse_state_line(line: str):
    """'S:f0,f1,f2,f3,a0,a1,a2,a3' -> (flex[4], abduct[4]) or None."""
    line = line.strip()
    if not line.startswith("S:"):
        return None
    parts = line[2:].split(",")
    if len(parts) != 8:
        return None
    try:
        vals = [float(v) for v in parts]
    except ValueError:
        return None
    return vals[:4], vals[4:]
