"""Recorded joint-space motions and named poses, ROS-free.

A motion is JSON {"name", "rate", "q": [[j0..j5], ...], ...} as v1's ur_record_motion.py wrote it (one
cycle of the pendant's wave program sampled over RTDE at 50 Hz). The library is the package's motions/
folder plus $RD_MOTIONS_DIR (the robot's own recordings, from the v1 snapshot). Named poses live in
config/poses.yaml: vla_home is where policies start and where every gesture ends.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import yaml

JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def _share(*parts: str) -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        base = Path(get_package_share_directory("rd_arm"))
    except Exception:  # noqa: BLE001
        base = Path(__file__).resolve().parents[1]
    return base.joinpath(*parts)


def motion_dirs() -> list[Path]:
    dirs = []
    env = os.environ.get("RD_MOTIONS_DIR")
    if env:
        dirs.append(Path(env))
    dirs.append(_share("motions"))
    return dirs


def list_motions() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in reversed(motion_dirs()):          # the env dir wins
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                out[p.stem] = p
    return out


def load_motion(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.suffix == ".json" or not p.exists():
        found = list_motions().get(str(name_or_path))
        if found is None:
            raise FileNotFoundError(f"no motion {name_or_path!r} in {[str(d) for d in motion_dirs()]}")
        p = found
    m = json.loads(p.read_text())
    q = m.get("q")
    if not isinstance(q, list) or not q or any(len(s) != 6 for s in q):
        raise ValueError(f"{p}: 'q' must be a non-empty list of 6-joint samples")
    rate = float(m.get("rate") or 50.0)
    if rate <= 0:
        raise ValueError(f"{p}: bad rate {rate}")
    return {"name": m.get("name") or p.stem, "rate": rate, "q": [[float(v) for v in s] for s in q],
            "recorded": m.get("recorded", ""), "source": m.get("source", ""), "path": str(p)}


def duration(m: dict) -> float:
    return len(m["q"]) / m["rate"]


def max_joint_step(m: dict) -> float:
    return max((max(abs(a - b) for a, b in zip(s0, s1)) for s0, s1 in zip(m["q"], m["q"][1:])), default=0.0)


def cap_steps(q: list[list[float]], max_dq: float) -> tuple[list[list[float]], int]:
    """Bound the per-sample joint step (a corrupt sample can never yank the arm). Returns (q, capped)."""
    out = [list(q[0])]
    capped = 0
    for s in q[1:]:
        last = out[-1]
        if max(abs(a - b) for a, b in zip(s, last)) > max_dq:
            s = [a + max(-max_dq, min(max_dq, b - a)) for a, b in zip(last, s)]
            capped += 1
        out.append(list(s))
    return out, capped


def resample(q: list[list[float]], rate: float, speed_scale: float) -> list[tuple[float, list[float]]]:
    """(time_from_start, joints) for every sample at `speed_scale` of the recording's speed."""
    dt = 1.0 / (rate * max(0.05, float(speed_scale)))
    return [(i * dt, list(s)) for i, s in enumerate(q)]


def load_poses(path: str | Path | None = None) -> dict[str, list[float]]:
    p = Path(path) if path else _share("config", "poses.yaml")
    data = yaml.safe_load(p.read_text()) or {}
    out = {}
    for name, q in (data.get("poses") or {}).items():
        if isinstance(q, dict):
            q = [q[j] for j in JOINTS]
        if len(q) != 6:
            raise ValueError(f"pose {name!r} needs 6 joints")
        out[str(name)] = [float(v) for v in q]
    return out


def joint_distance(a, b) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def deg(x: float) -> float:
    return math.degrees(x)
