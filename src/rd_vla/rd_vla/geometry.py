"""Rotation-vector <-> quaternion, pose clamps. numpy only."""
from __future__ import annotations

import math

import numpy as np


def rotvec_to_quat(r):
    r = np.asarray(r, dtype=float)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    axis = r / angle
    s = math.sin(angle / 2.0)
    return [float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), float(math.cos(angle / 2.0))]


def quat_to_rotvec(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    if w < 0:                       # the short way round
        x, y, z, w = -x, -y, -z, -w
    s = math.sqrt(max(0.0, 1.0 - w * w))
    angle = 2.0 * math.atan2(s, w)
    if s < 1e-9:
        return [0.0, 0.0, 0.0]
    return [x / s * angle, y / s * angle, z / s * angle]


def clamp_step(target_xyz, current_xyz, max_step: float):
    """Pull a Cartesian target to within max_step of the current position (v1 rtde_arm's max_step)."""
    t = np.asarray(target_xyz, dtype=float)
    c = np.asarray(current_xyz, dtype=float)
    d = t - c
    n = float(np.linalg.norm(d))
    if n > max_step > 0:
        t = c + d * (max_step / n)
    return [float(v) for v in t]


def clamp_box(xyz, lo, hi):
    return [float(min(max(v, a), b)) for v, a, b in zip(xyz, lo, hi)]
