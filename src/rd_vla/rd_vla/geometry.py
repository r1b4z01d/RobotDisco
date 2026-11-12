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


def keep_out(xyz, no_go_xy=None, margin: float = 0.03, z_floor=None):
    """A Cartesian target kept off the robot's own body (v1 workspace.clamp_out_of_nogo and rtde_arm's z floor).

    no_go_xy = (xmin, xmax, ymin, ymax) in the arm's base frame: a target whose (x, y) falls inside it, grown by
    `margin`, is pushed to the nearest edge. z_floor lifts a target below it. Returns (xyz, was_changed)."""
    x, y, z = (float(v) for v in xyz)
    changed = False
    if no_go_xy is not None and len(no_go_xy) == 4:
        xmin, xmax, ymin, ymax = no_go_xy[0] - margin, no_go_xy[1] + margin, no_go_xy[2] - margin, no_go_xy[3] + margin
        if xmin <= x <= xmax and ymin <= y <= ymax:
            dist = {"xmin": x - xmin, "xmax": xmax - x, "ymin": y - ymin, "ymax": ymax - y}
            edge = min(dist, key=dist.get)
            x = xmin if edge == "xmin" else xmax if edge == "xmax" else x
            y = ymin if edge == "ymin" else ymax if edge == "ymax" else y
            changed = True
    if z_floor is not None and z < z_floor:
        z, changed = float(z_floor), True
    return [x, y, z], changed


def workspace_limits() -> dict:
    """keep_out's arguments from robot.yaml arm.workspace ({} entries missing = that limit off)."""
    from rd_bringup import config
    w = config.get("arm.workspace", None) or {}
    box = w.get("no_go_xy")
    return {"no_go_xy": [float(v) for v in box] if box and len(box) == 4 else None,
            "margin": float(w.get("no_go_margin", 0.03)),
            "z_floor": float(w["z_floor"]) if w.get("z_floor") is not None else None}
