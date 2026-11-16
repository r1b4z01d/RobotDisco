"""The last few centimetres to a landmark.

Nav2 calls a goal reached inside its tolerance (0.25 m, 0.25 rad here), and a differential drive that arrives beside
its goal cannot sidestep onto it. A manipulation policy recorded from one spot wants the robot within a few
centimetres and a degree or two of that spot (the picks of 2026-11-14 missed with the robot 10 to 20 cm off). So once
Nav2 has arrived, the landmark server finishes the job itself, slowly, in three plain moves:

    turn    on the spot to face the goal point (or to face away from it, when backing up is the shorter way)
    drive   straight until the goal point is abeam, steering lightly to hold the line
    align   on the spot to the landmark's own heading

`step()` is the whole controller: pure, one call per control tick, tested against a simulated robot. The velocities
go out on the navigation's own topic, through the twist mux and the collision monitor like any other.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class Params:
    pos_tol: float = 0.03          # m: nearer than this to the goal point is there
    yaw_tol: float = 0.025         # rad (1.4 deg): the final heading
    max_start: float = 0.6         # m: further than this is not a final approach (Nav2 did not arrive)
    v_max: float = 0.08            # m/s
    v_min: float = 0.03            # below this the base does not move at all
    w_max: float = 0.5             # rad/s
    w_min: float = 0.12
    k_v: float = 1.2               # 1/s: speed per metre still to go
    k_w: float = 2.0               # 1/s: turn rate per radian of heading error
    face_tol: float = 0.06         # rad: facing the goal point well enough to start driving
    reverse_beyond: float = 1.75   # rad (100 deg): a goal further round than this is backed onto
    max_passes: int = 3            # turn-and-drive attempts before settling for the heading alone


@dataclass
class State:
    phase: str = "turn"            # turn | drive | align | done
    passes: int = 0
    reverse: bool = False
    note: str = ""
    last_along: float = field(default=float("nan"))


def _spin(err: float, p: Params) -> float:
    w = max(p.w_min, min(p.w_max, abs(err) * p.k_w))
    return math.copysign(w, err)


def step(state: State, pose: tuple, goal: tuple, p: Params = Params()) -> tuple:
    """One tick: (v, w) for the robot at pose (x, y, yaw) heading for goal (x, y, yaw). Advances `state`."""
    x, y, yaw = pose
    dx, dy = goal[0] - x, goal[1] - y
    dist = math.hypot(dx, dy)
    if state.phase == "turn":
        if dist <= p.pos_tol or state.passes >= p.max_passes:
            state.phase = "align"
        else:
            bearing = math.atan2(dy, dx)
            if state.passes == 0 or math.isnan(state.last_along):
                state.reverse = abs(wrap(bearing - yaw)) > p.reverse_beyond
            err = wrap(bearing + (math.pi if state.reverse else 0.0) - yaw)
            if abs(err) <= p.face_tol:
                state.phase, state.last_along = "drive", float("nan")
            else:
                return 0.0, _spin(err, p)
    if state.phase == "drive":
        along = dx * math.cos(yaw) + dy * math.sin(yaw)          # how far ahead (+) or behind (-) the goal point is
        passed = not math.isnan(state.last_along) and along * state.last_along < 0.0
        state.last_along = along
        if abs(along) <= p.pos_tol * 0.5 or passed:
            state.passes += 1
            state.phase = "turn" if dist > p.pos_tol else "align"   # still off to one side: once more round
            return 0.0, 0.0
        v = math.copysign(max(p.v_min, min(p.v_max, abs(along) * p.k_v)), along)
        bearing = math.atan2(dy, dx)
        err = wrap(bearing + (math.pi if along < 0 else 0.0) - yaw)
        w = max(-0.3, min(0.3, err * p.k_w)) if dist > 2.0 * p.pos_tol else 0.0   # no steering at the very end: the bearing swings
        return v, w
    if state.phase == "align":
        err = wrap(goal[2] - yaw)
        if abs(err) <= p.yaw_tol:
            state.phase = "done"
            state.note = f"{dist * 100:.0f} cm, {abs(math.degrees(err)):.1f} deg off"
            return 0.0, 0.0
        return 0.0, _spin(err, p)
    return 0.0, 0.0
