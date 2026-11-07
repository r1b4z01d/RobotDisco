"""A scripted policy for the rig: holds the pose, drifts the end effector by `drift` metres per step along
`axis`, and closes the hand once `close_after` steps have been seen (so done_when=closed ends a run).
Deterministic, ROS-free, instant."""
from __future__ import annotations

import numpy as np

from ..schema import ACTION_DIM, ARM_POSE_DIM


class FakeBackend:
    name = "fake"

    def __init__(self, drift: float = 0.002, axis: int = 2, close_after: int = 10, hold: bool = False, **_):
        self.drift, self.axis, self.close_after, self.hold = float(drift), int(axis), int(close_after), bool(hold)
        self.steps = 0
        self.loaded = None

    def load(self, checkpoint):
        self.loaded = checkpoint or "fake"

    def reset(self):
        self.steps = 0

    def act(self, state, images, task):
        state = np.asarray(state, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[:ARM_POSE_DIM] = state[:ARM_POSE_DIM]           # the current pose, from the observation
        if not self.hold:
            a[self.axis] += self.drift * (1 if self.steps < 1000 else 0)
        a[ARM_POSE_DIM:] = 1.0 if self.steps >= self.close_after else 0.0
        self.steps += 1
        return a

    def info(self):
        return {"backend": self.name, "checkpoint": self.loaded, "steps": self.steps, "cameras": "any", "chunk": 1}
