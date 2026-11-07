"""Policy backends behind one interface: fake (scripted, the rig), smolvla (LeRobot), groot (phase 9)."""
from __future__ import annotations

from .base import Backend


def make_backend(name: str, checkpoint: str | None = None, device: str = "cpu", **kw) -> Backend:
    name = (name or "fake").lower()
    if name == "fake":
        from .fake import FakeBackend
        return FakeBackend(**kw)
    if name == "smolvla":
        from .smolvla import SmolVLABackend
        return SmolVLABackend(checkpoint, device=device, **kw)
    if name == "groot":
        from .groot import GrootBackend
        return GrootBackend(checkpoint, device=device, **kw)
    raise ValueError(f"unknown backend {name!r} (fake | smolvla | groot)")
