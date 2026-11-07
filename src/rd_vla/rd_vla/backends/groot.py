"""GR00T N1.7 backend (phase 9): NVIDIA's policy serves over ZeroMQ from its own process (Isaac-GR00T's
inference service); this backend is its client. Not implemented until the fine-tune exists."""
from __future__ import annotations


class GrootBackend:
    name = "groot"

    def __init__(self, checkpoint, device="cuda", host="127.0.0.1", port=5555, **_):
        raise NotImplementedError("GR00T backend: phase 9 (needs a fine-tuned checkpoint and the GR00T inference service)")
