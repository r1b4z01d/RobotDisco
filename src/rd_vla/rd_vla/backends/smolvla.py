"""SmolVLA (LeRobot) backend: the checkpoint's policy + its pre/post processors, one observation in, an
action chunk out, served one action per step with the chunk refilled on demand (v1 infer_server's scheduler,
without the prefetch thread: Isaac's Orin runs it in ~0.3 s a chunk at 12 Hz). Imports lerobot lazily.

Reactivity: `n_action_steps` executes only the first N of each chunk before re-observing (closed loop
without retraining); the default is the policy's own chunk (fully open loop, as v1 shipped).
"""
from __future__ import annotations

import time
from collections import deque


class SmolVLABackend:
    name = "smolvla"

    def __init__(self, checkpoint: str | None, device: str = "cpu", n_action_steps: int | None = None, **_):
        self.device = device
        self.n_action_steps = n_action_steps
        self.checkpoint = None
        self.policy = self.pre = self.post = None
        self.queue: deque = deque()
        self.last_infer_s = 0.0
        self.chunks = 0
        if checkpoint:
            self.load(checkpoint)

    def load(self, checkpoint: str | None) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        if not checkpoint:
            raise ValueError("smolvla needs a checkpoint")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        if self.n_action_steps:
            self.policy.config.n_action_steps = int(self.n_action_steps)
        self.checkpoint = checkpoint
        self.torch = torch
        self.reset()

    def reset(self) -> None:
        self.queue.clear()
        if self.policy is not None:
            self.policy.reset()

    def _observation(self, state, images: dict, task: str) -> dict:
        import numpy as np
        torch = self.torch
        obs = {"observation.state": torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0), "task": task}
        for name, img in images.items():
            a = np.ascontiguousarray(img[..., ::-1]) if img.shape[-1] == 3 else img   # the wire JPEG decodes BGR
            obs[f"observation.images.{name}"] = torch.as_tensor(a).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return obs

    def act(self, state, images, task):
        if self.policy is None:
            raise RuntimeError("no checkpoint loaded")
        if not self.queue:
            t0 = time.monotonic()
            obs = self._observation(state, images, task)
            with self.torch.inference_mode():
                batch = self.pre(obs)
                for k, v in batch.items():
                    if hasattr(v, "to"):
                        batch[k] = v.to(self.device)
                chunk = self.policy.predict_action_chunk(batch)
                chunk = self.post({"action": chunk})["action"] if self.post is not None else chunk
            n = int(self.n_action_steps or chunk.shape[1])
            for a in chunk[0, :n].cpu().numpy():
                self.queue.append(a)
            self.last_infer_s = time.monotonic() - t0
            self.chunks += 1
        return self.queue.popleft()

    def info(self) -> dict:
        return {"backend": self.name, "checkpoint": self.checkpoint, "device": self.device, "chunks": self.chunks,
                "last_infer_s": round(self.last_infer_s, 3), "n_action_steps": self.n_action_steps,
                "queued": len(self.queue)}
