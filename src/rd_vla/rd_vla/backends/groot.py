"""GR00T N1.5 (LeRobot's `groot` policy) backend: the checkpoint's policy + its pre/post processors, one observation in,
a chunk of 16 actions out, served one action per step. Imports lerobot lazily, like the SmolVLA backend.

Needs, in the interpreter that serves it, LeRobot's groot extras (peft, timm, dm-tree; flash-attn where it installs)
and, on machines where flash-attn does not build and for cameras of unequal size, the two local patches of LeRobot's
groot code (provision/patches/lerobot-groot-*.patch).

Timing: a chunk takes 0.05 s on the 4090 (7 GB of GPU memory, 2026-11-20); the Orin is not measured yet. A chunk is 16
actions, 1.3 s at 12 Hz. With `prefetch` the next chunk is computed in a thread once `prefetch_at` actions remain
(rd_vla/chunks.py), which must cover the forward pass in steps. There is no Real-Time Chunking here.

Two things LeRobot's own flow does not do for a chunk: its post-processor keeps only the LAST step of a 3-D chunk (so
each step is un-normalised by itself here), and `select_action` would run all 50 of `n_action_steps` open loop.
"""
from __future__ import annotations

import time
from collections import deque


class GrootBackend:
    name = "groot"

    def __init__(self, checkpoint: str | None, device: str = "cuda", n_action_steps: int | None = None,
                 prefetch: bool = False, prefetch_at: int = 2, **_):
        self.device = device
        self.n_action_steps = n_action_steps
        self.prefetch, self.prefetch_at = bool(prefetch), int(prefetch_at)
        self.scheduler = None
        self.checkpoint = None
        self.policy = self.pre = self.post = None
        self.queue: deque = deque()
        self.last_infer_s = 0.0
        self.chunks = 0
        self.horizon = 16
        if checkpoint:
            self.load(checkpoint)

    def load(self, checkpoint: str | None) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        if not checkpoint:
            raise ValueError("groot needs a checkpoint")
        self.policy = GrootPolicy.from_pretrained(checkpoint)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        self.checkpoint = checkpoint
        self.torch = torch
        self.horizon = 0                 # the chunk's real length (16) is only known from the first forward pass
        if self.prefetch:
            from ..chunks import ChunkScheduler
            chunk = 16
            at = max(1, min(self.prefetch_at, chunk // 2))
            horizon = min(int(self.n_action_steps or chunk), chunk - at - 1)   # horizon + the skipped stretch fit in a chunk
            self.scheduler = ChunkScheduler(self._compute, horizon, at, pass_tail=False, log=lambda m: print(m, flush=True))
        self.reset()

    def reset(self) -> None:
        self.queue.clear()
        if self.scheduler is not None:
            self.scheduler.reset()
        if self.policy is not None:
            self.policy.reset()

    def _observation(self, state, images: dict, task: str) -> dict:
        import numpy as np
        torch = self.torch
        obs = {"observation.state": torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0), "task": [task]}
        for name, img in images.items():      # RGB in, RGB to the policy (rd_vla/backends/smolvla.py on why)
            obs[f"observation.images.{name}"] = torch.as_tensor(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return obs

    def _forward(self, obs: dict):
        """One forward pass: (the chunk in the policy's normalised units, the same chunk as a list of 10-D actions)."""
        batch = self.pre(obs)
        for k, v in batch.items():
            if hasattr(v, "to"):
                batch[k] = v.to(self.device)
        raw = self.policy.predict_action_chunk(batch)
        phys = [self.post(raw[:, t, :].clone())[0].float().cpu().numpy() for t in range(raw.shape[1])]
        self.horizon = len(phys)
        return raw, phys

    def _compute(self, obs, prev_left):
        with self.torch.inference_mode():
            raw, phys = self._forward(obs())
        keep = min(self.scheduler.keep, len(phys))
        return raw[:, :keep].detach(), phys[:keep]

    def act(self, state, images, task):
        if self.policy is None:
            raise RuntimeError("no checkpoint loaded")
        if self.scheduler is not None:
            action = self.scheduler.next_action(lambda: self._observation(state, images, task))
            self.last_infer_s = self.scheduler.last_compute_s
            return action
        if not self.queue:
            t0 = time.monotonic()
            with self.torch.inference_mode():
                _, chunk = self._forward(self._observation(state, images, task))
            for a in chunk[:int(self.n_action_steps or len(chunk))]:
                self.queue.append(a)
            self.last_infer_s = time.monotonic() - t0
            self.chunks += 1
        return self.queue.popleft()

    def info(self) -> dict:
        d = {"backend": self.name, "checkpoint": self.checkpoint, "device": self.device, "chunks": self.chunks,
             "last_infer_s": round(self.last_infer_s, 3), "n_action_steps": self.n_action_steps or self.horizon,
             "queued": len(self.queue), "prefetch": self.prefetch}
        if self.scheduler is not None:
            s = self.scheduler
            d.update(n_action_steps=s.horizon, prefetch_at=s.prefetch_at, queued=len(s),
                     chunks=sum(s.stats[k] for k in ("inline", "prefetched", "waited")), served=dict(s.stats))
        return d
