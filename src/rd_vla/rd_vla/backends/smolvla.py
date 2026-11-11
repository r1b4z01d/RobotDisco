"""SmolVLA (LeRobot) backend: the checkpoint's policy + its pre/post processors, one observation in, an
action chunk out, served one action per step. Imports lerobot lazily.

Timing: a chunk takes about 0.14 s on the 4090 and 0.73 s on the robot's AGX Orin (JetPack 7.2.1, torch 2.10 cu130,
MAXN, measured 2026-11-10; v1 on JetPack 6 measured 0.875 s). At 12 Hz that is nine steps. Without `prefetch` the
refill is synchronous and the loop pauses that long at every chunk boundary; with it the next chunk is computed in
a thread once `prefetch_at` actions remain (rd_vla/chunks.py), which must cover the forward pass in steps: 2 on the
4090, 10 on the Orin. `rtc` (Real-Time Chunking, needs prefetch) pulls each new chunk towards the unplayed tail of
the one before, so the seam blends. `num_steps` overrides the denoising steps (10): fewer is faster and coarser.

Reactivity: `n_action_steps` executes only the first N of each chunk before re-observing (closed loop
without retraining); the default is the policy's own chunk (fully open loop, as v1 shipped).
"""
from __future__ import annotations

import time
from collections import deque


class SmolVLABackend:
    name = "smolvla"

    def __init__(self, checkpoint: str | None, device: str = "cpu", n_action_steps: int | None = None,
                 prefetch: bool = False, prefetch_at: int = 2, rtc: bool = False, rtc_inference_delay: int | None = None,
                 rtc_execution_horizon: int | None = None, rtc_max_guidance: float = 10.0, num_steps: int | None = None, **_):
        if rtc and not prefetch:
            raise ValueError("rtc needs prefetch: it blends a chunk computed ahead into the one still playing")
        self.device = device
        self.n_action_steps = n_action_steps
        self.prefetch, self.prefetch_at, self.rtc, self.num_steps = bool(prefetch), int(prefetch_at), bool(rtc), num_steps
        self.rtc_inference_delay, self.rtc_execution_horizon, self.rtc_max_guidance = rtc_inference_delay, rtc_execution_horizon, rtc_max_guidance
        self.scheduler = None
        self.rtc_guided = 0
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
        cfg = self.policy.config
        if self.n_action_steps:
            cfg.n_action_steps = int(self.n_action_steps)
        if self.num_steps and hasattr(cfg, "num_steps"):
            cfg.num_steps = int(self.num_steps)
        self.checkpoint = checkpoint
        self.torch = torch
        if self.prefetch:
            from ..chunks import ChunkScheduler
            chunk = int(cfg.chunk_size)
            at = max(1, min(self.prefetch_at, chunk // 2))
            horizon = min(int(self.n_action_steps or chunk), chunk - at - 1)  # horizon + the skipped stretch fit in a chunk
            self.scheduler = ChunkScheduler(self._compute, horizon, at, pass_tail=self.rtc,
                                            log=lambda m: print(m, flush=True))
            if self.rtc:
                from lerobot.policies.rtc.configuration_rtc import RTCConfig
                # guidance reaches execution_horizon actions into the new chunk: the whole horizon anchors every
                # served action to the old plan (smooth, deaf to a moved object); just above the delay leaves the rest free
                self._rtc = {"inference_delay": int(self.rtc_inference_delay or self.scheduler.prefetch_at + 1),
                             "execution_horizon": int(self.rtc_execution_horizon or horizon)}
                cfg.rtc_config = RTCConfig(enabled=True, execution_horizon=self._rtc["execution_horizon"],
                                           max_guidance_weight=float(self.rtc_max_guidance))
                self.policy.init_rtc_processor()
        self.reset()

    def reset(self) -> None:
        self.queue.clear()
        if self.scheduler is not None:
            self.scheduler.reset()       # drops a prefetch in flight too
        if self.policy is not None:
            self.policy.reset()

    def _forward(self, obs: dict, **kwargs):
        """One forward pass: (the chunk in the policy's normalized units, the same chunk in joint units)."""
        batch = self.pre(obs)
        for k, v in batch.items():
            if hasattr(v, "to"):
                batch[k] = v.to(self.device)
        raw = self.policy.predict_action_chunk(batch, **kwargs)
        return raw, (self.post(raw) if self.post is not None else raw)   # lerobot 0.4: the unnormalizer takes the action tensor

    def _compute(self, obs, prev_left):
        """The scheduler's forward pass, possibly in its thread. no_grad, not inference_mode: RTC differentiates inside."""
        kwargs = {}
        if self.rtc and prev_left is not None and prev_left.shape[1] > 0:
            kwargs = {"prev_chunk_left_over": prev_left, **self._rtc}
            self.rtc_guided += 1
        with self.torch.no_grad():
            raw, phys = self._forward(obs(), **kwargs)
        keep = min(self.scheduler.keep, raw.shape[1])
        return raw[:, :keep].detach(), list(phys[0, :keep].cpu().numpy())

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
        if self.scheduler is not None:
            # the tensors are built only for the step that starts a forward pass (18 ms of every step otherwise)
            action = self.scheduler.next_action(lambda: self._observation(state, images, task))
            self.last_infer_s = self.scheduler.last_compute_s
            return action
        if not self.queue:
            t0 = time.monotonic()
            obs = self._observation(state, images, task)
            with self.torch.inference_mode():
                _, chunk = self._forward(obs)
            n = int(self.n_action_steps or chunk.shape[1])
            for a in chunk[0, :n].cpu().numpy():
                self.queue.append(a)
            self.last_infer_s = time.monotonic() - t0
            self.chunks += 1
        return self.queue.popleft()

    def info(self) -> dict:
        d = {"backend": self.name, "checkpoint": self.checkpoint, "device": self.device, "chunks": self.chunks,
             "last_infer_s": round(self.last_infer_s, 3), "n_action_steps": self.n_action_steps,
             "queued": len(self.queue), "prefetch": self.prefetch}
        if self.scheduler is not None:
            s = self.scheduler
            d.update(n_action_steps=s.horizon, prefetch_at=s.prefetch_at, queued=len(s), chunks=sum(s.stats[k] for k in ("inline", "prefetched", "waited")),
                     served=dict(s.stats), rtc=self.rtc, rtc_guided=self.rtc_guided)
        return d
