"""The action-chunk queue with a background refill (v1 infer_server's _ChunkScheduler, without torch in it).

A chunked policy computes a whole chunk the moment its queue drains, on the request path: the step that hits the
boundary waits the full forward pass (0.14 s on the 4090, 0.73 s on the Orin: nine steps at 12 Hz). Here the next
chunk is computed in a thread as soon as `prefetch_at` actions remain, so every request is answered from memory.

The price: the new chunk is planned from an observation a few steps old. The refill starts on the step that finds
`prefetch_at` + 1 actions queued (the one it is about to serve and `prefetch_at` behind it); the new chunk's action 0
belongs to that step, so by the time the queue is empty its first `prefetch_at` + 1 actions describe a stretch the
arm has driven, and they are skipped. (v1 before 2026-10-11 replayed them all: a 46 mm seam on the GPU box, a 200 mm
snap back per chunk on the Jetson at prefetch_at 21. v1 after it skipped `prefetch_at`, one short without RTC.)
So `prefetch_at` must cover the forward pass in steps, and horizon + prefetch_at + 1 must fit in the policy's chunk.

`compute(obs, prev_left)` -> (raw, actions): one forward pass. `actions` is what gets served, `raw` is the same
chunk in the policy's own units, sliceable as raw[:, a:b]; `prev_left` is the tail of the outgoing raw chunk from
the action of the observation's step on (None unless `pass_tail`): it lines up with the new chunk index for index,
and Real-Time Chunking pulls the new chunk towards it so the seam blends.
"""
from __future__ import annotations

import threading
import time


class ChunkScheduler:
    def __init__(self, compute, horizon: int, prefetch_at: int = 2, pass_tail: bool = False, log=print):
        self.compute = compute
        self.horizon = max(1, int(horizon))
        # a horizon of one has nothing to refill ahead of: it stays one inline pass per step
        self.prefetch_at = max(1, min(int(prefetch_at), self.horizon - 1)) if self.horizon > 1 else 0
        self.keep = self.horizon + self.prefetch_at + (1 if self.prefetch_at else 0)     # actions a compute should return
        self.pass_tail = bool(pass_tail)
        self.log = log
        self._q: list = []
        self._raw = None
        self._pending = None         # (a computed chunk or None, the actions to skip): not installed yet
        self._thread = None
        # reset() bumps it; a prefetch drops its result when the epoch moved, so a chunk planned at the end of one
        # episode is never installed into the next (v1 on hardware: the fingers grasping, opening and fluttering
        # around step 10 of every run after the first)
        self._epoch = 0
        self._lock = threading.Lock()        # the queue, the pending chunk, the thread, the epoch
        self._gpu = threading.Lock()         # one forward pass at a time
        self.stats = {"inline": 0, "prefetched": 0, "waited": 0, "failed": 0}
        self.last_compute_s = 0.0

    def __len__(self) -> int:
        return len(self._q)

    def _run(self, obs, prev_left):
        with self._gpu:
            t0 = time.monotonic()
            out = self.compute(obs, prev_left)
            self.last_compute_s = time.monotonic() - t0
        return out

    def _spawn(self, obs, prev_left, skip: int) -> None:
        epoch = self._epoch

        def work():
            try:
                c = self._run(obs, prev_left)
            except Exception as e:  # noqa: BLE001  the serving thread computes inline instead
                self.log(f"[chunks] prefetch failed: {type(e).__name__}: {e}")
                c = None
            with self._lock:
                if epoch != self._epoch:
                    return           # reset() ran meanwhile; _thread may already belong to a newer prefetch
                if c is None:
                    self.stats["failed"] += 1
                self._pending, self._thread = (c, skip) if c is not None else None, None

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def _install(self, computed, skip: int) -> bool:
        """Make a computed chunk the live one, from its action `skip` on. The caller holds the lock."""
        if computed is None:
            return False
        raw, actions = computed
        if skip and len(actions) - skip < 1:
            self.log(f"[chunks] chunk of {len(actions)} too short to skip {skip} elapsed actions: served from its start")
            skip = 0
        self._raw = raw[:, skip:skip + self.horizon] if raw is not None else None
        self._q = list(actions[skip:skip + self.horizon])
        return bool(self._q)

    def next_action(self, obs):
        with self._lock:
            epoch = self._epoch
            thread = None
            if not self._q:
                if self._pending is not None:
                    if self._install(*self._pending):
                        self.stats["prefetched"] += 1
                    self._pending = None
                else:
                    thread = self._thread
        if thread is not None:       # dry with a prefetch in flight (prefetch_at too small for this machine): wait for it
            thread.join()
            with self._lock:
                if self._pending is not None and epoch == self._epoch:
                    self._install(*self._pending)
                    self._pending = None
                    self.stats["waited"] += 1
        with self._lock:
            inline = not self._q
        if inline:                   # the first step of an episode, or the prefetch failed
            c = self._run(obs, None)
            with self._lock:
                if not self._install(c, 0):
                    raise RuntimeError("the policy returned an empty chunk")
                self.stats["inline"] += 1
        with self._lock:
            left = len(self._q)      # with the action this step serves: what plays before the queue is empty
            if self.prefetch_at and left <= self.prefetch_at + 1 and self._thread is None and self._pending is None:
                prev_left = self._raw[:, self._raw.shape[1] - left:] if self.pass_tail and self._raw is not None else None
                self._spawn(obs, prev_left, left)
            return self._q.pop(0)

    def reset(self) -> None:
        with self._lock:
            self._epoch += 1
            self._q, self._raw, self._pending, self._thread = [], None, None, None
