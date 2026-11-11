"""The inference server: holds the policy, answers obs -> action over the wire (rd_vla.wire), switches
checkpoints on request. ROS-free; on the robot it runs under supervisor with the LeRobot venv's python.

    python3 -m rd_vla.infer_server --backend fake --port 8777
    /opt/vla/bin/python -m rd_vla.infer_server --backend smolvla --ckpt tigger_1_merged_smolvla_1 --device cuda

Commands over the wire: {cmd: info} -> what is served; {cmd: select, model: <spec>} -> load another
checkpoint (resolved under the models dir); {cmd: reset} -> clear the action queue.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import argparse
import sys
import threading
import time

from . import checkpoints, wire
from .backends import make_backend


class Server:
    def __init__(self, backend: str, ckpt: str | None, device: str, models_dir=None, **kw):
        self.backend_name, self.device, self.models_dir, self.kw = backend, device, models_dir, kw
        self.lock = threading.Lock()
        self.backend = None
        self.checkpoint = None
        self.load(ckpt)

    def load(self, spec: str | None) -> dict:
        path = None
        if spec and self.backend_name != "fake":
            path = str(checkpoints.resolve(spec, self.models_dir))
        if self.backend is not None and spec and (path or spec) == self.checkpoint:
            return self.info()             # already serving it: a run that names the loaded checkpoint costs nothing
        b = make_backend(self.backend_name, path or spec, device=self.device, **self.kw)
        if self.backend_name == "fake":
            b.load(spec)
        with self.lock:
            self.backend, self.checkpoint = b, (path or spec)
        return self.info()

    def infer(self, state, images, task):
        with self.lock:
            return self.backend.act(state, images, task)

    def reset(self):
        with self.lock:
            self.backend.reset()

    def info(self) -> dict:
        d = {"checkpoint": self.checkpoint, "resolved": self.checkpoint, "device": self.device}
        if self.checkpoint and self.backend_name != "fake":
            d.update(checkpoints.describe(self.checkpoint))
        d.update(self.backend.info())
        return d

    def command(self, msg: dict) -> dict:
        if msg.get("cmd") == "select":
            try:
                return {"ok": True, **self.load(str(msg.get("model", "")))}
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                return {"ok": False, "error": str(e)}
        return {"ok": False, "error": f"unknown command {msg.get('cmd')!r}"}


def main(argv=None) -> int:
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="fake", choices=["fake", "smolvla", "groot"])
    ap.add_argument("--ckpt", default=None, help="run[@step] under the models dir, or a path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--n-action-steps", type=int, default=None, help="execute only the first N actions of each chunk before re-observing; 0/None = the whole chunk")
    ap.add_argument("--prefetch", action="store_true", help="smolvla: compute the next chunk in a thread before the queue drains, so no step waits for a forward pass")
    ap.add_argument("--prefetch-at", type=int, default=2, help="refill once this many actions remain: the forward pass in steps (2 on the 4090, 10 on the Orin at 12 Hz)")
    ap.add_argument("--rtc", action="store_true", help="Real-Time Chunking: blend each prefetched chunk into the tail of the one playing (needs --prefetch)")
    ap.add_argument("--rtc-inference-delay", type=int, default=None, help="leading actions pinned to the old chunk: the steps that play during a forward pass (default: --prefetch-at + 1)")
    ap.add_argument("--rtc-execution-horizon", type=int, default=None, help="leading actions RTC pulls towards the old chunk (default: all served; just above the delay leaves the rest free to react)")
    ap.add_argument("--rtc-max-guidance", type=float, default=10.0, help="how hard RTC pulls the seam towards the old chunk")
    ap.add_argument("--num-steps", type=int, default=None, help="smolvla: denoising steps per chunk (the policy's 10); fewer is faster and coarser")
    ap.add_argument("--fake-close-after", type=int, default=10)
    ap.add_argument("--fake-drift", type=float, default=0.002)
    args = ap.parse_args(argv)
    if args.rtc and not args.prefetch:
        ap.error("--rtc needs --prefetch")
    kw = {}
    if args.backend == "smolvla":
        kw = {"n_action_steps": args.n_action_steps or None,      # 0 = the policy's own chunk
              "prefetch": args.prefetch, "prefetch_at": args.prefetch_at, "rtc": args.rtc, "num_steps": args.num_steps,
              "rtc_inference_delay": args.rtc_inference_delay, "rtc_execution_horizon": args.rtc_execution_horizon,
              "rtc_max_guidance": args.rtc_max_guidance}
    if args.backend == "fake":
        kw = {"close_after": args.fake_close_after, "drift": args.fake_drift}
    server = Server(args.backend, args.ckpt, args.device, args.models_dir, **kw)
    print(f"[infer_server] {args.backend} serving {server.checkpoint} on {args.host}:{args.port}", flush=True)
    wire.serve(args.host, args.port, server.infer, reset_fn=server.reset, info_fn=server.info, cmd_fn=server.command,
               ready_fn=lambda p: print(f"[infer_server] ready on port {p}", flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
