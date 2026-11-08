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
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--fake-close-after", type=int, default=10)
    ap.add_argument("--fake-drift", type=float, default=0.002)
    args = ap.parse_args(argv)
    kw = {"n_action_steps": args.n_action_steps} if args.backend == "smolvla" else {}
    if args.backend == "fake":
        kw = {"close_after": args.fake_close_after, "drift": args.fake_drift}
    server = Server(args.backend, args.ckpt, args.device, args.models_dir, **kw)
    print(f"[infer_server] {args.backend} serving {server.checkpoint} on {args.host}:{args.port}", flush=True)
    wire.serve(args.host, args.port, server.infer, reset_fn=server.reset, info_fn=server.info, cmd_fn=server.command,
               ready_fn=lambda p: print(f"[infer_server] ready on port {p}", flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
