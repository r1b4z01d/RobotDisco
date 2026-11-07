"""The observation -> action RPC between the policy runner and the inference server (v1 remote.py).

4-byte big-endian length prefix + a pickled dict of plain Python types (lists / str / bytes); images go
as JPEG, shrunk to fit inside 512x512 (the policy's own fit) first. Same process, loopback on the robot;
the GPU box over the LAN as the fallback. Trusted LAN only: pickle is not safe against a hostile peer.
"""
from __future__ import annotations

import pickle
import socket
import struct

WIRE_MAX_WH = 512
JPEG_QUALITY = 92


def send_msg(sock, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_all(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf += chunk
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack(">I", _recv_all(sock, 4))
    return pickle.loads(_recv_all(sock, n))


def _fit_for_wire(a, max_wh: int = WIRE_MAX_WH):
    import cv2
    h, w = a.shape[:2]
    scale = min(max_wh / w, max_wh / h)
    if scale >= 1.0:
        return a
    return cv2.resize(a, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def encode_obs(state, images: dict, task: str, max_wh: int = WIRE_MAX_WH, quality: int = JPEG_QUALITY) -> dict:
    import cv2

    def jpg(a) -> bytes:
        if max_wh:
            a = _fit_for_wire(a, max_wh)
        ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()
    return {"state": [float(v) for v in state], "task": task, "images": {name: jpg(rgb) for name, rgb in images.items()}}


def decode_obs(msg: dict):
    import cv2
    import numpy as np

    def unjpg(b):
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    return (np.asarray(msg["state"], dtype=np.float32), {name: unjpg(b) for name, b in msg["images"].items()}, msg["task"])


class PolicyClient:
    """obs -> action RPC; the server holds the policy and its action-chunk queue."""

    def __init__(self, host: str, port: int, timeout: float = 15.0):
        self.host, self.port, self.timeout = host, int(port), timeout
        self.sock = None

    def connect(self) -> "PolicyClient":
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self

    def _rpc(self, msg: dict) -> dict:
        send_msg(self.sock, msg)
        rep = recv_msg(self.sock)
        return rep if isinstance(rep, dict) else {"error": "bad reply"}

    def infer(self, state, images: dict, task: str):
        import numpy as np
        rep = self._rpc(encode_obs(state, images, task))
        if "error" in rep:
            raise RuntimeError(f"policy server: {rep['error']}")
        return np.asarray(rep["action"], dtype=np.float32)

    def reset(self) -> None:
        self._rpc({"cmd": "reset"})

    def info(self) -> dict:
        rep = self._rpc({"cmd": "info"})
        return rep if rep.get("ok") else {}

    def select(self, model: str) -> dict:
        return self._rpc({"cmd": "select", "model": model})

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None


def serve(host: str, port: int, infer_fn, reset_fn=None, info_fn=None, cmd_fn=None, ready_fn=None, stop_event=None) -> None:
    """One client at a time; the next connection replaces a dead one. infer_fn(state, images, task) -> action."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(2)
    srv.settimeout(0.5)
    if ready_fn:
        ready_fn(srv.getsockname()[1])
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(None)
                while stop_event is None or not stop_event.is_set():
                    try:
                        msg = recv_msg(conn)
                    except (ConnectionError, OSError, EOFError, pickle.UnpicklingError):
                        break
                    try:
                        if isinstance(msg, dict) and "cmd" in msg:
                            cmd = msg["cmd"]
                            if cmd == "reset":
                                if reset_fn:
                                    reset_fn()
                                rep = {"ok": True}
                            elif cmd == "info":
                                rep = {"ok": True, **(info_fn() if info_fn else {})}
                            elif cmd_fn is not None:
                                rep = cmd_fn(msg)
                            else:
                                rep = {"ok": False, "error": f"unknown command {cmd!r}"}
                        else:
                            state, images, task = decode_obs(msg)
                            action = infer_fn(state, images, task)
                            rep = {"action": [float(v) for v in action]}
                    except Exception as e:  # noqa: BLE001 - report, keep serving
                        rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    try:
                        send_msg(conn, rep)
                    except OSError:
                        break
    finally:
        srv.close()
