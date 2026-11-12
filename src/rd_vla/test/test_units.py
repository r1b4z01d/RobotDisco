"""ROS-free: schema, geometry, the wire RPC against the fake backend server, checkpoints, evals, the recorder."""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from rd_vla import checkpoints, wire
from rd_vla.backends import make_backend
from rd_vla.evals import EvalRegistry
from rd_vla.geometry import clamp_box, clamp_step, quat_to_rotvec, rotvec_to_quat
from rd_vla.infer_server import Server
from rd_vla.recorder import EpisodeWriter, lerobot_available, load_episode_npz
from rd_vla.schema import ACTION_DIM, STATE_DIM, features, make_state, split_action


def test_schema():
    st = make_state([0.1] * 6, [0.2] * 6, [0.3] * 4)
    assert len(st) == STATE_DIM
    pose, curls = split_action(list(range(ACTION_DIM)))
    assert pose == [0, 1, 2, 3, 4, 5] and curls == [6, 7, 8, 9]
    ft = features(960, 540)
    assert ft["action"]["shape"] == (10,) and ft["observation.images.scene"]["shape"] == (540, 960, 3)
    with pytest.raises(ValueError):
        split_action([1, 2])


def test_geometry():
    for r in ([0, 0, 0], [0.5, 0, 0], [0, 1.2, -0.3], [3.0, 0.1, 0.1]):
        back = quat_to_rotvec(rotvec_to_quat(r))
        assert np.allclose(back, r, atol=1e-9)
    assert clamp_step([1, 0, 0], [0, 0, 0], 0.1) == pytest.approx([0.1, 0, 0])
    assert clamp_step([0.05, 0, 0], [0, 0, 0], 0.1) == pytest.approx([0.05, 0, 0])
    assert clamp_box([2, -2, 0.5], [-1, -1, 0], [1, 1, 1]) == [1, -1, 0.5]


def test_fake_backend_closes_the_hand():
    b = make_backend("fake", close_after=3, drift=0.01)
    b.load(None)
    st = [0.1] * 16
    acts = [b.act(st, {}, "t") for _ in range(5)]
    assert acts[0][2] == pytest.approx(0.11) and acts[0][6:].tolist() == [0, 0, 0, 0]
    assert acts[3][6:].tolist() == [1, 1, 1, 1]


def test_wire_roundtrip_with_server():
    server = Server("fake", "demo", "cpu", close_after=2)
    stop = threading.Event()
    ports = []
    t = threading.Thread(target=wire.serve, args=("127.0.0.1", 0, server.infer),
                         kwargs=dict(reset_fn=server.reset, info_fn=server.info, cmd_fn=server.command,
                                     ready_fn=ports.append, stop_event=stop), daemon=True)
    t.start()
    for _ in range(50):
        if ports:
            break
        time.sleep(0.05)
    client = wire.PolicyClient("127.0.0.1", ports[0]).connect()
    try:
        img = np.zeros((540, 960, 3), np.uint8)
        img[100:200, 100:300] = 200
        info = client.info()
        assert info["backend"] == "fake" and info["checkpoint"] == "demo"
        a1 = client.infer([0.0] * 16, {"scene": img, "wrist": img, "side": img}, "pick")
        a2 = client.infer([0.0] * 16, {"scene": img, "wrist": img, "side": img}, "pick")
        a3 = client.infer([0.0] * 16, {"scene": img, "wrist": img, "side": img}, "pick")
        assert a1.shape == (10,) and a3[6:].tolist() == [1, 1, 1, 1] and a2[6:].tolist() == [0, 0, 0, 0]
        client.reset()
        assert client.info()["steps"] == 0
        rep = client.select("nonexistent_run")
        assert rep.get("ok") is True            # the fake backend accepts any name
        assert client.info()["checkpoint"] == "nonexistent_run"
    finally:
        client.close()
        stop.set()
        t.join(timeout=3)
    enc = wire.encode_obs([1.0] * 16, {"scene": img}, "t")
    state, images, task = wire.decode_obs(enc)
    assert images["scene"].shape == (288, 512, 3) and task == "t" and state.shape == (16,)   # 960x540 fit inside 512


def test_checkpoints(tmp_path):
    run = tmp_path / "train" / "tigger_1" / "checkpoints"
    for step in ("005000", "020000"):
        d = run / step / "pretrained_model"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"type": "smolvla"}))
    (run / "last").symlink_to(run / "020000")
    assert checkpoints.resolve("tigger_1", tmp_path / "train").name == "pretrained_model"
    assert "020000" in str(checkpoints.resolve("tigger_1", tmp_path / "train"))
    assert "005000" in str(checkpoints.resolve("tigger_1@5000", tmp_path / "train"))
    d = checkpoints.describe(checkpoints.resolve("tigger_1@5000", tmp_path / "train"))
    assert d["type"] == "smolvla" and d["run"] == "tigger_1" and d["step"] == "005000"
    runs = checkpoints.list_runs(tmp_path / "train")
    assert runs[0]["run"] == "tigger_1" and runs[0]["last"] == "020000"
    with pytest.raises(FileNotFoundError):
        checkpoints.resolve("nope", tmp_path / "train")
    # a run copied without its `last` symlink (the v1 mirror on the GPU box): the newest numbered step is last
    nolink = tmp_path / "train" / "tigger_2" / "checkpoints"
    for step in ("005000", "010000"):
        d = nolink / step / "pretrained_model"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"type": "smolvla"}))
    assert "010000" in str(checkpoints.resolve("tigger_2", tmp_path / "train"))
    assert "005000" in str(checkpoints.resolve("tigger_2@5000", tmp_path / "train"))
    link = checkpoints.set_served(tmp_path, checkpoints.resolve("tigger_1", tmp_path / "train"))
    assert checkpoints.served(tmp_path) == link.resolve()


def test_eval_registry(tmp_path):
    reg = EvalRegistry(tmp_path / "evals")
    reg.create("20261107_a", {"task": "pick", "model": "m1", "checkpoint": "/c/m1", "steps": 40, "episodes": 1, "successes": 1})
    reg.create("20261107_b", {"task": "pick", "model": "m1", "checkpoint": "/c/m1", "steps": 50, "episodes": 1, "successes": 0})
    reg.annotate("20261107_a", {"rating": 4, "notes": "clean grasp"})
    items = reg.list()
    assert len(items) == 2 and items[0]["name"] == "20261107_b"
    g = reg.groups()
    assert g[0]["model"] == "/c/m1" and g[0]["runs"] == 2 and g[0]["success_rate"] == 0.5 and g[0]["rating"] == 4.0
    reg.rename("20261107_b", "20261107_c")
    assert {i["name"] for i in reg.list()} == {"20261107_a", "20261107_c"}
    reg.delete("20261107_c")
    assert len(reg.list()) == 1
    with pytest.raises(ValueError):
        reg.create("../x", {})


def test_recorder_npz(tmp_path):
    w = EpisodeWriter(tmp_path, "demo", fps=10, width=64, height=32, cameras=("scene", "wrist"), backend="npz")
    img = np.full((48, 96, 3), 7, np.uint8)
    for i in range(5):
        w.add_frame({"scene": img, "wrist": img}, [float(i)] * 16, [float(i)] * 10, "pick")
    info = w.save_episode()
    assert info["frames"] == 5 and info["backend"] == "npz"
    ep = load_episode_npz(tmp_path / "demo" / "episodes" / "episode_000000.npz")
    assert ep["action"].shape == (5, 10) and ep["observation.images.scene"].shape == (5, 32, 64, 3)
    meta = json.loads((tmp_path / "demo" / "meta.json").read_text())
    assert meta["episodes"] == 1 and meta["fps"] == 10
    with pytest.raises(KeyError):
        w.add_frame({"scene": img}, [0.0] * 16, [0.0] * 10, "t")


def test_recorder_keeps_each_cameras_size(tmp_path):
    w = EpisodeWriter(tmp_path, "shapes", fps=10, cameras=("scene", "wrist"), backend="npz")      # no size given: native per camera
    w.add_frame({"scene": np.zeros((54, 54, 3), np.uint8), "wrist": np.zeros((54, 96, 3), np.uint8)}, [0.0] * 16, [0.0] * 10, "t")
    w.add_frame({"scene": np.zeros((108, 108, 3), np.uint8), "wrist": np.zeros((54, 96, 3), np.uint8)}, [0.0] * 16, [0.0] * 10, "t")
    w.save_episode()
    ep = load_episode_npz(tmp_path / "shapes" / "episodes" / "episode_000000.npz")
    assert ep["observation.images.scene"].shape == (2, 54, 54, 3) and ep["observation.images.wrist"].shape == (2, 54, 96, 3)
    meta = json.loads((tmp_path / "shapes" / "meta.json").read_text())
    assert meta["features"]["observation.images.scene"]["shape"] == [54, 54, 3]
    assert meta["features"]["observation.images.wrist"]["shape"] == [54, 96, 3]


@pytest.mark.skipif(not lerobot_available(), reason="lerobot not importable in this interpreter")
def test_recorder_lerobot_sidesteps_a_foreign_dir(tmp_path):
    (tmp_path / "taken" / "episodes").mkdir(parents=True)            # an npz recording already at this name
    w = EpisodeWriter(tmp_path, "taken", fps=10, cameras=("scene",), use_videos=False)
    assert w.backend == "lerobot" and w.repo_id == "taken_2"
    w.add_frame({"scene": np.zeros((32, 64, 3), np.uint8)}, [0.0] * 16, [0.0] * 10, "t")
    w.save_episode()
    w.close()
    assert (tmp_path / "taken_2" / "meta").is_dir()


@pytest.mark.skipif(not lerobot_available(), reason="lerobot not importable in this interpreter")
def test_recorder_lerobot(tmp_path):
    w = EpisodeWriter(tmp_path, "demo_lr", fps=10, width=64, height=32, cameras=("scene",), use_videos=False)
    assert w.backend == "lerobot"
    img = np.zeros((32, 64, 3), np.uint8)
    for i in range(4):
        w.add_frame({"scene": img}, [float(i)] * 16, [float(i)] * 10, "pick")
    info = w.save_episode()
    assert info["backend"] == "lerobot" and info["frames"] == 4
    assert (tmp_path / "demo_lr" / "meta").is_dir()
    w.close()                                   # finalize: the parquet footers, else it cannot be re-opened
    w2 = EpisodeWriter(tmp_path, "demo_lr", fps=10, width=64, height=32, cameras=("scene",), use_videos=False)
    assert w2.episodes == 1
    for i in range(3):
        w2.add_frame({"scene": img}, [0.0] * 16, [0.0] * 10, "pick")
    assert w2.save_episode()["episode"] == 1
    w2.close()


def test_selecting_the_loaded_checkpoint_does_not_reload_it():
    from rd_vla import infer_server
    made = []
    real = infer_server.make_backend

    def counting(name, spec, **kw):
        made.append(spec)
        return real(name, spec, **kw)
    infer_server.make_backend = counting
    try:
        s = infer_server.Server("fake", "one", "cpu")
        assert made == ["one"]
        assert s.command({"cmd": "select", "model": "one"})["ok"] and made == ["one"]          # the same: nothing loaded
        assert s.command({"cmd": "select", "model": "two"})["ok"] and made == ["one", "two"]   # another: loaded
        assert s.checkpoint == "two"
    finally:
        infer_server.make_backend = real


class _Raw:
    """A stand-in for the policy's (1, n, dof) chunk tensor: sliceable as raw[:, a:b], with a shape."""

    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))

    def __getitem__(self, key):
        return _Raw(self.values[key[1]])


def _chunk_policy(length=50, delay=0.0, fail_on=()):
    """compute(obs, prev_left): chunk k planned at observation `obs` holds the actions (obs, 0), (obs, 1), ..."""
    calls = []

    def compute(obs, prev_left):
        calls.append((obs, None if prev_left is None else list(prev_left.values)))
        if len(calls) in fail_on:
            raise RuntimeError("boom")
        time.sleep(delay)
        return _Raw([(obs, i) for i in range(length)]), [(obs, i) for i in range(length)]
    return compute, calls


def _settle(s):
    t = s._thread
    if t is not None:
        t.join(timeout=5.0)


def test_chunk_scheduler_prefetches_and_skips_the_elapsed_actions():
    from rd_vla.chunks import ChunkScheduler
    compute, calls = _chunk_policy()
    s = ChunkScheduler(compute, horizon=8, prefetch_at=2, log=lambda m: None)
    out = []
    for step in range(24):
        out.append(s.next_action(step))
        _settle(s)                                   # the forward pass is faster than a step here
    # the first chunk inline from observation 0; the next planned at step 5 (actions 5, 6, 7 still to play), served from its action 3
    assert out[:8] == [(0, i) for i in range(8)]
    assert out[8:16] == [(5, i) for i in range(3, 11)]
    assert out[16:24] == [(13, i) for i in range(3, 11)]
    assert s.stats["inline"] == 1 and s.stats["prefetched"] == 2 and s.stats["waited"] == 0
    assert [c[0] for c in calls] == [0, 5, 13, 21]


def test_chunk_scheduler_waits_for_a_slow_prefetch_instead_of_computing_twice():
    from rd_vla.chunks import ChunkScheduler
    compute, calls = _chunk_policy(delay=0.05)
    s = ChunkScheduler(compute, horizon=4, prefetch_at=1, log=lambda m: None)
    out = [s.next_action(step) for step in range(12)]              # no pause between steps: the queue runs dry
    assert out[:4] == [(0, i) for i in range(4)] and out[4:8] == [(2, i) for i in range(2, 6)]
    assert s.stats["waited"] >= 1 and s.stats["inline"] == 1 and len(calls) <= 4


def test_chunk_scheduler_reset_drops_a_prefetch_in_flight():
    from rd_vla.chunks import ChunkScheduler
    compute, calls = _chunk_policy(delay=0.1)
    s = ChunkScheduler(compute, horizon=4, prefetch_at=2, log=lambda m: None)
    for step in range(2):
        s.next_action(("old", step))                              # the second step spawns a prefetch
    stale = s._thread
    assert stale is not None
    s.reset()
    stale.join(timeout=5.0)
    assert s._pending is None and len(s) == 0                      # the old episode's chunk went nowhere
    out = [s.next_action(("new", step)) for step in range(2)]
    assert out == [(("new", 0), 0), (("new", 0), 1)]
    _settle(s)


def test_chunk_scheduler_retries_a_failed_prefetch_and_passes_the_tail():
    from rd_vla.chunks import ChunkScheduler
    compute, calls = _chunk_policy(fail_on=(2,))
    logs = []
    s = ChunkScheduler(compute, horizon=4, prefetch_at=2, pass_tail=True, log=logs.append)
    out = []
    for step in range(8):
        out.append(s.next_action(step))
        _settle(s)
    # the prefetch of step 1 fails; step 2 tries again with two actions left to play, so its chunk starts at action 2
    assert out[:4] == [(0, i) for i in range(4)] and out[4:8] == [(2, i) for i in range(2, 6)]
    assert s.stats["inline"] == 1 and s.stats["failed"] == 1 and any("boom" in m for m in logs)
    assert calls[1] == (1, [(0, 1), (0, 2), (0, 3)])               # the prefetch got the outgoing chunk from its own step on
    one = ChunkScheduler(_chunk_policy()[0], horizon=1, prefetch_at=2, log=lambda m: None)   # nothing to refill ahead of
    assert [one.next_action(i) for i in range(3)] == [(0, 0), (1, 0), (2, 0)] and one.stats["inline"] == 3


def test_spacemouse_reports():
    import struct
    from rd_vla.spacemouse import SpaceMouseState, normalise
    assert normalise(0, 350, 0.12) == 0.0 and normalise(30, 350, 0.12) == 0.0            # inside the deadband
    assert abs(normalise(350, 350, 0.12) - 1.0) < 1e-9 and abs(normalise(-700, 350, 0.12) + 1.0) < 1e-9
    assert 0.0 < normalise(60, 350, 0.12) < 0.07                                          # ramps from the deadband's edge
    s = SpaceMouseState()
    s.feed(bytes([1]) + struct.pack("<hhh", 350, -350, 0))                                # translation
    s.feed(bytes([2]) + struct.pack("<hhh", 0, 175, -350))                                # rotation
    assert s.axes["x"] == 1.0 and s.axes["y"] == -1.0 and s.axes["z"] == 0.0
    assert s.axes["roll"] == 0.0 and 0.4 < s.axes["pitch"] < 0.5 and s.axes["yaw"] == -1.0
    s.feed(bytes([1]) + struct.pack("<hhhhhh", 0, 0, 350, 350, 0, 0))                     # both in one report
    assert s.axes["x"] == 0.0 and s.axes["z"] == 1.0 and s.axes["roll"] == 1.0 and s.axes["yaw"] == 0.0
    s.feed(bytes([3, 0b10])); assert s.buttons[:2] == [0, 1]
    s.feed(bytes([23, 1, 2])); s.feed(b""); assert s.reports == 4                         # battery and such: ignored
    s.zero(); assert not any(s.axes.values()) and not any(s.buttons)


def test_spacemouse_finds_only_spacemice(tmp_path):
    from rd_vla.spacemouse import find_nodes
    for name, hid_id in (("hidraw0", "0003:00008086:00000B3A"), ("hidraw4", "0003:0000256F:0000C652"),
                         ("hidraw9", "0003:0000046D:0000C534"), ("hidraw11", "0003:0000046D:0000C626")):
        d = tmp_path / name / "device"
        d.mkdir(parents=True)
        (d / "uevent").write_text(f"DRIVER=hid-generic\nHID_ID={hid_id}\nHID_NAME=x\n")
    # 3Dconnexion's own id, and the old Logitech-branded SpaceNavigator; never Logitech's mouse receiver
    assert find_nodes(str(tmp_path)) == ["/dev/hidraw11", "/dev/hidraw4"]


def test_teleop_integration_and_grasp():
    import importlib.util
    import math
    import re
    # the two pure functions, without importing rclpy: lifted out of the module's source
    src = (Path(__file__).resolve().parents[1] / "rd_vla" / "teleop.py").read_text()
    from rd_vla.geometry import quat_to_rotvec, rotvec_to_quat
    ns = {"quat_to_rotvec": quat_to_rotvec, "rotvec_to_quat": rotvec_to_quat}
    start, end = src.index("AXIS_MAP = "), src.index("class Teleop")
    exec(src[start:end], ns)
    integrate, step_grasp, amap = ns["integrate"], ns["step_grasp"], ns["AXIS_MAP"]
    still = dict.fromkeys(("x", "y", "z", "roll", "pitch", "yaw"), 0.0)
    home = [0.1, 0.2, 0.9, 0.0, 0.0, 0.0]
    assert integrate(home, still, amap, 0.131, 0.45, 1 / 12) == home                      # puck at rest: nothing moves
    out = integrate(home, {**still, "x": 1.0, "z": -1.0}, amap, 0.12, 0.45, 0.5)          # the map inverts x and z
    assert abs(out[0] - (0.1 - 0.06)) < 1e-9 and abs(out[2] - (0.9 + 0.06)) < 1e-9 and out[1] == 0.2 and out[3:] == [0.0, 0.0, 0.0]
    out = integrate(home, {**still, "pitch": 1.0}, amap, 0.12, 0.5, 1.0)                  # puck pitch -> robot yaw, sign -1
    assert out[:3] == home[:3] and abs(out[5] + 0.5) < 1e-6 and abs(out[3]) < 1e-9 and abs(out[4]) < 1e-9
    # rotations compose about the BASE axes: a quarter turn about z, then a quarter turn about base x
    pose = [0, 0, 0, 0.0, 0.0, math.pi / 2]
    out = integrate(pose, {**still, "yaw": -1.0}, amap, 0.0, math.pi / 2, 1.0)            # puck yaw -> robot roll, sign -1: +90 deg about x
    q = rotvec_to_quat(out[3:])
    # the tool's x axis, turned to +y by the first rotation, now points along +z
    x, y, z, w = q
    tool_x = (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w))
    assert abs(tool_x[0]) < 1e-6 and abs(tool_x[1]) < 1e-6 and abs(tool_x[2] - 1.0) < 1e-6
    g = 0.0
    for _ in range(5):
        g = step_grasp(g, [1, 0], 0.104, 0.060)
    assert abs(g - 0.52) < 1e-9 and step_grasp(g, [0, 1], 0.104, 0.060) == g - 0.060
    assert step_grasp(g, [1, 1], 0.1, 0.1) == g and step_grasp(g, [], 0.1, 0.1) == g      # both or none: hold
    assert step_grasp(0.98, [1, 0], 0.104, 0.06) == 1.0 and step_grasp(0.03, [0, 1], 0.104, 0.06) == 0.0


def test_keep_out_pushes_a_target_off_the_body_and_above_the_floor():
    from rd_vla.geometry import keep_out
    box = (-0.27, 0.27, 0.0, 0.88)
    assert keep_out([0.5, 0.5, 0.3], box, 0.03, -0.2) == ([0.5, 0.5, 0.3], False)          # beside the body: untouched
    xyz, changed = keep_out([0.25, 0.4, 0.3], box, 0.03, -0.2)                             # inside, nearest the +x edge
    assert changed and abs(xyz[0] - 0.30) < 1e-9 and xyz[1:] == [0.4, 0.3]
    xyz, changed = keep_out([0.0, 0.01, 0.3], box, 0.03, -0.2)                             # just inside the front edge
    assert changed and abs(xyz[1] + 0.03) < 1e-9 and xyz[0] == 0.0
    assert keep_out([0.5, 0.5, -0.4], box, 0.03, -0.2) == ([0.5, 0.5, -0.2], True)         # under the floor: lifted
    assert keep_out([0.0, 0.4, -0.4], None, 0.03, None) == ([0.0, 0.4, -0.4], False)       # no limits configured


def test_robot_yaml_carries_the_arms_workspace_limits():
    from rd_vla.geometry import workspace_limits
    w = workspace_limits()
    assert w["z_floor"] is not None and w["z_floor"] < 0 and len(w["no_go_xy"]) == 4 and 0 < w["margin"] < 0.2
    assert w["no_go_xy"][0] < w["no_go_xy"][1] and w["no_go_xy"][2] < w["no_go_xy"][3]


class _FakePad:
    key_count = 15

    def __init__(self):
        self.labels, self.cb = {}, None

    def set_label(self, key, text, bg=None, fg=None):
        self.labels[key] = text

    def on_key(self, cb):
        self.cb = cb

    def close(self):
        pass


class _FakeTeleop:
    def __init__(self):
        self.recording = self.freedrive = self.enabled = False
        self._regrasp_open, self.regrasp_ranges = None, []
        self.commands, self.nudges, self.rated, self.logs = [], [], [], []

    def p(self, name):
        return {"grasp_close_rate": 0.104, "grasp_open_rate": 0.060, "fps": 12.0}[name]

    def command(self, name, a=None):
        self.commands.append(name)

    def nudge(self, delta, finger=-1):
        self.nudges.append((round(delta, 4), finger))

    def rate(self, stars):
        self.rated.append(stars)
        return f"rated {stars}"

    def log(self, line):
        self.logs.append(line)


def test_stream_deck_keys():
    from rd_vla.deck import Deck
    now = [100.0]
    t, pad = _FakeTeleop(), _FakePad()
    d = Deck(t, pad=pad, clock=lambda: now[0])
    press = lambda k: (pad.cb(k, True), pad.cb(k, False), d.poll())  # noqa: E731
    d.poll()
    assert pad.labels[0] == "●" and pad.labels[1] == "⌂" and pad.labels[2] == "🆓" and pad.labels[10] == "✊" and pad.labels[11] == "1"
    press(0); press(1); press(2); press(3); press(4)                # idle: REC, home, freedrive, recover the arm, teleop off
    assert t.commands == ["rec", "home", "freedrive", "reconnect", "quit"]
    t.commands.clear(); t.recording = True
    d.poll()
    assert pad.labels[0] == "+\n1" and pad.labels[1] == "💾" and pad.labels[2] == "🗑"
    press(0); t._regrasp_open = 5; d.poll()
    assert pad.labels[0] == "END\n2"                                 # a flagged stretch is open: the key says how to close it
    press(2); press(1)
    assert t.commands == ["grasp_mark", "discard", "save"]
    # fingers: held keys ramp, the thumb is key 1 -> channel 3, the whole hand is the first column
    pad.cb(11, True); d.poll(); d.poll(); pad.cb(11, False); d.poll()
    assert t.nudges == [(0.1248, 3), (0.1248, 3)]                    # 0.104 per teleop tick, this poll runs every 0.1 s at 12 Hz
    t.nudges.clear(); pad.cb(5, True); d.poll(); pad.cb(5, False); d.poll()
    assert t.nudges == [(-0.072, -1)]
    pad.cb(7, True); d.poll(); pad.cb(7, False); assert t.nudges[-1] == (-0.072, 2)      # finger 2 = the index = channel 2
    # after a save the bottom row is the stars for 30 s, then the fingers again
    t.recording = False; t.nudges.clear()
    d.enter_rating(); d.poll()
    assert pad.labels[10] == "1\n★" and pad.labels[14] == "5\n★"
    press(13)
    assert t.rated == [4] and t.nudges == [] and pad.labels[13] == "3"
    d.enter_rating(); d.poll(); now[0] += 31; d.poll()
    assert pad.labels[10] == "✊"
    pad.cb(12, True); d.poll(); assert t.nudges and t.nudges[-1][1] == 2 and t.rated == [4]


def test_tool_voltage_is_twelve_or_nothing(monkeypatch):
    import importlib.util
    src = (Path(__file__).resolve().parents[2] / "rd_arm" / "rd_arm" / "freedrive.py").read_text()
    ns = {}
    start, end = src.index("HAND_VOLTAGE = 12"), src.index("class Freedrive")
    from rd_bringup import config
    exec("from rd_bringup import config\n" + src[start:end], ns)
    assert ns["tool_voltage"]() == 12                                # robot.yaml as it is
    monkeypatch.setattr(config, "get", lambda key, default=None: 24 if key == "arm.tool_voltage" else default)
    ns["config"] = config
    with pytest.raises(ValueError):
        ns["tool_voltage"]()


def test_camera_squares_follow_the_pages_file(tmp_path, monkeypatch):
    from rd_bringup import config
    monkeypatch.setattr(config, "data_dir", lambda: tmp_path)
    assert config.camera_squares() == {"scene": 0.4, "wrist": None, "side": 0.3}
    f = config.camera_squares_file()
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"scene": 0.75, "wrist": 0.5, "side": 7}))
    assert config.camera_squares() == {"scene": 0.75, "wrist": None, "side": 1.0}     # the wrist is never cropped; clamped to 0..1
    f.write_text("not json")
    assert config.camera_squares()["scene"] == 0.4
