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
