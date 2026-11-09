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
