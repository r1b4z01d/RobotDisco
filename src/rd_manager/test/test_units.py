"""ROS-free: route files and the runner with fakes, the supervisor client against the fake supervisord,
the map / costmap renderer on a synthetic grid."""
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import yaml

from rd_manager.grids import MapView
from rd_manager.routes import RouteRunner, check_route, load_route, route_map, save_route, segments, valid_route_name
from rd_manager.supervisor import FakeSupervisor, Supervisor

PKG = Path(__file__).resolve().parents[1]


def test_route_roundtrip(tmp_path):
    wps = [{"name": "a", "position": {"x": 1, "y": 2, "z": 0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
           {"name": "b", "position": {"x": 3, "y": 4, "z": 0}, "orientation": {"x": 0, "y": 0, "z": 0.7071, "w": 0.7071},
            "post_actions": [{"vla": "m1", "task": "grab", "seconds": 30, "done": "closed"}, {"say": "done"}]},
           {"name": "c", "position": {"x": 5, "y": 6, "z": 0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}, "pause": 2}]
    p = tmp_path / "r.yaml"
    save_route(p, wps, map_name="demo")
    assert route_map(p) == "demo"
    loaded, init = load_route(p)
    assert [w.name for w in loaded] == ["a", "b", "c"] and init is None
    assert abs(loaded[1].yaw - 1.5708) < 1e-3 and loaded[1].actions[0].kind == "vla" and loaded[2].pause == 2
    assert segments(loaded) == [(0, 1), (2, 2)]
    assert "2 segment" in check_route(p)
    assert loaded[1].as_dict()["post_actions"][0]["vla"] == "m1"
    (tmp_path / "old.yaml").write_text(yaml.safe_dump({"waypoints": [{"name": "x", "position": {}, "orientation": {}, "post_actions": ["ls", {"run": "echo", "wait": True}]}]}))
    assert "shell action" in check_route(tmp_path / "old.yaml") and not load_route(tmp_path / "old.yaml")[0][0].blocking
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"waypoints": [{"name": "x", "position": {}, "orientation": {}, "post_actions": [{"fly": 1}]}]}))
    assert check_route(tmp_path / "bad.yaml").startswith("ERROR")
    assert valid_route_name("tigger route") and not valid_route_name("../x")


def test_v1_route_files_load():
    for p in (PKG / "routes").glob("*.yaml"):
        wps, _ = load_route(p)
        assert wps, p


def test_route_runner_runs_segments_and_actions():
    log = []
    r = RouteRunner(follow=lambda poses: (log.append(("follow", len(poses))), (True, "arrived"))[1],
                    run_action=lambda a: (log.append(("action", a.kind)), (True, "ok"))[1],
                    cancel=lambda: log.append(("cancel",)), set_initial_pose=lambda x, y, yaw: log.append(("init", x)))
    import tempfile
    p = Path(tempfile.mkdtemp()) / "r.yaml"
    save_route(p, [{"name": "a", "position": {"x": 1, "y": 2, "z": 0}, "orientation": {"w": 1}, "post_actions": [{"say": "hi"}]},
                   {"name": "b", "position": {"x": 3, "y": 4, "z": 0}, "orientation": {"w": 1}}],
               initial_pose={"position": {"x": 0.5, "y": 0.5, "z": 0}, "orientation": {"w": 1}})
    r.start(p)
    for _ in range(50):
        if not r.busy:
            break
        time.sleep(0.05)
    assert not r.busy and r.state["phase"] == "done" and r.state["segments"] == 2
    assert log == [("init", 0.5), ("follow", 1), ("action", "say"), ("follow", 1)]


def test_route_runner_stop():
    gate = threading.Event()
    r = RouteRunner(follow=lambda poses: (gate.wait(5), (False, "cancelled"))[1], run_action=lambda a: (True, ""), cancel=gate.set,
                    set_initial_pose=lambda *a: None)
    import tempfile
    p = Path(tempfile.mkdtemp()) / "r.yaml"
    save_route(p, [{"name": "a", "position": {"x": 1, "y": 2, "z": 0}, "orientation": {"w": 1}}])
    r.start(p)
    time.sleep(0.2)
    assert r.stop() == "stopping"
    time.sleep(0.3)
    assert not r.busy and r.state["phase"] == "failed"


def test_supervisor_against_the_fake():
    fake = FakeSupervisor(["rd_base", "rd_nav", "rd_perception"]).start()
    try:
        sv = Supervisor(fake.url)
        sv.settle_scale = 0.0
        d = sv.services()
        assert d["reachable"] and {s["name"] for s in d["services"]} == {"rd_base", "rd_nav", "rd_perception"}
        assert sv.action("rd_nav", "start")["ok"] and sv.states()["rd_nav"] == "RUNNING"
        assert sv.action("rd_nav", "restart")["ok"] and ("stop", "rd_nav") in fake.calls
        log = sv.log("rd_nav")
        assert log["lines"] and "started" in log["lines"][0]
        act = sv.activity()
        assert act["reachable"] and any(u["unit"] == "nav" for u in act["units"])
        sv.restart_base()
        for _ in range(100):
            if not sv.job["running"]:
                break
            time.sleep(0.1)
        assert any("done" in line for line in sv.job["log"])
        assert not Supervisor("http://127.0.0.1:1/RPC2", timeout=0.3).services()["reachable"]
    finally:
        fake.stop()


def test_mapview_file_and_grid(tmp_path):
    shutil.copytree(PKG.parent / "rd_nav" / "maps", tmp_path / "maps")
    mv = MapView()
    meta = mv.load_file(tmp_path / "maps" / "demo.yaml")
    assert meta["resolution"] == 0.05 and meta["width"] > 100 and mv.png[:4] == b"\x89PNG" and meta["rev"] == 1
    grid = NS(data=[0] * 40 + [100] * 10 + [-1] * 50, info=NS(width=10, height=10, resolution=0.1, origin=NS(position=NS(x=1.0, y=2.0), orientation=NS(x=0, y=0, z=0, w=1))),
              header=NS(frame_id="map", stamp=NS(sec=5, nanosec=0)))
    meta = mv.load_grid(grid, "slam:test")
    assert meta["live"] and meta["origin"] == [1.0, 2.0] and meta["rev"] == 2
    cm = NS(data=[0] * 20 + [50] * 30 + [99] * 25 + [100] * 25, info=grid.info, header=NS(frame_id="odom", stamp=NS(sec=7, nanosec=1)))
    png, cmeta = mv.costmap_png("local", {"msg": cm, "t": time.time()}, {}, "/local_costmap/costmap")
    assert png[:4] == b"\x89PNG" and "error" in cmeta
    png2, cmeta2 = mv.costmap_png("local", {"msg": cm, "t": time.time()}, {"odom": (1.0, 0.0, 0.0)}, "/local_costmap/costmap")
    assert png2 == png and cmeta2["origin"] == [2.0, 2.0]


def test_typed_text_gets_the_wake_phrase_once():
    from rd_manager.ros_bridge import with_wake_phrase
    assert with_wake_phrase("what time is it") == "robot disco, what time is it"
    assert with_wake_phrase("  Robot Disco what time is it ") == "Robot Disco what time is it"


def test_exposure_lock_freezes_each_camera_where_auto_left_it():
    from rd_manager import exposure
    cfg = {"by_path": "/dev/fake-{port}", "scene": {"port": "4.2"}, "wrist": {"port": "1.1"}, "side": {"port": "2.4"}}
    ctrls = {d: {"auto_exposure": 3, "exposure_time_absolute": 150 + i, "white_balance_automatic": 1, "white_balance_temperature": 4600}
             for i, d in enumerate(exposure.devices(cfg).values())}
    calls = []

    def run(args):
        dev = args[2]
        if "--list-ctrls" in args:
            return "\n".join(f"   {k} 0x009a0901 (int)    : min=0 max=5000 step=1 default=3 value={v} flags=inactive" for k, v in ctrls[dev].items())
        calls.append(args)
        for kv in args[4].split(","):
            k, v = kv.split("=")
            ctrls[dev][k] = int(v)
        return ""
    assert exposure.state(cfg, run)["locked"] is False
    s = exposure.set_locked(cfg, True, run)
    assert s["locked"] and [c["exposure"] for c in s["cameras"].values()] == [150, 151, 152]     # what auto had chosen
    assert all(c["auto_exposure"] == 1 and c["white_balance_automatic"] == 0 and c["white_balance_temperature"] == 4600 for c in ctrls.values())
    s = exposure.set_locked(cfg, False, run)
    assert not s["locked"] and all(c["auto_exposure"] == 3 and c["white_balance_automatic"] == 1 for c in ctrls.values())
    assert exposure.state({"by_path": "/dev/x-{port}", "scene": {"port": "1"}}, lambda a: "")["cameras"]["scene"]["found"] is False


def test_exposure_keys_of_the_teleop_page():
    from rd_manager import exposure
    cfg = {"by_path": "/dev/fake-{port}", "scene": {"port": "4.2"}, "wrist": {"port": "1.1"}, "side": {"port": "2.4"}}
    ctrls = {d: {"auto_exposure": 3, "exposure_time_absolute": 156} for d in exposure.devices(cfg).values()}

    def run(args):
        if "--list-ctrls" in args:
            return "\n".join(f"  {k} 0x1 (int) : min=1 max=5000 value={v}" for k, v in ctrls[args[2]].items())
        for kv in args[4].split(","):
            k, v = kv.split("=")
            ctrls[args[2]][k] = int(v)
        return ""
    s = exposure.adjust(cfg, "plus", "scene", run)                 # one camera brighter: manual, the others untouched
    assert s["cameras"]["scene"]["exposure"] > 156 and s["cameras"]["scene"]["locked"]
    assert s["cameras"]["side"]["exposure"] == 156 and not s["cameras"]["side"]["locked"] and not s["locked"]
    s = exposure.adjust(cfg, "minus", None, run)                   # all darker
    assert s["cameras"]["side"]["exposure"] < 156 and s["locked"]
    up = exposure.adjust(cfg, "plus", "side", run)["cameras"]["side"]["exposure"]
    assert up > s["cameras"]["side"]["exposure"]
    assert exposure.adjust(cfg, "auto", None, run)["locked"] is False
    assert exposure.adjust(cfg, "lock", None, run)["locked"] is True


def test_training_runs_under_the_lerobot_python_and_its_own_script(tmp_path, monkeypatch):
    import json as _json
    from rd_manager import models
    ds = tmp_path / "outputs" / "datasets" / "demo"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(_json.dumps({"total_episodes": 2, "total_frames": 100, "features": {
        "observation.images.scene": {}, "observation.images.wrist": {}, "observation.images.side": {}}}))
    monkeypatch.setenv("RD_VLA_PYTHON", "/somewhere/vla/bin/python")
    monkeypatch.setattr(models, "_running_runs", lambda: set())
    r = models.start_training(str(tmp_path), str(tmp_path / "outputs" / "datasets"), "demo", "smolvla", 30, 4, 15, dry_run=True)
    cmd = r["command"].split()
    assert cmd[0] == "/somewhere/vla/bin/python"                      # not the manager's own interpreter, which has no lerobot
    script = Path(cmd[1])
    assert script.name == "train_regrasp.py" and script.is_file() and script.is_absolute()
    # the script's folder is first on sys.path when it runs: nothing in it may shadow a library lerobot imports
    assert {p.name for p in script.parent.glob("*.py")} == {"__init__.py", "train_regrasp.py", "hf_cache.py"}
    assert "--policy.path=lerobot/smolvla_base" in cmd and "--steps=30" in cmd and r["run"] == "demo_smolvla"


def test_push_checkpoint_copies_only_the_served_part(tmp_path):
    from rd_manager import models
    run = tmp_path / "outputs" / "train" / "run_a"
    for step in ("000010", "000020"):
        pm = run / "checkpoints" / step / "pretrained_model"
        pm.mkdir(parents=True)
        (pm / "config.json").write_text('{"type": "smolvla"}')
        (pm / "model.safetensors").write_bytes(step.encode())
        (run / "checkpoints" / step / "training_state").mkdir()
        (run / "checkpoints" / step / "training_state" / "optimizer.bin").write_bytes(b"x" * 10)
    (run / "capture.json").write_text("{}")
    dest = tmp_path / "robot" / "outputs" / "train"
    r = models.push_checkpoint(str(tmp_path), "run_a", None, str(dest))
    assert r["step"] == "000020" and r["spec"] == "run_a@000020"
    got = dest / "run_a" / "checkpoints" / "000020" / "pretrained_model"
    assert (got / "model.safetensors").read_bytes() == b"000020" and (dest / "run_a" / "capture.json").is_file()
    assert not (dest / "run_a" / "checkpoints" / "000020" / "training_state").exists()
    assert not (dest / "run_a" / "checkpoints" / "000010").exists()
    assert models.push_checkpoint(str(tmp_path), "run_a", "000010", str(dest))["step"] == "000010"
    from rd_vla import checkpoints
    assert checkpoints.resolve("run_a", dest).parent.name == "000020"       # what the robot's server would load


def test_uncropped_view_draws_the_square_on_the_upright_frame():
    from rd_manager.ros_bridge import upright_with_square
    frame = np.full((720, 1280, 3), 100, np.uint8)                       # as the camera gives it: landscape
    out = upright_with_square(frame, "scene", squares={"scene": 0.4}, rotate=270)
    assert out.shape == (640, 360, 3)                                    # turned upright, half size
    off = int(round((640 - 360) * 0.4))
    assert tuple(out[off + 180, 180]) == (100, 100, 100)                 # inside the square: as it is
    assert tuple(out[off + 1, 180]) == (255, 200, 0)                     # its edge
    assert tuple(out[5, 180]) == (50, 50, 50) and tuple(out[635, 180]) == (50, 50, 50)     # outside it: dimmed
    whole = upright_with_square(frame, "wrist", squares={"wrist": None}, rotate=0)
    assert whole.shape == (360, 640, 3) and (whole == 100).all()         # an uncropped camera: nothing drawn
