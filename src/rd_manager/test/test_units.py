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
    save_route(p, wps, map_name="zone_one")
    assert route_map(p) == "zone_one"
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
    meta = mv.load_file(tmp_path / "maps" / "zone_one.yaml")
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
