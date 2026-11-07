"""The HTTP API without ROS (the GPU box's mode): pages, maps, landmarks, routes, evals, services, datasets."""
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rd_manager.server import build_app
from rd_manager.supervisor import FakeSupervisor

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path, monkeypatch):
    shutil.copytree(PKG.parent / "rd_nav" / "maps", tmp_path / "maps")
    shutil.copytree(PKG.parent / "rd_nav" / "landmarks", tmp_path / "landmarks")
    (tmp_path / "routes").mkdir()
    (tmp_path / "outputs" / "datasets").mkdir(parents=True)
    monkeypatch.delenv("RD_EVALS_DIR", raising=False)
    fake = FakeSupervisor(["rd_base", "rd_nav", "rd_slam"]).start()
    app = build_app(ros=False, sim=False, data_dir=str(tmp_path), robot_url="http://robot:8092", supervisor_url=fake.url,
                    maps_dir=tmp_path / "maps", landmarks_dir=tmp_path / "landmarks", routes_dir=tmp_path / "routes")
    try:
        yield TestClient(app)
    finally:
        fake.stop()


def test_pages_and_static(client):
    for path in ("/", "/live/", "/live/3d", "/arm/", "/folks/"):
        r = client.get(path)
        assert r.status_code == 200 and "<html" in r.text.lower(), path
    assert client.get("/app.js").status_code == 200
    assert client.get("/static/drivepad.js").status_code == 200
    assert client.get("/static/vendor/three.module.min.js").status_code == 200
    assert client.get("/healthz").json()["ros"] is False


def test_map_and_pose_without_ros(client):
    m = client.get("/api/map").json()
    assert m["name"] == "zone_one.yaml" and m["resolution"] == 0.05
    assert client.get("/map.png").content[:4] == b"\x89PNG"
    p = client.get("/api/pose").json()
    assert p["pose"] is None and p["ros_up"] is False
    assert client.get("/api/scans").json()["scans"] == {}
    assert client.post("/api/drive", json={"vx": 0.1, "wz": 0}).status_code == 503
    assert client.get("/api/cameras").json()["cameras"][0]["alive"] is False


def test_maps_tab(client):
    d = client.get("/api/maps").json()
    assert d["active"] == "zone_one" and d["units"]["rd_nav"] == "STOPPED" and d["source"] == "live"
    assert client.get("/api/maps/zone_one.png").status_code == 200
    r = client.post("/api/maps/slam/start", json={"mode": "new", "name": "kitchen"}).json()
    assert r["ok"] and client.get("/api/maps").json()["mapping"] is True
    assert client.post("/api/maps/slam/stop", json={}).json()["active"] == "zone_one"
    assert client.post("/api/maps/zone_one/activate", json={"restart": False}).json()["active"] == "zone_one"
    assert client.delete("/api/maps/zone_one").status_code == 400          # the active map
    png = client.get("/api/maps/zone_one.png").content
    import base64
    r = client.post("/api/maps/zone_one/draw", json={"png": "data:image/png;base64," + base64.b64encode(png).decode(), "save_as": "zone_copy"}).json()
    assert r["map"]["name"] == "zone_copy"
    assert client.delete("/api/maps/zone_copy").json()["ok"]
    assert client.post("/api/maps/zone_one/keepout").json()["cells"]["top_box"] > 0


def test_landmarks_and_tasks(client):
    d = client.get("/api/landmarks").json()
    assert {lm["id"] for lm in d["landmarks"]} >= {"charger", "dining-room"} and d["map"] == "zone_one"
    assert client.post("/api/landmarks", json={}).status_code == 503                        # no ROS: no capture
    r = client.post("/api/landmarks", json={"name": "Kitchen", "x": 1.0, "y": 2.0, "yaw": 0.5}).json()
    assert r["landmark"]["id"] == "kitchen" and r["landmark"]["map"] == "zone_one"
    t = client.post("/api/landmarks/kitchen/tasks", json={"name": "Make coffee", "model": "coffee_v1", "task": "make coffee", "seconds": 45}).json()["task"]
    assert t["id"] == "make-coffee" and t["seconds"] == 45.0
    t2 = client.post("/api/landmarks/kitchen/tasks/make-coffee", json={"done_when": "closed"}).json()["task"]
    assert t2["done_when"] == "closed"
    assert client.post("/api/landmarks/kitchen/tasks/make-coffee/run").status_code == 503     # no ROS
    assert client.delete("/api/landmarks/kitchen/tasks/make-coffee").json()["ok"]
    assert client.post("/api/landmarks/kitchen", json={"name": "Kitchen 2"}).json()["landmark"]["name"] == "Kitchen 2"
    assert client.delete("/api/landmarks/kitchen").json()["ok"]
    assert client.delete("/api/landmarks/kitchen").status_code == 404


def test_routes(client):
    assert client.get("/api/routes").json()["routes"] == []
    wps = [{"name": "a", "position": {"x": 1, "y": 2, "z": 0}, "orientation": {"w": 1}, "post_actions": [{"say": "hi"}]},
           {"name": "b", "position": {"x": 3, "y": 4, "z": 0}, "orientation": {"w": 1}}]
    r = client.post("/api/route?name=test route", json={"waypoints": wps}).json()
    assert r["ok"] and "2 segment" in r["check"]
    d = client.get("/api/route?name=test route").json()
    assert [w["name"] for w in d["waypoints"]] == ["a", "b"] and d["map"] == "zone_one"
    assert client.get("/api/routes").json()["routes"] == ["test route"]
    assert client.post("/api/route/run", json={"name": "test route"}).status_code == 503       # no ROS
    assert client.delete("/api/route?name=test route").json()["ok"]


def test_evals_models_datasets(client):
    assert client.get("/api/evals").json()["evals"] == []
    r = client.post("/api/evals", json={"name": "20261107_x", "task": "pick", "model": "m1", "episodes": 2, "successes": 1}).json()
    assert r["success_rate"] == 0.5
    client.post("/api/evals/20261107_x/annotate", json={"rating": 3, "notes": "ok"})
    assert client.get("/api/evals/groups").json()["groups"][0]["rating"] == 3.0
    assert client.get("/api/evals/20261107_x/video").status_code == 404
    m = client.get("/api/models").json()
    assert m["models"] == [] and m["runs"] == []
    assert client.get("/api/datasets").json()["datasets"] == []


def test_services_and_robot_tabs(client):
    s = client.get("/api/services").json()
    assert s["reachable"] and {x["name"] for x in s["services"]} == {"rd_base", "rd_nav", "rd_slam"}
    assert client.post("/api/services/rd_nav", json={"action": "start"}).json()["ok"]
    assert client.get("/api/services/rd_nav/log").json()["lines"]
    assert client.post("/api/services/rd_nav", json={"action": "dance"}).status_code == 400
    assert client.get("/api/activity").json()["reachable"]
    r = client.get("/api/robot").json()
    assert r["reachable"] is False and r["url"] == "http://robot:8092/arm/"
    assert client.get("/api/route_editor").json()["url"] == "http://robot:8092/live/"
    assert client.get("/api/folks").json()["reachable"] is False
    assert client.get("/api/volume").json()["volume"] is None
