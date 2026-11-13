"""The HTTP API without ROS (the GPU box's mode): pages, maps, landmarks, routes, evals, services, datasets, and
the Panels tab against a fake WLED (its JSON API and its live pixel WebSocket on one loopback port)."""
import asyncio
import http
import json
import shutil
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rd_manager.server import build_app
from rd_manager.supervisor import FakeSupervisor

PKG = Path(__file__).resolve().parents[1]


def _app(tmp_path, monkeypatch, supervisor_url, **kw):
    shutil.copytree(PKG.parent / "rd_nav" / "maps", tmp_path / "maps")
    shutil.copytree(PKG.parent / "rd_nav" / "landmarks", tmp_path / "landmarks")
    (tmp_path / "routes").mkdir()
    (tmp_path / "outputs" / "datasets").mkdir(parents=True)
    monkeypatch.delenv("RD_EVALS_DIR", raising=False)
    return build_app(ros=False, sim=False, data_dir=str(tmp_path), robot_url="http://robot:8092", supervisor_url=supervisor_url,
                     maps_dir=tmp_path / "maps", landmarks_dir=tmp_path / "landmarks", routes_dir=tmp_path / "routes", **kw)


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake = FakeSupervisor(["rd_base", "rd_nav", "rd_slam"]).start()
    try:
        yield TestClient(_app(tmp_path, monkeypatch, fake.url))
    finally:
        fake.stop()


class FakeWled:
    """A WLED on loopback: /json/si and /json/eff over HTTP, and on /ws the live pixel stream once asked ({"lv":true}):
    binary "L", 2, width, height, RGB per pixel. The frames count up in their first pixel."""
    W, H = 32, 16
    SI = {"info": {"name": "Disco Panels", "ver": "0.15.0", "live": False, "leds": {"fps": 30, "matrix": {"w": W, "h": H}}},
          "state": {"on": True, "bri": 128, "seg": [{"id": 0, "n": "Right", "start": 0, "stop": 16, "startY": 0, "stopY": 16, "on": True, "fx": 1},
                                                     {"id": 1, "start": 16, "stop": 32, "startY": 0, "stopY": 16, "on": True, "fx": 7},
                                                     {"id": 2, "start": 0, "stop": 0}]}}
    EFF = ["Solid", "Blink", "Breathe"]

    def start(self):
        self.asked, self.released, self.ready = [], threading.Event(), threading.Event()
        threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True).start()
        assert self.ready.wait(5.0)
        return self

    def stop(self):
        self.loop.call_soon_threadsafe(self.done.set)

    async def _main(self):
        from websockets.legacy.server import serve
        self.loop, self.done = asyncio.get_running_loop(), asyncio.Event()
        async with serve(self._ws, "127.0.0.1", 0, process_request=self._http) as server:
            self.url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            self.ready.set()
            await self.done.wait()

    async def _http(self, path, headers):
        body = {"/json/si": self.SI, "/json/eff": self.EFF}.get(path)
        if body is not None:
            return http.HTTPStatus.OK, [("Content-Type", "application/json")], json.dumps(body).encode()
        return None                                         # /ws: go on with the WebSocket handshake

    async def _ws(self, ws, path=None):
        stream = None

        async def frames():
            n = 0
            while True:
                await ws.send(b"L" + bytes([2, self.W, self.H]) + bytes([n % 256, 0, 40]) * (self.W * self.H))
                n += 1
                await asyncio.sleep(0.03)
        try:
            async for msg in ws:
                lv = json.loads(msg).get("lv")
                self.asked.append(lv)
                if lv and stream is None:
                    stream = asyncio.ensure_future(frames())
                elif lv is False:
                    self.released.set()
        finally:
            if stream:
                stream.cancel()


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
    assert m["name"] == "demo.yaml" and m["resolution"] == 0.05
    assert client.get("/map.png").content[:4] == b"\x89PNG"
    p = client.get("/api/pose").json()
    assert p["pose"] is None and p["ros_up"] is False
    assert client.get("/api/scans").json()["scans"] == {}
    assert client.post("/api/drive", json={"vx": 0.1, "wz": 0}).status_code == 503
    assert client.post("/api/voice/text", json={"text": "robot disco hello"}).status_code == 503
    assert client.post("/api/voice/text", json={"text": "   "}).status_code == 400
    assert client.get("/api/voice").json() == {}
    assert client.get("/api/cameras").json()["cameras"][0]["alive"] is False


def test_maps_tab(client):
    d = client.get("/api/maps").json()
    assert d["active"] == "demo" and d["units"]["rd_nav"] == "STOPPED" and d["source"] == "live"
    assert client.get("/api/maps/demo.png").status_code == 200
    r = client.post("/api/maps/slam/start", json={"mode": "new", "name": "kitchen"}).json()
    assert r["ok"] and client.get("/api/maps").json()["mapping"] is True
    assert client.post("/api/maps/slam/stop", json={}).json()["active"] == "demo"
    assert client.post("/api/maps/demo/activate", json={"restart": False}).json()["active"] == "demo"
    assert client.delete("/api/maps/demo").status_code == 400          # the active map
    png = client.get("/api/maps/demo.png").content
    import base64
    r = client.post("/api/maps/demo/draw", json={"png": "data:image/png;base64," + base64.b64encode(png).decode(), "save_as": "zone_copy"}).json()
    assert r["map"]["name"] == "zone_copy"
    assert client.delete("/api/maps/zone_copy").json()["ok"]
    assert client.post("/api/maps/demo/keepout").json()["cells"]["top_box"] > 0


def test_landmarks_and_tasks(client):
    d = client.get("/api/landmarks").json()
    assert {lm["id"] for lm in d["landmarks"]} >= {"charger", "dining-room"} and d["map"] == "demo"
    assert client.post("/api/landmarks", json={}).status_code == 503                        # no ROS: no capture
    r = client.post("/api/landmarks", json={"name": "Kitchen", "x": 1.0, "y": 2.0, "yaw": 0.5}).json()
    assert r["landmark"]["id"] == "kitchen" and r["landmark"]["map"] == "demo"
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
    assert [w["name"] for w in d["waypoints"]] == ["a", "b"] and d["map"] == "demo"
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
    assert r["reachable"] is False and r["url"] == "http://robot:8092/teleop/"
    assert client.get("/api/route_editor").json()["url"] == "http://robot:8092/live/"
    assert client.get("/api/folks").json()["reachable"] is False
    assert client.get("/api/volume").json()["volume"] is None
    assert client.get("/api/panels").json() == {"url": "", "reachable": False, "info": None}       # no panels configured
    assert 'data-view="panels"' in client.get("/").text and "loadPanelsView" in client.get("/app.js").text


def test_panels_tab(tmp_path, monkeypatch):
    wled = FakeWled().start()
    try:
        with TestClient(_app(tmp_path, monkeypatch, "", panels_url=wled.url)) as c:
            r = c.get("/api/panels").json()
            assert r["reachable"] and r["url"] == wled.url + "/"
            i = r["info"]
            assert (i["name"], i["w"], i["h"], i["on"], i["bri"], i["live"]) == ("Disco Panels", 32, 16, True, 128, False)
            assert [(g["name"], g["x0"], g["x1"], g["y0"], g["y1"], g["effect"]) for g in i["segments"]] == \
                [("Right", 0, 16, 0, 16, "Blink"), ("segment 1", 16, 32, 0, 16, "7")]       # unnamed -> "segment n"; unknown effect -> its number
            # two tabs share the one stream the WLED gives out
            with c.websocket_connect("/ws/panels") as a, c.websocket_connect("/ws/panels") as b:
                fa, fb = a.receive_bytes(), b.receive_bytes()
                for f in (fa, fb):
                    assert f[:4] == b"L\x02\x20\x10" and len(f) == 4 + 32 * 16 * 3 and f[5:7] == bytes([0, 40])
                assert a.receive_bytes()[4] != fa[4]                                        # it keeps coming
            assert wled.asked.count(True) == 1                                              # one upstream connection for both
            assert wled.released.wait(8.0), "the relay did not hand the stream back after the last tab left"
    finally:
        wled.stop()


def test_panels_unreachable(tmp_path, monkeypatch):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{s.getsockname()[1]}"
    s.close()                                                                               # nothing listens here
    c = TestClient(_app(tmp_path, monkeypatch, "", panels_url=url))
    assert c.get("/api/panels").json() == {"url": url + "/", "reachable": False, "info": None}


def test_teleop_page_and_its_config(client):
    r = client.get("/teleop/")
    assert r.status_code == 200 and "btn-teleop" in r.text and "/static/pages/robot/robot.js" in r.text
    assert client.get("/static/pages/robot/robot.js").status_code == 200 and client.get("/static/pages/robot/robot.css").status_code == 200
    c = client.get("/api/teleop/config").json()
    assert c["mode"] == "teleop" and c["api"] >= 2 and set(c["cams"]) == {"scene", "side", "wrist"}
    assert client.get("/api/teleop").json()["running"] is False                   # no ROS in this test
    assert client.post("/api/teleop/cmd", json={"name": "rec"}).status_code == 503
    assert client.get("/api/robot").json()["url"].endswith("/teleop/")


def test_dataset_import_passes_the_pages_fields(client, tmp_path):
    src = tmp_path / "elsewhere" / "ds_a"
    (src / "meta").mkdir(parents=True)
    (src / "meta" / "info.json").write_text("{}")
    r = client.post("/api/datasets/add", json={"remote": None, "source": str(src), "as_name": "ds_copy"})
    assert r.status_code == 200 and r.json()["name"] == "ds_copy"       # was a TypeError: the server passed name=
    assert client.post("/api/datasets/add", json={"remote": None, "source": None, "as_name": None}).status_code == 400
