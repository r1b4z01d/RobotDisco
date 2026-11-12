"""The manager's HTTP side: FastAPI routes over the ROS bridge, the stores (maps, landmarks, routes, evals,
checkpoints, datasets) and supervisor. v1 webui/server.py + Scripts/route_editor.py in one process.

Pages:  /            the manager (tabs)        /live/        the 2D map + route editor      /live/3d   the 3D view
        /arm/        arm + hand + policy       /folks/       the people the robot knows
The Panels tab (the WLED LED matrices) is a view of the manager page: /api/panels and the /ws/panels pixel relay.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rd_nav.keepout import build_mask
from rd_nav.landmarks import LandmarkStore, slug, task_from_body
from rd_nav.maps_store import MapsStore, valid_name as valid_map_name
from rd_vla import checkpoints
from rd_vla.evals import EvalRegistry

from rd_bringup import config as rd_config

from . import annotations as ann_mod, datasets as ds_mod, exposure as exposure_mod, hand_api, models as models_mod, ops as ops_mod
from .capture import LANDMARK_CAMS, capture_landmark
from .grids import MapView
from .panels import PanelsRelay, panels_info, slash
from .routes import RouteRunner, check_route, load_route, route_map, save_route, valid_route_name
from .supervisor import Supervisor

STATIC = Path(__file__).resolve().parents[1] / "static"
if not STATIC.is_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        STATIC = Path(get_package_share_directory("rd_manager")) / "static"
    except Exception:  # noqa: BLE001
        pass
MIME = {".js": "application/javascript", ".css": "text/css", ".html": "text/html; charset=utf-8", ".png": "image/png",
        ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".stl": "model/stl", ".dae": "model/vnd.collada+xml", ".json": "application/json",
        ".obj": "text/plain", ".mtl": "text/plain", ".glb": "model/gltf-binary", ".wasm": "application/wasm"}
MAPS_UNITS = {"slam": "rd_slam", "localization": "rd_nav", "navigation": "rd_nav"}


def _guard(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith(".") or name == "..":
        raise HTTPException(400, f"bad name {name!r}")
    return name


def _run(fn, *a, **k):
    try:
        return fn(*a, **k)
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError, LookupError, RuntimeError) as e:
        raise HTTPException(400 if not isinstance(e, FileNotFoundError) else 404, str(e))


def build_app(ros: bool = True, sim: bool = False, data_dir: str = "/mnt/data", robot_url: str = "", supervisor_url: str = "",
              maps_dir=None, landmarks_dir=None, routes_dir=None, bridge=None, panels_url: str = "") -> FastAPI:
    app = FastAPI(title="Robot Disco")
    outputs = Path(data_dir) / "outputs"
    datasets_dir = outputs / "datasets"
    maps = MapsStore(maps_dir)
    landmarks = LandmarkStore(landmarks_dir)
    evals = EvalRegistry(outputs / "evals" if not os.environ.get("RD_EVALS_DIR") else None)
    routes_path = Path(routes_dir) if routes_dir else None
    if routes_path is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            share_routes = Path(get_package_share_directory("rd_manager")) / "routes"
            probe = next(iter(share_routes.glob("*.yaml")), None)
            routes_path = probe.resolve().parent if probe and probe.is_symlink() else share_routes
        except Exception:  # noqa: BLE001
            routes_path = Path(__file__).resolve().parents[1] / "routes"
    routes_path.mkdir(parents=True, exist_ok=True)
    sv = Supervisor(supervisor_url) if supervisor_url else None
    mapview = MapView()
    state = {"mapping": False, "slam": {}, "map_source": "sim" if sim else "live", "maps_job": {"running": False, "log": [], "t": 0.0}, "sim": sim}
    active = maps.active_yaml()
    if active:
        mapview.load_file(active)
    if bridge is None and ros:
        from .ros_bridge import RosBridge
        bridge = RosBridge(log=print)
        bridge.start()
    B = bridge
    runner = RouteRunner(follow=lambda poses: B.follow_waypoints(poses) if B else (False, "no ROS"),
                         run_action=lambda act: _route_action(B, act), cancel=lambda: B and B.halt(),
                         set_initial_pose=lambda x, y, yaw: B and B.publish_initialpose(x, y, yaw), log=print)
    app.state.bridge, app.state.maps, app.state.landmarks, app.state.runner, app.state.state = B, maps, landmarks, runner, state

    def resume_mapping():
        """A mapping session outlives a restart of the manager: slam_toolbox runs in its own unit. Whether one is on
        was only in this process's memory, so after a restart the live map stopped being shown and Save had no
        session to save. The unit's state and the request file say it."""
        try:
            if sv and sv.states({MAPS_UNITS["slam"]}).get(MAPS_UNITS["slam"]) == "RUNNING":
                req = maps.slam_request()
                if req:
                    state["mapping"] = True
                    state["slam"] = {"request": req, "started": time.time(), "seed": None, "resumed": True}
                    print(f"mapping session {req.get('name')!r} ({req.get('mode')}) is running: resumed")
        except Exception as e:  # noqa: BLE001  no supervisor: nothing to resume
            print(f"resume_mapping: {type(e).__name__}: {e}")
    threading.Thread(target=resume_mapping, daemon=True, name="resume-mapping").start()

    def ros_up() -> bool:
        return bool(B and B.up)

    def pose():
        return B.pose_fresh() if B else None

    def ros_status() -> str:
        return B.status if B else "no ROS bridge"

    def live_map_tick():
        if state["mapping"] and B and B.live_map.get("t", 0) > state["slam"].get("served_t", 0) and time.time() - state["slam"].get("served_t", 0) > 1.0:
            try:
                mapview.load_grid(B.live_map["msg"], f"slam:{(state['slam'].get('request') or {}).get('name') or 'new'} (live)")
                state["slam"]["served_t"] = time.time()
            except Exception as e:  # noqa: BLE001
                state["slam"]["live_error"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ pages and static
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/live/", response_class=HTMLResponse)
    @app.get("/live", response_class=HTMLResponse)
    def live_page():
        return (STATIC / "pages" / "live.html").read_text()

    @app.get("/live/3d", response_class=HTMLResponse)
    @app.get("/3d", response_class=HTMLResponse)
    def map3d_page():
        return (STATIC / "pages" / "map3d.html").read_text()

    @app.get("/arm/", response_class=HTMLResponse)
    def arm_page():
        return (STATIC / "pages" / "arm.html").read_text()

    @app.get("/teleop/", response_class=HTMLResponse)
    def teleop_page_html():
        return (STATIC / "pages" / "robot" / "index.html").read_text()

    @app.get("/folks/", response_class=HTMLResponse)
    def folks_page():
        return (STATIC / "pages" / "folks.html").read_text()

    for name in ("app.js", "style.css", "favicon.svg"):
        app.add_api_route(f"/{name}", (lambda n: (lambda: FileResponse(STATIC / n, media_type=MIME[Path(n).suffix])))(name), methods=["GET"])
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/mesh/{pkg}/{rel:path}")
    def mesh(pkg: str, rel: str):
        try:
            from ament_index_python.packages import get_package_share_directory
            base = Path(get_package_share_directory(pkg)).resolve()
        except Exception:  # noqa: BLE001
            raise HTTPException(404, "no such package")
        p = Path(os.path.normpath(str(base / rel)))
        if not str(p).startswith(str(base) + os.sep) or not p.is_file():
            raise HTTPException(404, "no such mesh")
        return FileResponse(p, media_type=MIME.get(p.suffix.lower(), "application/octet-stream"), headers={"Cache-Control": "max-age=86400"})

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "ros": ros_up(), "sim": sim, "pose": pose() is not None, "nav2": bool(B and B.nav2_active)}

    # ------------------------------------------------------------------ the map and the live state (the pages)
    @app.get("/map.png")
    def map_png():
        live_map_tick()
        return Response(mapview.png, media_type="image/png")

    @app.get("/api/map")
    def api_map():
        live_map_tick()
        return mapview.meta or {"error": "no map"}

    @app.get("/api/pose")
    def api_pose():
        live_map_tick()      # a page that only polls the state still gets the map slam is drawing (its map_rev moves)
        return {"pose": pose(), "ros": ros_status(), "sim": sim, "halted": False, "ros_up": ros_up(), "map_rev": mapview.rev, "mapping": state["mapping"]}

    @app.get("/api/scans")
    def api_scans(layers: str = ""):
        if not B:
            return {"scans": {}, "topics": {}, "ros": ros_status(), "obstacles": []}
        from .ros_bridge import CLOUD_TOPICS, SCAN_TOPICS
        now = time.time()
        names = [n for n in layers.split(",") if n]
        B.want = {"names": names or list(SCAN_TOPICS) + list(CLOUD_TOPICS), "t": now}
        B.touch(*B.want["names"])
        return {"scans": {k: v for k, v in B.scans.items() if now - v["t"] < 3.0 and (not names or k in names)},
                "topics": {**SCAN_TOPICS, **CLOUD_TOPICS}, "ros": ros_status(), "obstacles": []}

    @app.get("/api/costmap")
    def api_costmap(name: str = "global", color: Optional[str] = None):
        if B:
            B.touch("costmap:" + name)         # subscribed on the first ask; the latched grid arrives for the next
        if not B or name not in B.costmaps:
            return {"error": "no costmap received yet (is navigation running?)"}
        from .ros_bridge import COSTMAP_TOPICS
        return mapview.costmap_png(name, B.costmaps[name], B.frame_tf, COSTMAP_TOPICS.get(name, ""), color)[1]

    @app.get("/costmap/{name}.png")
    def costmap_png(name: str, color: Optional[str] = None):
        if B:
            B.touch("costmap:" + name)
        if not B or name not in B.costmaps:
            raise HTTPException(404, "no costmap")
        from .ros_bridge import COSTMAP_TOPICS
        return Response(mapview.costmap_png(name, B.costmaps[name], B.frame_tf, COSTMAP_TOPICS.get(name, ""), color)[0], media_type="image/png")

    @app.get("/api/urdf")
    def api_urdf():
        if not B or not B.urdf:
            return JSONResponse({"error": "no /robot_description received yet (is the base launch running?)"}, 503)
        return Response(B.urdf, media_type="application/xml", headers={"Cache-Control": "max-age=60"})

    @app.get("/api/tf")
    def api_tf():
        if not B:
            return {"frames": {}, "joints": {}, "t": time.time(), "ros": ros_status(), "ros_up": False}
        B.touch("joints")
        live_map_tick()
        return {"frames": B.frames, "joints": B.joints, "t": time.time(), "ros": ros_status(), "published": B.published, "sim": sim,
                "halted": False, "tf_debug": B.tf_debug, "map_rev": mapview.rev, "mapping": state["mapping"]}

    @app.get("/api/nav")
    def api_nav(particles: str = ""):
        if not B:
            return {}
        nav = dict(B.nav)
        nav["nav2_active"] = B.nav2_active
        nav["goal_active"] = B._nav_handle is not None
        nav["goal_state"] = B.nav_goal
        nav["route"] = runner.state
        nav["policy"] = B.policy_feedback
        if particles:
            B.want3d["particles"] = time.time()
            B.touch("particles")
            nav["particles"] = B.particles
        return nav

    @app.get("/api/cloud.bin")
    def api_cloud(name: str = "velodyne", max: int = 60000):
        if B:
            B.touch(name)
        b = B.cloud_bin(name, int(max)) if B else None
        if not b:
            raise HTTPException(404, f"no {name} data yet")
        return Response(b, media_type="application/octet-stream")

    @app.get("/api/apriltags")
    def api_apriltags():
        from rd_nav.tag_corrector import load_tag_poses, mat_to_xy_yaw
        try:
            from ament_index_python.packages import get_package_share_directory
            path = Path(get_package_share_directory("rd_nav")) / "config" / "tag_poses.yaml"
        except Exception:  # noqa: BLE001
            return {"tags": {}}
        tags = {}
        for tid, M in load_tag_poses(str(path)).items():
            x, y, yaw = mat_to_xy_yaw(M)
            tags[str(tid)] = {"x": round(x, 3), "y": round(y, 3), "z": round(float(M[2, 3]), 3), "yaw": round(yaw, 3)}
        return {"tags": tags, "file": str(path)}

    @app.get("/api/obstacles")
    def api_obstacles():
        return {"obstacles": [], "sim": sim}

    # ------------------------------------------------------------------ cameras
    @app.get("/api/cameras")
    def api_cameras():
        from .ros_bridge import CAMERA_LABELS, CAMERA_TOPICS
        out = []
        for n, t in CAMERA_TOPICS.items():
            out.append({"name": n, "label": CAMERA_LABELS[n], "topic": t, "url": f"/cam/{n}.mjpg", "snapshot": f"/cam/{n}.jpg",
                        "alive": B.cam_alive(n) if B else False})
        return {"cameras": out}

    @app.get("/cam/{name}.jpg")
    def cam_jpg(name: str):
        if not B:
            raise HTTPException(404, "no ROS")
        B.want_cam(name)
        e = B.jpeg.get(name)
        if not e or not e["data"]:
            raise HTTPException(404, "no frame yet")
        return Response(e["data"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/cam/{name}.mjpg")
    def cam_mjpg(name: str, fps: float = 10.0):
        if not B:
            raise HTTPException(404, "no ROS")

        def gen():
            last = 0.0
            deadline = time.time() + 3600
            while time.time() < deadline:
                B.want_cam(name)
                e = B.jpeg.get(name)
                if e and e["data"] and e["t"] != last:
                    last = e["t"]
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(e["data"])).encode() + b"\r\n\r\n" + e["data"] + b"\r\n"
                time.sleep(1.0 / max(1.0, fps))
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/face.jpg")
    def face_jpg():
        return cam_jpg("face")

    # ------------------------------------------------------------------ doing things
    @app.post("/api/halt")
    def api_halt(body: dict = Body(default={})):
        did = B.halt() if B else ["no ROS bridge"]
        if runner.busy:
            did.append(runner.stop())
        return {"ok": True, "did": did, "sim": sim}

    @app.post("/api/drive")
    def api_drive(body: dict = Body(...)):
        if not ros_up():
            raise HTTPException(503, "the ROS bridge is not up")
        try:
            vx, wz = float(body.get("vx", 0.0)), float(body.get("wz", 0.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "vx and wz must be numbers")
        return _run(B.set_drive, vx, wz, bool(body.get("turbo")), bool(body.get("stop")))

    @app.post("/api/publish")
    def api_publish(body: dict = Body(...)):
        kind = body.get("type")
        if kind not in ("goal", "initialpose"):
            raise HTTPException(400, "type must be goal|initialpose")
        if not ros_up():
            raise HTTPException(503, "the ROS bridge is not up")
        x, y, yaw = float(body["x"]), float(body["y"]), float(body.get("yaw", 0.0))
        return {"ok": True, "queued": B.publish_goal(x, y, yaw) if kind == "goal" else B.publish_initialpose(x, y, yaw)}

    @app.post("/api/say")
    def api_say(body: dict = Body(...)):
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "text")
        return B.say(text, wait=bool(body.get("wait"))) if B else {"ok": False, "message": "no ROS"}

    @app.post("/api/voice/text")
    def api_voice_text(body: dict = Body(...)):
        """The Live tab's text box: typed text goes to the voice assistant as if Whisper had heard it."""
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "text")
        if not ros_up():
            raise HTTPException(503, "the ROS bridge is not up")
        return B.voice_text(text)

    @app.get("/api/voice")
    def api_voice():
        """What the voice assistant last heard / took as the command / answered (each with a server timestamp
        <key>_t), its state, and listeners = whether it is subscribed to the text box's topic."""
        return {**B.voice, "listeners": B.voice_text_pub.get_subscription_count()} if B else {}

    # ------------------------------------------------------------------ the arm, the hand, policies
    @app.get("/api/arm")
    def api_arm():
        out = {"ros": ros_up(), "stream": B.stream_state if B else "", "hand": B.hand_state if B else [], "joints": {}, "policy": B.policy_feedback if B else {},
               "status": B.arm_status if B else {}, "speaking": B.speaking if B else False}
        if B:
            B.touch("joints")
            for _ in range(8):                 # the first ask subscribes: give the 100 Hz states a moment
                if any(k.startswith("ur_") for k in B.joints):
                    break
                time.sleep(0.05)
            out["joints"] = {k: v for k, v in B.joints.items() if k.startswith("ur_")}
        return out

    @app.post("/api/arm/power")
    def api_arm_power(body: dict = Body(...)):
        if not B:
            raise HTTPException(503, "no ROS")
        return B.arm_power(int(body.get("command", 0)), float(body.get("timeout_s", 60.0)))

    @app.post("/api/arm/home")
    def api_arm_home():
        return B.call_trigger(B.home_client, timeout=3.0) if B else {"success": False, "message": "no ROS"}

    @app.post("/api/arm/stream")
    def api_arm_stream(body: dict = Body(...)):
        return B.set_stream(bool(body.get("on"))) if B else {"success": False, "message": "no ROS"}

    @app.get("/api/teleop")
    def api_teleop():
        """The SpaceMouse teleop and its recorder: device, enabled, recording, frames, dataset, task, the last message."""
        return B.teleop_state() if B else {"running": False, "error": "no ROS"}

    @app.get("/api/teleop/config")
    def api_teleop_config():
        """What v1's robot page asks first: the mode, its API level, where the camera streams are."""
        return {"mode": "teleop", "api": 2, "title": rd_config.get("robot.host", "robot"), "cams": {n: f"/cam/{n}.mjpg" for n in ("scene", "side", "wrist")}}

    teleop_page = {"exposure": {}, "exposure_t": 0.0, "status_t": 0.0, "status_busy": False}

    def _teleop_last(t: dict) -> dict:
        """The last saved episode with its rating and notes from the dataset's annotations sidecar."""
        last = dict(t.get("last") or {})
        if last.get("ep") is not None and last.get("dataset"):
            a = ann_mod.load(str(datasets_dir / last["dataset"])).get(str(last["ep"]), {})
            last.update({k: a[k] for k in ("rating", "notes") if k in a})
        return last

    def teleop_page_state() -> dict:
        """The state v1's robot page renders, composed from the teleop node, the arm, the hand and the cameras."""
        t = B.teleop_state() if B else {"running": False}
        now = time.time()
        if B and now - teleop_page["status_t"] > 3.0 and not teleop_page["status_busy"]:     # the dashboard, off this thread
            teleop_page["status_t"], teleop_page["status_busy"] = now, True

            def poll():
                try:
                    B.arm_power(0, 5.0)
                except Exception:  # noqa: BLE001
                    pass
                teleop_page["status_busy"] = False
            threading.Thread(target=poll, daemon=True).start()
        if now - teleop_page["exposure_t"] > 2.0:
            teleop_page["exposure_t"] = now
            try:
                teleop_page["exposure"] = exposure_mod.state(rd_config.get("sensors.usb_cameras", {}) or {})
            except Exception:  # noqa: BLE001  no v4l2-ctl, no cameras: the card stays empty
                teleop_page["exposure"] = {}
        arm = (B.arm_status if B else {}) or {}
        fresh = bool(arm) and now - arm.get("t", 0) < 15.0
        safety, mode = str(arm.get("safety_status", "")), str(arm.get("robot_mode", ""))
        fault = None if not fresh or safety in ("", "NORMAL", "REDUCED") else f"{safety} ({mode}): {arm.get('message', '')}".strip()
        pose = t.get("pose") or []
        ex = teleop_page["exposure"].get("cameras", {})
        cams = rd_config.get("sensors.usb_cameras", {}) or {}
        hand = (B.hand_state if B else []) or []
        stream = (B.stream_state if B else "") or ""
        return {
            "mode": "teleop", "running": bool(t.get("running")), "device": bool(t.get("device")), "enabled": bool(t.get("enabled")),
            "input": "freedrive" if t.get("freedrive") else "spacemouse" if t.get("device") else "no spacemouse", "task": t.get("task", ""),
            "recording": bool(t.get("recording")), "frames": t.get("frames", 0), "episodes": t.get("episodes", 0),
            "grasps": t.get("grasps", 1), "regrasp_open": bool(t.get("regrasp_open")), "rec_s": t.get("rec_s", 0.0),
            "save_in_flight": bool(t.get("saving")),
            "dataset": {"name": t.get("dataset", ""), "root": str(datasets_dir / t.get("dataset", "")), "storage": t.get("backend") or ""},
            "ur": {"connected": fresh and mode == "RUNNING", "ready": fresh and bool(arm.get("program_running")), "pose": pose,
                   "reach": round(math.hypot(pose[0], pose[1]), 3) if len(pose) >= 2 else None, "base_deg": None,
                   "fault": fault, "freedrive": bool(t.get("freedrive")), "homing": False, "mode": mode or None},
            "hand": {"ok": bool(hand) or None, "grasp": t.get("grasp")},
            "tick": {"hz": t.get("tick_hz", 0.0), "ms": 0.0},
            "servo": {"short": "stream " + (stream.split() or ["off"])[0], "on": stream.startswith("on"), "slow": False, "detail": stream},
            "exposure": {"values": {n: c["exposure"] for n, c in ex.items() if c.get("exposure") is not None},
                         "locked": bool(teleop_page["exposure"].get("locked")), "phase": False,
                         "medians": {n: m[0] for n, m in (B.cam_median if B else {}).items() if now - m[1] < 5.0}},
            "crop": {n: v for n, v in rd_config.camera_squares().items() if v is not None},
            "crop_locked": bool(t.get("recording") or t.get("saving") or t.get("episodes", 0) > 0),
            "deck": bool(t.get("deck")), "last": _teleop_last(t), "status": t.get("message", ""),
        }

    @app.post("/api/teleop/cmd")
    def api_teleop_cmd(body: dict = Body(...)):
        """A command from the teleop page: most go to the teleop node's own loop (/teleop/cmd); the cameras' exposure,
        the last episode's rating and notes, and the arm's recovery are done here."""
        if not B:
            raise HTTPException(503, "no ROS")
        name, a = str(body.get("name", "")), body.get("args") or {}
        if name == "expo":
            try:
                teleop_page["exposure"] = exposure_mod.adjust(rd_config.get("sensors.usb_cameras", {}) or {}, str(a.get("action", "")), a.get("cam"))
                teleop_page["exposure_t"] = time.time()
                B.teleop_note(time.strftime("%H:%M:%S ") + f"exposure {a.get('action')}{' ' + a['cam'] if a.get('cam') else ''}: "
                              + ", ".join(f"{n} {c['exposure']}" for n, c in teleop_page["exposure"].get("cameras", {}).items()))
            except Exception as e:  # noqa: BLE001
                B.teleop_note(time.strftime("%H:%M:%S ") + f"exposure failed: {e}")
        elif name in ("rating", "notes"):
            last = (B.teleop_state().get("last") or {})
            if last.get("ep") is None or not last.get("dataset"):
                B.teleop_note(time.strftime("%H:%M:%S ") + "no saved episode to annotate yet")
            else:
                fields = {"rating": int(a.get("stars", 0))} if name == "rating" else {"notes": str(a.get("text", "")).strip()}
                ann_mod.set_episode(str(datasets_dir / last["dataset"]), int(last["ep"]), fields)
                B.teleop_note(time.strftime("%H:%M:%S ") + f"episode {last['ep'] + 1}: {name} saved")
        elif name == "crop":
            cam, at = str(a.get("cam", "")), float(a.get("at", -1))
            t = B.teleop_state()
            squares = rd_config.camera_squares()
            if squares.get(cam) is None or not 0.0 <= at <= 1.0:
                note = f"crop: {cam or '?'} is not a cropped camera"
            elif t.get("recording") or t.get("saving") or t.get("episodes", 0) > 0:
                note = f"crop: fixed, the dataset {t.get('dataset') or ''} already has episodes cut at {squares[cam]:.2f} (choose a new dataset name first)"
            else:
                r = B.set_camera_square(cam, at)
                if r["success"]:
                    squares[cam] = at
                    f = rd_config.camera_squares_file()
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps({k: v for k, v in squares.items() if v is not None}, indent=1))
                    note = f"crop {cam}: {at:.2f} (kept in {f.name}; datasets and policies must agree on it)"
                else:
                    note = f"crop {cam}: {r['message']}"
            B.teleop_note(time.strftime("%H:%M:%S ") + note)
        elif name == "input_mode":
            B.teleop_note(time.strftime("%H:%M:%S ") + "the SpaceMouse (or freedrive) is the only input")
        else:
            B.teleop_cmd(name, a)
        return {"ok": True}

    @app.get("/api/teleop/events")
    async def api_teleop_events(request: Request):
        """Server-Sent Events for the teleop page: `state` five times a second, `log` lines as they come."""
        import asyncio

        async def gen():
            seq = B._teleop_seq - min(len(B.teleop_log), 50) if B else 0        # the page starts with the recent lines
            while not await request.is_disconnected():
                if B:
                    for n, line in list(B.teleop_log):
                        if n > seq:
                            seq = n
                            yield f"event: log\ndata: {json.dumps(line)}\n\n"
                st = await asyncio.get_running_loop().run_in_executor(None, teleop_page_state)
                yield f"event: state\ndata: {json.dumps(st)}\n\n"
                await asyncio.sleep(0.2)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/cameras/exposure")
    def api_exposure():
        return exposure_mod.state(rd_config.get("sensors.usb_cameras", {}) or {})

    @app.post("/api/cameras/exposure")
    def api_exposure_set(body: dict = Body(...)):
        """{locked: true} freezes each policy camera at the exposure and white balance it has now; false = auto again."""
        return exposure_mod.set_locked(rd_config.get("sensors.usb_cameras", {}) or {}, bool(body.get("locked")))

    @app.post("/api/arm/wave")
    def api_arm_wave(body: dict = Body(default={})):
        if not B:
            raise HTTPException(503, "no ROS")
        threading.Thread(target=B.wave, args=(str(body.get("motion", "wave")), str(body.get("return_to", "home")), float(body.get("speed_scale", 1.0))), daemon=True).start()
        return {"ok": True, "message": "waving"}

    @app.post("/api/policy/run")
    def api_policy_run(body: dict = Body(...)):
        if not B:
            raise HTTPException(503, "no ROS")
        if B._task_handle is not None:
            raise HTTPException(409, "a policy run is active")
        kw = {k: body[k] for k in ("task", "checkpoint", "backend", "max_seconds", "record", "landmark", "done_when", "still_seconds", "home_first", "max_steps") if k in body}
        if not kw.get("task"):
            raise HTTPException(400, "task")
        result: dict = {"running": True}
        B.policy_result = result

        def run():
            result.update(B.run_policy(**kw))
            result["running"] = False
        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, "started": kw}

    @app.get("/api/policy")
    def api_policy():
        return {"feedback": B.policy_feedback if B else {}, "result": getattr(B, "policy_result", {}) if B else {}, "active": bool(B and B._task_handle is not None)}

    @app.post("/api/policy/stop")
    def api_policy_stop():
        h = B._task_handle if B else None
        if h is None:
            return {"ok": False, "message": "no run active"}
        h.cancel_goal_async()
        return {"ok": True}

    hand_state = hand_api.HandState()
    hand_host = "127.0.0.1" if sim else str(rd_config.get("hand.ip", "127.0.0.1"))
    hand_port = int(rd_config.get("hand.ports.udp", 8766))

    def hand_publish(pub, msg, line: str, **fields) -> None:
        if not B:
            raise HTTPException(503, "no ROS")
        if pub.get_subscription_count() == 0:
            hand_state.sent(line, error="the hand node is not running")
            raise HTTPException(502, "the hand node is not running (unit rd_arm), nothing was sent")
        pub.publish(msg)
        hand_state.sent(line, **fields)

    @app.get("/api/hand")
    def api_hand(host: Optional[str] = None):
        """The hand's liveness (UDP probe), the last command this server sent it, and the gesture table."""
        h = (host or hand_host).strip()
        p = hand_api.probe(h, hand_port)
        if p["alive"]:
            hand_state.alive_t = time.time()
        seen = None if hand_state.alive_t is None else round(time.time() - hand_state.alive_t, 1)
        node = B.hand_state if B else []
        return {"host": h, "udp_port": hand_port, "alive": p["alive"], "uptime_ms": p["uptime_ms"], "error": p["error"],
                "last_alive_s": seen, "console_url": f"http://{h}/webserial", "gestures": hand_api.GESTURES,
                "speed_range": [hand_api.SPEED_MIN, hand_api.SPEED_MAX], "state": hand_state.snapshot(),
                "reported": node[4:8] if len(node) >= 8 else None}

    @app.post("/api/hand/curls")
    def api_hand_curls(body: dict = Body(...)):
        """Per-finger curl and abduction -> one J: line (v1's mapping), sent by the hand node."""
        from std_msgs.msg import Float32MultiArray
        try:
            cmd = hand_api.build_command(body.get("curls", []), body.get("abduct"), body.get("speed") or hand_state.snapshot()["speed"])
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))
        hand_publish(B.hand_offsets_pub if B else None, Float32MultiArray(data=cmd["offsets"] + [float(cmd["speed"])]), cmd["line"],
                     curls=cmd["curls"], abduct=cmd["abduct"], speed=cmd["speed"])
        return cmd

    def hand_line(line: str) -> None:
        from std_msgs.msg import String
        hand_publish(B.hand_raw_pub if B else None, String(data=line), line)

    @app.post("/api/hand/gesture")
    def api_hand_gesture(body: dict = Body(...)):
        """One of the firmware's gestures (a single character; it moves at the hand's own speed)."""
        c = str(body.get("line", "")).strip()
        if c not in hand_api.GESTURES:
            raise HTTPException(400, f"gesture must be one of: {' '.join(hand_api.GESTURES)}")
        hand_line(c)
        return {"line": c, "gesture": hand_api.GESTURES[c]}

    @app.post("/api/hand/raw")
    def api_hand_raw(body: dict = Body(...)):
        """Any one line as typed (J:..., a gesture character). Q is answered by the probe."""
        line = str(body.get("line", "")).strip()
        if not line or len(line) > 150 or not line.isascii() or not line.isprintable():
            raise HTTPException(400, "one printable ASCII line of at most 150 characters")
        if line == "Q":
            p = hand_api.probe((body.get("host") or hand_host).strip(), hand_port)
            return {"line": line, "reply": f"K:{p['uptime_ms']}" if p["alive"] else None, "error": p["error"]}
        hand_line(line)
        return {"line": line, "reply": None, "error": None}

    @app.post("/api/hand/{action}")
    def api_hand_action(action: str):
        if not B:
            raise HTTPException(503, "no ROS")
        if action == "open":
            return B.call_trigger(B.hand_open)
        if action == "close":
            return B.call_trigger(B.hand_close)
        raise HTTPException(404, "open|close")

    # ------------------------------------------------------------------ routes (the editor pages)
    def route_files() -> list:
        return sorted(p.stem for p in routes_path.glob("*.yaml") if not p.name.endswith(".raw.yaml"))

    @app.get("/api/routes")
    def api_routes():
        return {"routes": route_files(), "maps": {n: route_map(routes_path / f"{n}.yaml") for n in route_files()}, "current": maps.active(),
                "runner": runner.state}

    @app.get("/api/route")
    def api_route(name: str):
        if not valid_route_name(name) or not (routes_path / f"{name}.yaml").exists():
            raise HTTPException(404, f"no route {name!r}")
        wps, init = _run(load_route, routes_path / f"{name}.yaml")
        return {"name": name, "waypoints": [w.as_dict() for w in wps], "initial_pose": init.as_dict() if init else None,
                "map": route_map(routes_path / f"{name}.yaml"), "check": check_route(routes_path / f"{name}.yaml")}

    @app.post("/api/route")
    def api_route_save(name: str = "", body: dict = Body(...)):
        name = name or str(body.get("name", ""))
        if not valid_route_name(name):
            raise HTTPException(400, "bad route name")
        save_route(routes_path / f"{name}.yaml", list(body.get("waypoints") or []), body.get("initial_pose"), maps.active() or "")
        return {"ok": True, "check": check_route(routes_path / f"{name}.yaml")}

    @app.delete("/api/route")
    def api_route_delete(name: str):
        p = routes_path / f"{_guard(name)}.yaml"
        if not p.exists():
            raise HTTPException(404, "no such route")
        p.unlink()
        return {"ok": True}

    @app.post("/api/route/run")
    def api_route_run(body: dict = Body(...)):
        name = _guard(str(body.get("name", "")))
        p = routes_path / f"{name}.yaml"
        if not p.exists():
            raise HTTPException(404, "no such route")
        if not ros_up():
            raise HTTPException(503, "the ROS bridge is not up")
        return _run(runner.start, p, str(body.get("from", "")))

    @app.post("/api/route/stop")
    def api_route_stop():
        return {"ok": True, "message": runner.stop()}

    @app.post("/api/tool")
    def api_tool(tool: str = "", name: str = "", body: dict = Body(default={})):
        return {"ok": False, "output": f"the {tool} tool is not ported yet (v1 record_path.py)", "route": api_route(name)}

    # ------------------------------------------------------------------ maps
    @app.get("/api/maps")
    def api_maps():
        d = maps.list()
        units = sv.states(set(MAPS_UNITS.values())) if sv else {}
        return {**d, "current": maps.active(), "mapping": state["mapping"], "slam": {k: v for k, v in state["slam"].items() if k != "served_t"},
                "ros_up": ros_up(), "map_rev": mapview.rev, "pose": pose(), "units": units, "job": state["maps_job"], "source": state["map_source"]}

    @app.get("/api/maps/{name}.png")
    def api_map_image(name: str):
        if not valid_map_name(name) or not maps.yaml_path(name).exists():
            raise HTTPException(404, f"no map {name!r}")
        return Response(maps.png(name), media_type="image/png")

    @app.post("/api/maps/slam/start")
    def api_slam_start(body: dict = Body(...)):
        mode, name = str(body.get("mode") or "new"), str(body.get("name") or "").strip()
        start_pose, seed = None, None
        if mode == "continue":
            p = pose()
            if body.get("seed_pose", True) and p and maps.active() == name:
                start_pose = [p["x"], p["y"], p["yaw"]]
                seed = {"seeded": True, "pose": start_pose}
            else:
                seed = {"seeded": False, "why": "no live pose" if not p else "not the active map" if maps.active() != name else "seed_pose off"}
        req = _run(maps.set_slam_request, mode, name, start_pose)
        state["mapping"] = True
        state["slam"] = {"request": req, "started": time.time(), "seed": seed}
        units = []
        if sv:
            try:
                for u in (MAPS_UNITS["localization"],):
                    sv.action(u, "stop")
                    units.append(f"stop {u}")
                sv.action(MAPS_UNITS["slam"], "start")
                units.append(f"start {MAPS_UNITS['slam']}")
            except Exception as e:  # noqa: BLE001
                units.append(f"supervisor: {e}")
        return {"ok": True, "request": req, "seed": seed, "units": units}

    @app.post("/api/maps/slam/save")
    def api_slam_save(body: dict = Body(...)):
        name = str(body.get("name") or (state["slam"].get("request") or {}).get("name") or "").strip()
        if not valid_map_name(name):
            raise HTTPException(400, f"bad map name {name!r}")
        if not B:
            raise HTTPException(503, "no ROS")
        res = B.slam_save(str(maps.dir / name))
        if res.get("error"):
            raise HTTPException(502, res["error"])
        state["slam"]["last_save"] = {**res, "name": name, "t": time.time()}
        return {"ok": True, "saved": res, "map": maps.info(name, maps.active()) if maps.yaml_path(name).exists() else None}

    @app.post("/api/maps/slam/stop")
    def api_slam_stop():
        state["mapping"] = False
        maps.clear_slam_request()
        state["slam"] = {k: v for k, v in state["slam"].items() if k == "last_save"}
        if sv:
            try:
                sv.action(MAPS_UNITS["slam"], "stop")
                sv.action(MAPS_UNITS["localization"], "start")
            except Exception as e:  # noqa: BLE001
                return {"ok": True, "active": maps.active(), "units": f"supervisor: {e}"}
        a = maps.active()
        if a:
            mapview.load_file(maps.yaml_path(a))
        return {"ok": True, "active": a}

    @app.post("/api/maps/{name}/activate")
    def api_map_activate(name: str, body: dict = Body(default={})):
        if not valid_map_name(name) or not maps.yaml_path(name).exists():
            raise HTTPException(404, f"no map {name!r}")
        maps.set_active(name)
        if not state["mapping"]:
            mapview.load_file(maps.yaml_path(name))
        job = state["maps_job"]
        if sv and body.get("restart", True) and not job["running"]:
            def run():
                job.update(running=True, log=[], t=time.time())
                try:
                    sv.action(MAPS_UNITS["localization"], "restart")
                    job["log"].append("restarted rd_nav on " + name)
                except Exception as e:  # noqa: BLE001
                    job["log"].append(f"failed: {e}")
                finally:
                    job["running"] = False
            threading.Thread(target=run, daemon=True).start()
        return {"ok": True, "active": name, "map": maps.info(name, name), "note": "localization + navigation restart on it" if sv else "restart rd_nav to run on it"}

    @app.post("/api/maps/{name}/draw")
    def api_map_draw(name: str, body: dict = Body(...)):
        if not valid_map_name(name) or not maps.yaml_path(name).exists():
            raise HTTPException(404, f"no map {name!r}")
        png = str(body.get("png") or "")
        if "," in png and png.startswith("data:"):
            png = png.split(",", 1)[1]
        info = _run(maps.save_drawn, name, base64.b64decode(png), (body.get("save_as") or "").strip() or None)
        if info["name"] == maps.active() and not state["mapping"]:
            mapview.load_file(maps.yaml_path(info["name"]))
        return {"ok": True, "map": info}

    @app.post("/api/maps/{name}/keepout")
    def api_map_keepout(name: str):
        return {"ok": True, "cells": _run(build_mask, maps.dir)}

    @app.delete("/api/maps/{name}")
    def api_map_delete(name: str):
        return {"ok": True, "deleted": _run(maps.delete, _guard(name))}

    @app.get("/api/map_source")
    def api_map_source(request: Request):
        return {"source": state["map_source"], "options": ["live", "sim"], "url": "/live/"}

    @app.post("/api/map_source")
    def api_map_source_set(body: dict = Body(...)):
        state["map_source"] = str(body.get("source", "live"))
        return {"ok": True, "source": state["map_source"]}

    # ------------------------------------------------------------------ landmarks
    def lm_payload(items):
        return {"landmarks": items, "map": maps.active(), "pose": pose(), "ros": ros_status(), "ros_up": ros_up(), "sim": sim, "source": state["map_source"],
                "voice": bool(B and B.voice_task_pub.get_subscription_count()) if B else False,
                "cameras": {n: bool(B and B.cam_alive(n)) for n in LANDMARK_CAMS}}     # there to be photographed (a capture asks for frames itself)

    @app.get("/api/landmarks")
    def api_landmarks():
        return lm_payload(landmarks.all())

    @app.get("/api/landmarks/img/{name}")
    def api_landmark_image(name: str):
        p = (landmarks.img_dir / _guard(name)).resolve()
        if p.parent != landmarks.img_dir.resolve() or p.suffix != ".jpg" or not p.is_file():
            raise HTTPException(404, "no such image")
        return FileResponse(p, media_type="image/jpeg")

    @app.get("/landmarks/img/{name}")
    def landmark_image_v1(name: str):
        return api_landmark_image(name)

    @app.get("/api/landmarks/map")
    def api_landmark_map():
        return {"map": maps.active(), "meta": mapview.meta}

    @app.get("/api/landmarks/map.png")
    def api_landmark_map_png():
        return Response(mapview.png, media_type="image/png")

    @app.post("/api/landmarks")
    def api_landmark_add(body: dict = Body(...)):
        name = str(body.get("name") or "").strip() or f"landmark {len(landmarks.all()) + 1}"
        lid = slug(name)
        if "x" in body and "y" in body:
            cap = {"x": float(body["x"]), "y": float(body["y"]), "yaw": float(body.get("yaw", 0.0))}
            if B:
                B.fresh_frames(LANDMARK_CAMS)
                cap.update(capture_landmark(lid, landmarks.img_dir, None, B.jpeg, want_pose=False))
        else:
            if not B:
                raise HTTPException(503, "no ROS bridge")
            try:
                B.fresh_frames(LANDMARK_CAMS)
                cap = capture_landmark(lid, landmarks.img_dir, pose(), B.jpeg)
            except LookupError as e:
                raise HTTPException(409, f"{e}: {ros_status()}")
        item = {"id": lid, "name": name, "map": maps.active(), "x": cap["x"], "y": cap["y"], "yaw": cap["yaw"],
                "images": cap.get("images") or {}, "cameras": cap.get("cameras") or [], "photo": (cap.get("images") or {}).get("stitched"),
                "created": time.strftime("%Y-%m-%d %H:%M"), "updated": time.strftime("%Y-%m-%d %H:%M"), "tasks": []}
        return {"ok": True, "landmark": landmarks.upsert(item), "capture": cap}

    @app.post("/api/landmarks/{lid}")
    def api_landmark_edit(lid: str, body: dict = Body(...)):
        lm = landmarks.get(lid)
        if lm is None:
            raise HTTPException(404, "no such landmark")
        for k in ("name", "x", "y", "yaw", "map"):
            if k in body:
                lm[k] = body[k]
        lm["updated"] = time.strftime("%Y-%m-%d %H:%M")
        return {"ok": True, "landmark": landmarks.upsert(lm)}

    @app.post("/api/landmarks/{lid}/goto")
    def api_landmark_goto(lid: str):
        lm = landmarks.get(lid)
        if lm is None:
            raise HTTPException(404, "no such landmark")
        if not ros_up():
            raise HTTPException(503, "the ROS bridge is not up")
        threading.Thread(target=B.go_to_landmark, args=(lid,), daemon=True).start()
        return {"ok": True, "landmark": lid, "message": f"heading to {lm.get('name', lid)}"}

    @app.post("/api/landmarks/{lid}/recapture")
    def api_landmark_recapture(lid: str, body: dict = Body(default={})):
        lm = landmarks.get(lid)
        if lm is None:
            raise HTTPException(404, "no such landmark")
        if not B:
            raise HTTPException(503, "no ROS bridge")
        want_pose = bool(body.get("pose", True))
        try:
            B.fresh_frames(LANDMARK_CAMS)
            cap = capture_landmark(lid, landmarks.img_dir, pose(), B.jpeg, want_pose=want_pose)
        except LookupError as e:
            raise HTTPException(409, str(e))
        if want_pose:
            lm.update(x=cap["x"], y=cap["y"], yaw=cap["yaw"], map=maps.active())
        if cap.get("images"):
            lm["images"] = cap["images"]
            lm["cameras"] = cap.get("cameras") or []
            lm["photo"] = cap["images"].get("stitched")
        lm["updated"] = time.strftime("%Y-%m-%d %H:%M")
        return {"ok": True, "landmark": landmarks.upsert(lm), "capture": cap}

    @app.delete("/api/landmarks/{lid}")
    def api_landmark_delete(lid: str):
        if not landmarks.delete(lid):
            raise HTTPException(404, "no such landmark")
        return {"ok": True}

    @app.post("/api/landmarks/{lid}/tasks")
    def api_task_add(lid: str, body: dict = Body(...)):
        return {"ok": True, "task": _run(landmarks.set_task, lid, body)}

    @app.post("/api/landmarks/{lid}/tasks/{tid}")
    def api_task_edit(lid: str, tid: str, body: dict = Body(...)):
        return {"ok": True, "task": _run(landmarks.set_task, lid, body, tid)}

    @app.post("/api/landmarks/{lid}/tasks/{tid}/run")
    def api_task_run(lid: str, tid: str):
        from std_msgs.msg import String
        if not B:
            raise HTTPException(503, "no ROS")
        if B.voice_task_pub.get_subscription_count():
            B.voice_task_pub.publish(String(data=f"{lid}/{tid}"))
            return {"ok": True, "via": "voice", "message": "the voice assistant drives there and runs it"}
        lm = landmarks.get(lid)
        task = next((t for t in (lm or {}).get("tasks", []) if t.get("id") == tid), None)
        if lm is None or task is None:
            raise HTTPException(404, "no such task")

        def run():
            r = B.go_to_landmark(lid)
            if r["ok"]:
                B.run_policy(task=str(task.get("task") or task["name"]), checkpoint=str(task.get("model", "")), max_seconds=float(task.get("seconds") or 120),
                             landmark=lid, done_when=str(task.get("done_when") or ""), still_seconds=float(task.get("still") or 0), record=bool(task.get("record", False)))
        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, "via": "manager", "message": f"driving to {lm.get('name', lid)}, then {task['name']}"}

    @app.delete("/api/landmarks/{lid}/tasks/{tid}")
    def api_task_delete(lid: str, tid: str):
        if not landmarks.delete_task(lid, tid):
            raise HTTPException(404, "no such task")
        return {"ok": True}

    # ------------------------------------------------------------------ folks + volume
    @app.get("/api/folks")
    def api_folks(request: Request):
        people = (B.people or {}) if B else {}
        reachable = bool(B and B.people)
        return {"reachable": reachable, "url": "/folks/", "info": {"count": len(people.get("people", [])), "live": people.get("live", {})}}

    @app.get("/api/people")
    def api_people():
        return (B.people if B else None) or {"people": [], "live": {}}

    @app.get("/thumb/{pid}.jpg")
    def api_thumb(pid: str):
        people = (B.people or {}).get("people", []) if B else []
        p = next((x for x in people if x["id"] == _guard(pid)), None)
        if not p or not p.get("thumb") or not Path(p["thumb"]).is_file():
            raise HTTPException(404, "no thumb")
        return FileResponse(p["thumb"], media_type="image/jpeg")

    @app.post("/api/people/{pid}/{action}")
    def api_people_action(pid: str, action: str, body: dict = Body(default={})):
        from std_msgs.msg import String
        if not B:
            raise HTTPException(503, "no ROS")
        if action == "name":
            B.face_pubs["set_person_name"].publish(String(data=f"{pid}={body.get('name', '')}"))
        elif action == "forget":
            B.face_pubs["forget_person"].publish(String(data=pid))
        elif action == "reset_greeting":
            B.face_pubs["reset_greeting"].publish(String(data=pid))
        elif action == "wink":
            return B.call_trigger(B.wink_client)
        else:
            raise HTTPException(404, "name|forget|reset_greeting|wink")
        return {"ok": True}

    @app.get("/api/battery")
    def api_battery():
        """The pack as rd_base's battery_monitor last reported it; `age` in seconds, null fields where it has no value."""
        b = getattr(B, "battery", None) if B else None
        if not b:
            return {"present": False, "error": "no ROS" if not B else "nothing on /battery yet"}
        return {**{k: v for k, v in b.items() if k != "t"}, "age": round(time.time() - b["t"], 1)}

    @app.get("/api/volume")
    def api_volume():
        if not B:
            return {"sink": None, "volume": None, "muted": False, "error": "no ROS"}
        v = getattr(B, "volume", None)
        return {"sink": "say_server", "volume": int(round(v * 100)) if v is not None else None, "muted": getattr(B, "muted", False)}

    @app.post("/api/volume")
    def api_volume_set(body: dict = Body(...)):
        if not B:
            raise HTTPException(503, "no ROS")
        if "volume" in body:
            level = max(0.0, min(1.0, float(body["volume"]) / 100.0))
            r = B.set_volume(level)
            if r.get("ok"):
                B.volume = level
        if "muted" in body:
            from std_srvs.srv import SetBool
            cl = B.node.create_client(SetBool, "/audio/mute", callback_group=B.cb)
            if cl.wait_for_service(timeout_sec=2.0):
                cl.call(SetBool.Request(data=bool(body["muted"])))
                B.muted = bool(body["muted"])
        return api_volume()

    # ------------------------------------------------------------------ the other instances (the GPU box's tabs)
    @app.get("/api/robot")
    def api_robot(request: Request):
        return {"reachable": ros_up(), "url": "/teleop/" if ros else robot_url + "/teleop/", "config": {"mode": "sim" if sim else "robot", "title": "teleop + recording"}}

    @app.get("/api/route_editor")
    def api_route_editor(request: Request, scope: str = "routes"):
        base = "/live/" if ros else robot_url + "/live/"
        return {"reachable": ros_up(), "url": base, "source": state["map_source"], "info": {"map": maps.active(), "pose": pose(), "ros": ros_status()}}

    # ------------------------------------------------------------------ services + activity
    @app.get("/api/services")
    def api_services():
        if not sv:
            return {"reachable": False, "error": "no supervisor (sim)", "url": "", "job": {"running": False, "log": []}}
        return sv.services()

    @app.post("/api/services/restart_base")
    def api_services_restart_base():
        if not sv:
            raise HTTPException(503, "no supervisor")
        return _run(sv.restart_base)

    @app.post("/api/services/{name}")
    def api_service_action(name: str, body: dict = Body(...)):
        if not sv:
            raise HTTPException(503, "no supervisor")
        try:
            return sv.action(name, str(body.get("action", "")))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"supervisor: {e}")

    @app.get("/api/services/{name}/log")
    def api_service_log(name: str, stream: str = "stdout", lines: int = 80):
        if not sv:
            raise HTTPException(503, "no supervisor")
        return _run(sv.log, name, stream, lines)

    @app.get("/api/activity")
    def api_activity(lines: int = 200, units: str = ""):
        if not sv:
            return {"reachable": False, "error": "no supervisor (sim)", "lines": [], "units": []}
        return sv.activity(lines, set(units.split(",")) if units else None)

    # ------------------------------------------------------------------ datasets / models / evals (v1 webui, over the data dir)
    @app.get("/api/datasets")
    def api_datasets():
        try:
            items = ds_mod.list_datasets(str(datasets_dir))
        except Exception as e:  # noqa: BLE001
            return {"datasets": [], "error": str(e), "dir": str(datasets_dir)}
        return {"datasets": items, "dir": str(datasets_dir)}

    @app.get("/api/datasets/{name}/episodes")
    def api_episodes(name: str):
        return {"episodes": _run(ds_mod.list_episodes, str(datasets_dir), _guard(name))}

    for op, fn, key in (("rename_task", "rename_task", "task"), ("delete_episodes", "delete_episodes", "episodes"), ("merge", None, None)):
        pass

    @app.post("/api/datasets/{name}/annotate")
    def api_annotate(name: str, payload: dict = Body(...)):
        from . import annotations as anno
        return _run(anno.set_episode, str(datasets_dir / _guard(name)), int(payload.get("episode", 0)), payload.get("fields") or {})

    @app.post("/api/datasets/{name}/rename_task")
    def api_rename_task(name: str, payload: dict = Body(...)):
        return _run(ops_mod.rename_task, str(datasets_dir), _guard(name), payload.get("task"))

    @app.post("/api/datasets/{name}/delete_episodes")
    def api_delete_episodes(name: str, payload: dict = Body(...)):
        return _run(ops_mod.delete_episodes, str(datasets_dir), _guard(name), [int(e) for e in payload.get("episodes", [])])

    @app.post("/api/datasets/{name}/trim")
    def api_trim(name: str, payload: dict = Body(...)):
        return _run(ops_mod.trim_episode, str(datasets_dir), _guard(name), int(payload["episode"]), int(payload.get("start", 0)), payload.get("end"))

    @app.post("/api/datasets/{name}/move")
    def api_move(name: str, payload: dict = Body(...)):
        return _run(ops_mod.move_episode, str(datasets_dir), _guard(name), int(payload["episode"]), _guard(str(payload["dst"])))

    @app.post("/api/datasets/merge")
    def api_merge(payload: dict = Body(...)):
        return _run(ops_mod.merge, str(datasets_dir), [_guard(n) for n in payload.get("names", [])], _guard(str(payload.get("out_name", ""))))

    @app.post("/api/datasets/{name}/to_video")
    def api_to_video(name: str, payload: dict = Body(default={})):
        return _run(ops_mod.to_video, str(datasets_dir), _guard(name), payload.get("out_name"))

    @app.post("/api/datasets/{name}/delete")
    def api_dataset_delete(name: str, payload: dict = Body(default={})):
        return _run(ops_mod.delete_dataset, str(datasets_dir), _guard(name))

    @app.post("/api/datasets/{name}/restore")
    def api_restore(name: str):
        return _run(ops_mod.restore_backup, str(datasets_dir), _guard(name))

    @app.post("/api/datasets/{name}/rename")
    def api_rename_dataset(name: str, payload: dict = Body(...)):
        return _run(ops_mod.rename_dataset, str(datasets_dir), _guard(name), _guard(str(payload.get("new_name", ""))))

    @app.post("/api/datasets/add")
    def api_add(payload: dict = Body(...)):
        return _run(ops_mod.add_dataset, str(datasets_dir), source=payload.get("source"), name=payload.get("name"))

    @app.post("/api/datasets/{name}/train")
    def api_train(name: str, payload: dict = Body(default={})):
        return _run(models_mod.start_training, str(Path(data_dir)), str(datasets_dir), _guard(name), str(payload.get("policy", "smolvla")),
                    int(payload.get("steps", 20000)), int(payload.get("batch_size", 64)), int(payload.get("save_freq", 2500)), bool(payload.get("dry_run", False)))

    @app.get("/api/models")
    def api_models():
        return {"models": models_mod.list_models(str(Path(data_dir))), "runs": checkpoints.list_runs(outputs / "train"),
                "served": str(checkpoints.served(outputs) or "")}

    @app.post("/api/models/{name}/rename")
    def api_model_rename(name: str, payload: dict = Body(...)):
        return _run(models_mod.rename_model, str(Path(data_dir)), _guard(name), _guard(str(payload.get("new_name", ""))))

    @app.post("/api/models/{name}/delete")
    def api_model_delete(name: str, payload: dict = Body(default={})):
        return _run(models_mod.delete_model, str(Path(data_dir)), _guard(name))

    @app.post("/api/models/{name}/serve")
    def api_model_serve(name: str, payload: dict = Body(default={})):
        spec = _guard(name) + (f"@{payload['step']}" if payload.get("step") else "")
        path = _run(checkpoints.resolve, spec, outputs / "train")
        checkpoints.set_served(outputs, path)
        return {"ok": True, "served": str(path)}

    @app.get("/api/evals")
    def api_evals():
        return {"evals": evals.list(), "dir": str(evals.dir)}

    @app.post("/api/evals")
    def api_eval_create(payload: dict = Body(...)):
        return _run(evals.create, _guard(str(payload.get("name", ""))), payload.get("fields") or payload)

    @app.get("/api/evals/groups")
    def api_eval_groups():
        return {"groups": evals.groups()}

    @app.post("/api/evals/bulk_annotate")
    def api_evals_bulk(payload: dict = Body(...)):
        out = [evals.annotate(_guard(n), payload.get("fields") or {}) for n in payload.get("names", [])]
        return {"ok": True, "updated": len(out)}

    @app.post("/api/evals/{name}/annotate")
    def api_eval_annotate(name: str, payload: dict = Body(...)):
        return _run(evals.annotate, _guard(name), payload.get("fields") or payload)

    @app.post("/api/evals/{name}/rename")
    def api_eval_rename(name: str, payload: dict = Body(...)):
        return _run(evals.rename, _guard(name), _guard(str(payload.get("new_name", ""))))

    @app.post("/api/evals/{name}/delete")
    def api_eval_delete(name: str, payload: dict = Body(default={})):
        evals.delete(_guard(name))
        return {"ok": True}

    @app.get("/api/evals/{name}/video")
    def api_eval_video(name: str):
        d = evals.path(_guard(name))
        vids = sorted(d.glob("*.mp4"))
        if not vids:
            raise HTTPException(404, "no video")
        return FileResponse(vids[0], media_type="video/mp4")

    @app.get("/api/evals/{name}/csv")
    def api_eval_csv(name: str):
        d = evals.path(_guard(name))
        csvs = sorted(d.glob("*.csv"))
        if not csvs:
            raise HTTPException(404, "no csv")
        return FileResponse(csvs[0], media_type="text/csv")

    # ------------------------------------------------------------------ panels: the WLED matrices (rd_manager.panels)
    panels_relay = PanelsRelay(panels_url)

    @app.get("/api/panels")
    def api_panels():
        """Where the WLED is and what it is set to; the Panels tab draws the pixels from /ws/panels."""
        info = panels_info(panels_url) if panels_url else None
        return {"url": slash(panels_url) if panels_url else "", "reachable": info is not None, "info": info}

    @app.websocket("/ws/panels")
    async def ws_panels(socket: WebSocket):
        await socket.accept()
        await panels_relay.join(socket)
        try:
            while True:
                await socket.receive_text()                 # nothing is expected; this notices the close
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            panels_relay.leave(socket)

    # ------------------------------------------------------------------ the WebSocket: live data, 5 Hz
    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                live_map_tick()
                if B:
                    from .ros_bridge import SCAN_TOPICS
                    B.touch(*SCAN_TOPICS)      # an open Live page shows the laser scans
                msg = {"t": time.time(), "pose": pose(), "ros": ros_status(), "ros_up": ros_up(), "map_rev": mapview.rev, "mapping": state["mapping"],
                       "nav": {k: v for k, v in (B.nav.items() if B else []) if k in ("goal", "polygons")}, "goal_state": B.nav_goal if B else {},
                       "route": runner.state, "policy": B.policy_feedback if B else {}, "speaking": B.speaking if B else False,
                       "voice": B.voice if B else {}, "stream": B.stream_state if B else "", "hand": B.hand_state if B else [],
                       "scans": {k: v for k, v in (B.scans.items() if B else []) if time.time() - v["t"] < 3.0}}
                await socket.send_text(json.dumps(msg))
                await asyncio.sleep(0.2)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def _route_action(bridge, act) -> tuple[bool, str]:
    s = act.spec
    if act.kind == "pause":
        time.sleep(float(s.get("pause", 0.0)))
        return True, "paused"
    if bridge is None:
        return False, "no ROS"
    if act.kind == "say":
        return (lambda r: (r["ok"], r["message"]))(bridge.say(str(s.get("say", "")), wait=True))
    if act.kind == "vla":
        r = bridge.run_policy(task=str(s.get("task", "")), checkpoint=str(s.get("vla", "")), max_seconds=float(s.get("seconds") or 120),
                              done_when=str(s.get("done") or ""), still_seconds=float(s.get("still") or 0), record=bool(s.get("record", False)),
                              home_first=bool(s.get("home", True)), max_steps=int(s.get("steps") or 0))
        return r["ok"], r["message"]
    return False, f"unknown action {act.kind}"
