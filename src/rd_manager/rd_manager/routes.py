"""Routes: a YAML of waypoints with per-waypoint actions (v1 path.py's format, kept so the editor pages and the
old route files work), and the runner that drives them with Nav2's FollowWaypoints and runs the actions
through the actions the rest of v2 exposes: `vla:` -> RunPolicy, `say:` -> Say, `pause:` -> a wait.

    waypoints:
      - name: Charger
        position: {x: 0.69, y: 0.92, z: 0}
        orientation: {x: 0, y: 0, z: 0.027, w: 0.999}
        pause: 2.0                         # optional: stop here that long before the actions
        post_actions:
          - vla: tigger_1_merged_smolvla_1   # blocking: the policy run (task string, seconds, done ...)
            task: Grab and hold tigger
            seconds: 60
            done: closed
          - say: "I'm here"                  # blocking: the sentence is spoken before driving on
    initial_pose: {...}                      # optional: a pose to set before the route (an initialpose)

A waypoint with a blocking action ends a FollowWaypoints segment: the robot arrives, the actions run in
order, then the next segment starts. v1's shell actions (`run:` / a bare string) still load but are never
run (nothing shells out any more); check_route says so.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_ .-]{0,63}$")


def valid_route_name(name: str) -> bool:
    return bool(name) and bool(_NAME.match(name)) and not name.startswith(".") and "/" not in name


@dataclass
class Action:
    kind: str                 # vla | say | pause
    spec: dict
    blocking: bool = True
    on_fail: str = "stop"     # stop | continue

    @property
    def label(self) -> str:
        if self.kind == "vla":
            return f"vla {self.spec.get('vla')} ({self.spec.get('task', '')})"
        if self.kind == "say":
            return f"say {self.spec.get('say')!r}"
        if self.kind == "shell":
            return f"shell {self.spec.get('run')!r} (v1 action, not run)"
        return f"pause {self.spec.get('pause')}s"


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    yaw: float
    actions: list = field(default_factory=list)
    pause: float = 0.0

    @property
    def blocking(self) -> list:
        return [a for a in self.actions if a.blocking] + ([Action("pause", {"pause": self.pause})] if self.pause else [])

    def as_dict(self) -> dict:
        d = {"name": self.name, "position": {"x": self.x, "y": self.y, "z": 0.0},
             "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(self.yaw / 2), "w": math.cos(self.yaw / 2)}}
        if self.pause:
            d["pause"] = self.pause
        if self.actions:
            d["post_actions"] = [a.spec for a in self.actions]
        return d


def yaw_of(o: dict) -> float:
    z, w = float(o.get("z", 0.0)), float(o.get("w", 1.0))
    x, y = float(o.get("x", 0.0)), float(o.get("y", 0.0))
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def parse_action(raw, where: str) -> Action:
    if isinstance(raw, str):
        return Action("shell", {"run": raw}, False, "continue")          # v1 shell action: kept in the file, never run
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: an action must be a mapping")
    on_fail = str(raw.get("on_fail", "stop"))
    if on_fail not in ("stop", "continue"):
        raise ValueError(f"{where}: on_fail must be stop or continue")
    if "vla" in raw:
        if not raw.get("task"):
            raise ValueError(f"{where}: a vla action needs 'task: ...'")
        return Action("vla", dict(raw), True, on_fail)
    if "say" in raw:
        return Action("say", dict(raw), bool(raw.get("wait", True)), on_fail)
    if "pause" in raw:
        float(raw["pause"])
        return Action("pause", dict(raw), True, on_fail)
    if "run" in raw:
        return Action("shell", dict(raw), False, "continue")             # v1 shell action: kept, never run
    raise ValueError(f"{where}: unknown action {sorted(raw)}")


def load_route(path: Path) -> tuple[list[Waypoint], Optional[Waypoint]]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    wps = []
    for i, raw in enumerate(data.get("waypoints") or []):
        name = str(raw.get("name") or f"wp_{i + 1}")
        pos, ori = raw.get("position") or {}, raw.get("orientation") or {}
        actions = [parse_action(a, f"{name}.post_actions[{j}]") for j, a in enumerate(raw.get("post_actions") or [])]
        wps.append(Waypoint(name, float(pos.get("x", 0)), float(pos.get("y", 0)), yaw_of(ori), actions, float(raw.get("pause") or 0.0)))
    ip = data.get("initial_pose")
    init = None
    if isinstance(ip, dict):
        init = Waypoint("initial_pose", float((ip.get("position") or {}).get("x", 0)), float((ip.get("position") or {}).get("y", 0)),
                        yaw_of(ip.get("orientation") or {}))
    return wps, init


def save_route(path: Path, waypoints: list[dict], initial_pose: Optional[dict] = None, map_name: str = "") -> None:
    out = {"waypoints": waypoints}
    if initial_pose:
        out["initial_pose"] = initial_pose
    head = f"# {Path(path).stem}: written by rd_manager's route editor on {time.strftime('%Y-%m-%d %H:%M')}" + (f" (map {map_name})" if map_name else "") + "\n"
    Path(path).write_text(head + yaml.safe_dump(out, sort_keys=False))


def check_route(path: Path) -> str:
    try:
        wps, init = load_route(path)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    segs = segments(wps)
    lines = [f"{len(wps)} waypoints, {len(segs)} segment(s)" + (", initial pose set" if init else "")]
    for a, b in segs:
        lines.append(f"  drive {wps[a].name} .. {wps[b].name}")
        for act in wps[b].blocking:
            lines.append(f"    then {act.label}")
    legacy = [a for w in wps for a in w.actions if a.kind == "shell"]
    if legacy:
        lines.append(f"  NOTE: {len(legacy)} v1 shell action(s) are kept in the file but not run")
    return "\n".join(lines)


def segments(waypoints: list[Waypoint]) -> list[tuple[int, int]]:
    """(first, last) index pairs: a segment ends at a waypoint with a blocking action, or at the end."""
    out, start = [], 0
    for i, wp in enumerate(waypoints):
        if wp.blocking or i == len(waypoints) - 1:
            out.append((start, i))
            start = i + 1
    return out


def route_map(path: Path) -> str:
    try:
        first = Path(path).read_text().split("\n", 1)[0]
    except OSError:
        return ""
    m = re.search(r"\(map ([A-Za-z0-9_-]+)\)", first)
    return m.group(1) if m else ""


class RouteRunner:
    """Drives a route. `follow(poses) -> (ok, why)` and `run_action(action) -> (ok, why)` are the node's
    blocking callables; `cancel()` stops whichever is running. One route at a time."""

    def __init__(self, follow: Callable, run_action: Callable, cancel: Callable, set_initial_pose: Callable, log=None):
        self.follow, self.run_action, self.cancel_fn, self.set_initial_pose = follow, run_action, cancel, set_initial_pose
        self._log = log or (lambda m: None)
        self.state = {"running": False, "route": "", "segment": 0, "segments": 0, "waypoint": "", "phase": "", "log": [], "t": 0.0}
        self._stop = False
        self._thread = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, path: Path, from_wp: str = "") -> dict:
        import threading
        if self.busy:
            raise RuntimeError(f"a route is running ({self.state['route']})")
        wps, init = load_route(path)
        if from_wp:
            idx = next((i for i, w in enumerate(wps) if w.name == from_wp), None)
            if idx is None:
                raise ValueError(f"no waypoint {from_wp!r}")
            wps, init = wps[idx:], None
        if not wps:
            raise ValueError("the route has no waypoints")
        self._stop = False
        self.state.update(running=True, route=Path(path).stem, segment=0, segments=len(segments(wps)), waypoint="", phase="starting", log=[], t=time.time())
        self._thread = threading.Thread(target=self._run, args=(wps, init), daemon=True, name="route")
        self._thread.start()
        return dict(self.state)

    def stop(self) -> str:
        if not self.busy:
            return "no route running"
        self._stop = True
        self.cancel_fn()
        return "stopping"

    def _note(self, msg: str):
        self.state["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")
        self.state["log"] = self.state["log"][-100:]
        self._log(msg)

    def _run(self, wps, init):
        ok, why = True, "done"
        try:
            if init is not None:
                self.set_initial_pose(init.x, init.y, init.yaw)
                time.sleep(1.0)
            for n, (a, b) in enumerate(segments(wps)):
                if self._stop:
                    ok, why = False, "stopped"
                    return
                self.state.update(segment=n + 1, phase="driving", waypoint=wps[b].name)
                self._note(f"segment {n + 1}: {wps[a].name} .. {wps[b].name}")
                ok, why = self.follow([(w.x, w.y, w.yaw) for w in wps[a:b + 1]])
                if not ok:
                    self._note(f"drive failed: {why}")
                    return
                for act in wps[b].blocking:
                    if self._stop:
                        ok, why = False, "stopped"
                        return
                    self.state["phase"] = act.kind
                    self._note(f"at {wps[b].name}: {act.label}")
                    ok2, why2 = self.run_action(act)
                    if not ok2:
                        self._note(f"{act.label} failed: {why2}")
                        if act.on_fail == "stop":
                            ok, why = False, why2
                            return
            self._note("route done")
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"{type(e).__name__}: {e}"
            self._note(f"route crashed: {why}")
        finally:
            self.state.update(running=False, phase="done" if ok else "failed", result=why)
