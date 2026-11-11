"""Landmarks: named places on a map the robot navigates to, each with the tasks that happen there.

    landmarks/landmarks.yaml
      landmarks:
        - id: right-bookshelf
          name: Right Bookshelf
          map: demo
          x: 3.1
          y: -0.4
          yaw: 1.57
          photo: right-bookshelf.jpg        # landmarks/img/, taken when the landmark was made (optional)
          tasks:
            - id: pick-up-tigger
              name: pick up tigger              # what you say
              model: tigger_1_merged_smolvla_1  # checkpoint the inference server serves
              task: Grab and hold tigger        # the instruction the policy was trained with
              seconds: 60
              done_when: closed                 # none | closed | open  (grasp level that ends the run)
              keep_grasp: true
              home: true

The file is the one source for rd_nav's GoToLandmark server, rd_voice's task matcher and rd_manager's
Landmarks tab (v1 kept it in Scripts/landmarks/). ROS-free; re-read when the file changes on disk.
"""
from __future__ import annotations

import difflib
import os
import re
import time
from pathlib import Path
from typing import Optional

import yaml

TASK_FIELDS = ("name", "model", "task", "backend", "seconds", "still", "done_when", "keep_grasp", "home", "steps")
TASK_DEFAULTS = {"seconds": 120.0, "still": 0.0, "done_when": "none", "keep_grasp": False, "home": True}
_ID = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    return _ID.sub("-", str(text).strip().lower()).strip("-") or "x"


def default_landmarks_dir() -> Path:
    env = os.environ.get("RD_LANDMARKS_DIR")
    if env:
        return Path(env)
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("rd_nav")) / "landmarks"
    except Exception:  # noqa: BLE001
        share = Path(__file__).resolve().parents[1] / "landmarks"
    p = share / "landmarks.yaml"
    if p.is_symlink():
        return p.resolve().parent
    return share


def task_from_body(body: dict, cur: dict | None = None) -> dict:
    """A task entry from a POST body (all fields optional when editing `cur`). ValueError = bad input."""
    out = dict(cur or {})
    for k in TASK_FIELDS:
        if k not in body:
            continue
        v = body[k]
        if k in ("name", "model", "task", "backend", "done_when"):
            v = str(v or "").strip()
            if k == "name":
                v = v.rstrip(" .!?,").lower()
            if k == "done_when" and v not in ("", "none", "closed", "open"):
                raise ValueError("done_when must be none, closed or open")
            out[k] = v
        elif k in ("seconds", "still"):
            if v in ("", None):
                out.pop(k, None)
            else:
                out[k] = round(float(v), 2)
        elif k == "steps":
            out[k] = int(v or 0)
        else:   # keep_grasp, home
            out[k] = bool(v) if not isinstance(v, str) else v.strip().lower() in ("1", "true", "yes", "on")
    if not out.get("name"):
        raise ValueError("a task needs a name (what you say)")
    if not out.get("model") and not out.get("task"):
        raise ValueError("a task needs a model and a task string")
    out.setdefault("id", slug(out["name"]))
    return out


class LandmarkStore:
    def __init__(self, directory: Path | str | None = None):
        self.dir = Path(directory) if directory else default_landmarks_dir()
        self.file = self.dir / "landmarks.yaml"
        self.img_dir = self.dir / "img"
        self._mtime: float | None = None
        self._items: list[dict] = []

    # ---- file
    def load(self) -> list[dict]:
        try:
            m = self.file.stat().st_mtime
        except FileNotFoundError:
            self._items, self._mtime = [], None
            return self._items
        if m != self._mtime:
            data = yaml.safe_load(self.file.read_text()) or {}
            self._items = [i for i in (data.get("landmarks") or []) if isinstance(i, dict) and i.get("id")]
            self._mtime = m
        return self._items

    def save(self, items: list[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        head = (f"# landmarks.yaml - named places the robot can navigate to; written by rd_nav/rd_manager on "
                f"{time.strftime('%Y-%m-%d %H:%M')} ({len(items)} landmarks).\n")
        tmp = self.file.with_name("landmarks.yaml.tmp")
        tmp.write_text(head + yaml.safe_dump({"landmarks": items}, sort_keys=False, allow_unicode=True))
        os.replace(tmp, self.file)
        self._mtime = None

    # ---- queries
    def all(self, map_name: str | None = None) -> list[dict]:
        items = self.load()
        return [i for i in items if not map_name or i.get("map") in (None, map_name)]

    def get(self, landmark_id: str) -> Optional[dict]:
        for i in self.load():
            if i["id"] == landmark_id:
                return i
        return None

    def find(self, text: str, map_name: str | None = None, cutoff: float = 0.6) -> Optional[dict]:
        """A landmark by id or name, exact first, then the closest fuzzy match."""
        key = str(text).strip().lower()
        items = self.all(map_name)
        for i in items:
            if key in (i["id"], str(i.get("name", "")).lower()):
                return i
        names = {str(i.get("name", "")).lower(): i for i in items}
        m = difflib.get_close_matches(key, list(names), n=1, cutoff=cutoff)
        return names[m[0]] if m else None

    def tasks(self, map_name: str | None = None) -> list[tuple[dict, dict]]:
        return [(lm, t) for lm in self.all(map_name) for t in (lm.get("tasks") or []) if isinstance(t, dict) and t.get("id")]

    def find_task(self, phrase: str, map_name: str | None = None, cutoff: float = 0.75) -> Optional[tuple[dict, dict]]:
        """(landmark, task) whose task name matches the phrase: exact or contained, else the closest match."""
        key = str(phrase).strip().lower().rstrip(" .!?,")
        pairs = self.tasks(map_name)
        for lm, t in pairs:
            n = str(t.get("name", "")).lower()
            if n and (n == key or n in key):
                return lm, t
        names = {str(t.get("name", "")).lower(): (lm, t) for lm, t in pairs}
        m = difflib.get_close_matches(key, list(names), n=1, cutoff=cutoff)
        return names[m[0]] if m else None

    # ---- edits
    def upsert(self, item: dict) -> dict:
        items = list(self.load())
        item = dict(item)
        if not item.get("id"):
            item["id"] = slug(item.get("name", "landmark"))
            base, n = item["id"], 2
            while any(i["id"] == item["id"] for i in items):
                item["id"] = f"{base}-{n}"
                n += 1
        for k in ("x", "y", "yaw"):
            item[k] = round(float(item.get(k, 0.0)), 4)
        item.setdefault("tasks", [])
        for idx, i in enumerate(items):
            if i["id"] == item["id"]:
                items[idx] = {**i, **item}
                break
        else:
            items.append(item)
        self.save(items)
        return item

    def delete(self, landmark_id: str) -> bool:
        items = self.load()
        keep = [i for i in items if i["id"] != landmark_id]
        if len(keep) == len(items):
            return False
        self.save(keep)
        return True

    def set_task(self, landmark_id: str, body: dict, task_id: str | None = None) -> dict:
        lm = self.get(landmark_id)
        if lm is None:
            raise KeyError(landmark_id)
        tasks = [t for t in (lm.get("tasks") or []) if isinstance(t, dict)]
        cur = next((t for t in tasks if t.get("id") == task_id), None) if task_id else None
        t = task_from_body(body, cur)
        if cur is None:
            if any(x.get("id") == t["id"] for x in tasks):
                raise ValueError(f"task {t['id']!r} exists")
            tasks.append(t)
        else:
            tasks[tasks.index(cur)] = t
        lm["tasks"] = tasks
        self.upsert(lm)
        return t

    def delete_task(self, landmark_id: str, task_id: str) -> bool:
        lm = self.get(landmark_id)
        if lm is None:
            return False
        tasks = [t for t in (lm.get("tasks") or []) if t.get("id") != task_id]
        if len(tasks) == len(lm.get("tasks") or []):
            return False
        lm["tasks"] = tasks
        self.upsert(lm)
        return True
