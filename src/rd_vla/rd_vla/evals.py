"""The eval registry (v1 webui/evals.py): runs under <evals_dir>/<name>/eval.json plus artifacts (mp4, csv).

Auto runs are written by the policy runner (one folder per rollout); manual ones by the manager. Runs are
grouped by the resolved checkpoint; the operator rates and annotates from the UI.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

META_KEYS = ("model", "checkpoint", "task", "episodes", "successes", "success_rate", "rating", "notes", "operator",
             "checkpoint_inferred", "landmark")
RUN_KEYS = ("steps", "duration_s", "exposure", "fps", "n_action_steps", "remote", "ended", "backend", "dataset")


def evals_dir() -> Path:
    env = os.environ.get("RD_EVALS_DIR")
    if env:
        return Path(env)
    from rd_bringup import config
    return Path(config.get("robot.data", "/mnt/data")) / "outputs" / "evals"


def _guard(name: str) -> None:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"bad eval name: {name!r}")


def _read(run: Path) -> dict:
    p = run / "eval.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except ValueError:
            return {}
    return {}


def _write(run: Path, data: dict) -> None:
    run.mkdir(parents=True, exist_ok=True)
    tmp = run / "eval.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, run / "eval.json")


def _artifacts(run: Path) -> dict:
    return {"videos": sorted(os.path.basename(x) for x in glob.glob(str(run / "*.mp4"))),
            "csvs": sorted(os.path.basename(x) for x in glob.glob(str(run / "*.csv")))}


class EvalRegistry:
    def __init__(self, directory: Path | str | None = None):
        self.dir = Path(directory) if directory else evals_dir()

    def path(self, name: str) -> Path:
        _guard(name)
        return self.dir / name

    def list(self) -> list[dict[str, Any]]:
        out = []
        if not self.dir.is_dir():
            return out
        for run in sorted(p for p in self.dir.iterdir() if p.is_dir()):
            d = _read(run)
            item = {"name": run.name, "created": d.get("created") or run.stat().st_mtime, **_artifacts(run)}
            for k in META_KEYS + RUN_KEYS:
                if k in d:
                    item[k] = d[k]
            out.append(item)
        out.sort(key=lambda x: x["created"], reverse=True)
        return out

    def create(self, name: str, fields: dict) -> dict:
        run = self.path(name)
        if (run / "eval.json").exists():
            raise FileExistsError(name)          # a folder with artifacts but no eval.json yet is fine (the runner's)
        data = {"created": time.time(), "episodes": 0, "successes": 0}
        data.update({k: v for k, v in fields.items() if k in META_KEYS + RUN_KEYS})
        if data.get("episodes"):
            data["success_rate"] = round(float(data.get("successes", 0)) / float(data["episodes"]), 3)
        _write(run, data)
        return {"name": name, **data}

    def annotate(self, name: str, fields: dict) -> dict:
        run = self.path(name)
        if not run.is_dir():
            raise FileNotFoundError(name)
        d = _read(run)
        for k, v in fields.items():
            if k in META_KEYS:
                d[k] = v
        if d.get("episodes"):
            d["success_rate"] = round(float(d.get("successes", 0)) / float(d["episodes"]), 3)
        _write(run, d)
        return {"name": name, **d}

    def rename(self, name: str, new_name: str) -> dict:
        src, dst = self.path(name), self.path(new_name)
        if not src.is_dir():
            raise FileNotFoundError(name)
        if dst.exists():
            raise FileExistsError(new_name)
        os.rename(src, dst)
        return {"name": new_name, **_read(dst)}

    def delete(self, name: str) -> None:
        run = self.path(name)
        if run.is_dir():
            shutil.rmtree(run)

    @staticmethod
    def model_key(meta: dict) -> str:
        return str(meta.get("checkpoint") or meta.get("model") or "?")

    def groups(self) -> list[dict[str, Any]]:
        """Per checkpoint: how many runs, mean rating, success rate over the rated runs."""
        groups: dict[str, dict] = {}
        for e in self.list():
            key = self.model_key(e)
            g = groups.setdefault(key, {"model": key, "runs": 0, "ratings": [], "episodes": 0, "successes": 0, "latest": 0})
            g["runs"] += 1
            if e.get("rating"):
                g["ratings"].append(float(e["rating"]))
            g["episodes"] += int(e.get("episodes") or 0)
            g["successes"] += int(e.get("successes") or 0)
            g["latest"] = max(g["latest"], float(e.get("created") or 0))
        out = []
        for g in groups.values():
            r = g.pop("ratings")
            g["rating"] = round(sum(r) / len(r), 2) if r else None
            g["success_rate"] = round(g["successes"] / g["episodes"], 3) if g["episodes"] else None
            out.append(g)
        out.sort(key=lambda g: g["latest"], reverse=True)
        return out
