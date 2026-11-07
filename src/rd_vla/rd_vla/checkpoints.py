"""Checkpoint specs and the serve link (v1 services.resolve_checkpoint / served_checkpoint), ROS-free.

A spec is a run name, run@step, or a path:  tigger_1_merged_smolvla_1  |  tigger_1_merged_smolvla_1@020000  |
/abs/path/to/pretrained_model. Runs live under <models_dir>/<run>/checkpoints/<step>/pretrained_model with
`last` a symlink to the newest step. `serve_link` (<outputs>/serve_ckpt) names what the server serves.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def models_dir() -> Path:
    env = os.environ.get("RD_MODELS_DIR")
    if env:
        return Path(env)
    from rd_bringup import config
    data = config.get("robot.data", "/mnt/data")
    return Path(data) / "outputs" / "train"


def resolve(spec: str, train_dir: Path | str | None = None) -> Path:
    """The pretrained_model directory for a spec. FileNotFoundError with the candidates tried."""
    train_dir = Path(train_dir) if train_dir else models_dir()
    spec = str(spec).strip()
    if not spec:
        raise FileNotFoundError("empty checkpoint spec")
    p = Path(spec).expanduser()
    tried = []
    if p.is_absolute() or spec.startswith(("./", "../")):
        for cand in (p, p / "pretrained_model", p / "checkpoints" / "last" / "pretrained_model"):
            tried.append(cand)
            if (cand / "config.json").is_file():
                return cand.resolve()
        raise FileNotFoundError(f"no checkpoint at {spec} (tried {[str(t) for t in tried]})")
    run, _, step = spec.partition("@")
    step = step or "last"
    if step.isdigit():
        step = f"{int(step):06d}"
    for cand in (train_dir / run / "checkpoints" / step / "pretrained_model", train_dir / run / "pretrained_model", train_dir / run):
        tried.append(cand)
        if (cand / "config.json").is_file():
            return cand.resolve()
    raise FileNotFoundError(f"no checkpoint for {spec!r} under {train_dir} (tried {[str(t) for t in tried]})")


def describe(path: Path | str) -> dict:
    """What a checkpoint is: policy type, run name, step, the training dataset if recorded."""
    p = Path(path)
    out = {"path": str(p), "type": None, "run": None, "step": None, "dataset": None}
    try:
        cfg = json.loads((p / "config.json").read_text())
        out["type"] = cfg.get("type") or cfg.get("policy", {}).get("type")
    except (OSError, ValueError):
        pass
    parts = p.resolve().parts
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        out["run"] = parts[i - 1] if i >= 1 else None
        out["step"] = parts[i + 1] if i + 1 < len(parts) else None
    for tc in (p / "train_config.json", p.parent / "train_config.json"):
        try:
            t = json.loads(tc.read_text())
            out["dataset"] = (t.get("dataset") or {}).get("repo_id") or (t.get("dataset") or {}).get("root")
            break
        except (OSError, ValueError):
            continue
    return out


def list_runs(train_dir: Path | str | None = None) -> list[dict]:
    train_dir = Path(train_dir) if train_dir else models_dir()
    out = []
    if not train_dir.is_dir():
        return out
    for run in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        ck = run / "checkpoints"
        steps = sorted(s.name for s in ck.iterdir() if s.is_dir() and s.name != "last") if ck.is_dir() else []
        if steps or (run / "config.json").is_file():
            out.append({"run": run.name, "steps": steps, "last": steps[-1] if steps else None,
                        "mtime": run.stat().st_mtime})
    return out


def serve_link(outputs_dir: Path | str) -> Path:
    return Path(outputs_dir) / "serve_ckpt"


def served(outputs_dir: Path | str) -> Path | None:
    link = serve_link(outputs_dir)
    try:
        return link.resolve() if link.exists() else None
    except OSError:
        return None


def set_served(outputs_dir: Path | str, checkpoint: Path | str) -> Path:
    link = serve_link(outputs_dir)
    tmp = link.with_name("serve_ckpt.tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(Path(checkpoint).resolve())
    os.replace(tmp, link)
    return link
