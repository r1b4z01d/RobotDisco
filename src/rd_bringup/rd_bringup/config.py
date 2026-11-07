"""The one place addresses, ports and device paths come from: config/robot.yaml at the workspace root.

    from rd_bringup import config
    cfg = config.load()                     # the whole file as a dict (cached)
    config.get("arm.ip")                    # "192.168.11.21"
    config.get("hand.ports.udp", 8766)      # with a default
    config.get("robot.ports.manager", as_type=int)

Where the file is, in order: $RD_ROBOT_CONFIG; a `config/robot.yaml` found walking up from this module (the
source tree with --symlink-install, or the install tree, both sit under the workspace root); then
`/home/rd/RobotDisco/config/robot.yaml`. Any value can be overridden for one process with an environment
variable spelled RD__<path with __ for dots>, e.g. RD__arm__ip=127.0.0.1 — that is how the rig points the
packages at mocks without editing the file.

`rd_config arm.ip` prints a value from the shell (launch files and supervisor units use it).
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ENV_PATH = "RD_ROBOT_CONFIG"
ENV_PREFIX = "RD__"
FALLBACK = Path("/home/rd/RobotDisco/config/robot.yaml")

_MISSING = object()


def path() -> Path:
    """The config file in use (raises FileNotFoundError when none exists)."""
    p = os.environ.get(ENV_PATH)
    if p:
        return Path(p)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "config" / "robot.yaml"
        if cand.is_file():
            return cand
    if FALLBACK.is_file():
        return FALLBACK
    raise FileNotFoundError(f"no config/robot.yaml: set {ENV_PATH}")


@functools.lru_cache(maxsize=4)
def _load(p: str, mtime: float) -> dict:
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load(reload: bool = False) -> dict:
    p = path()
    mtime = p.stat().st_mtime if not reload else -1.0
    if reload:
        _load.cache_clear()
        mtime = p.stat().st_mtime
    return _load(str(p), mtime)


def get(key: str, default: Any = _MISSING, as_type: type | None = None) -> Any:
    """Dotted lookup with env override. KeyError when absent and no default given."""
    env = os.environ.get(ENV_PREFIX + key.replace(".", "__"))
    if env is not None:
        value: Any = env
    else:
        value = load()
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                if default is _MISSING:
                    raise KeyError(f"robot.yaml has no '{key}'")
                return default
            value = value[part]
    if as_type is not None and not isinstance(value, as_type):
        if as_type is bool and isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return as_type(value)
    return value


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: rd_config <dotted.key> [default]   # prints the value;  rd_config --path", file=sys.stderr)
        return 2
    if argv[0] == "--path":
        print(path())
        return 0
    try:
        v = get(argv[0]) if len(argv) < 2 else get(argv[0], argv[1])
    except KeyError as e:
        print(e, file=sys.stderr)
        return 1
    print(yaml.safe_dump(v, default_flow_style=True).strip() if isinstance(v, (dict, list)) else v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
