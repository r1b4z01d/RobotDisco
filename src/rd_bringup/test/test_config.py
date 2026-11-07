import os
from pathlib import Path

import pytest

from rd_bringup import config

ROOT = Path(__file__).resolve().parents[3]          # src/rd_bringup/test -> workspace root


@pytest.fixture(autouse=True)
def _point_at_repo_config(monkeypatch):
    monkeypatch.setenv(config.ENV_PATH, str(ROOT / "config" / "robot.yaml"))
    config._load.cache_clear()


def test_path_is_the_repo_file():
    assert config.path() == ROOT / "config" / "robot.yaml"


@pytest.mark.parametrize("key,expect", [
    ("arm.ip", "192.168.11.21"),
    ("hand.ip", "192.168.11.117"),
    ("robot.ip", "192.168.11.80"),
    ("gpu.ip", "192.168.11.119"),
    ("hand.ports.udp", 8766),
    ("arm.ports.dashboard", 29999),
    ("robot.ports.manager", 8092),
    ("robot.ros.rmw", "rmw_cyclonedds_cpp"),
])
def test_known_values(key, expect):
    assert config.get(key) == expect


def test_every_section_present():
    cfg = config.load()
    for section in ("network", "robot", "arm", "hand", "gpu", "sensors", "audio", "gamepad"):
        assert section in cfg, section


def test_missing_key_raises_and_default_works():
    with pytest.raises(KeyError):
        config.get("arm.nope")
    assert config.get("arm.nope", 7) == 7


def test_env_override(monkeypatch):
    monkeypatch.setenv("RD__arm__ip", "127.0.0.1")
    assert config.get("arm.ip") == "127.0.0.1"
    monkeypatch.setenv("RD__hand__ports__udp", "9000")
    assert config.get("hand.ports.udp", as_type=int) == 9000


def test_cli(capsys):
    assert config.main(["arm.ip"]) == 0
    assert capsys.readouterr().out.strip() == "192.168.11.21"
    assert config.main(["nope.nope"]) == 1
    assert config.main(["nope.nope", "fallback"]) == 0


def test_no_ip_typed_anywhere_else():
    """The rule from the plan: addresses live in robot.yaml only. Scan every tracked source file."""
    import re
    import subprocess
    files = subprocess.run(["git", "ls-files", "src", "rig"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    ip = re.compile(r"\b192\.168\.\d+\.\d+\b")
    offenders = []
    for f in files:
        p = ROOT / f
        if p.suffix in (".stl", ".png", ".pgm", ".jpg", ".md", ".json", ".js", ".html", ".css", ".svg") or not p.is_file():
            continue
        if "/static/" in f or "/vendor/" in f:          # the pages' hint texts and vendored libraries
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if ip.search(line) and "test_config.py" not in f and not line.lstrip().startswith("#"):
                offenders.append(f"{f}:{n}: {line.strip()[:80]}")
    assert not offenders, "\n".join(offenders)
