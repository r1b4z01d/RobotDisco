"""localization.launch.py's odom_alpha argument: empty keeps nav2.yaml's AMCL motion noise, a value overrides alpha1..5."""
import importlib.util
from pathlib import Path

from launch import LaunchContext

LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "localization.launch.py"


def _load():
    spec = importlib.util.spec_from_file_location("localization_launch", LAUNCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _context(value):
    ctx = LaunchContext()
    ctx.launch_configurations["odom_alpha"] = value
    return ctx


def test_empty_keeps_the_yaml_values():
    assert _load().odom_alpha_params(_context("")) == []
    assert _load().odom_alpha_params(_context("  ")) == []


def test_value_overrides_all_five_alphas():
    got = _load().odom_alpha_params(_context("0.05"))
    assert got == [{"alpha1": 0.05, "alpha2": 0.05, "alpha3": 0.05, "alpha4": 0.05, "alpha5": 0.05}]


def test_the_yaml_default_is_the_real_robots_noise():
    import yaml
    cfg = yaml.safe_load((LAUNCH.parents[1] / "config" / "nav2.yaml").read_text())["amcl"]["ros__parameters"]
    assert all(cfg[f"alpha{i}"] == 0.2 for i in range(1, 6))
