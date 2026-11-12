"""Lock or release the three policy cameras' exposure and white balance with v4l2-ctl (v1 sensors/cameras.py's
manual-exposure switch). Auto exposure meters the whole frame, so the same shelf looks different once the hand is in
view or a lamp goes on; locked, every frame of a dataset is exposed alike.

lock: each camera keeps the exposure and colour temperature its auto modes sit at right now. release: auto again.
The devices are robot.yaml sensors.usb_cameras (by USB port); `run` is injectable for the tests.
"""
from __future__ import annotations

import os
import re
import subprocess

CAMERAS = ("scene", "wrist", "side")
MANUAL, AUTO = 1, 3          # V4L2 auto_exposure menu: 1 = manual, 3 = aperture priority (auto)


def devices(cfg: dict) -> dict:
    by_path = str(cfg.get("by_path", ""))
    return {c: os.path.realpath(by_path.format(port=cfg[c]["port"])) for c in CAMERAS if c in cfg and by_path}


def _run(args: list) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout


def read(device: str, run=_run) -> dict:
    """{control: value} for the controls this module touches (absent ones are left out)."""
    out = run(["v4l2-ctl", "-d", device, "--list-ctrls"])
    found = {}
    for name in ("auto_exposure", "exposure_time_absolute", "white_balance_automatic", "white_balance_temperature", "gain"):
        m = re.search(rf"^\s*{name}\s.*?\bvalue=(-?\d+)", out, re.M)
        if m:
            found[name] = int(m.group(1))
    return found


def state(cfg: dict, run=_run) -> dict:
    cams = {}
    for cam, dev in devices(cfg).items():
        c = read(dev, run)
        cams[cam] = {"device": dev, "locked": c.get("auto_exposure") == MANUAL, "exposure": c.get("exposure_time_absolute"),
                     "white_balance": c.get("white_balance_temperature"), "found": bool(c)}
    return {"cameras": cams, "locked": bool(cams) and all(c["locked"] for c in cams.values())}


def set_locked(cfg: dict, locked: bool, run=_run) -> dict:
    for dev in devices(cfg).values():
        c = read(dev, run)
        if not c:
            continue
        if locked:
            # the value first read while auto is still on, then manual with exactly that value
            sets = [f"auto_exposure={MANUAL}"]
            if "exposure_time_absolute" in c:
                sets.append(f"exposure_time_absolute={c['exposure_time_absolute']}")
            run(["v4l2-ctl", "-d", dev, "-c", ",".join(sets)])
            if "white_balance_automatic" in c:
                run(["v4l2-ctl", "-d", dev, "-c", "white_balance_automatic=0"])
                if "white_balance_temperature" in c:
                    run(["v4l2-ctl", "-d", dev, "-c", f"white_balance_temperature={c['white_balance_temperature']}"])
        else:
            run(["v4l2-ctl", "-d", dev, "-c", f"auto_exposure={AUTO}"])
            if "white_balance_automatic" in c:
                run(["v4l2-ctl", "-d", dev, "-c", "white_balance_automatic=1"])
    return state(cfg, run)


def adjust(cfg: dict, action: str, cam: str | None = None, run=_run) -> dict:
    """v1's exposure keys: auto | lock | plus | minus, for every policy camera or just `cam`. plus / minus switch a
    camera to manual and scale its exposure time by a quarter stop either way."""
    if action in ("auto", "lock") and cam is None:
        return set_locked(cfg, action == "lock", run)
    for name, dev in devices(cfg).items():
        if cam is not None and name != cam:
            continue
        c = read(dev, run)
        if "exposure_time_absolute" not in c:
            continue
        if action in ("auto", "lock"):
            set_locked({"by_path": cfg.get("by_path", ""), name: cfg[name]}, action == "lock", run)
            continue
        value = c["exposure_time_absolute"] * (1.19 if action == "plus" else 1 / 1.19)
        value = max(1, min(5000, int(round(value)) + (1 if action == "plus" else -1 if value > 2 else 0)))
        run(["v4l2-ctl", "-d", dev, "-c", f"auto_exposure={MANUAL},exposure_time_absolute={value}"])
    return state(cfg, run)
