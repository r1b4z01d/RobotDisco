"""Launch-time exposure for a policy run (v1 sensors/cameras.py ExposureSession.auto_expose).

Before the first observation each policy camera is set to the manual exposure whose frame median lands nearest
TARGET_MEDIAN, and left locked there for the run. Why not the driver's auto: it meters the whole sensor, re-meters as
the arm sweeps through the frame, and its converged value cannot be read back (the control reports its last manual
value while auto runs, so "auto, then lock" locks a stale number: the cameras sat at a morning's exposure in full
daylight, the shelf a white slab, 2026-11-14). The median ignores a small blown window and exposes for what fills the
frame. White balance is left automatic, as v1 left it.

`frame(cam)` -> (monotonic time, RGB array) of the newest frame; `set_exposure(cam, value)` applies one. Both are
injectable; `v4l2_setter` is the real one. Cameras without a device (the rig, the sim) are skipped.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

import numpy as np

LADDER = (625.0, 312.0, 156.0, 78.0, 39.0, 20.0, 10.0, 5.0)     # v4l2 exposure_time_absolute, 100 us units
TARGET_MEDIAN = 120.0
STEP = 1.25
SETTLE_FRAMES = 2                 # new frames to see once SETTLE_S has passed
SETTLE_S = 0.35                   # the sensor takes two frames to show a new exposure and the camera node and the
                                  # rotate node hold a few more: frames counted from the set alone read the old value


def devices(cfg: dict, cameras) -> dict:
    by_path = str((cfg or {}).get("by_path", ""))
    out = {}
    for c in cameras:
        port = ((cfg or {}).get(c) or {}).get("port")
        if by_path and port:
            dev = by_path.format(port=port)
            if os.path.exists(dev):
                out[c] = os.path.realpath(dev)
    return out


def v4l2_setter(devs: dict):
    def set_exposure(cam: str, value: float) -> None:
        subprocess.run(["v4l2-ctl", "-d", devs[cam], "-c", f"auto_exposure=1,exposure_time_absolute={int(round(value))}"],
                       capture_output=True, timeout=5)
    return set_exposure


def median_of(img) -> float:
    nz = img.max(axis=2) > 12                         # the black of a letterbox or a covered lens is not content
    px = img[nz] if nz.any() else img.reshape(-1, 3)
    return float(np.median(px))


def _measure(cam, value, frame, set_exposure, timeout: float = 2.0):
    set_exposure(cam, value)
    t_set = time.monotonic()
    seen, last_t, end = 0, None, t_set + timeout
    while time.monotonic() < end:
        got = frame(cam)
        if got is not None and got[0] > t_set + SETTLE_S and got[0] != last_t:
            last_t = got[0]
            seen += 1
            if seen >= SETTLE_FRAMES:
                return median_of(got[1])
        time.sleep(0.01)
    return None


def _search(cam, frame, set_exposure, out: dict, target: float) -> None:
    best, best_err, best_med = None, None, None
    for v in LADDER:
        med = _measure(cam, v, frame, set_exposure)
        if med is None:
            break
        if best_err is None or abs(med - target) < best_err:
            best, best_err, best_med = v, abs(med - target), med
    if best is None:
        return
    for _ in range(3):                                 # the ladder is a factor of two coarse: walk a quarter stop at a time
        if best_err < 8:
            break
        cand = max(1.0, best * (STEP if best_med < target else 1.0 / STEP))
        med = _measure(cam, cand, frame, set_exposure)
        if med is None or abs(med - target) >= best_err:
            break
        best, best_err, best_med = cand, abs(med - target), med
    set_exposure(cam, best)
    out[cam] = {"exposure": int(round(best)), "median": round(best_med, 1)}


def auto_expose(cameras, frame, set_exposure, target: float = TARGET_MEDIAN) -> dict:
    """{camera: {exposure, median}} for the cameras that were set; one thread per camera (independent devices)."""
    out: dict = {}
    threads = [threading.Thread(target=_search, args=(c, frame, set_exposure, out, target), daemon=True) for c in cameras]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=25.0)
    time.sleep(0.25)                                   # the chosen exposure in the frames before anyone observes
    return out
