"""The WLED controller behind the robot's LED matrix panels, for the manager's Panels tab (v1 webui/server.py's
/api/panels and /ws/panels, 2026-11-09). ROS-free: the manager talks to the WLED itself, over HTTP for what it is
set to and over the WLED's WebSocket for the live pixels. The address is robot.yaml's panels.ip.
"""
from __future__ import annotations

import asyncio
import json
import re
import urllib.request

_EFFECTS: dict[str, list] = {}       # per WLED: the effect names never change


def slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def panels_info(url: str, timeout: float = 3.0) -> dict | None:
    """The matrix size, the segments (one per panel) and the effect each one runs; None when the WLED does not answer."""
    url = slash(url)
    try:
        with urllib.request.urlopen(url + "json/si", timeout=timeout) as r:        # state + info: the small one
            d = json.loads(r.read())
        if not _EFFECTS.get(url):
            with urllib.request.urlopen(url + "json/eff", timeout=timeout) as r:
                _EFFECTS[url] = list(json.loads(r.read()))
        i, st, fx = d["info"], d["state"], _EFFECTS[url]
        m = i["leds"].get("matrix") or {}
        return {
            "name": i.get("name"), "ver": i.get("ver"), "on": st.get("on"), "bri": st.get("bri"),
            "live": bool(i.get("live")), "fps": i["leds"].get("fps"),
            "w": m.get("w"), "h": m.get("h"),
            "segments": [
                {"name": s.get("n") or f"segment {s.get('id')}", "x0": s["start"], "x1": s["stop"],
                 "y0": s.get("startY", 0), "y1": s.get("stopY", m.get("h") or 1), "on": s.get("on"),
                 "effect": fx[s["fx"]] if 0 <= s.get("fx", -1) < len(fx) else str(s.get("fx"))}
                for s in st.get("seg", []) if s.get("stop", 0) > s.get("start", 0)],
        }
    except Exception:  # noqa: BLE001 - the panels are off, or not on this network
        return None


class PanelsRelay:
    """One connection to the WLED's live pixel stream, shared by every open Panels tab.

    The WLED streams to a single live client (the last one that asked), so the tabs cannot each connect to it; the
    relay also keeps the page working where the browser cannot reach the WLED itself. It connects when the first tab
    arrives and hangs up when the last one leaves. Frames are relayed as they are: binary "L", 2, width, height,
    then RGB per pixel (the WLED thins a large matrix, so a frame can be smaller than the matrix)."""

    def __init__(self, url: str):
        self.url = re.sub(r"^http", "ws", slash(url)) + "ws" if url else ""
        self.viewers: set = set()
        self.task = None

    async def join(self, ws) -> None:
        self.viewers.add(ws)
        if self.url and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self._pump())

    def leave(self, ws) -> None:
        self.viewers.discard(ws)

    async def _pump(self) -> None:
        import websockets
        while self.viewers:
            try:
                async with websockets.connect(self.url, open_timeout=4, max_size=1 << 20) as up:
                    await up.send('{"lv":true}')
                    while self.viewers:
                        try:
                            frame = await asyncio.wait_for(up.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            await up.send('{"lv":true}')    # another client took the stream: ask for it back
                            continue
                        if not isinstance(frame, (bytes, bytearray)):
                            continue                        # state updates (JSON text); the tab polls /api/panels
                        for v in list(self.viewers):
                            try:
                                await v.send_bytes(bytes(frame))
                            except Exception:  # noqa: BLE001 - that tab went away
                                self.viewers.discard(v)
                    await up.send('{"lv":false}')
            except Exception:  # noqa: BLE001 - the WLED is off or restarting: try again while a tab waits
                await asyncio.sleep(2.0)
