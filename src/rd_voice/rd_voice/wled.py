"""The LED panels follow the music: while something is playing the WLED shows its audio-reactive GEQ effect, and
when the music stops it goes back to what it was showing.

WLED's JSON API on the panels' controller (robot.yaml panels.ip): /json/effects names the effects, /json/state has
each segment's `fx`. What was showing before the music is remembered in a small file, so a restart of the voice
assistant in the middle of a song still puts the right effect back afterwards.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional


def _http(url: str, body: Optional[dict] = None, timeout: float = 3.0) -> dict | list:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


class MusicPanels:
    def __init__(self, host: str, effect: str = "GEQ", memory: Optional[Path] = None, http=_http, log=lambda s: None):
        self.url = f"http://{host}" if host and "://" not in host else host
        self.effect, self.memory, self._http, self.log = effect, memory, http, log
        self._fx_id: Optional[int] = None
        self._names: list = []
        self._was_playing: Optional[bool] = None

    def _effect_id(self) -> Optional[int]:
        if self._fx_id is None:
            self._names = list(self._http(self.url + "/json/effects"))
            self._fx_id = next((i for i, n in enumerate(self._names) if str(n).strip().lower() == self.effect.lower()), None)
        return self._fx_id

    def _remembered(self) -> dict:
        try:
            return {int(k): int(v) for k, v in json.loads(self.memory.read_text()).items()} if self.memory else {}
        except (OSError, ValueError, AttributeError):
            return {}

    def _remember(self, before: dict) -> None:
        if self.memory and before:
            try:
                self.memory.parent.mkdir(parents=True, exist_ok=True)
                self.memory.write_text(json.dumps(before))
            except OSError:
                pass

    def update(self, playing: bool) -> None:
        """Call every few seconds with whether music is playing. Does nothing while that has not changed."""
        if not self.url or playing == self._was_playing:
            return
        first = self._was_playing is None
        try:
            fx = self._effect_id()
            if fx is None:
                self.log(f"panels: the WLED has no effect named {self.effect}")
                self._was_playing = playing
                return
            segs = [s for s in self._http(self.url + "/json/state").get("seg", []) if s.get("stop", 1) > s.get("start", 0)]
            if playing:
                before = {int(s["id"]): int(s.get("fx", 0)) for s in segs if int(s.get("fx", 0)) != fx}
                self._remember(before)                       # what to go back to; nothing new when it was already showing
                if before:
                    self._http(self.url + "/json/state", {"seg": [{"id": i, "fx": fx} for i in before]})
                    self.log(f"panels: {self.effect} while the music plays")
            else:
                back = self._remembered()
                todo = [{"id": int(s["id"]), "fx": back[int(s["id"])]} for s in segs if int(s.get("fx", 0)) == fx and int(s["id"]) in back]
                if todo:
                    self._http(self.url + "/json/state", {"seg": todo})
                    name = self._names[todo[0]["fx"]] if todo[0]["fx"] < len(self._names) else todo[0]["fx"]
                    self.log(f"panels: back to {name}" + (" (the music had stopped before I started)" if first else ""))
            self._was_playing = playing
        except (OSError, ValueError, KeyError) as e:        # the controller is away: try again on the next call
            self.log(f"panels: {type(e).__name__}: {e}")
