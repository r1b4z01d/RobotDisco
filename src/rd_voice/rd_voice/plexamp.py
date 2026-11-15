"""Music from a Plex server, played by Plexamp headless on the robot (the `plexamp` unit, :32500).

The same face as skills.MusicSkill (play / next / previous / now / stop / is_playing / duck / unduck), so the brain
does not care which one it has. play("") shuffles the whole music library; play("<words>") asks the server's search
and takes the best artist, else album, else track. The server builds the play queue, Plexamp is told to play it
(the Plex companion calls its own apps use), and next / previous / stop / the volume go to Plexamp.

Ducking is Plexamp's own volume: it plays straight to the sound device, so it has no PulseAudio stream to turn down.

The account token is Plexamp's own, read from its settings when needed and never logged. `MusicWithFallback` falls
back to the local player when Plexamp or the server does not answer.
"""
from __future__ import annotations

import glob
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable, Optional

SETTINGS = "~/.local/share/Plexamp/Settings"


def read_token(settings_dir: str = SETTINGS) -> str:
    """Plexamp's account token: the file holds one type character, then the token."""
    for f in glob.glob(os.path.join(os.path.expanduser(settings_dir), "*user%3Atoken")):
        try:
            raw = open(f).read().strip()
        except OSError:
            continue
        return raw[1:] if len(raw) > 1 else ""
    return ""


def squash(name: str) -> str:
    """A name as its letters and digits only, lower case, a leading "the" dropped: what two spellings share."""
    k = "".join(c for c in name.lower() if c.isalnum())
    return k[3:] if k.startswith("the") and len(k) > 6 else k


def _http(url: str, method: str = "GET", headers: Optional[dict] = None, timeout: float = 8.0) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}, method=method), timeout=timeout) as r:
        return r.read()


class PlexampSkill:
    def __init__(self, server: str, player: str = "http://127.0.0.1:32500", duck_level: float = 0.25,
                 token: Optional[Callable[[], str]] = None, http=_http, start_unit: Optional[Callable[[], None]] = None,
                 start_wait_s: float = 25.0):
        self.server, self.player = server.rstrip("/"), player.rstrip("/")
        self.duck_level = max(0.0, min(float(duck_level), 1.0))
        self._token, self._http = token or read_token, http
        self._start_unit, self._start_wait_s = start_unit, float(start_wait_s)
        self._cmd = 0
        self._machine = self._section = None
        self._artists = None                 # (squashed name, item) of every artist, read once: for names heard wrong
        self._volume_before_duck: Optional[int] = None
        self._state = ("", 0.0)

    # ---- the two ends
    def _headers(self) -> dict:
        return {"Accept": "application/json", "X-Plex-Client-Identifier": "robot-disco-voice", "X-Plex-Token": self._token()}

    def _server(self, path: str, method: str = "GET", **q) -> dict:
        b = self._http(self.server + path + ("?" + urllib.parse.urlencode(q) if q else ""), method, self._headers())
        return json.loads(b or b"{}").get("MediaContainer", {})

    def _player(self, path: str, **q) -> bytes:
        self._cmd += 1
        q.update(commandID=self._cmd, type="music")
        return self._http(self.player + path + "?" + urllib.parse.urlencode(q), "GET", self._headers())

    def _timeline(self) -> dict:
        root = ET.fromstring(self._player("/player/timeline/poll", wait=0, includeMetadata=1))
        for t in root.iter("Timeline"):
            if t.get("type") == "music":
                d = dict(t.attrib)
                track = t.find("Track")
                if track is not None:
                    d.update(title=track.get("title", ""), artist=track.get("grandparentTitle", ""), album=track.get("parentTitle", ""))
                return d
        return {}

    def available(self) -> bool:
        """The player answers (started first where a unit was given) and it is signed in."""
        if not self._token():
            return False
        try:
            self._player("/player/timeline/poll", wait=0)
            return True
        except (OSError, ET.ParseError):
            pass
        if self._start_unit is None:
            return False
        self._start_unit()
        end = time.monotonic() + self._start_wait_s
        while time.monotonic() < end:
            time.sleep(1.0)
            try:
                self._player("/player/timeline/poll", wait=0)
                return True
            except (OSError, ET.ParseError):
                continue
        return False

    def _library(self) -> tuple:
        if self._machine is None:
            self._machine = self._server("/identity")["machineIdentifier"]
            self._section = next(d["key"] for d in self._server("/library/sections").get("Directory", []) if d.get("type") == "artist")
        return self._machine, self._section

    # ---- the skill
    def find(self, query: str) -> Optional[dict]:
        """The best match for the words: an artist, else an album, else a track."""
        _machine, section = self._library()
        hubs = {h.get("type"): h.get("Metadata") or [] for h in self._server("/hubs/search", query=query, limit=3, sectionId=section).get("Hub", [])}
        for kind in ("artist", "album", "track"):
            if hubs.get(kind):
                return hubs[kind][0]
        return self.find_artist_by_sound(query)

    def find_artist_by_sound(self, query: str, cutoff: float = 0.68) -> Optional[dict]:
        """An artist whose name the ears got roughly right: "dead mouse" for deadmau5, "dead mile 5" too. The search
        above matches spelling; this compares the letters and digits of what was heard with every artist's name."""
        import difflib
        _machine, section = self._library()
        if self._artists is None:
            self._artists = [(squash(m.get("title", "")), m) for m in self._server(f"/library/sections/{section}/all", type=8).get("Metadata", [])]
            self._artists = [(k, m) for k, m in self._artists if len(k) >= 3]
        heard = squash(query)
        if len(heard) < 3:
            return None
        best, score = None, cutoff
        for key, m in self._artists:
            if abs(len(key) - len(heard)) > max(3, len(heard) // 2):
                continue
            r = difflib.SequenceMatcher(None, heard, key).ratio()
            if r > score:
                best, score = m, r
        return best

    def play(self, query: str = "") -> str:
        machine, section = self._library()
        query = (query or "").strip()
        if query:
            item = self.find(query)
            if item is None:
                return f"I couldn't find {query} in the music library."
            source, kind = item["key"].replace("/children", ""), item.get("type")
            said = item.get("title", query) if kind == "artist" else f"{item.get('title', query)} by {item.get('grandparentTitle') or item.get('parentTitle') or 'an unknown artist'}"
        else:
            source, kind, said = f"/library/sections/{section}/all?type=10", "library", "music"
        pq = self._server("/playQueues", "POST", type="audio", uri=f"server://{machine}/com.plexapp.plugins.library{source}",
                          shuffle=0 if kind in ("album", "track") else 1, continuous=1)
        if not pq.get("Metadata"):
            return "The music library gave me nothing to play."
        host = urllib.parse.urlparse(self.server)
        self._player("/player/playback/playMedia", address=host.hostname, port=host.port or 32400, protocol=host.scheme or "http",
                     machineIdentifier=machine, key=pq["Metadata"][0]["key"], containerKey=f"/playQueues/{pq['playQueueID']}?window=100&own=1",
                     token=self._token())
        self._state = ("playing", time.monotonic())
        self._volume_before_duck = None
        return f"Playing {said}."

    def next(self, auto: bool = False) -> str:
        self._player("/player/playback/skipNext")
        return "Next song."

    def previous(self) -> str:
        self._player("/player/playback/skipPrevious")
        return "Previous song."

    def now(self) -> str:
        t = self._timeline()
        if t.get("state") not in ("playing", "paused") or not t.get("title"):
            return "Nothing is playing."
        return f"This is {t['title']}" + (f" by {t['artist']}" if t.get("artist") else "") + "."

    def stop(self, quiet: bool = False) -> str:
        try:
            was = self.is_playing(fresh=True)
            self._player("/player/playback/stop")
        except (OSError, ET.ParseError):
            was = False
        self._state = ("stopped", time.monotonic())
        return "" if quiet else ("Music stopped." if was else "No music is playing.")

    def is_playing(self, fresh: bool = False) -> bool:
        if not fresh and time.monotonic() - self._state[1] < 1.5:
            return self._state[0] == "playing"
        try:
            state = self._timeline().get("state", "")
        except (OSError, ET.ParseError):
            state = ""
        self._state = (state, time.monotonic())
        return state == "playing"

    def duck(self) -> None:
        if self._volume_before_duck is not None:
            return
        try:
            if not self.is_playing():
                return
            vol = int(float(self._timeline().get("volume", 100)))
            self._volume_before_duck = vol
            self._player("/player/playback/setParameters", volume=max(3, int(vol * self.duck_level)))
        except (OSError, ET.ParseError, ValueError):
            pass

    def unduck(self) -> None:
        vol, self._volume_before_duck = self._volume_before_duck, None
        if vol is None:
            return
        try:
            self._player("/player/playback/setParameters", volume=vol)
        except (OSError, ET.ParseError):
            pass


class MusicWithFallback:
    """Plexamp when it answers, the local player when it does not; whichever started the music is the one that is
    skipped, asked and stopped."""

    def __init__(self, plex: PlexampSkill, local, log=lambda s: None):
        self.plex, self.local, self.log = plex, local, log
        self.active = None

    def play(self, query: str = "") -> str:
        try:
            if self.plex.available():
                self.local.stop(quiet=True)
                r = self.plex.play(query)
                self.active = self.plex
                return r
            self.log("plexamp does not answer: playing from the local music folder")
        except (OSError, KeyError, StopIteration, ValueError, ET.ParseError) as e:
            self.log(f"plex: {type(e).__name__}: {e}; playing from the local music folder")
        self.active = self.local
        return self.local.play(query)

    def _on(self, name: str, *a, **k):
        target = self.active or self.local
        try:
            return getattr(target, name)(*a, **k)
        except (OSError, ET.ParseError) as e:
            self.log(f"music {name}: {type(e).__name__}: {e}")
            return "The music player did not answer."

    def next(self, auto: bool = False) -> str:
        return self._on("next", auto)

    def previous(self) -> str:
        return self._on("previous")

    def now(self) -> str:
        return self._on("now")

    def stop(self, quiet: bool = False) -> str:
        return self._on("stop", quiet)

    def is_playing(self) -> bool:
        return bool(self.active is not None and self._on("is_playing") is True) or self.local.is_playing()

    def duck(self) -> None:
        (self.active or self.local).duck()

    def unduck(self) -> None:
        (self.active or self.local).unduck()
