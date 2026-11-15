"""ROS-free: the wake matcher, task matching, the task runner's state machine with fakes, the brain's rules."""
import threading
import time
from pathlib import Path

from rd_nav.landmarks import LandmarkStore
from rd_voice.brain import Brain
from rd_voice.skills import ClockSkill, HomeAssistantSkill, MusicSkill, NewsSkill, WeatherSkill
from rd_voice.tasks import NoTaskRunner, TaskRunner, match_task, normalise
from rd_voice.wake import WakeMatcher

LANDMARKS = Path(__file__).resolve().parents[2] / "rd_nav" / "landmarks"


def test_wake_matcher():
    w = WakeMatcher(["robot disco"], 0.74)
    r = w.match("robot disco what time is it")
    assert r.woke and r.command == "what time is it"
    assert w.match("robot disco").woke and w.match("robot disco").command == ""
    assert w.match("hey robot disco, turn on the light").woke
    assert w.match("robot disco turn on the light").command == "turn on the light"
    assert not w.match("the weather is nice").woke


def test_normalise_and_match():
    store = LandmarkStore(LANDMARKS)
    tasks = store.tasks()
    assert normalise("Could you please pick up tigger now?") == "pick up tigger"
    lm, t = match_task("could you pick up tiger please", tasks)      # the command after the wake phrase, Whisper's spelling
    assert lm["id"] == "dining-room" and t["id"] == "pick-up-tigger"
    assert match_task("did you pick up tigger?", tasks) is None       # a question is not an order
    assert match_task("make coffee", tasks) is None


def test_task_runner_drives_then_runs():
    store = LandmarkStore(LANDMARKS)
    said, log = [], []
    gate = threading.Event()

    def navigate(lm):
        log.append(("nav", lm["id"]))
        return True, "arrived"

    def run_policy(lm, task):
        log.append(("policy", task["model"], task["task"]))
        gate.wait(5)
        return True, "done"
    r = TaskRunner(store, navigate, run_policy, said.append, log=lambda m: None)
    assert "pick up tigger" in r.names()[0]
    assert r.run("make coffee").startswith("I don't know how to make coffee")
    reply = r.run("pick up tigger")
    assert reply == "Heading to the Dining Room to pick up tigger."
    time.sleep(0.3)
    assert r.busy and "driving" not in r.status() and "doing pick up tigger" in r.status()
    assert r.run("pick up tigger").startswith("I'm busy")
    gate.set()
    time.sleep(0.3)
    assert not r.busy and r.last["ok"] and log == [("nav", "dining-room"), ("policy", "tigger_1_merged_smolvla_1", "Grab and hold tigger")]
    assert said == ["I'm at the Dining Room. Here goes: pick up tigger.", "Done with pick up tigger."]
    assert "last job was pick up tigger" in r.status()


def test_task_runner_stop_and_failure():
    store = LandmarkStore(LANDMARKS)
    said = []
    cancelled = threading.Event()

    def navigate(lm):
        cancelled.wait(5)
        return False, "cancelled"
    r = TaskRunner(store, navigate, lambda lm, t: (True, "done"), said.append, cancel=cancelled.set)
    r.run("pick up tigger")
    time.sleep(0.2)
    assert r.stop().startswith("Stopping on the way")
    time.sleep(0.3)
    assert not r.busy and r.last["reason"] == "cancelled on the way"
    r2 = TaskRunner(store, lambda lm: (False, "not localized"), lambda lm, t: (True, "done"), said.append)
    said.clear()
    r2.run("pick up tigger")
    time.sleep(0.3)
    assert said == ["I couldn't get to the Dining Room: not localized."] and not r2.last["ok"]


class FakeNav:
    def __init__(self):
        self.calls = []

    def go_to(self, place):
        self.calls.append(place)
        return f"Heading to the {place}."

    def stop(self, quiet=False):
        return "Stopping."

    def places(self):
        return ["Charger", "Dining Room", "Hallway"]

    def status(self):
        return "I'm not moving."


def test_brain_rules_without_an_llm():
    nav = FakeNav()
    store = LandmarkStore(LANDMARKS)
    tasks = TaskRunner(store, lambda lm: (True, "arrived"), lambda lm, t: (True, "done"), lambda s: None)
    b = Brain(WeatherSkill(), NewsSkill(), HomeAssistantSkill(), MusicSkill(music_dir="/nonexistent"), ClockSkill(),
              navigator=nav, tasks=tasks, llm_enabled=False, llm_backend="none")
    assert b.active_backend in ("rules", "none", "rules only") or not b.llm_available
    assert ":" in b.handle("what time is it")
    assert b.handle("go to the hallway") == "Heading to the Hallway." and nav.calls == ["Hallway"]
    assert b.handle("pick up tigger").startswith("Heading to the Dining Room")
    time.sleep(0.3)
    assert b.handle("stop") in ("Stopping.", "")
    reply = b.handle("how far is the moon")
    assert isinstance(reply, str) and reply
    assert NoTaskRunner().run("x").startswith("Arm jobs")


def test_a_task_prepares_the_policy_before_it_drives():
    """The inference server does not run at boot: the task runner asks for it as it sets off, ahead of the drive."""
    import threading
    order, done = [], threading.Event()
    store = type("S", (), {"load": lambda self: [], "tasks": lambda self: []})()
    r = TaskRunner(store, lambda lm: (order.append("navigate"), (True, "arrived"))[1],
                   lambda lm, t: (order.append("run"), (True, "done"))[1], lambda s: None,
                   on_done=lambda *a: done.set(), prepare=lambda lm, t: order.append("prepare"))
    r.start({"id": "left-bookshelf", "name": "Left Bookshelf"}, {"id": "pick-up-tigger", "name": "pick up tigger"})
    assert done.wait(5.0)
    assert order == ["prepare", "navigate", "run"]
    # a failing prepare costs time at the landmark, not the task
    order.clear(); done.clear()
    def boom(lm, t):
        raise RuntimeError("no supervisor")
    r2 = TaskRunner(store, lambda lm: (order.append("navigate"), (True, "arrived"))[1],
                    lambda lm, t: (order.append("run"), (True, "done"))[1], lambda s: None, on_done=lambda *a: done.set(), prepare=boom)
    r2.start({"id": "a", "name": "A"}, {"id": "b", "name": "b"})
    assert done.wait(5.0) and order == ["navigate", "run"]


def test_music_player_falls_back_to_gstreamer_without_ffplay():
    from rd_voice.skills import default_player
    assert default_player(lambda name: "/usr/bin/" + name)[0] == "ffplay"
    assert default_player(lambda name: "/usr/bin/gst-play-1.0" if name == "gst-play-1.0" else None) == ["gst-play-1.0", "--no-interactive", "-q"]
    assert default_player(lambda name: None)[0] == "ffplay"


def test_plexamp_skill_plays_from_the_plex_server_and_falls_back():
    """rd_voice/plexamp.py against a fake server and player: search, the play queue, the companion call, the volume
    for ducking, and the local player when Plexamp does not answer."""
    import json
    import urllib.parse
    from rd_voice.plexamp import MusicWithFallback, PlexampSkill
    calls, state = [], {"state": "stopped", "volume": 80, "up": True}

    def http(url, method="GET", headers=None, timeout=8.0):
        u = urllib.parse.urlparse(url); q = dict(urllib.parse.parse_qsl(u.query))
        calls.append((u.netloc, method, u.path, q))
        assert headers["X-Plex-Token"] == "tok"
        if u.netloc == "player:32500":
            if not state["up"]:
                raise OSError("refused")
            if u.path.endswith("playMedia"):
                state["state"] = "playing"
            elif u.path.endswith("/stop"):
                state["state"] = "stopped"
            elif u.path.endswith("setParameters"):
                state["volume"] = int(q["volume"])
            track = '<Track title="Taxman" grandparentTitle="The Beatles" parentTitle="Revolver"/>' if state["state"] == "playing" else ""
            return f'<MediaContainer><Timeline type="music" state="{state["state"]}" volume="{state["volume"]}">{track}</Timeline></MediaContainer>'.encode()
        if u.path == "/identity":
            return json.dumps({"MediaContainer": {"machineIdentifier": "abc"}}).encode()
        if u.path == "/library/sections":
            return json.dumps({"MediaContainer": {"Directory": [{"key": "1", "type": "movie"}, {"key": "3", "type": "artist"}]}}).encode()
        if u.path == "/hubs/search":
            hubs = [{"type": "artist", "Metadata": [{"type": "artist", "title": "The Beatles", "key": "/library/metadata/7/children"}]}] if q["query"] == "beatles" else []
            return json.dumps({"MediaContainer": {"Hub": hubs}}).encode()
        if u.path == "/playQueues":
            return json.dumps({"MediaContainer": {"playQueueID": 9, "Metadata": [{"key": "/library/metadata/55"}]}}).encode()
        raise AssertionError(url)

    class Local:
        def __init__(self): self.played, self.stopped = [], 0
        def play(self, q=""): self.played.append(q); return "local"
        def stop(self, quiet=False): self.stopped += 1; return ""
        def is_playing(self): return False
        def duck(self): pass
        def unduck(self): pass
    plex = PlexampSkill("http://server:32400", "http://player:32500", duck_level=0.25, token=lambda: "tok", http=http)
    local = Local()
    m = MusicWithFallback(plex, local)
    assert m.play("beatles") == "Playing The Beatles."
    pq = [c for c in calls if c[2] == "/playQueues"][-1]
    assert pq[1] == "POST" and pq[3]["uri"] == "server://abc/com.plexapp.plugins.library/library/metadata/7" and pq[3]["shuffle"] == "1"
    pm = [c for c in calls if c[2].endswith("playMedia")][-1][3]
    assert pm["address"] == "server" and pm["port"] == "32400" and pm["containerKey"].startswith("/playQueues/9") and pm["key"] == "/library/metadata/55"
    assert m.is_playing() and m.now() == "This is Taxman by The Beatles."
    m.duck(); assert state["volume"] == 20
    m.unduck(); assert state["volume"] == 80
    assert m.play("nobody at all") == "I couldn't find nobody at all in the music library."
    assert m.play("") == "Playing music." and [c for c in calls if c[2] == "/playQueues"][-1][3]["uri"].endswith("/library/sections/3/all?type=10")
    assert m.stop() == "Music stopped." and state["state"] == "stopped"
    state["up"] = False                                   # Plexamp gone: the local folder plays
    assert m.play("beatles") == "local" and local.played == ["beatles"]
