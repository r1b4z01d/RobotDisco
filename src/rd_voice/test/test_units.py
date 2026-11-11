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
