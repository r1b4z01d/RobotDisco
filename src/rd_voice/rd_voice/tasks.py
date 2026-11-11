"""Landmark tasks: a policy run tied to the place it happens at ("Robot Disco, pick up tigger" = drive to the
dining room, run the tigger model). v1 spawned vla_action.py; 5.0 calls rd_nav's GoToLandmark and rd_vla's
RunPolicy through callables the node provides (`navigate(landmark) -> (ok, why)`, `run_policy(lm, task) ->
(ok, why)`, both blocking and cancellable through `cancel()`), so the matching and the state machine are
ROS-free and tested with fakes. The landmarks come from rd_nav.landmarks.LandmarkStore.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable, List, Optional, Tuple

_PREFIX = re.compile(r"^(?:(?:hey|ok|okay|please|so|now|then|and)[, ]+)*"
                     r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+|go\s+(?:and\s+)?|go\s+ahead\s+and\s+|i\s+want\s+you\s+to\s+|i\s+need\s+you\s+to\s+)?")
_QUESTION = re.compile(r"^(?:(?:hey|ok|okay|so|um|uh)[, ]+)*(?:did|have|had|do|does|are|were|was|is|what|where|when|why|how|who|which)\b")
_SUFFIX = re.compile(r"(?:\s+(?:please|for\s+me|now|right\s+now|again|would\s+you|will\s+you|ok|okay))*[.!?,\s]*$")


def normalise(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").lower().strip())
    t = _PREFIX.sub("", t)
    t = _SUFFIX.sub("", t)
    return t.strip()


def match_task(text: str, tasks: List[Tuple[dict, dict]], cutoff: float = 0.78) -> Optional[Tuple[dict, dict]]:
    """The task whose spoken name the utterance is (or contains); a question about a job is not an order."""
    import difflib
    q = normalise(text)
    if not q or not tasks or _QUESTION.match((text or "").lower().strip()):
        return None
    names = [(normalise(t["name"]), lm, t) for lm, t in tasks]
    names = [n for n in names if n[0]]
    hits = [n for n in names if n[0] == q or re.search(r"(?<![a-z])" + re.escape(n[0]) + r"(?![a-z])", q)]
    if hits:
        return max(hits, key=lambda n: len(n[0]))[1:]
    close = difflib.get_close_matches(q, [n[0] for n in names], n=1, cutoff=cutoff)
    if close:
        return next(n[1:] for n in names if n[0] == close[0])
    return None


class TaskRunner:
    def __init__(self, store, navigate: Callable[[dict], Tuple[bool, str]], run_policy: Callable[[dict, dict], Tuple[bool, str]],
                 say: Callable[[str], None], cancel: Callable[[], None] = lambda: None, log: Optional[Callable[[str], None]] = None,
                 on_done: Optional[Callable[[dict, dict, bool, str], None]] = None,
                 prepare: Optional[Callable[[dict, dict], None]] = None):
        self.store = store
        self.navigate, self.run_policy, self.say, self.cancel_fn = navigate, run_policy, say, cancel
        self._log = log or (lambda m: None)
        self.on_done = on_done
        self.prepare = prepare       # called as the task sets off: whatever the policy needs can warm up during the drive
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._current: Optional[Tuple[dict, dict]] = None
        self._phase = ""
        self._cancel = threading.Event()
        self.last: dict = {}

    def _tasks(self) -> List[Tuple[dict, dict]]:
        return [(lm, t) for lm, t in self.store.tasks() if t.get("name") and (t.get("model") or t.get("task"))]

    def names(self) -> List[str]:
        return [f"{t['name']} (at the {lm['name']})" for lm, t in self._tasks()]

    def match(self, text: str) -> Optional[Tuple[dict, dict]]:
        return match_task(text, self._tasks())

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> str:
        cur = self._current
        if not self.busy or cur is None:
            if self.last:
                return f"The last job was {self.last['task']} at the {self.last['landmark']}: {self.last['reason']}."
            return "I'm not working on anything."
        lm, t = cur
        return (f"I'm driving to the {lm['name']} to {t['name']}." if self._phase == "driving"
                else f"I'm at the {lm['name']} doing {t['name']}.")

    def run(self, text: str) -> str:
        hit = self.match(text)
        if hit is None:
            known = ", ".join(t["name"] for _lm, t in self._tasks())
            return f"I don't know how to {normalise(text) or 'do that'}. " + (f"I can {known}." if known else "No jobs are set up yet.")
        return self.start(*hit)

    def run_ids(self, landmark_id: str, task_id: str) -> str:
        for lm, t in self._tasks():
            if lm["id"] == landmark_id and t["id"] == task_id:
                return self.start(lm, t)
        return f"There is no task {task_id} at {landmark_id}."

    def start(self, lm: dict, task: dict) -> str:
        with self._lock:
            if self.busy and self._current is not None:
                clm, ct = self._current
                return f"I'm busy: {ct['name']} at the {clm['name']}. Say stop first."
            self._cancel.clear()
            self._current = (lm, task)
            self._phase = "driving"
            self._thread = threading.Thread(target=self._worker, args=(lm, task), daemon=True, name="landmark-task")
            self._thread.start()
        return f"Heading to the {lm['name']} to {task['name']}."

    def _worker(self, lm: dict, task: dict) -> None:
        ok, reason = False, "unknown"
        try:
            self._log(f"task {task['name']!r}: driving to {lm['name']!r}")
            if self.prepare is not None:
                try:
                    self.prepare(lm, task)
                except Exception as e:  # noqa: BLE001  a cold server only costs time at the landmark
                    self._log(f"prepare failed: {e}")
            ok, why = self.navigate(lm)
            if self._cancel.is_set():
                ok, reason = False, "cancelled on the way"
                return
            if not ok:
                reason = f"could not get there ({why})"
                self.say(f"I couldn't get to the {lm['name']}: {why}.")
                return
            self._phase = "running"
            self.say(f"I'm at the {lm['name']}. Here goes: {task['name']}.")
            ok, reason = self.run_policy(lm, task)
            if self._cancel.is_set():
                ok, reason = False, "stopped"
                self.say(f"Stopped {task['name']}.")
            elif ok:
                self.say(f"Done with {task['name']}.")
            else:
                self.say(f"I couldn't finish {task['name']}: {reason}.")
        except Exception as e:  # noqa: BLE001
            ok, reason = False, f"{type(e).__name__}: {e}"
            self._log(f"task {task['name']!r} crashed: {reason}")
            self.say(f"Something went wrong with {task['name']}.")
        finally:
            self.last = {"landmark": lm["name"], "task": task["name"], "ok": ok, "reason": reason, "when": time.time()}
            self._log(f"task {task['name']!r} at {lm['name']!r}: {'ok' if ok else 'failed'}: {reason}")
            self._phase, self._current = "", None
            if self.on_done:
                try:
                    self.on_done(lm, task, ok, reason)
                except Exception:  # noqa: BLE001
                    pass

    def stop(self) -> str:
        if not self.busy or self._current is None:
            return ""
        lm, t = self._current
        self._cancel.set()
        phase = self._phase
        self.cancel_fn()
        return (f"Stopping on the way to the {lm['name']}." if phase == "driving" else f"Stopping {t['name']}; the arm is going home.")


class NoTaskRunner:
    busy = False

    def names(self):
        return []

    def match(self, text):
        return None

    def run(self, text):
        return "Arm jobs are not set up."

    def start(self, lm, task):
        return "Arm jobs are not set up."

    def stop(self):
        return ""

    def status(self):
        return "I'm not working on anything."
