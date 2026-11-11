"""The voice assistant: ears (Whisper in its venv; the rig types on /voice/text_in) -> the wake phrase ->
the brain (rules, then Claude or Ollama with the skills as tools) -> the Say action. Navigation is
rd_nav's GoToLandmark (the places are the landmarks), landmark tasks are GoToLandmark + rd_vla's RunPolicy.

  sub  /voice/text_in     String   a transcript (the rig, the manager's text box)
  sub  /voice/run_task    String   "landmark_id/task_id" (the manager's Landmarks tab)
  sub  /speaking          Bool     the Say server: mute the ears while the robot talks
  sub  /face/person       String   who is in view (the brain's "person" context)
  pub  /voice/heard, /voice/command, /voice/reply, /voice/state   String
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import json
import os
import subprocess
import threading
import time
from typing import List, Optional

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from rd_bringup import config
from rd_interfaces.action import GoToLandmark, RunPolicy, Say
from rd_nav.landmarks import LandmarkStore

from .brain import Brain
from .skills import ClockSkill, HomeAssistantSkill, MusicSkill, NewsSkill, VolumeSkill, WeatherSkill, default_player
from .tasks import TaskRunner
from .wake import WakeMatcher


class ActionCall:
    """One blocking action call with a cancel handle (the task thread waits on it)."""

    def __init__(self, client, goal, timeout: float):
        self.client, self.goal, self.timeout = client, goal, timeout
        self.handle = None
        self.done = threading.Event()
        self.result = None
        self.status = None

    def run(self) -> tuple[bool, str]:
        if not self.client.wait_for_server(timeout_sec=5.0):
            return False, "the server is not running"
        fut = self.client.send_goal_async(self.goal)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > 10:
                return False, "no answer from the server"
            time.sleep(0.05)
        self.handle = fut.result()
        if not self.handle.accepted:
            return False, "the server is busy"
        res = self.handle.get_result_async()
        while not res.done():
            if time.monotonic() - t0 > self.timeout:
                self.cancel()
                return False, "it took too long"
            time.sleep(0.1)
        r = res.result()
        self.status, self.result = r.status, r.result
        if r.status == GoalStatus.STATUS_SUCCEEDED and getattr(r.result, "success", True):
            return True, getattr(r.result, "message", "done")
        return False, getattr(r.result, "message", f"status {r.status}")

    def cancel(self):
        if self.handle is not None:
            self.handle.cancel_goal_async()


class Navigator:
    """The brain's Navigator over GoToLandmark; places = the landmarks of the active map."""

    def __init__(self, node, store: LandmarkStore, timeout: float = 300.0):
        self.node, self.store, self.timeout = node, store, timeout
        self.client = ActionClient(node, GoToLandmark, "go_to_landmark", callback_group=node.cb)
        self._call: Optional[ActionCall] = None
        self._last = ""

    def places(self) -> List[str]:
        return [str(lm.get("name") or lm["id"]) for lm in self.store.all()]

    def go_to(self, place: str) -> str:
        lm = self.store.find(place)
        if lm is None:
            return f"I don't know where {place} is."
        self._last = str(lm.get("name") or lm["id"])
        threading.Thread(target=self._drive, args=(lm,), daemon=True).start()
        return f"Heading to the {self._last}."

    def _drive(self, lm):
        ok, why = self.go_to_landmark(lm)
        self.node.on_navigation_done(self._last if ok else None, why)

    def go_to_landmark(self, lm: dict) -> tuple[bool, str]:
        call = ActionCall(self.client, GoToLandmark.Goal(landmark=lm["id"]), self.timeout)
        self._call = call
        try:
            return call.run()
        finally:
            self._call = None

    def stop(self, quiet: bool = False) -> str:
        c = self._call
        if c is not None:
            c.cancel()
            return "Stopping."
        return "" if quiet else "I'm not moving."

    def status(self) -> str:
        return f"I'm driving to the {self._last}." if self._call is not None else "I'm not moving."


class VoiceNode(Node):
    def __init__(self):
        super().__init__("voice")
        dp = self.declare_parameter
        dp("ears_enabled", True)
        dp("ears_python", "~/.robot_face/voice-venv/bin/python")
        dp("asr_model", "base.en"); dp("asr_device", "cpu"); dp("asr_compute_type", "int8"); dp("asr_threads", 4)
        dp("audio_source", str(config.get("audio.mic", ""))); dp("vad_min_rms", 0.012); dp("vad_end_ms", 900)
        dp("wake_phrases", ["robot disco"]); dp("wake_threshold", 0.74); dp("follow_up_seconds", 6.0)
        dp("llm_enabled", True); dp("llm_model", "claude-opus-5-5"); dp("llm_backend", "auto")
        dp("ollama_url", f"http://{config.get('gpu.ip')}:{config.get('gpu.ports.ollama', 11434)},http://127.0.0.1:{config.get('robot.ports.ollama', 11434)}")
        dp("ollama_model", "qwen2.5:7b"); dp("persona", "Robot Disco"); dp("location_name", str(config.get("site.name", "")))
        dp("weather_latitude", float(config.get("site.weather.latitude", 0.0))); dp("weather_longitude", float(config.get("site.weather.longitude", 0.0)))
        dp("weather_place", str(config.get("site.weather.place", ""))); dp("units", "fahrenheit")
        dp("news_feeds", ["NPR=https://feeds.npr.org/1001/rss.xml", "BBC=https://feeds.bbci.co.uk/news/world/rss.xml"])
        dp("hass_url", ""); dp("hass_token", ""); dp("hass_aliases", [""])
        dp("music_dir", "~/Music"); dp("player_cmd", default_player())      # ffplay, or gst-play-1.0 where there is none
        dp("music_continuous", True); dp("music_duck", 0.6)
        dp("volume_sink", str(config.get("audio.sink", ""))); dp("volume_step", 10); dp("volume_max", 100)
        dp("landmarks_dir", ""); dp("task_nav_timeout", 300.0); dp("task_run_timeout", 600.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cb = ReentrantCallbackGroup()
        self.pub_heard = self.create_publisher(String, "/voice/heard", 10)
        self.pub_command = self.create_publisher(String, "/voice/command", 10)
        self.pub_reply = self.create_publisher(String, "/voice/reply", 10)
        self.pub_state = self.create_publisher(String, "/voice/state", 10)
        self.create_subscription(String, "/voice/text_in", self.on_text_in, 10, callback_group=self.cb)
        self.create_subscription(String, "/voice/run_task", self.on_run_task, 10, callback_group=self.cb)
        self.create_subscription(Bool, "/speaking", self.on_speaking, 10, callback_group=self.cb)
        self.create_subscription(String, "/face/person", self.on_person, 10, callback_group=self.cb)
        self.say_client = ActionClient(self, Say, "say", callback_group=self.cb)
        self.policy_client = ActionClient(self, RunPolicy, "run_policy", callback_group=self.cb)
        feeds = dict(item.split("=", 1) for item in (p("news_feeds") or []) if "=" in item)
        aliases = dict(item.split("=", 1) for item in (p("hass_aliases") or []) if "=" in item)
        self.weather = WeatherSkill(float(p("weather_latitude")), float(p("weather_longitude")), str(p("weather_place")), str(p("units")))
        self.news = NewsSkill({k.strip(): v.strip() for k, v in feeds.items()} or None)
        self.home = HomeAssistantSkill(str(p("hass_url")), str(p("hass_token")), {k.strip(): v.strip() for k, v in aliases.items()})
        self.music = MusicSkill(str(p("music_dir")), list(p("player_cmd")), sink=str(p("volume_sink")),
                                continuous=bool(p("music_continuous")), duck_level=float(p("music_duck")))
        self.clock = ClockSkill()
        self.volume = VolumeSkill(str(p("volume_sink")) or "@DEFAULT_SINK@", int(p("volume_step")), int(p("volume_max")), defer=self.after_speech)
        self.landmarks = LandmarkStore(p("landmarks_dir") or None)
        self.nav = Navigator(self, self.landmarks, float(p("task_nav_timeout")))
        self._prepare = self.create_client(Trigger, "/vla/prepare_server")
        self._policy_call: Optional[ActionCall] = None
        self.tasks = TaskRunner(self.landmarks, navigate=self.nav.go_to_landmark, run_policy=self.run_policy, say=self.say,
                                cancel=self.cancel_task, log=lambda m: self.get_logger().info(m), prepare=self.prepare_policy)
        self.brain = Brain(self.weather, self.news, self.home, self.music, self.clock, self.nav, volume=self.volume, tasks=self.tasks,
                           persona=str(p("persona")), location_name=str(p("location_name")), model=str(p("llm_model")),
                           llm_enabled=bool(p("llm_enabled")), llm_backend=str(p("llm_backend")), ollama_url=str(p("ollama_url")),
                           ollama_model=str(p("ollama_model")), log=lambda m: self.get_logger().info(m))
        self.wake = WakeMatcher(list(p("wake_phrases")), float(p("wake_threshold")))
        self.follow_up_seconds = float(p("follow_up_seconds"))
        self._follow_up_until = 0.0
        self._person: Optional[str] = None
        self._person_time = 0.0
        self._speaking = False
        self._after: List = []
        self._after_lock = threading.Lock()
        self._queue: List[str] = []
        self._queue_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._ears: Optional[subprocess.Popen] = None
        if bool(p("ears_enabled")):
            threading.Thread(target=self.ears_loop, daemon=True).start()
        self.set_state("idle")
        self.get_logger().info(f"brain: {self.brain.active_backend}; places: {', '.join(self.nav.places()) or 'none'}; "
                               f"tasks: {', '.join(self.tasks.names()) or 'none'}")

    # ---- ears (the Whisper worker in its venv)
    def ears_loop(self):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        python = os.path.expanduser(str(p("ears_python")))
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ears_worker.py")
        if not os.path.exists(python):
            self.get_logger().warn(f"ears disabled: no {python} (provision setup_voice.sh); use /voice/text_in")
            return
        args = [python, worker, "--model", str(p("asr_model")), "--device", str(p("asr_device")), "--compute-type", str(p("asr_compute_type")),
                "--threads", str(p("asr_threads")), "--min-rms", str(p("vad_min_rms")), "--end-ms", str(p("vad_end_ms"))]
        if p("audio_source"):
            args += ["--source", str(p("audio_source"))]
        while rclpy.ok():
            try:
                self._ears = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
                for line in self._ears.stdout:
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    self.on_ears_event(ev)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"ears: {e}")
            if not rclpy.ok():
                return
            time.sleep(3.0)

    def ears_cmd(self, cmd: dict):
        proc = self._ears
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write(json.dumps(cmd) + "\n")
                proc.stdin.flush()
            except OSError:
                pass

    def on_ears_event(self, ev: dict):
        kind = ev.get("event")
        if kind == "ready":
            self.get_logger().info(f"ears ready: {ev.get('model')} on {ev.get('device')} from {ev.get('source')}")
        elif kind == "utterance":
            text = str(ev.get("text", "")).strip()
            if text:
                self.pub_heard.publish(String(data=text))
                self.handle_transcript(text)
        elif kind == "error":
            self.get_logger().warn(f"ears: {ev.get('error')}")

    # ---- inputs
    def on_text_in(self, m: String):
        text = m.data.strip()
        if text:
            self.pub_heard.publish(String(data=text))
            self.handle_transcript(text)

    def on_run_task(self, m: String):
        if "/" in m.data:
            lm, t = m.data.strip().split("/", 1)
            self.say(self.tasks.run_ids(lm, t))
        else:
            self.say(self.tasks.run(m.data))

    def on_person(self, m: String):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        if d.get("name"):
            self._person, self._person_time = str(d["name"]), time.monotonic()

    def handle_transcript(self, text: str):
        now = time.monotonic()
        res = self.wake.match(text)
        if res.woke:
            command = res.command
            self.get_logger().info(f"wake ({res.matched!r}, {res.score:.2f}): {command!r}")
        elif now < self._follow_up_until:
            command = text
            self._follow_up_until = 0.0
        else:
            return
        if not command.strip():
            self._follow_up_until = now + self.follow_up_seconds
            self.set_state("listening")
            return
        self.pub_command.publish(String(data=command))
        with self._queue_lock:
            if len(self._queue) >= 2:
                return
            self._queue.append(command)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self.respond_loop, daemon=True)
                self._worker.start()

    def respond_loop(self):
        while True:
            with self._queue_lock:
                if not self._queue:
                    return
                command = self._queue.pop(0)
            try:
                self.set_state("thinking")
                person = self._person if (time.monotonic() - self._person_time) < 60.0 else None
                reply = self.brain.handle(command, person)
                self.pub_reply.publish(String(data=reply))
                self.say(reply)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"responding failed: {e}")
                self.say("Something went wrong.")
            finally:
                self.set_state("idle")

    # ---- actions
    def run_policy(self, lm: dict, task: dict) -> tuple[bool, str]:
        goal = RunPolicy.Goal(task=str(task.get("task") or task.get("name")), checkpoint=str(task.get("model", "")),
                              backend=str(task.get("backend", "")), max_seconds=float(task.get("seconds") or 120.0),
                              max_steps=int(task.get("steps") or 0), home_first=bool(task.get("home", True)),
                              record=bool(task.get("record", False)), landmark=str(lm.get("id", "")),
                              done_when=str(task.get("done_when", "") or ""), still_seconds=float(task.get("still") or 0.0))
        call = ActionCall(self.policy_client, goal, float(task.get("seconds") or 120.0) + float(self.get_parameter("task_run_timeout").value))
        self._policy_call = call
        try:
            ok, msg = call.run()
            return ok, msg
        finally:
            self._policy_call = None

    def prepare_policy(self, lm: dict, task: dict) -> None:
        """A landmark task is setting off: ask the policy runner to start its server now, so the model is loaded by
        the time the robot arrives (the server does not run at boot). Fire and forget."""
        if self._prepare.service_is_ready():
            self._prepare.call_async(Trigger.Request())
        else:
            self.get_logger().info("no /vla/prepare_server (rd_vla not running): the policy server starts when the run does")

    def cancel_task(self):
        self.nav.stop(quiet=True)
        c = self._policy_call
        if c is not None:
            c.cancel()

    def on_navigation_done(self, where: Optional[str], why: str = ""):
        if where:
            self.say("I'm at the charger." if where.lower() == "charger" else f"I've arrived at the {where}.")
        elif why and why not in ("cancelled",):
            self.say(f"I couldn't get there: {why}.")

    # ---- output
    def say(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        self.get_logger().info(f"say: {text!r}")
        self.ears_cmd({"cmd": "mute"})
        if self.say_client.wait_for_server(timeout_sec=1.0):
            self.say_client.send_goal_async(Say.Goal(text=text))
        else:
            self.get_logger().warn("no Say server")
            self.ears_cmd({"cmd": "unmute"})

    def on_speaking(self, m: Bool):
        self._speaking = bool(m.data)
        if m.data:
            self.ears_cmd({"cmd": "mute"})
        else:
            self.ears_cmd({"cmd": "unmute"})
            with self._after_lock:
                fns, self._after = self._after, []
            for fn in fns:
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    self.get_logger().warn(f"after-speech action failed: {e}")

    def after_speech(self, fn):
        with self._after_lock:
            self._after.append(fn)

    def set_state(self, state: str):
        self.pub_state.publish(String(data=state))

    def destroy_node(self):
        if self._ears is not None and self._ears.poll() is None:
            self.ears_cmd({"cmd": "quit"})
            try:
                self._ears.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._ears.kill()
        super().destroy_node()


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = VoiceNode()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
