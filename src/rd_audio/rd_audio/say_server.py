"""The Say action server: the one voice of the robot (v1 had the TTS engines in both the face and the voice
nodes). Text -> WAV (Kokoro in its venv, espeak as the fallback, the fake voice in the rig) -> playback on
the speaker sink, with the samples streamed to /mouth/audio in time with the audio so the face's waveform
tracks what comes out of the speaker.

  action /say            rd_interfaces/action/Say   (text, voice, speed, interrupt) -> (success, message, seconds)
  sub    /say_text       std_msgs/String            fire-and-forget (the manager's Say box, scripts)
  pub    /speaking       std_msgs/Bool              true while audio plays (the voice node mutes its ears)
  pub    /mouth/audio    std_msgs/Float32MultiArray 40 ms chunks in [-1, 1]
  pub    /audio/volume   std_msgs/Float32           the sink's volume 0..1 (1 Hz); param `volume` sets it
  srv    /audio/mute     std_srvs/SetBool

One sentence at a time; a goal with interrupt=true stops the current one. `player` = aplay (robot, pinned to
the sink through PULSE_SINK) | none (the rig: no audio device, the timing is kept).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import SetBool

from rd_bringup import config
from rd_interfaces.action import Say

from . import audio_sink
from .audio_utils import chunk_level, iter_chunks, load_wav_mono, prepend_silence, write_wav_mono16
from .tts_engines import EspeakTTS, FakeTTS, KokoroTTS


class SayServer(Node):
    def __init__(self):
        super().__init__("say_server")
        dp = self.declare_parameter
        dp("engine", "auto")                  # auto | kokoro | espeak | fake
        dp("voice", str(config.get("audio.tts_voice", "af_heart")))
        dp("speed", 1.0)
        dp("kokoro_python", "~/.robot_face/kokoro-venv/bin/python")
        dp("kokoro_model_dir", "~/.robot_face/kokoro")
        dp("player", "aplay")                 # aplay | none
        dp("sink", str(config.get("audio.sink", "")))
        dp("sink_watchdog", True)
        dp("lead_silence", 1.5)               # seconds of faint tone before each utterance (the USB adapter wakes up)
        dp("lead_level", 0.01)
        dp("mouth_gain", 1.0)
        dp("volume", -1.0)                    # 0..1 sets the sink; -1 = leave it
        self.cb = ReentrantCallbackGroup()
        self.speaking_pub = self.create_publisher(Bool, "/speaking", 10)
        self.mouth_pub = self.create_publisher(Float32MultiArray, "/mouth/audio", 10)
        self.volume_pub = self.create_publisher(Float32, "/audio/volume", 10)
        self.create_subscription(String, "/say_text", self.on_text, 10, callback_group=self.cb)
        self.create_service(SetBool, "/audio/mute", self.on_mute, callback_group=self.cb)
        self._lock = threading.Lock()
        self._current = None          # the running goal handle
        self._current_abort = None    # its interrupt flag (set by a later goal with interrupt=true)
        self._proc = None
        self.server = ActionServer(self, Say, "say", execute_callback=self.execute, goal_callback=self.on_goal,
                                   cancel_callback=lambda h: CancelResponse.ACCEPT, callback_group=self.cb)
        self.engine = None
        self.engine_name = "none"
        threading.Thread(target=self._init_engine, daemon=True).start()
        sink = self.get_parameter("sink").value
        self.watchdog = None
        if sink and self.get_parameter("sink_watchdog").value and self.get_parameter("player").value != "none":
            self.watchdog = audio_sink.SinkWatchdog(sink, log=lambda m: self.get_logger().info(m))
            self.watchdog.start()
        self.add_on_set_parameters_callback(self.on_params)
        self.create_timer(1.0, self.publish_volume, callback_group=self.cb)

    # ---- engines
    def _init_engine(self):
        pref = self.get_parameter("engine").value
        log = lambda m: self.get_logger().info(m)  # noqa: E731
        if pref in ("auto", "kokoro"):
            k = KokoroTTS(self.get_parameter("kokoro_python").value, self.get_parameter("kokoro_model_dir").value,
                          voice=self.get_parameter("voice").value, speed=float(self.get_parameter("speed").value), log=log)
            if k.available() and k.start():
                self.engine, self.engine_name = k, "kokoro"
                self.get_logger().info(f"Kokoro ready in {k.load_seconds:.1f}s ({len(k.voices)} voices)")
                return
            if pref == "kokoro":
                self.get_logger().warn("Kokoro unavailable; falling back to espeak")
        if pref == "fake":
            self.engine, self.engine_name = FakeTTS(), "fake"
            return
        e = EspeakTTS()
        self.engine, self.engine_name = e, "espeak"
        self.get_logger().info("using espeak")

    def on_params(self, params):
        for p in params:
            if p.name == "volume" and 0.0 <= float(p.value) <= 1.0 and self.get_parameter("sink").value:
                audio_sink.set_volume(self.get_parameter("sink").value, float(p.value) * 100.0)
        return SetParametersResult(successful=True)

    def publish_volume(self):
        sink = self.get_parameter("sink").value
        if not sink or self.get_parameter("player").value == "none":
            return
        v = audio_sink.get_volume(sink)
        if v and v.get("volume") is not None:
            self.volume_pub.publish(Float32(data=float(v["volume"]) / 100.0))

    def on_mute(self, req, res):
        sink = self.get_parameter("sink").value
        if sink and self.get_parameter("player").value != "none":
            audio_sink.set_mute(sink, bool(req.data))
        res.success, res.message = True, "muted" if req.data else "unmuted"
        return res

    # ---- goals
    def on_goal(self, goal):
        with self._lock:
            if self._current is not None and not goal.interrupt:
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_text(self, m: String):
        text = m.data.strip()
        if text:
            threading.Thread(target=self.speak, args=(text, None, None), daemon=True).start()

    def execute(self, handle):
        goal = handle.request
        abort = threading.Event()
        with self._lock:
            if self._current is not None and self._current_abort is not None:
                self._current_abort.set()              # interrupt=true: stop the current one
                proc = self._proc
                if proc is not None and proc.poll() is None:
                    proc.terminate()
            self._current, self._current_abort = handle, abort
        res = Say.Result()
        t0 = time.monotonic()
        try:
            ok, msg = self.speak(goal.text, goal.voice or None, goal.speed or None,
                                 cancelled=lambda: handle.is_cancel_requested or abort.is_set(),
                                 interrupted=abort.is_set,
                                 progress=lambda f: handle.publish_feedback(Say.Feedback(progress=float(f))))
        finally:
            with self._lock:
                if self._current is handle:
                    self._current, self._current_abort = None, None
        res.success, res.message, res.seconds = ok, msg, time.monotonic() - t0
        if handle.is_cancel_requested:
            handle.canceled()
        elif ok:
            handle.succeed()
        else:
            handle.abort()
        return res

    # ---- the voice
    def speak(self, text: str, voice=None, speed=None, cancelled=lambda: False, interrupted=lambda: False,
              progress=lambda f: None) -> tuple[bool, str]:
        text = (text or "").strip()
        if not text:
            return False, "nothing to say"
        for _ in range(100):
            if self.engine is not None:
                break
            time.sleep(0.1)
        if self.engine is None:
            return False, "tts_failed: no engine"
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            if voice and hasattr(self.engine, "voice"):
                self.engine.voice = voice
            if speed and hasattr(self.engine, "speed"):
                self.engine.speed = float(speed)
            if not self.engine.synthesize(text, path):
                return False, "tts_failed"
            audio, rate = load_wav_mono(path)
            lead = float(self.get_parameter("lead_silence").value)
            if lead > 0 and self.get_parameter("player").value != "none":
                audio = prepend_silence(audio, rate, lead, level=float(self.get_parameter("lead_level").value))
                write_wav_mono16(path, audio, rate)
            self.speaking_pub.publish(Bool(data=True))
            try:
                return self._play(path, audio, rate, cancelled, interrupted, progress)
            finally:
                self.speaking_pub.publish(Bool(data=False))
        except Exception as e:  # noqa: BLE001
            if rclpy.ok():
                self.get_logger().error(f"say failed: {e}")
            return False, f"tts_failed: {e}"
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _play(self, path, audio, rate, cancelled, interrupted, progress) -> tuple[bool, str]:
        player = self.get_parameter("player").value
        proc = None
        if player == "aplay":
            proc = subprocess.Popen(["aplay", "-q", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    env=audio_sink.player_env(self.get_parameter("sink").value))
            self._proc = proc
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 1e-3:
            audio = audio / peak
        total = audio.size / rate
        gain = float(self.get_parameter("mouth_gain").value)
        t0 = time.monotonic()
        for chunk, end_t in iter_chunks(audio, rate, 0.04):
            if cancelled():
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                return False, "interrupted" if interrupted() else "cancelled"
            data, level = chunk_level(chunk, gain)
            if data.size:
                self.mouth_pub.publish(Float32MultiArray(data=[float(v) for v in data]))
            progress(min(1.0, end_t / max(total, 1e-6)))
            delay = t0 + end_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if proc is not None and proc.poll() is not None and end_t < total - 0.1:
                return False, "no_sink: the player exited early"
        if proc is not None:
            proc.wait()
            self._proc = None
        return True, "done"


def main():
    rclpy.init()
    node = SayServer()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.watchdog is not None:
            node.watchdog.stop()
        node.destroy_node()
        rclpy.try_shutdown()
