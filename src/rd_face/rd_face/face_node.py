"""The face: eyes that track whoever is in view, a mouth that moves with the speech, idle gestures, face
recognition with a learned gallery, and the greeting (hello by name, a wave through rd_arm, a wink for someone
back within the hour). v1's combined_face_hand_node without the TTS (rd_audio's Say), the wave (rd_arm's
Wave) and the HTTP Folks page (rd_manager; the data is on topics here).

  sub  <image_topic>               rgb8 Image (the D435 colour stream)
  sub  /mouth/audio, /mouth/level  from the Say server
  pub  /face/person   String json  {id, name, score, new, box} whenever someone is identified
  pub  /face/people   String json  the gallery (1 Hz and on change): [{id, name, seen_count, last_seen, last_greeted, thumb}]
  pub  /face/image    rgb8 Image   the rendered face (headless renderer; off with the GL window)
  sub  /face/set_person_name  String   "Name" (the person in view) or "person_3=Name"
  sub  /face/forget_person    String   id, name, or "all"
  sub  /face/reset_greeting   String   id
  sub  /face/debug_person     String json {id, name, score} - the rig's stand-in for a recognised face
  srv  /face/wink, /face/list_people   std_srvs/Trigger
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String
from std_srvs.srv import Trigger

from rd_interfaces.action import Say, Wave

from .face_utils import pick_face, schedule_blink
from .mouth_gestures import gesture_duration, gesture_frame, pick_gesture


def image_to_bgr(m: Image) -> Optional[np.ndarray]:
    data = np.frombuffer(bytes(m.data), dtype=np.uint8)
    if m.encoding in ("rgb8", "bgr8"):
        a = data.reshape(m.height, m.step // 3, 3)[:, :m.width, :]
        return np.ascontiguousarray(a[..., ::-1] if m.encoding == "rgb8" else a)
    if m.encoding == "mono8":
        a = data.reshape(m.height, m.step)[:, :m.width]
        return np.ascontiguousarray(np.repeat(a[..., None], 3, axis=2))
    return None


class FaceNode(Node):
    def __init__(self):
        super().__init__("face")
        dp = self.declare_parameter
        dp("image_topic", "/camera/camera/color/image_raw")
        dp("renderer", "gl")                 # gl | headless | none
        dp("render_fps", 30.0)
        dp("publish_image", True)            # headless: publish /face/image
        dp("width", 1080); dp("height", 1900); dp("fullscreen", True)
        dp("camera_timeout", 3.0); dp("face_stick_seconds", 1.5)
        dp("blink_min", 2.5); dp("blink_max", 15.5); dp("blink_duration", 0.14); dp("blink_closed_hold", 0.05)
        dp("gaze_speed", 0.3); dp("gaze_idle_speed", 0.01)
        dp("idle_scan_delay", 5.0); dp("idle_scan_period", 20.0)
        dp("mouth_enabled", True); dp("mouth_gain", 1.0); dp("mouth_attack", 0.6); dp("mouth_release", 0.12)
        dp("mouth_timeout", 0.4); dp("mouth_smile", 25.0); dp("mouth_smile_happy", 70.0)
        dp("mouth_gestures", True); dp("mouth_gesture_min_interval", 4.0); dp("mouth_gesture_max_interval", 12.0)
        dp("greeting_enabled", True); dp("greeting_text", "hello"); dp("greeting_known_text", "hello {name}")
        dp("greeting_late_text", "oh, hi {name}"); dp("greeting_wait", 2.0); dp("greeting_name_window", 10.0)
        dp("greeting_happy_seconds", 30.0); dp("greeting_cooldown", 3600.0)
        dp("wave_enabled", True); dp("wave_known_only", True); dp("wave_cooldown", 20.0)
        dp("wink_eye", "right"); dp("wink_duration", 0.7); dp("wink_depth", 0.9)
        dp("face_id_enabled", True); dp("face_id_backend", "auto"); dp("face_id_python", "~/.robot_face/faceid-venv/bin/python")
        dp("face_id_models_dir", ""); dp("face_id_gallery", "~/.robot_face/people.json"); dp("face_thumb_dir", "~/.robot_face/faces")
        dp("face_id_min_period", 0.5); dp("face_id_match_threshold", 0.0); dp("face_id_auto_enroll", True)
        dp("face_id_min_face_px", 40); dp("face_id_enroll_min_face_px", 64); dp("face_id_enroll_frames", 6)
        dp("face_id_gpu", True); dp("face_id_det_size", 640); dp("face_id_threads", 2)
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.p = p
        self.cb = ReentrantCallbackGroup()
        # outputs
        self.person_pub = self.create_publisher(String, "/face/person", 10)
        self.people_pub = self.create_publisher(String, "/face/people", 10)
        self.image_pub = self.create_publisher(Image, "/face/image", QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.say = ActionClient(self, Say, "say", callback_group=self.cb)
        self.wave = ActionClient(self, Wave, "wave", callback_group=self.cb)
        # inputs
        self.create_subscription(Image, p("image_topic"), self.on_image, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT), callback_group=self.cb)
        self.create_subscription(Float32MultiArray, "/mouth/audio", self.on_mouth_audio, 20, callback_group=self.cb)
        self.create_subscription(Float32, "/mouth/level", self.on_mouth_level, 10, callback_group=self.cb)
        self.create_subscription(String, "/face/set_person_name", self.on_set_name, 10, callback_group=self.cb)
        self.create_subscription(String, "/face/forget_person", self.on_forget, 10, callback_group=self.cb)
        self.create_subscription(String, "/face/reset_greeting", self.on_reset_greeting, 10, callback_group=self.cb)
        self.create_subscription(String, "/face/debug_person", self.on_debug_person, 10, callback_group=self.cb)
        self.create_service(Trigger, "/face/wink", self.on_wink_srv, callback_group=self.cb)
        self.create_service(Trigger, "/face/list_people", self.on_list_people, callback_group=self.cb)
        # state
        self.lock = threading.Lock()
        self.camera_w, self.camera_h = 1280, 720
        self.last_frame_mono = 0.0
        self.tracked_face = None
        self.last_seen = 0.0
        self.track_hint = None
        self.current_person = None       # the face_id Match (or a debug stand-in)
        self.last_person_time = 0.0
        self.last_person_label = None
        self.happy_until = 0.0
        self.greeting_due = None
        self.generic_greeted_at = None
        self.last_wave = 0.0
        self.smoothed_center = None
        self.blink_start = None
        self.next_blink = schedule_blink(float(p("blink_min")), float(p("blink_max")))
        self.wink_start = None
        self.wink_eye_index = 0 if str(p("wink_eye")).lower().startswith("l") else 1
        self.mouth_level = 0.0
        self.mouth_target = 0.0
        self.mouth_samples = None
        self.mouth_last_input = 0.0
        self.mouth_smile_current = float(p("mouth_smile"))
        self.gesture_active = None
        self.gesture_next = time.monotonic() + random.uniform(float(p("mouth_gesture_min_interval")), float(p("mouth_gesture_max_interval")))
        self.start_mono = time.monotonic()
        self.camera_warned = False
        self.said = []                   # what was said (the last few), for the log and the tests
        # face id
        self.face_id = None
        self._vision_job = None
        self._vision_result = None
        self._last_embed = 0.0
        if p("face_id_enabled"):
            self._init_face_id()
            threading.Thread(target=self._vision_loop, daemon=True, name="face_vision").start()
        # renderer
        self.renderer = None
        self._init_renderer()
        self.render_dt = 1.0 / float(p("render_fps"))
        self.create_timer(self.render_dt, self.render_tick, callback_group=self.cb)
        self.create_timer(1.0, self.publish_people, callback_group=self.cb)
        self.get_logger().info(f"renderer {p('renderer')}, face id {'on' if self.face_id else 'off'}, image {p('image_topic')}")

    # ---- setup
    def _init_face_id(self):
        from .face_id import FaceIdentifier, InsightFaceBackend
        p = self.p
        backend = p("face_id_backend")
        python = os.path.expanduser(p("face_id_python"))
        if backend == "auto":
            backend = "insightface" if InsightFaceBackend.available(python) else "opencv"
        models = p("face_id_models_dir") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        try:
            from ament_index_python.packages import get_package_share_directory
            share_models = os.path.join(get_package_share_directory("rd_face"), "models")
            if not p("face_id_models_dir") and os.path.isdir(share_models):
                models = share_models
        except Exception:  # noqa: BLE001
            pass
        try:
            self.face_id = FaceIdentifier.create(
                backend, models, os.path.expanduser(p("face_id_gallery")), faceid_python=python, gpu=bool(p("face_id_gpu")),
                det_size=int(p("face_id_det_size")), threads=int(p("face_id_threads")), log=lambda m: self.get_logger().info(m),
                thumb_dir=os.path.expanduser(p("face_thumb_dir")), match_threshold=float(p("face_id_match_threshold")) or None,
                auto_enroll=bool(p("face_id_auto_enroll")), min_face_px=int(p("face_id_min_face_px")),
                enroll_min_face_px=int(p("face_id_enroll_min_face_px")), enroll_frames=int(p("face_id_enroll_frames")))
            self.get_logger().info(f"face id: {backend}, {len(self.face_id.people())} people known")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"face id unavailable: {e}")
            self.face_id = None

    def _init_renderer(self):
        p = self.p
        mode = p("renderer")
        if mode == "gl":
            try:
                from .gl_eye_renderer import GlEyeRenderer
                self.renderer = GlEyeRenderer(width=int(p("width")), height=int(p("height")), fullscreen=bool(p("fullscreen")),
                                              mouth_enabled=bool(p("mouth_enabled")), log=self.get_logger())
                return
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"GL renderer failed ({e}); headless")
                mode = "headless"
        if mode == "headless":
            from .headless_renderer import HeadlessRenderer
            self.renderer = HeadlessRenderer(width=int(p("width")) // 2, height=int(p("height")) // 2, mouth_enabled=bool(p("mouth_enabled")))

    # ---- camera + vision
    def on_image(self, m: Image):
        frame = image_to_bgr(m)
        if frame is None:
            return
        now = time.monotonic()
        self.camera_h, self.camera_w = frame.shape[:2]
        if self.last_frame_mono == 0.0 or self.camera_warned:
            self.get_logger().info(f"camera frames on {self.p('image_topic')} ({self.camera_w}x{self.camera_h})")
            self.camera_warned = False
        self.last_frame_mono = now
        if self.face_id is None:
            return
        with self.lock:
            self._vision_job = (np.ascontiguousarray(frame), now)
            result, self._vision_result = self._vision_result, None
        if result is None:
            return
        faces, match = result
        boxes = [f.box for f in faces if f.score >= self.face_id.min_detect_score and f.box[2] >= self.face_id.min_face_px]
        selected = pick_face(boxes, self.tracked_face)
        if selected is not None:
            self.on_person_seen(selected, now)
            self.track_hint = (selected[0] + selected[2] / 2.0, selected[1] + selected[3] / 2.0)
        if match is not None:
            self.apply_match(match, now)

    def _vision_loop(self):
        while rclpy.ok():
            with self.lock:
                job, self._vision_job = self._vision_job, None
            if job is None:
                time.sleep(0.005)
                continue
            frame, t = job
            now = time.monotonic()
            do_embed = now - self._last_embed >= float(self.p("face_id_min_period"))
            faces, match = [], None
            try:
                faces = self.face_id.backend.analyze(frame, embed=do_embed)
                if do_embed and faces:
                    self._last_embed = now
                    match = self.face_id.identify(frame, near=self.track_hint, faces=faces)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"face tracking failed: {e}", throttle_duration_sec=5.0)
            with self.lock:
                self._vision_result = (faces, match)

    def on_person_seen(self, box, now):
        if self.tracked_face is None:
            self.happy_until = now + float(self.p("greeting_happy_seconds"))
            self.current_person = None
            self.generic_greeted_at = None
            if self.p("greeting_enabled"):
                self.greeting_due = now + (float(self.p("greeting_wait")) if self.face_id else 0.0)
        self.tracked_face = box
        self.last_seen = now

    def apply_match(self, match, now):
        self.current_person = match
        if match.known:
            self.last_person_time = now
            label = match.label
            if getattr(match, "is_new", False):
                self.get_logger().info(f"new person enrolled as {match.person_id}")
            elif label != self.last_person_label:
                self.get_logger().info(f"recognised {label} (score {match.score:.2f})")
            self.last_person_label = label
        self.person_pub.publish(String(data=json.dumps({"id": match.person_id, "name": match.name, "score": round(float(match.score), 3),
                                                        "new": bool(getattr(match, "is_new", False)), "box": list(match.box)})))
        self.publish_people()

    def on_debug_person(self, m: String):
        """The rig's recognised face: {id, name, score, box?}. Marks the person as seen and identified."""
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        now = time.monotonic()
        box = tuple(d.get("box") or (self.camera_w // 2 - 60, self.camera_h // 2 - 80, 120, 160))
        self.on_person_seen(box, now)
        if d.get("id"):
            class _M:
                pass
            mm = _M()
            mm.person_id, mm.name, mm.score, mm.box, mm.known, mm.is_new = d["id"], d.get("name"), float(d.get("score", 0.9)), box, True, bool(d.get("new", False))
            mm.label = mm.name or mm.person_id
            self.apply_match(mm, now)

    # ---- people (the Folks data)
    def people_json(self) -> list:
        if self.face_id is None:
            return []
        out = []
        cooldown = float(self.p("greeting_cooldown"))
        for pp in self.face_id.people():
            lg = float(getattr(pp, "last_greeted", 0.0) or 0.0)
            out.append({"id": pp.person_id, "name": pp.name, "seen_count": int(getattr(pp, "seen_count", 0)),
                        "views": len(getattr(pp, "embeddings", []) or []), "first_seen": getattr(pp, "first_seen", 0.0),
                        "last_seen": getattr(pp, "last_seen", 0.0), "last_greeted": lg,
                        "next_hello_in": max(0.0, lg + cooldown - time.time()) if lg else 0.0,
                        "thumb": self.face_id.gallery.thumb_path(pp.person_id) if hasattr(self.face_id, "gallery") else None})
        return out

    def publish_people(self):
        self.people_pub.publish(String(data=json.dumps({"people": self.people_json(), "live": self.live_json()})))

    def live_json(self) -> dict:
        m = self.current_person
        if m is None or not m.known or time.monotonic() - self.last_person_time > 10.0:
            return {}
        return {"id": m.person_id, "name": m.name, "score": round(float(m.score), 3)}

    def on_set_name(self, m: String):
        if self.face_id is None:
            return
        text = m.data.strip()
        if "=" in text:
            target, name = (s.strip() for s in text.split("=", 1))
        else:
            name = text
            if not (self.current_person is not None and self.current_person.known and time.monotonic() - self.last_person_time < 10.0):
                self.get_logger().warn("no person in view to name; use person_N=Name")
                return
            target = self.current_person.person_id
        person = self.face_id.name_person(target, name)
        if person is None:
            self.get_logger().warn(f"unknown person {target!r}")
        else:
            self.get_logger().info(f"{person.person_id} is now {person.name!r}")
            self.last_person_label, self.current_person = None, None
        self.publish_people()

    def on_forget(self, m: String):
        if self.face_id is None:
            return
        who = m.data.strip()
        if who.lower() == "all":
            for pp in list(self.face_id.people()):
                self.face_id.forget(pp.person_id)
        elif not self.face_id.forget(who):
            self.get_logger().warn(f"unknown person {who!r}")
        self.current_person = None
        self.publish_people()

    def on_reset_greeting(self, m: String):
        if self.face_id is not None and hasattr(self.face_id, "gallery"):
            self.face_id.gallery.reset_greeting(m.data.strip())
            self.publish_people()

    def on_list_people(self, req, res):
        if self.face_id is None:
            res.success, res.message = False, "face id is disabled"
            return res
        lines = self.face_id.gallery.summary() if hasattr(self.face_id, "gallery") else []
        res.success, res.message = True, "\n".join(lines) if lines else "no people learned yet"
        return res

    # ---- greeting
    def greeted_recently(self, pid) -> bool:
        if self.face_id is None or not hasattr(self.face_id, "gallery"):
            return pid in self._debug_greeted and time.time() - self._debug_greeted[pid] < float(self.p("greeting_cooldown"))
        person = self.face_id.gallery.find(pid)
        lg = float(getattr(person, "last_greeted", 0.0) or 0.0) if person else 0.0
        return bool(lg) and time.time() - lg < float(self.p("greeting_cooldown"))

    _debug_greeted: dict = {}

    def mark_greeted(self, pid):
        if pid is None:
            return
        if self.face_id is not None and hasattr(self.face_id, "gallery"):
            try:
                self.face_id.gallery.mark_greeted(pid)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"could not record the greeting: {e}")
        else:
            self._debug_greeted[pid] = time.time()

    def speak(self, text: str):
        self.said.append(text)
        self.said = self.said[-20:]
        self.get_logger().info(f"say: {text!r}")
        if self.say.wait_for_server(timeout_sec=0.5):
            self.say.send_goal_async(Say.Goal(text=text, interrupt=False))
        else:
            self.get_logger().warn("no Say server; silent")

    def do_wave(self, reason: str) -> str:
        if not self.p("wave_enabled"):
            return "wave disabled"
        if time.monotonic() - self.last_wave < float(self.p("wave_cooldown")):
            return "wave cooldown"
        if not self.wave.wait_for_server(timeout_sec=0.5):
            return "no wave server"
        self.last_wave = time.monotonic()
        self.get_logger().info(f"wave ({reason})")
        self.wave.send_goal_async(Wave.Goal(motion="wave", return_to="home", speed_scale=1.0, wiggle_fingers=True))
        return "waving"

    def wink(self):
        if self.wink_start is None:
            self.wink_start = time.monotonic()

    def on_wink_srv(self, req, res):
        self.wink()
        res.success, res.message = True, "winking"
        return res

    def person_name(self):
        m = self.current_person
        return m.name if (m is not None and m.known and m.name) else None

    def greeting_step(self, now):
        p = self.p
        name = self.person_name()
        pid = self.current_person.person_id if (self.current_person is not None and self.current_person.known) else None
        if self.greeting_due is not None:
            if pid is not None and self.greeted_recently(pid):
                self.greeting_due = None
                self.wink()
            elif name is not None:
                self.speak(str(p("greeting_known_text")).format(name=name))
                self.greeting_due = None
                self.mark_greeted(pid)
                self.do_wave(f"recognised {name}")
            elif now >= self.greeting_due:
                self.speak(str(p("greeting_text")))
                self.greeting_due = None
                self.generic_greeted_at = now
                if pid is not None:
                    self.mark_greeted(pid)
                if not p("wave_known_only"):
                    self.do_wave("someone is here")
        elif self.generic_greeted_at is not None:
            if pid is not None:
                recent = self.greeted_recently(pid)
                self.mark_greeted(pid)
                self.generic_greeted_at = None
                if name is not None and not recent:
                    self.speak(str(p("greeting_late_text")).format(name=name))
                    self.do_wave(f"recognised {name}")
                elif recent:
                    self.wink()
            elif now - self.generic_greeted_at > float(p("greeting_name_window")):
                self.generic_greeted_at = None

    # ---- mouth
    def on_mouth_audio(self, m: Float32MultiArray):
        a = np.asarray(m.data, dtype=np.float32)
        if a.size == 0:
            return
        gain = float(self.p("mouth_gain"))
        self.mouth_target = min(1.0, float(np.abs(a).max()) * gain)
        self.mouth_samples = a * gain
        self.mouth_last_input = time.monotonic()

    def on_mouth_level(self, m: Float32):
        self.mouth_target = min(max(float(m.data) * float(self.p("mouth_gain")), 0.0), 1.0)
        self.mouth_last_input = time.monotonic()

    def idle_gesture(self, now, idle):
        p = self.p
        if not p("mouth_gestures"):
            return (0.0, 0.0, 0.0, None, 0.0)
        if not idle:
            if self.gesture_active is not None:
                self.gesture_active = None
                self.gesture_next = now + random.uniform(float(p("mouth_gesture_min_interval")), float(p("mouth_gesture_max_interval")))
            return (0.0, 0.0, 0.0, None, 0.0)
        if self.gesture_active is None:
            if now < self.gesture_next:
                return (0.0, 0.0, 0.0, None, 0.0)
            name, side = pick_gesture()
            self.gesture_active = (name, now, gesture_duration(name), side)
        name, start, duration, side = self.gesture_active
        t = (now - start) / duration
        if t >= 1.0:
            self.gesture_active = None
            self.gesture_next = now + random.uniform(float(p("mouth_gesture_min_interval")), float(p("mouth_gesture_max_interval")))
            return (0.0, 0.0, 0.0, None, 0.0)
        return gesture_frame(name, t, now, side)

    def update_mouth(self, now, seeing_person):
        p = self.p
        if not p("mouth_enabled"):
            return 0.0, None, 0.0, 0.0, None, 0.0
        speaking = (now - self.mouth_last_input) <= float(p("mouth_timeout"))
        if not speaking:
            self.mouth_target = 0.0
            self.mouth_samples = None
        goal = float(p("mouth_smile_happy")) if seeing_person else float(p("mouth_smile"))
        self.mouth_smile_current += (goal - self.mouth_smile_current) * 0.08
        g_level, g_smile, g_skew, g_points, g_blend = self.idle_gesture(now, idle=(not seeing_person and not speaking))
        target = max(self.mouth_target, g_level)
        k = float(p("mouth_attack")) if target > self.mouth_level else float(p("mouth_release"))
        self.mouth_level += (target - self.mouth_level) * k
        if self.mouth_level < 1e-3:
            self.mouth_level = 0.0
        return self.mouth_level, self.mouth_samples, self.mouth_smile_current + g_smile, g_skew, g_points, g_blend

    # ---- the frame
    def render_tick(self):
        p = self.p
        now = time.monotonic()
        if (self.last_frame_mono == 0.0 or now - self.last_frame_mono > float(p("camera_timeout"))) and not self.camera_warned \
                and now - self.start_mono > float(p("camera_timeout")):
            self.get_logger().warn(f"no camera frames on {p('image_topic')}; the face animates but cannot see anyone")
            self.camera_warned = True
        if self.tracked_face is not None and now - self.last_seen > float(p("face_stick_seconds")):
            self.happy_until, self.tracked_face, self.current_person, self.greeting_due, self.generic_greeted_at = 0.0, None, None, None, None
        self.greeting_step(now)
        # gaze
        since = now - self.last_seen
        if self.tracked_face is not None:
            bx, by, bw, bh = self.tracked_face
            center = (bx + bw / 2.0, by + bh / 2.0)
        elif since > float(p("idle_scan_delay")):
            e = since - float(p("idle_scan_delay"))
            a = (2 * math.pi / float(p("idle_scan_period"))) * e
            center = (self.camera_w / 2.0 + math.sin(a) * self.camera_w * 0.3 + math.sin(e * 1.9 + 0.7) * self.camera_w * 0.05,
                      self.camera_h / 2.0 + math.cos(a * 0.7) * self.camera_h * 0.15 + math.sin(e * 2.6) * self.camera_h * 0.05)
        else:
            center = (self.camera_w / 2.0, self.camera_h / 2.0)
        if self.smoothed_center is None:
            self.smoothed_center = center
        else:
            k = float(p("gaze_speed")) if self.tracked_face is not None else float(p("gaze_idle_speed"))
            k = 1.0 - (1.0 - k) ** (self.render_dt * 30.0)
            self.smoothed_center = (self.smoothed_center[0] + (center[0] - self.smoothed_center[0]) * k,
                                    self.smoothed_center[1] + (center[1] - self.smoothed_center[1]) * k)
        # blink / wink
        if self.blink_start is None and now >= self.next_blink and self.wink_start is None:
            self.blink_start = now
        blink = 0.0
        if self.blink_start is not None:
            el = now - self.blink_start
            closing = float(p("blink_duration")) / 2.0
            hold = max(float(p("blink_closed_hold")), 0.0)
            if el >= closing * 2 + hold:
                self.blink_start = None
                self.next_blink = schedule_blink(float(p("blink_min")), float(p("blink_max")))
            elif el <= closing:
                blink = min(1.0, el / closing)
            elif el <= closing + hold:
                blink = 1.0
            else:
                blink = max(0.0, 1.0 - (el - closing - hold) / closing)
        wink = 0.0
        if self.wink_start is not None:
            w = now - self.wink_start
            d = float(p("wink_duration"))
            close_t, open_t = 0.18 * d, 0.25 * d
            hold_t = d - close_t - open_t
            if w >= d:
                self.wink_start = None
                self.next_blink = max(self.next_blink, now + 1.0)
            elif w <= close_t:
                wink = w / close_t
            elif w <= close_t + hold_t:
                wink = 1.0
            else:
                wink = max(0.0, 1.0 - (w - close_t - hold_t) / open_t)
        nx = max(-1.0, min(1.0, (self.smoothed_center[0] - self.camera_w / 2.0) / (self.camera_w / 2.0)))
        ny = max(-1.0, min(1.0, (self.smoothed_center[1] - self.camera_h / 2.0) / (self.camera_h / 2.0)))
        travel = 40.0
        gaze = (-nx * travel * 0.7, ny * travel * 0.6)
        level, samples, smile, skew, path, blend = self.update_mouth(now, seeing_person=self.tracked_face is not None)
        if self.renderer is None:
            return
        frame = self.renderer.render(blink=blink, wink=wink * float(p("wink_depth")), wink_eye=self.wink_eye_index, gaze_offset=gaze,
                                     mood=3 if now < self.happy_until else 0, dt=self.render_dt, mouth_level=level, mouth_samples=samples,
                                     mouth_smile=smile, mouth_skew=skew, mouth_shape=blend, mouth_path=path)
        if frame is not None and p("publish_image") and p("renderer") == "headless":
            m = Image(height=frame.shape[0], width=frame.shape[1], encoding="bgr8", is_bigendian=0, step=frame.shape[1] * 3, data=frame.tobytes())
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "face"
            self.image_pub.publish(m)
        if self.renderer.should_close():
            self.get_logger().info("window closed; shutting down")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = FaceNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.renderer is not None:
            node.renderer.destroy()
        if node.face_id is not None:
            node.face_id.stop()
        node.destroy_node()
        rclpy.try_shutdown()
