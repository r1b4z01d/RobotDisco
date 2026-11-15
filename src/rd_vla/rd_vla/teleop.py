"""SpaceMouse teleop of the arm and the hand through rd_arm's streamer, and the recording of demonstrations
(v1 manual_panel.py's SpaceMouse path). The SpaceMouse is the only teleop device: the gamepad drives the base.

  puck            -> the tool's position and orientation in the base frame (gains and axis map: parameters)
  button 0 / 1    -> held: close / open the hand, slowly
  /teleop/enable          std_srvs/SetBool   streaming on / off (nothing moves while off)
  /teleop/home            std_srvs/Trigger   the arm to its home pose (streaming goes off first)
  /teleop/episode/start   std_srvs/Trigger   record from now: every tick one frame (three cameras, the state, and
                                             as the action the pose and curls just commanded), parameters `dataset`, `task`
  /teleop/episode/save    std_srvs/Trigger   close the episode and write it (LeRobot v3 under <data>/outputs/datasets)
  /teleop/episode/discard std_srvs/Trigger
  /teleop/cmd             std_msgs/String    JSON {name, args}: what the console's teleop page sends (v1's panel commands):
                                             enable {on}, rec, save, discard, home, quit, grasp_hold {dir}, grasp {delta},
                                             grasp_mark (flag a re-grasp stretch), task {text}, dataset {name},
                                             freedrive (toggle hand-guiding: the arm is moved by hand, the hand by the
                                             buttons, and the recorded action is the pose the arm is actually at),
                                             recover_hand (power-cycle the tool connector), reconnect (recover the arm)
  Stream Deck     -> v1's keys when a deck is plugged in (rd_vla/deck.py): REC / +GRASP, SAVE / HOME, DISCARD / FREEDRIVE,
                     reconnect, teleop off, per-finger open and close, and the stars after a save
  /teleop/state           std_msgs/String    JSON five times a second: device, enabled, recording, frames, episodes, grasps,
                                             dataset, task, the tool's pose, the grasp, the tick rate, the last saved episode
  /teleop/log             std_msgs/String    one line per event, for the page's log

Runs under the LeRobot venv's python (vla.launch.py), like the policy runner: the recorder needs lerobot.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import json
import os
import queue
import threading
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger

from rd_bringup import config

from .geometry import clamp_box, clamp_step, keep_out, quat_to_rotvec, rotvec_to_quat, workspace_limits
from .observer import Observer
from .schema import CAMERA_KEYS
from .spacemouse import SpaceMouse

# robot axis -> (puck axis, sign): v1's calibration for this robot seen from the operator's place (sm_calib.json)
AXIS_MAP = {"x": ("x", -1), "y": ("y", 1), "z": ("z", -1), "roll": ("yaw", -1), "pitch": ("roll", 1), "yaw": ("pitch", -1)}
AXES = ("x", "y", "z", "roll", "pitch", "yaw")
AFTER_SAVE = ("stay_same", "stay_open", "home_same", "home_open")
# The calibration's six steps (v1 teleop/sm_calibrate.py): the gesture the operator wants to mean each motion of the
# gripper, and the base-frame axis and sign that motion is. The operator stands behind the robot's arm looking the
# way it reaches: their right is the base frame's -x, away from them is -y, and what they call the gripper's roll
# (about the line away from them) turns about the base y axis, which integrate() names pitch. Bryan's first
# calibration, taken with every prompt mapped to its own name and +, came out with x, y and yaw inverted and roll
# and pitch exchanged (2026-11-15).
CALIBRATION = (("x", -1, "SLIDE the puck the way the gripper should move RIGHT"),
               ("y", -1, "SLIDE the puck the way the gripper should move AWAY from you"),
               ("z", 1, "LIFT the puck the way the gripper should move UP"),
               ("pitch", 1, "TILT the puck the way the gripper should ROLL"),
               ("roll", -1, "TILT the puck the way the gripper should PITCH (nose up)"),
               ("yaw", -1, "TWIST the puck the way the gripper should YAW (spin)"))


def dominant(samples: list) -> tuple:
    """(puck axis, mean deflection) of the axis that moved most over a list of {axis: value} readings."""
    if not samples:
        return AXES[0], 0.0
    mean = {k: sum(s.get(k, 0.0) for s in samples) / len(samples) for k in AXES}
    k = max(AXES, key=lambda a: abs(mean[a]))
    return k, mean[k]


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def integrate(target, axes: dict, axis_map: dict, linear: float, angular: float, dt: float):
    """The target pose [x, y, z, rx, ry, rz] moved by the puck's deflection for one tick: translation along the base
    axes, rotation about the base axes through the tool point (composed, not added to the rotation vector)."""
    v = {k: axes[src] * sign for k, (src, sign) in axis_map.items()}
    out = [target[0] + v["x"] * linear * dt, target[1] + v["y"] * linear * dt, target[2] + v["z"] * linear * dt]
    w = [v["roll"] * angular * dt, v["pitch"] * angular * dt, v["yaw"] * angular * dt]
    if any(w):
        out += list(quat_to_rotvec(quat_multiply(rotvec_to_quat(w), rotvec_to_quat(target[3:]))))
    else:
        out += list(target[3:])
    return out


def write_annotation(dataset_root, episode: int, **fields) -> None:
    """Merge fields into <dataset>/meta/annotations.json[episode] (rd_manager annotations.py's sidecar: grasps,
    regrasp_ranges, rating, notes). Outside the LeRobot schema, so it never touches the dataset's format."""
    p = Path(dataset_root) / "meta" / "annotations.json"
    try:
        data = json.loads(p.read_text()) if p.is_file() else {}
    except (ValueError, OSError):
        data = {}
    data.setdefault(str(int(episode)), {}).update(fields)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, p)


def write_capture(dataset_root) -> None:
    """<dataset>/meta/capture.json, once, when the dataset gets its first episode: how the cameras were cut and
    turned for it (v1's file; rd_manager's dataset cards and anyone serving a policy trained on it read it)."""
    p = Path(dataset_root) / "meta" / "capture.json"
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        data = json.loads(p.read_text()) if p.is_file() else None
    except (ValueError, OSError):
        data = None
    if data is None:
        cams = config.get("sensors.usb_cameras", {}) or {}
        data = {"version": 1, "host": str(config.get("robot.host", "")),
                "squares": {k: v for k, v in config.camera_squares().items()},
                "rotate": {k: int(c.get("rotate", 0)) for k, c in cams.items() if isinstance(c, dict) and "port" in c},
                "recorded": {"first": now}}
    data.setdefault("recorded", {})["last"] = now
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def episodes_on_disk(dataset_root) -> int:
    """How many episodes a dataset already holds (LeRobot's meta/info.json, or the npz fallback's meta.json)."""
    for rel, key in (("meta/info.json", "total_episodes"), ("meta.json", "episodes")):
        try:
            return int(json.loads((Path(dataset_root) / rel).read_text())[key])
        except (OSError, ValueError, KeyError):
            continue
    return 0


def step_grasp(grasp: float, buttons, close_rate: float, open_rate: float, close_button: int = 0, open_button: int = 1) -> float:
    close = bool(buttons[close_button]) if close_button < len(buttons) else False
    opened = bool(buttons[open_button]) if open_button < len(buttons) else False
    if close and not opened:
        return min(1.0, grasp + close_rate)
    if opened and not close:
        return max(0.0, grasp - open_rate)
    return grasp


class TopicDevice:
    """The rig's puck: /teleop/debug_input, Float32MultiArray [x, y, z, roll, pitch, yaw, button0, button1]."""

    def __init__(self, node):
        self.axes = dict.fromkeys(AXES, 0.0)
        self.buttons = [0] * 8
        node.create_subscription(Float32MultiArray, "/teleop/debug_input", self._on, 10)

    def _on(self, m):
        d = list(m.data) + [0.0] * 8
        self.axes = {k: max(-1.0, min(1.0, float(v))) for k, v in zip(AXES, d[:6])}
        self.buttons = [int(d[6] > 0.5), int(d[7] > 0.5)] + [0] * 6

    def open(self) -> bool:
        return True

    def read(self) -> bool:
        return True

    def close(self) -> None:
        pass


class Teleop(Node):
    def __init__(self, device=None):
        super().__init__("vla_teleop")
        dp = self.declare_parameter
        dp("fps", 12.0)                       # the tick: commands and recorded frames (the datasets' rate)
        dp("linear_speed", 0.131)             # m/s at full deflection (v1 SM_POS_GAIN)
        dp("angular_speed", 0.45)             # rad/s (v1 SM_ROT_GAIN)
        dp("grasp_close_rate", 0.104); dp("grasp_open_rate", 0.060)      # per tick while the button is held (v1)
        dp("button_close", 0); dp("button_open", 1)
        dp("max_curls", [1.0, 1.0, 1.0, 1.0])  # index, middle, ring, thumb at a full grasp
        dp("axis_map", json.dumps(AXIS_MAP))
        dp("deadband", 0.12); dp("scale", 350.0)
        dp("max_step_m", 0.05)     # as the policy runner and v1: recording and playing back go through the same limits
        dp("workspace_min", [-1.0, -1.0, -0.2]); dp("workspace_max", [1.0, 1.0, 1.0])
        dp("base_frame", "ur_base"); dp("tool_frame", "ur_tool0")
        dp("cameras", list(CAMERA_KEYS)); dp("camera_topic", "/cameras/{name}/image")
        dp("dataset", ""); dp("task", "")
        # what the arm and the hand do once an episode is saved: stay | home, with the grip as it is | opened
        dp("after_save", "stay_same")
        dp("record_root", "")
        dp("debug_input", False)              # the rig: the puck is the topic /teleop/debug_input
        dp("deck", True); dp("deck_flip", True)   # the Stream Deck, mounted upside down on the robot (v1)
        p = self.p = lambda name: self.get_parameter(name).value
        self.cb = ReentrantCallbackGroup()
        self.axis_map = {k: (str(v[0]), int(v[1])) for k, v in json.loads(p("axis_map")).items()}
        try:                                    # the operator's own calibration, when one was made here (the page's button)
            saved = json.loads(self._calib_file().read_text())
            if set(saved) == set(AXES):
                self.axis_map = {k: (str(v[0]), int(v[1])) for k, v in saved.items()}
        except (OSError, ValueError, TypeError, IndexError):
            pass
        self.calib = ""                         # the calibration's current prompt, "" when none is running
        self._calib_stop = threading.Event()
        if device is None:
            device = TopicDevice(self) if bool(p("debug_input")) else SpaceMouse(float(p("scale")), float(p("deadband")))
        self.device = device
        self.device_ok = False
        self._device_tried = -1e9
        self.observer = Observer(self, cameras=tuple(p("cameras")), camera_topic=str(p("camera_topic")),
                                 base_frame=str(p("base_frame")), tool_frame=str(p("tool_frame")))
        self.pose_pub = self.create_publisher(PoseStamped, "/arm/target_pose", 10)
        self.hand_pub = self.create_publisher(Float32MultiArray, "/hand/curls", 10)
        self.state_pub = self.create_publisher(String, "/teleop/state", 10)
        self.log_pub = self.create_publisher(String, "/teleop/log", 50)
        self.create_subscription(String, "/teleop/cmd", self.on_cmd, 20, callback_group=self.cb)
        self.stream = self.create_client(SetBool, "/arm/stream", callback_group=self.cb)
        self.freedrive_client = self.create_client(SetBool, "/arm/freedrive", callback_group=self.cb)
        self.hand_power = self.create_client(Trigger, "/hand/power_cycle", callback_group=self.cb)
        self.arm_power = None
        self.home = self.create_client(Trigger, "/arm/go_home", callback_group=self.cb)
        self.create_service(SetBool, "/teleop/enable", self.on_enable, callback_group=self.cb)
        self.create_service(Trigger, "/teleop/home", self.on_home, callback_group=self.cb)
        self.create_service(Trigger, "/teleop/episode/start", self.on_start, callback_group=self.cb)
        self.create_service(Trigger, "/teleop/episode/save", self.on_save, callback_group=self.cb)
        self.create_service(Trigger, "/teleop/episode/discard", self.on_discard, callback_group=self.cb)
        self.limits = workspace_limits()
        self.enabled = False
        self._pending_rating = None
        self.homing = False                # a go-home is under way: the puck has let go and takes over after it
        self.freedrive = False                # hand-guiding: the arm is moved by hand, nothing is streamed
        self.target = None
        self.curls = [0.0] * 4                # each finger's grasp 0..1 (index, middle, ring, thumb), before max_curls
        self.lock = threading.Lock()          # the writer: the tick adds frames, the services open / save / discard
        self.writer = None
        self.recording = False
        # A take is kept in memory while it is recorded and written by one worker thread once SAVE is pressed: the
        # videos take 11 to 14 s to encode on the Orin, and REC was refused for as long (2026-11-15). The next take
        # can start at once; the saves queue up and are written in order.
        self.take: list = []                  # (images, state, action, task) per frame of the take being recorded
        self._jobs: queue.Queue = queue.Queue()
        self._unsaved = 0                     # takes handed over and not yet on disk
        self._last_job = None                 # the newest saved-or-saving take: what the stars rate
        threading.Thread(target=self._save_loop, daemon=True, name="episode-writer").start()
        self.frames = 0
        self.message = ""
        self.episodes = 0                     # in the current dataset
        self._episodes_of = None
        self.regrasp_ranges: list = []        # [start, end] frames flagged in this episode (the operator's "that grasp missed")
        self._regrasp_open = None
        self.last = {}                        # the last saved episode: ep, frames, grasps, dataset
        self.web_hold = (0, 0.0)              # the page's hold-to-close / hold-to-open: direction and when it was last asked
        self.pose = None
        self._ticks: list = []
        self.create_timer(1.0 / float(p("fps")), self.tick, callback_group=self.cb)
        self.create_timer(0.2, self.publish_state, callback_group=self.cb)
        self.deck = None
        if bool(p("deck")) and not bool(p("debug_input")):
            from .deck import Deck, Remote
            # teleop off, the deck is the robot's remote (halt, volume, the microphone, landmarks) through the manager
            remote = Remote(f"http://127.0.0.1:{int(config.get('robot.ports.manager', 8092))}", log=self.log)
            self.deck = Deck(self, flip=bool(p("deck_flip")), remote=remote)
            self.create_timer(0.1, self.deck.poll, callback_group=self.cb)
        self._restore_session()
        # LeRobot (and torch under it) takes many seconds to import on the Orin; the first REC paid for it
        threading.Thread(target=lambda: __import__("rd_vla.recorder", fromlist=["x"]).lerobot_warm(), daemon=True, name="lerobot-import").start()

    @property
    def saving(self) -> bool:
        return self._unsaved > 0

    @property
    def grasp(self) -> float:
        """The whole hand's grasp: the mean of the fingers (what the page shows and the puck's buttons move)."""
        return sum(self.curls) / len(self.curls)

    def nudge(self, delta: float, finger: int = -1) -> None:
        """Close (delta > 0) or open one finger, or all of them with finger < 0."""
        for i in (range(4) if finger < 0 else (finger,)):
            self.curls[i] = max(0.0, min(1.0, self.curls[i] + delta))

    # ---- the page's commands
    def log(self, line: str) -> None:
        self.message = line
        self.get_logger().info(line)
        self.log_pub.publish(String(data=time.strftime("%H:%M:%S ") + line))

    def _set(self, name: str, value: str, remember: bool = True) -> None:
        from rclpy.parameter import Parameter
        self.set_parameters([Parameter(name, Parameter.Type.STRING, value)])
        if remember:                               # the dataset and the task outlive a restart of this node
            try:
                f = self._session_file()
                d = json.loads(f.read_text()) if f.is_file() else {}
                d[name] = value
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(d))
            except (OSError, ValueError):
                pass

    def _calib_file(self) -> Path:
        return config.data_dir() / "outputs" / "sm_calib.json"

    def calibrate(self) -> None:
        """The puck's axis calibration, six gestures prompted on the page (v1 sm_calibrate.py). Run on a thread."""
        new: dict = {}
        try:
            for i, (axis, way, prompt) in enumerate(CALIBRATION, 1):
                head = f"puck calibration {i}/6: {prompt}"
                while True:
                    self.calib = head + ", and HOLD it"
                    end = time.monotonic() + 30.0          # wait for a firm push
                    while max(abs(v) for v in self.device.axes.values()) < 0.35:
                        if self._calib_stop.is_set() or time.monotonic() > end or not self.device_ok:
                            self.log("puck calibration stopped: " + ("cancelled" if self._calib_stop.is_set() else "no push seen" if self.device_ok else "the puck is not connected"))
                            return
                        time.sleep(0.02)
                    samples, end = [], time.monotonic() + 1.0
                    while time.monotonic() < end:
                        samples.append(dict(self.device.axes))
                        time.sleep(0.02)
                    src, val = dominant(samples)
                    self.calib = head + ": got it, let go"
                    end = time.monotonic() + 10.0
                    while max(abs(v) for v in self.device.axes.values()) > 0.08 and time.monotonic() < end and not self._calib_stop.is_set():
                        time.sleep(0.02)
                    time.sleep(0.4)
                    if abs(val) < 0.2:
                        self.log(f"puck calibration {i}/6: too weak ({src} {val:+.2f}), again")
                        continue
                    if src in [m[0] for m in new.values()]:
                        self.log(f"puck calibration {i}/6: that was the puck's {src} axis, already used for another motion; again, a cleaner gesture")
                        continue
                    new[axis] = (src, way * (1 if val > 0 else -1))
                    self.log(f"puck calibration {i}/6: {axis} = puck {src} {'+' if new[axis][1] > 0 else '-'}")
                    break
            self.axis_map = new
            try:
                f = self._calib_file()
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({k: list(v) for k, v in new.items()}, indent=1))
                self.log("puck calibration saved: " + ", ".join(f"{k}={v[0]}{'+' if v[1] > 0 else '-'}" for k, v in new.items()))
            except OSError as e:
                self.log(f"puck calibration in use but not saved: {e}")
        finally:
            self.calib = ""

    def _session_file(self) -> Path:
        return config.data_dir() / "outputs" / "teleop_session.json"

    def _restore_session(self) -> None:
        try:
            d = json.loads(self._session_file().read_text())
        except (OSError, ValueError):
            return
        if d.get("after_save") in AFTER_SAVE:
            self._set("after_save", d["after_save"], remember=False)
        for name in ("dataset", "task"):
            if d.get(name) and not str(self.p(name)).strip():
                self._set(name, str(d[name]), remember=False)

    def on_cmd(self, m: String):
        try:
            c = json.loads(m.data)
            name, a = str(c.get("name", "")), c.get("args") or {}
        except (ValueError, AttributeError):
            return
        self.command(name, a)

    def command(self, name: str, a: dict | None = None) -> None:
        """One of the panel's commands, from the page (/teleop/cmd) or a Stream Deck key."""
        a = a or {}
        tr = Trigger.Request()
        try:
            if name == "enable":
                r = self.on_enable(SetBool.Request(data=bool(a.get("on"))), SetBool.Response())
                if not r.success:
                    self.log(r.message)
            elif name == "quit":
                self.on_enable(SetBool.Request(data=False), SetBool.Response())
            elif name == "calibrate":
                if self.calib:
                    self._calib_stop.set()                  # pressed again: cancel
                elif self.enabled or self.freedrive or self.recording:
                    self.log("puck calibration: turn teleop off first (the arm must not follow the gestures)")
                elif not self.device_ok:
                    self.log("puck calibration: the puck is not connected")
                else:
                    self._calib_stop.clear()
                    self.calib = "puck calibration starting"
                    threading.Thread(target=self.calibrate, daemon=True, name="puck-calibration").start()
            elif name in ("rec", "save", "discard", "home"):
                handler = {"rec": self.on_start, "save": self.on_save, "discard": self.on_discard, "home": self.on_home}[name]
                r = handler(tr, Trigger.Response())
                if name != "rec" or not r.success:      # a started recording has logged itself
                    self.log(r.message)
            elif name == "grasp_hold":
                self.web_hold = (max(-1, min(1, int(a.get("dir", 0)))), time.monotonic())
            elif name == "grasp":
                self.nudge(float(a.get("delta", 0.1)))
            elif name == "freedrive":
                self.log(self.set_freedrive(not self.freedrive))
            elif name == "recover_hand":
                threading.Thread(target=self._recover_hand, daemon=True).start()
            elif name == "reconnect":
                threading.Thread(target=self._reconnect_arm, daemon=True).start()
            elif name == "rating":
                self.log(self.rate(int(a.get("stars", 0))))
            elif name == "grasp_mark":
                self.log(self.regrasp_mark())
            elif name == "after_save":
                mode = str(a.get("mode", ""))
                if mode in AFTER_SAVE:
                    self._set("after_save", mode)
                    self.log(f"after a save: {mode.replace('_', ', ')} grip")
            elif name == "task":
                text = str(a.get("text", "")).strip()
                if text:
                    self._set("task", text)
                    self.log(f"task: {text}")
            elif name == "dataset":
                text = str(a.get("name", "")).strip()
                if self.recording or self.saving:
                    self.log("dataset: not while recording or saving")
                elif text:
                    self._set("dataset", text)
                    self.log(f"dataset: {text} ({episodes_on_disk(self._root() / text)} episodes so far)")
            else:
                self.log(f"unknown command {name!r}")
        except Exception as e:  # noqa: BLE001  a bad command must never stop the node
            self.log(f"command {name} failed: {type(e).__name__}: {e}")

    def set_freedrive(self, on: bool) -> str:
        if on and self.recording and not self.freedrive:
            return "freedrive: not in the middle of a recording"
        if not self.freedrive_client.wait_for_service(timeout_sec=2.0):
            return "rd_arm's /arm/freedrive is not there"
        if on:
            if self.enabled and self.stream.service_is_ready():
                self.stream.call(SetBool.Request(data=False))      # the streamer lets go before the arm goes limp
            self.observer.cameras_on()
        r = self.freedrive_client.call(SetBool.Request(data=bool(on)))
        if not r.success:
            return f"freedrive: {r.message}"
        self.freedrive, self.target = bool(on), None
        if not on:
            if self.enabled and self.stream.service_is_ready():
                self.stream.call(SetBool.Request(data=True))       # back under the puck, from where the hand left it
            elif not self.recording:
                self.observer.cameras_off()
        return r.message

    def _recover_hand(self) -> None:
        if not self.hand_power.wait_for_service(timeout_sec=2.0):
            self.log("hand: rd_arm's /hand/power_cycle is not there")
            return
        self.log("hand: power-cycling the tool connector…")
        r = self.hand_power.call(Trigger.Request())
        self.log(f"hand: {r.message}")

    def _reconnect_arm(self) -> None:
        from rd_interfaces.srv import ArmPower
        if self.arm_power is None:
            self.arm_power = self.create_client(ArmPower, "/arm/power", callback_group=self.cb)
        if not self.arm_power.wait_for_service(timeout_sec=2.0):
            self.log("arm: no /arm/power service")
            return
        st = self.arm_power.call(ArmPower.Request(command=ArmPower.Request.STATUS, timeout_s=5.0))
        cmd = ArmPower.Request.POWER_ON if st.safety_status in ("NORMAL", "REDUCED") else ArmPower.Request.UNLOCK
        r = self.arm_power.call(ArmPower.Request(command=cmd, timeout_s=60.0))
        self.log(f"arm: {r.message} ({r.robot_mode} / {r.safety_status})")

    def rate(self, stars: int) -> str:
        """1..5 stars for the newest take: written now when it is on disk, with it when it is still being written."""
        stars = max(1, min(5, int(stars)))
        with self.lock:
            job = self._last_job
            if job is None:
                return "no saved episode to rate yet"
            job["rating"] = stars
            ep = job.get("ep")
        if ep is None:
            return f"rated {'★' * stars} (written once the episode is saved)"
        write_annotation(self._root() / job["dataset"], ep, rating=stars)
        self.last["rating"] = stars
        return f"episode {ep + 1}: {'★' * stars}"

    def regrasp_mark(self) -> str:
        """Toggle a re-grasp range: the first press opens it at this frame, the second closes it."""
        if not self.recording:
            return "+GRASP: not recording"
        if self._regrasp_open is None:
            self._regrasp_open = self.frames
            return f"re-grasp flagged from frame {self.frames} (press again when it holds)"
        self.regrasp_ranges.append([self._regrasp_open, self.frames])
        self._regrasp_open = None
        return f"re-grasp range closed: {self.regrasp_ranges[-1]} ({len(self.regrasp_ranges) + 1} grasps)"

    # ---- services
    def on_enable(self, req, res):
        on = bool(req.data)
        if on and self.freedrive:
            self.log(self.set_freedrive(False))        # the puck takes over from the hand
        if on:
            self.observer.cameras_on()        # the tool's pose (and the cameras a recording needs) only while in use
            end = time.monotonic() + 6.0
            while self.observer.ee_pose() is None and time.monotonic() < end:
                time.sleep(0.1)
            if self.observer.ee_pose() is None:
                self.observer.cameras_off()
                res.success, res.message = False, "no tool pose (is rd_arm up?)"
                return res
        if not self.stream.wait_for_service(timeout_sec=2.0):
            res.success, res.message = False, "rd_arm's /arm/stream is not there"
            return res
        self.stream.call_async(SetBool.Request(data=on))
        self.target = None
        self.enabled = on
        if not on and not self.recording:
            self.observer.cameras_off()
        res.success, res.message = True, "teleop " + ("on" if on else "off")
        self.get_logger().info(res.message)
        return res

    def on_home(self, _req, res):
        if self.recording:
            res.success, res.message = False, "recording: save or discard the episode first"
            return res
        if self.freedrive:
            self.set_freedrive(False)
        if not self.home.wait_for_service(timeout_sec=2.0):
            res.success, res.message = False, "rd_arm's /arm/go_home is not there"
            return res
        # The puck lets go for the move (it is a trajectory, not the stream) and takes the arm back when it is done:
        # going home used to leave teleop off, a second key press before every episode.
        was_on = self.enabled
        self.enabled, self.target, self.homing = False, None, True
        if self.stream.service_is_ready():
            self.stream.call_async(SetBool.Request(data=False))

        def done(_future):
            self.homing = False
            if was_on and not self.enabled and not self.freedrive:
                r = self.on_enable(SetBool.Request(data=True), SetBool.Response())
                self.log("home; teleop on again" if r.success else f"home; teleop stayed off: {r.message}")
        self.home.call_async(Trigger.Request()).add_done_callback(done)
        res.success, res.message = True, "going home" + ("" if was_on else " (teleop is off)")
        return res

    def _root(self) -> Path:
        return Path(self.p("record_root") or os.environ.get("RD_RECORD_ROOT") or (config.data_dir() / "outputs" / "datasets"))

    def on_start(self, _req, res):
        from .recorder import EpisodeWriter
        dataset, task = str(self.p("dataset")).strip(), str(self.p("task")).strip()
        if not dataset or not task:
            res.success, res.message = False, "set the parameters `dataset` and `task` first"
        elif self.recording:
            res.success, res.message = False, "an episode is already being recorded"
        elif not self.enabled and not self.freedrive:
            res.success, res.message = False, "teleop is off (TELEOP ON, or freedrive)"
        else:
            ok, why = self.observer.wait_ready(timeout=6.0)
            if not ok:
                res.success, res.message = False, why
                return res
            with self.lock:
                self.take = []
                self.frames, self.recording = 0, True
                self.regrasp_ranges, self._regrasp_open = [], None
            res.success, res.message = True, f"REC into {self._root() / dataset}" + (" (the take before is still being written)" if self.saving else "")
            self.log(res.message)
        self.message = res.message
        return res

    def on_save(self, _req, res):
        with self.lock:
            if not self.recording or self.frames == 0:
                self.recording = False
                res.success, res.message = False, "nothing recorded"
                return res
            self.recording = False
            frames = self.frames
            if self._regrasp_open is not None:               # a range left open ends with the episode
                self.regrasp_ranges.append([self._regrasp_open, frames])
                self._regrasp_open = None
            after = str(self.p("after_save"))             # as chosen when SAVE was pressed
            job = {"take": self.take, "frames": frames, "ranges": [list(r) for r in self.regrasp_ranges], "after_save": after,
                   "dataset": str(self.p("dataset")).strip(), "fps": int(round(float(self.p("fps")))), "cameras": tuple(self.p("cameras")),
                   "rating": None, "ep": None}
            self.take, self._last_job = [], job
            self._unsaved += 1
        self._jobs.put(job)
        if self.deck is not None:
            self.deck.enter_rating()          # the stars at once: the verdict is ready now, the videos are not
        res.success, res.message = True, f"saving {frames} frames"
        # the take is over: the grip and the arm as the operator chose for the end of a task
        if after.endswith("_open"):
            self.curls = [0.0] * 4
            self.hand_pub.publish(Float32MultiArray(data=[0.0, 0.0, 0.0, 0.0]))     # also when the puck is not driving (freedrive)
        if after.startswith("home"):
            r = self.on_home(Trigger.Request(), Trigger.Response())
            if not r.success:
                self.log(f"after the save: no homing ({r.message})")
        return res

    def _save_loop(self) -> None:
        """The one writer: each saved take in turn becomes an episode of its dataset (frames, videos, annotations)."""
        from .recorder import EpisodeWriter
        while True:
            job = self._jobs.get()
            ep = None
            try:
                writer = EpisodeWriter(self._root(), job["dataset"], fps=job["fps"], cameras=job["cameras"])
                for images, state, action, task in job["take"]:
                    writer.add_frame(images, state, action, task)
                saved = writer.save_episode()
                writer.close()                # finalize, so the dataset is loadable after every episode
                ep = int(saved["episode"])
                ranges = job["ranges"]
                write_annotation(saved["path"], ep, grasps=len(ranges) + 1, regrasp_ranges=ranges, after_save=job["after_save"])
                write_capture(saved["path"])
                line = f"episode {ep + 1} saved: {job['frames']} frames, {len(ranges) + 1} grasp(s), in {saved['path']}"
            except Exception as e:  # noqa: BLE001
                line = f"SAVE FAILED: {type(e).__name__}: {e}"
            job["take"] = None                # the frames' memory back
            with self.lock:
                self._unsaved -= 1
                job["ep"] = ep                # from here the stars are written straight to it
                stars = job["rating"]
                if ep is not None:
                    self.episodes, self._episodes_of = ep + 1, job["dataset"]
                    self.last = {"ep": ep, "frames": job["frames"], "grasps": len(job["ranges"]) + 1, "dataset": Path(saved["path"]).name}
            if stars and ep is not None:      # rated while it was being written
                write_annotation(saved["path"], ep, rating=stars)
                self.last["rating"] = stars
                line += f"; rated {'★' * stars}"
            self.log(line)

    def finish_saves(self, timeout: float = 180.0) -> bool:
        """Wait for the takes still being written (the node is going down). False when it gave up."""
        end = time.monotonic() + timeout
        while self.saving and time.monotonic() < end:
            time.sleep(0.2)
        return not self.saving

    def on_discard(self, _req, res):
        with self.lock:
            self.take = []
            was, self.recording, self.frames = self.recording, False, 0
        res.success, res.message = True, "episode discarded" if was else "nothing to discard"
        self.message = res.message
        return res

    # ---- the loop
    def _read_device(self) -> None:
        if not self.device_ok:
            now = time.monotonic()
            if now - self._device_tried < 2.0:
                return
            self._device_tried = now
            self.device_ok = bool(self.device.open())
            if self.device_ok:
                self.get_logger().info("SpaceMouse opened")
            return
        if not self.device.read():
            self.device_ok = False
            self.get_logger().warn("SpaceMouse lost; the arm holds its pose")

    def tick(self):
        self._read_device()
        if not self.enabled and not self.freedrive:
            return
        pose = self.pose = self.observer.ee_pose()
        if pose is None:
            return
        now = time.monotonic()
        self._ticks = [t for t in self._ticks if now - t < 2.0] + [now]
        p = self.p
        dt = 1.0 / float(p("fps"))
        axes = self.device.axes if self.device_ok else dict.fromkeys(AXES, 0.0)
        buttons = self.device.buttons if self.device_ok else []
        if self.freedrive:
            self.target = list(pose)            # hand-guided: the action recorded is the pose the arm is actually at
        else:
            if self.target is None:
                self.target = list(pose)
            self.target = integrate(self.target, axes, self.axis_map, float(p("linear_speed")), float(p("angular_speed")), dt)
            # the target never runs more than a step ahead of the tool or out of the box, so letting go stops the arm
            xyz = clamp_box(clamp_step(self.target[:3], pose[:3], float(p("max_step_m"))),
                            [float(v) for v in p("workspace_min")], [float(v) for v in p("workspace_max")])
            xyz, _ = keep_out(xyz, **self.limits)   # never over the robot's own body or under the floor (robot.yaml arm.workspace)
            self.target[:3] = list(xyz)
        close_rate, open_rate = float(p("grasp_close_rate")), float(p("grasp_open_rate"))
        # the puck's two buttons and the page's hold buttons move the whole hand; the deck's keys one finger each
        step = step_grasp(0.5, buttons, close_rate, open_rate, int(p("button_close")), int(p("button_open"))) - 0.5
        hold, asked = self.web_hold           # the page's hold buttons let go by themselves after 0.6 s
        if hold and time.monotonic() - asked < 0.6:
            step += close_rate if hold > 0 else -open_rate
        if step:
            self.nudge(step)
        curls = [c * float(m) for c, m in zip(self.curls, p("max_curls"))]
        if not self.freedrive:
            msg = PoseStamped()
            msg.header.frame_id = self.observer.base_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(v) for v in self.target[:3]]
            (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w) = rotvec_to_quat(self.target[3:])
            self.pose_pub.publish(msg)
        self.hand_pub.publish(Float32MultiArray(data=curls))
        if self.recording:
            obs = self.observer.observe()
            if obs is None:
                return
            state, images, _age = obs
            with self.lock:
                if self.recording:
                    self.take.append((images, state, [float(v) for v in self.target] + curls, str(p("task"))))
                    self.frames += 1

    def publish_state(self):
        dataset, fps = str(self.p("dataset")), float(self.p("fps"))
        if dataset != self._episodes_of and not self.saving:     # another dataset chosen: how much it already holds
            self.episodes, self._episodes_of = (episodes_on_disk(self._root() / dataset) if dataset else 0), dataset
        hz = (len(self._ticks) - 1) / (self._ticks[-1] - self._ticks[0]) if (self.enabled or self.freedrive) and len(self._ticks) > 2 else 0.0
        self.state_pub.publish(String(data=json.dumps({
            "device": self.device_ok, "enabled": self.enabled, "recording": self.recording, "saving": self.saving,
            "frames": self.frames, "rec_s": self.frames / fps, "episodes": self.episodes,
            "grasps": len(self.regrasp_ranges) + 1, "regrasp_open": self._regrasp_open is not None,
            "grasp": round(self.grasp, 3), "dataset": dataset, "task": str(self.p("task")),
            "freedrive": self.freedrive, "deck": bool(self.deck and self.deck.pad is not None),
            "pose": [round(float(v), 4) for v in self.pose] if (self.enabled or self.freedrive) and self.pose else [],
            "tick_hz": round(hz, 1), "last": self.last,
            "backend": "lerobot", "unsaved": self._unsaved, "message": self.message, "calib": self.calib, "after_save": str(self.p("after_save"))})))


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = Teleop()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.saving:                        # takes still being written: they are finished before the node goes
            print("[teleop] finishing the episodes still being written", flush=True)
            node.finish_saves()
        node.device.close()
        if node.deck is not None:
            node.deck.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
