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
                                             grasp_mark (flag a re-grasp stretch), task {text}, dataset {name}
  /teleop/state           std_msgs/String    JSON five times a second: device, enabled, recording, frames, episodes, grasps,
                                             dataset, task, the tool's pose, the grasp, the tick rate, the last saved episode
  /teleop/log             std_msgs/String    one line per event, for the page's log

Runs under the LeRobot venv's python (vla.launch.py), like the policy runner: the recorder needs lerobot.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import json
import os
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
        dp("max_step_m", 0.03)
        dp("workspace_min", [-1.0, -1.0, -0.2]); dp("workspace_max", [1.0, 1.0, 1.0])
        dp("base_frame", "ur_base"); dp("tool_frame", "ur_tool0")
        dp("cameras", list(CAMERA_KEYS)); dp("camera_topic", "/cameras/{name}/image")
        dp("dataset", ""); dp("task", "")
        dp("record_root", "")
        dp("debug_input", False)              # the rig: the puck is the topic /teleop/debug_input
        p = self.p = lambda name: self.get_parameter(name).value
        self.cb = ReentrantCallbackGroup()
        self.axis_map = {k: (str(v[0]), int(v[1])) for k, v in json.loads(p("axis_map")).items()}
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
        self.home = self.create_client(Trigger, "/arm/go_home", callback_group=self.cb)
        self.create_service(SetBool, "/teleop/enable", self.on_enable, callback_group=self.cb)
        self.create_service(Trigger, "/teleop/home", self.on_home, callback_group=self.cb)
        self.create_service(Trigger, "/teleop/episode/start", self.on_start, callback_group=self.cb)
        self.create_service(Trigger, "/teleop/episode/save", self.on_save, callback_group=self.cb)
        self.create_service(Trigger, "/teleop/episode/discard", self.on_discard, callback_group=self.cb)
        self.limits = workspace_limits()
        self.enabled = False
        self.target = None
        self.grasp = 0.0
        self.lock = threading.Lock()          # the writer: the tick adds frames, the services open / save / discard
        self.writer = None
        self.recording = False
        self.saving = False
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

    # ---- the page's commands
    def log(self, line: str) -> None:
        self.message = line
        self.get_logger().info(line)
        self.log_pub.publish(String(data=time.strftime("%H:%M:%S ") + line))

    def _set(self, name: str, value: str) -> None:
        from rclpy.parameter import Parameter
        self.set_parameters([Parameter(name, Parameter.Type.STRING, value)])

    def on_cmd(self, m: String):
        try:
            c = json.loads(m.data)
            name, a = str(c.get("name", "")), c.get("args") or {}
        except (ValueError, AttributeError):
            return
        tr = Trigger.Request()
        try:
            if name == "enable":
                r = self.on_enable(SetBool.Request(data=bool(a.get("on"))), SetBool.Response())
                if not r.success:
                    self.log(r.message)
            elif name == "quit":
                self.on_enable(SetBool.Request(data=False), SetBool.Response())
            elif name in ("rec", "save", "discard", "home"):
                handler = {"rec": self.on_start, "save": self.on_save, "discard": self.on_discard, "home": self.on_home}[name]
                r = handler(tr, Trigger.Response())
                if name != "rec" or not r.success:      # a started recording has logged itself
                    self.log(r.message)
            elif name == "grasp_hold":
                self.web_hold = (max(-1, min(1, int(a.get("dir", 0)))), time.monotonic())
            elif name == "grasp":
                self.grasp = max(0.0, min(1.0, self.grasp + float(a.get("delta", 0.1))))
            elif name == "grasp_mark":
                self.log(self.regrasp_mark())
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
        self.enabled, self.target = False, None
        if self.stream.service_is_ready():
            self.stream.call_async(SetBool.Request(data=False))
        if not self.home.wait_for_service(timeout_sec=2.0):
            res.success, res.message = False, "rd_arm's /arm/go_home is not there"
            return res
        self.home.call_async(Trigger.Request())
        res.success, res.message = True, "going home; teleop is off"
        return res

    def _root(self) -> Path:
        return Path(self.p("record_root") or os.environ.get("RD_RECORD_ROOT") or (config.data_dir() / "outputs" / "datasets"))

    def on_start(self, _req, res):
        from .recorder import EpisodeWriter
        dataset, task = str(self.p("dataset")).strip(), str(self.p("task")).strip()
        if not dataset or not task:
            res.success, res.message = False, "set the parameters `dataset` and `task` first"
        elif self.recording or self.saving:
            res.success, res.message = False, "an episode is already being recorded or saved"
        elif not self.enabled:
            res.success, res.message = False, "teleop is off (/teleop/enable)"
        else:
            ok, why = self.observer.wait_ready(timeout=6.0)
            if not ok:
                res.success, res.message = False, why
                return res
            with self.lock:
                if self.writer is None or self.writer.repo_id != dataset:
                    if self.writer is not None:
                        self.writer.close()
                    self.writer = EpisodeWriter(self._root(), dataset, fps=int(round(float(self.p("fps")))), cameras=tuple(self.p("cameras")))
                self.frames, self.recording = 0, True
                self.regrasp_ranges, self._regrasp_open = [], None
            res.success, res.message = True, f"REC into {self.writer.path} ({self.writer.backend})"
            self.log(res.message)
        self.message = res.message
        return res

    def on_save(self, _req, res):
        with self.lock:
            if not self.recording or self.frames == 0:
                self.recording = False
                res.success, res.message = False, "nothing recorded"
                return res
            self.recording, self.saving = False, True
            writer, frames = self.writer, self.frames
            if self._regrasp_open is not None:               # a range left open ends with the episode
                self.regrasp_ranges.append([self._regrasp_open, frames])
                self._regrasp_open = None
            ranges = [list(r) for r in self.regrasp_ranges]

        def work():                           # the video encoding takes seconds: not on a service thread
            try:
                saved = writer.save_episode()
                writer.close()                # finalize, so the dataset is loadable after every episode
                ep = int(saved["episode"])
                write_annotation(saved["path"], ep, grasps=len(ranges) + 1, regrasp_ranges=ranges)
                self.episodes, self._episodes_of = ep + 1, str(self.p("dataset"))
                self.last = {"ep": ep, "frames": frames, "grasps": len(ranges) + 1, "dataset": Path(saved["path"]).name}
                line = f"episode {ep + 1} saved: {frames} frames, {len(ranges) + 1} grasp(s), in {saved['path']}"
            except Exception as e:  # noqa: BLE001
                line = f"SAVE FAILED: {type(e).__name__}: {e}"
            with self.lock:
                self.writer = None            # re-opened (appending) by the next start
                self.saving = False
            self.log(line)

        threading.Thread(target=work, daemon=True).start()
        res.success, res.message = True, f"saving {frames} frames"
        return res

    def on_discard(self, _req, res):
        with self.lock:
            if self.recording and self.writer is not None:
                self.writer.discard_episode()
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
        if not self.enabled:
            return
        pose = self.pose = self.observer.ee_pose()
        if pose is None:
            return
        now = time.monotonic()
        self._ticks = [t for t in self._ticks if now - t < 2.0] + [now]
        if self.target is None:
            self.target = list(pose)
        p = self.p
        dt = 1.0 / float(p("fps"))
        axes = self.device.axes if self.device_ok else dict.fromkeys(AXES, 0.0)
        buttons = self.device.buttons if self.device_ok else []
        self.target = integrate(self.target, axes, self.axis_map, float(p("linear_speed")), float(p("angular_speed")), dt)
        # the target never runs more than a step ahead of the tool or out of the box, so letting go stops the arm
        xyz = clamp_box(clamp_step(self.target[:3], pose[:3], float(p("max_step_m"))),
                        [float(v) for v in p("workspace_min")], [float(v) for v in p("workspace_max")])
        xyz, _ = keep_out(xyz, **self.limits)   # never over the robot's own body or under the floor (robot.yaml arm.workspace)
        self.target[:3] = list(xyz)
        self.grasp = step_grasp(self.grasp, buttons, float(p("grasp_close_rate")), float(p("grasp_open_rate")),
                                int(p("button_close")), int(p("button_open")))
        hold, asked = self.web_hold           # the page's hold buttons, at the same rates; they let go by themselves after 0.6 s
        if hold and time.monotonic() - asked < 0.6:
            self.grasp = step_grasp(self.grasp, [hold > 0, hold < 0], float(p("grasp_close_rate")), float(p("grasp_open_rate")))
        curls = [self.grasp * float(m) for m in p("max_curls")]
        msg = PoseStamped()
        msg.header.frame_id = self.observer.base_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(v) for v in xyz]
        (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w) = rotvec_to_quat(self.target[3:])
        self.pose_pub.publish(msg)
        self.hand_pub.publish(Float32MultiArray(data=curls))
        if self.recording:
            obs = self.observer.observe()
            if obs is None:
                return
            state, images, _age = obs
            with self.lock:
                if self.recording and self.writer is not None:
                    self.writer.add_frame(images, state, [float(v) for v in self.target] + curls, str(p("task")))
                    self.frames += 1

    def publish_state(self):
        w = self.writer
        dataset, fps = str(self.p("dataset")), float(self.p("fps"))
        if dataset != self._episodes_of and not self.saving:     # another dataset chosen: how much it already holds
            self.episodes, self._episodes_of = (episodes_on_disk(self._root() / dataset) if dataset else 0), dataset
        hz = (len(self._ticks) - 1) / (self._ticks[-1] - self._ticks[0]) if self.enabled and len(self._ticks) > 2 else 0.0
        self.state_pub.publish(String(data=json.dumps({
            "device": self.device_ok, "enabled": self.enabled, "recording": self.recording, "saving": self.saving,
            "frames": self.frames, "rec_s": self.frames / fps, "episodes": self.episodes,
            "grasps": len(self.regrasp_ranges) + 1, "regrasp_open": self._regrasp_open is not None,
            "grasp": round(self.grasp, 3), "dataset": dataset, "task": str(self.p("task")),
            "pose": [round(float(v), 4) for v in self.pose] if self.enabled and self.pose else [],
            "tick_hz": round(hz, 1), "last": self.last,
            "backend": getattr(w, "backend", None), "message": self.message})))


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
        try:
            if node.writer is not None:
                node.writer.close()
        except Exception:  # noqa: BLE001
            pass
        node.device.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
