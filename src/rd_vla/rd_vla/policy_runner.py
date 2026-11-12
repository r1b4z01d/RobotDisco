"""RunPolicy action server: a VLA policy drives the arm and the hand until it finishes.

Per step at `fps`: observe (rd_vla.observer) -> the inference server (rd_vla.wire) -> the action's EE pose
(clamped: max step per tick, workspace box) to rd_arm's streamer on /arm/target_pose, the curls to
/hand/curls -> optional recording frame. Ends on max_seconds / max_steps / cancel / `done_when` (the grasp
level: closed = mean commanded curl >= 0.7 held for `still` seconds; open = <= 0.2 after having been
closed) / the inference server failing. Before: select the checkpoint on the server, go home (rd_arm's
/arm/go_home), streaming on. After: streaming off, home again when the goal says so, the eval.json.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import csv
import os
import threading
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray
from std_srvs.srv import SetBool, Trigger

from rd_bringup import config, units
from rd_interfaces.action import RunPolicy

from . import evals, wire
from .geometry import clamp_box, clamp_step, keep_out, rotvec_to_quat, workspace_limits
from .observer import Observer
from .recorder import EpisodeWriter
from .schema import CAMERA_KEYS, split_action


class PolicyRunner(Node):
    def __init__(self):
        super().__init__("policy_runner")
        dp = self.declare_parameter
        dp("fps", 12.0)
        dp("infer_host", "127.0.0.1")
        dp("infer_port", int(config.get("robot.ports.infer_server", 8777)))
        dp("cameras", list(CAMERA_KEYS))
        dp("camera_topic", "/cameras/{name}/image")
        dp("base_frame", "ur_base")      # the UR controller's base frame (v1's RTDE poses, the datasets): base_link turned 180 deg about Z
        dp("tool_frame", "ur_tool0")
        dp("max_step_m", 0.03)                       # Cartesian step cap per tick (v1 rtde_arm max_step)
        dp("workspace_min", [-1.0, -1.0, -0.2])      # UR base frame box the target is clamped into
        dp("workspace_max", [1.0, 1.0, 1.0])
        dp("obs_max_age_s", 0.5)
        dp("done_closed_level", 0.7)
        dp("done_open_level", 0.2)
        dp("done_still_s", 3.0)                     # default when the goal leaves still_seconds at 0
        dp("record_root", "")
        dp("evals_dir", "")
        dp("operator", "")
        # the supervisor unit that serves the policy: started for a run (or by ~/prepare_server, when a landmark task
        # sets off) and stopped once nothing has used it for a while. "" or "none" = whatever is serving stays as it is
        dp("server_unit", "")
        dp("server_start_timeout_s", 180.0)          # unit start + model load (14 s on the Orin once the weights are cached)
        dp("server_idle_stop_s", 300.0)              # after a run, and three times as long after a prepare nobody used
        self.cb = ReentrantCallbackGroup()
        self.observer = Observer(self, tuple(self.get_parameter("cameras").value), self.get_parameter("camera_topic").value,
                                 base_frame=self.get_parameter("base_frame").value, tool_frame=self.get_parameter("tool_frame").value,
                                 callback_group=self.cb)
        self.pose_pub = self.create_publisher(PoseStamped, "/arm/target_pose", 10)
        self.hand_pub = self.create_publisher(Float32MultiArray, "/hand/curls", 10)
        self.stream = self.create_client(SetBool, "/arm/stream", callback_group=self.cb)
        self.go_home = self.create_client(Trigger, "/arm/go_home", callback_group=self.cb)
        self._lock = threading.Lock()
        self._active = None
        self._program_running = None       # the UR driver's robot_program_running; None where nothing publishes it (the rig's mock arm)
        self.create_subscription(Bool, "/ur5e/io_and_status_controller/robot_program_running",
                                 lambda m: setattr(self, "_program_running", bool(m.data)),
                                 QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE),
                                 callback_group=self.cb)
        self.server = ActionServer(self, RunPolicy, "run_policy", execute_callback=self.execute, goal_callback=self.on_goal,
                                   cancel_callback=lambda h: CancelResponse.ACCEPT, callback_group=self.cb)
        unit = str(self.get_parameter("server_unit").value)
        self.server_unit = "" if unit.lower() in ("", "none") else unit
        self._server_stop_at = None              # monotonic time to stop the unit at; None = no stop pending
        self.create_service(Trigger, "/vla/prepare_server", self.on_prepare, callback_group=self.cb)
        self.create_timer(5.0, self._server_tick, callback_group=self.cb)
        self.get_logger().info(f"inference server {self.get_parameter('infer_host').value}:{self.get_parameter('infer_port').value}"
                               + (f", unit '{self.server_unit}' started on demand" if self.server_unit else ""))

    def on_goal(self, goal):
        with self._lock:
            return GoalResponse.REJECT if self._active is not None else GoalResponse.ACCEPT

    # ---- the server's unit, on demand
    def on_prepare(self, req, res):
        """Start the server's unit now and come back at once: the model loads while the robot drives to the task."""
        if not self.server_unit:
            res.success, res.message = True, "the server is not a unit here"
            return res
        res.success, res.message = units.start(self.server_unit)
        with self._lock:
            if self._active is None:             # nobody may ever come for it: stop it again later
                self._server_stop_at = time.monotonic() + 3.0 * float(self.get_parameter("server_idle_stop_s").value)
        self.get_logger().info(f"prepare: unit '{self.server_unit}' {res.message}")
        return res

    def _ensure_server(self, cancelled) -> tuple[bool, str]:
        """The unit up and its port accepting before a run connects. True at once where no unit is managed."""
        if not self.server_unit:
            return True, ""
        self._server_stop_at = None
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        if units.wait_port(p("infer_host"), int(p("infer_port")), 0.5):
            return True, "already serving"
        ok, why = units.start(self.server_unit)
        if not ok:
            return False, f"could not start the unit '{self.server_unit}': {why}"
        self.get_logger().info(f"unit '{self.server_unit}' {why}; waiting for the model")
        if units.wait_port(p("infer_host"), int(p("infer_port")), float(p("server_start_timeout_s")), cancelled):
            return True, "started"
        return False, f"the unit '{self.server_unit}' did not serve within {float(p('server_start_timeout_s')):.0f} s (state {units.state(self.server_unit)})"

    def _server_tick(self):
        with self._lock:
            due = self.server_unit and self._active is None and self._server_stop_at is not None and time.monotonic() >= self._server_stop_at
            if due:
                self._server_stop_at = None
        if due and units.state(self.server_unit) in units.ACTIVE:
            ok, why = units.stop(self.server_unit)
            self.get_logger().info(f"unit '{self.server_unit}' idle: {why if ok else 'stop failed: ' + why}")

    # ---- helpers
    def _stream(self, on: bool) -> tuple[bool, str]:
        if not self.stream.wait_for_service(timeout_sec=3.0):
            return False, "no /arm/stream service (rd_arm not running)"
        r = self.stream.call(SetBool.Request(data=on))
        return bool(r.success), r.message

    def _home(self) -> tuple[bool, str]:
        if not self.go_home.wait_for_service(timeout_sec=3.0):
            return False, "no /arm/go_home service"
        r = self.go_home.call(Trigger.Request())
        return bool(r.success), r.message

    def _feedback(self, handle, step, t0, phase, latency_ms):
        fb = RunPolicy.Feedback(step=int(step), seconds=float(time.monotonic() - t0), phase=phase, obs_latency_ms=float(latency_ms))
        handle.publish_feedback(fb)

    # ---- execution
    def execute(self, handle):
        with self._lock:
            self._active = handle
        t0 = time.monotonic()
        result = RunPolicy.Result()
        state = {"steps": 0, "run_dir": ""}
        self.observer.cameras_on()
        try:
            ok, msg = self._run(handle, handle.request, t0, state)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"error: {type(e).__name__}: {e}"
            self.get_logger().error(msg)
        finally:
            self.observer.cameras_off()
            with self._lock:
                self._active = None
                if self.server_unit:
                    self._server_stop_at = time.monotonic() + float(self.get_parameter("server_idle_stop_s").value)
        result.success, result.message = ok, msg
        result.steps = int(state["steps"])
        result.seconds = float(time.monotonic() - t0)
        result.run_dir = state["run_dir"]
        if handle.is_cancel_requested:
            handle.canceled()
        elif ok:
            handle.succeed()
        else:
            handle.abort()
        self.get_logger().info(f"run_policy '{handle.request.task}': {msg} ({result.steps} steps, {result.seconds:.1f}s)")
        return result

    def _run(self, handle, goal, t0, out) -> tuple[bool, str]:
        cancelled = lambda: handle.is_cancel_requested  # noqa: E731
        fps = float(self.get_parameter("fps").value)
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        # 1. the server and the checkpoint
        self._feedback(handle, 0, t0, "serving", 0)
        ok, why = self._ensure_server(cancelled)
        if not ok:
            return False, f"inference server: {why}"
        try:
            client = wire.PolicyClient(p("infer_host"), int(p("infer_port"))).connect()
        except OSError as e:
            return False, f"inference server unreachable: {e}"
        writer = None
        log = None
        step = 0
        ended = ""
        info: dict = {}
        try:
            if goal.checkpoint:
                rep = client.select(goal.checkpoint)
                if not rep.get("ok"):
                    return False, f"checkpoint: {rep.get('error', 'select failed')}"
            info = client.info()
            client.reset()
            # 2. home, observations
            if goal.home_first:
                self._feedback(handle, 0, t0, "homing", 0)
                ok, msg = self._home()
                if not ok:
                    return False, f"homing failed: {msg}"
            ready, why = self.observer.wait_ready(float(p("obs_max_age_s")) + 1.0, cancelled=cancelled)
            if not ready:
                return False, f"no observation: {why}"
            # 3. the eval record, the recording, streaming on. From here every exit, an exception included, goes
            # through step 5: the arm must not be left streaming and the run must leave its eval.json
            reg = evals.EvalRegistry(p("evals_dir") or None)
            run_name = time.strftime("%Y%m%d_%H%M%S_") + (goal.task.strip().lower().replace(" ", "_")[:30] or "run")
            run_dir = reg.path(run_name)
            run_dir.mkdir(parents=True, exist_ok=True)
            out["run_dir"] = str(run_dir)
            try:
                if goal.record:
                    root = Path(p("record_root") or os.environ.get("RD_RECORD_ROOT") or (config.data_dir() / "outputs" / "datasets"))
                    repo = f"rd/{time.strftime('%Y%m%d')}_{goal.task.strip().lower().replace(' ', '_')[:40] or 'run'}"
                    writer = EpisodeWriter(root, repo, fps=int(fps), cameras=self.observer.cameras)
                    out["dataset"] = str(writer.path)
                ok, msg = self._stream(True)
                if not ok:
                    ended = f"streaming: {msg}"
                else:
                    log = open(run_dir / "steps.csv", "w", newline="")
                    csvw = csv.writer(log)
                    csvw.writerow(["t", "step", "obs_age_ms", "infer_ms"] + [f"a{i}" for i in range(10)] + [f"s{i}" for i in range(16)])
                    # 4. the loop
                    period = 1.0 / fps
                    next_t = time.monotonic()
                    step = 0
                    closed_since = None
                    was_closed = False
                    ended = ""
                    max_step = float(p("max_step_m"))
                    limits = workspace_limits()      # the robot's own body and the floor (robot.yaml arm.workspace)
                    lo, hi = [float(v) for v in p("workspace_min")], [float(v) for v in p("workspace_max")]
                    clevel, olevel = float(p("done_closed_level")), float(p("done_open_level"))
                    still_s = float(goal.still_seconds) if goal.still_seconds > 0 else float(p("done_still_s"))
                    done_when = (goal.done_when or "").strip().lower()
                    self._feedback(handle, 0, t0, "running", 0)
                    was_running = False
                    while True:
                        if cancelled():
                            ended = "cancelled"
                            break
                        # The UR's control program stops on a protective stop. The first run on the robot went on
                        # sending targets to a stopped arm until the policy closed the hand and it was scored a success.
                        if self._program_running:
                            was_running = True
                        elif was_running:
                            ended = "arm stopped (protective stop?)"
                            break
                        if goal.max_seconds and time.monotonic() - t0 > goal.max_seconds:
                            ended = "timeout"
                            break
                        if goal.max_steps and step >= goal.max_steps:
                            ended = "max_steps"
                            break
                        obs = self.observer.observe()
                        if obs is None:
                            ended = "observation lost"
                            break
                        st, images, age = obs
                        if age > float(p("obs_max_age_s")):
                            self.get_logger().warn(f"observation {age * 1000:.0f} ms old", throttle_duration_sec=2.0)
                        ti = time.monotonic()
                        try:
                            action = client.infer(st, images, goal.task)
                        except (RuntimeError, OSError) as e:
                            ended = f"inference failed: {e}"
                            break
                        infer_ms = (time.monotonic() - ti) * 1000
                        pose, curls = split_action(action)
                        xyz, _ = keep_out(clamp_box(clamp_step(pose[:3], st[:3], max_step), lo, hi), **limits)
                        msg = PoseStamped()
                        msg.header.frame_id = self.observer.base_frame
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = xyz
                        qx, qy, qz, qw = rotvec_to_quat(pose[3:])
                        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = qx, qy, qz, qw
                        self.pose_pub.publish(msg)
                        self.hand_pub.publish(Float32MultiArray(data=[float(v) for v in curls]))
                        if writer is not None:
                            writer.add_frame(images, st, action, goal.task)
                        csvw.writerow([f"{time.monotonic() - t0:.3f}", step, f"{age * 1000:.0f}", f"{infer_ms:.0f}"]
                                      + [f"{v:.4f}" for v in action] + [f"{v:.4f}" for v in st])
                        step += 1
                        out["steps"] = step
                        self._feedback(handle, step, t0, "running", age * 1000)
                        # the grasp-level end condition (done_when closed | open)
                        level = sum(curls) / len(curls)
                        if level >= clevel:
                            closed_since = closed_since or time.monotonic()
                            if time.monotonic() - closed_since >= still_s:
                                was_closed = True
                                if done_when == "closed":
                                    ended = "done"
                                    break
                        else:
                            closed_since = None
                            if done_when == "open" and was_closed and level <= olevel:
                                ended = "done"
                                break
                        next_t += period
                        delay = next_t - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                        else:
                            next_t = time.monotonic()
            except Exception as e:  # noqa: BLE001 - reported on the eval record; the arm is stopped below
                ended = f"error: {type(e).__name__}: {e}"
                self.get_logger().error(ended)
        finally:
            client.close()
        # 5. after
        if log is not None:
            log.close()
        self._feedback(handle, step, t0, "returning", 0)
        self._stream(False)
        if goal.home_first and ended in ("done", "timeout", "max_steps"):
            self._home()
        if writer is not None:
            if step > 0 and ended in ("done", "timeout", "max_steps"):
                saved = writer.save_episode()
                self.get_logger().info(f"episode saved: {saved}")
            else:
                writer.discard_episode()
            writer.close()
        reg.create(run_name, {"task": goal.task, "dataset": out.get("dataset", ""), "model": goal.checkpoint or info.get("checkpoint") or "", "checkpoint": str(info.get("resolved") or ""),
                              "backend": info.get("backend", ""), "steps": step, "duration_s": round(time.monotonic() - t0, 1),
                              "fps": fps, "ended": ended, "landmark": goal.landmark, "operator": p("operator"), "episodes": 1,
                              "successes": 1 if ended == "done" else 0, "remote": f"{p('infer_host')}:{p('infer_port')}"})
        ok = ended in ("done", "timeout", "max_steps")
        return ok, ended


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = PolicyRunner()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
