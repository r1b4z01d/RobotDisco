"""Wave action server: play a recorded gesture collision-aware.

  1. planning   MoveIt plans from where the arm is to the recording's first sample (the chassis keepout box
                and the robot's own links are obstacles), executed on the trajectory controller
  2. moving     the recorded samples go to the controller as one trajectory at speed_scale, finger wiggle on
  3. returning  MoveIt plans back to `return_to` (vla_home | start | none)

Preconditions: joint states, the arm RUNNING (ArmPower STATUS; auto_power_up calls POWER_ON), move_group
and the controller up. Cancel stops the running trajectory. v1's ur_motion.py did the same over servoJ with
hand-tuned staged ramps; the planner replaces the staging.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import threading
import time

import rclpy
import yaml
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import SetBool, Trigger

from rd_interfaces.action import Wave
from rd_interfaces.srv import ArmPower

from . import motions
from .moveit_client import JOINT_NAMES, MoveItClient, trajectory_msg


class WaveServer(Node):
    def __init__(self):
        super().__init__("wave_server")
        self.declare_parameter("controller", "/ur5e/scaled_joint_trajectory_controller")
        self.declare_parameter("auto_power_up", True)
        self.declare_parameter("require_running", True)
        self.declare_parameter("approach_velocity_scale", 0.3)
        self.declare_parameter("max_dq", 0.05)              # per-sample joint step cap (rad)
        self.declare_parameter("start_tolerance", 0.01)     # rad: skip the approach plan when already there
        self.declare_parameter("keepout_file", "")
        self.declare_parameter("poses_file", "")
        self.cb = ReentrantCallbackGroup()
        self.moveit = MoveItClient(self, self.get_parameter("controller").value, self.cb)
        self.power = self.create_client(ArmPower, "/arm/power", callback_group=self.cb)
        self.wiggle = self.create_client(SetBool, "/hand/wiggle", callback_group=self.cb)
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=self.cb)
        self._q = None
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10, callback_group=self.cb)
        self._lock = threading.Lock()
        self._active = None
        self.server = ActionServer(self, Wave, "wave", execute_callback=self.execute, goal_callback=self.on_goal,
                                   cancel_callback=lambda h: CancelResponse.ACCEPT, callback_group=self.cb)
        self.poses = motions.load_poses(self.get_parameter("poses_file").value or None)
        self._scene_done = False
        self.create_timer(2.0, self.publish_keepout, callback_group=self.cb)
        self.create_service(Trigger, "/arm/go_home", self.on_go_home, callback_group=self.cb)
        self.get_logger().info(f"motions: {sorted(motions.list_motions())}; poses: {sorted(self.poses)}")

    # ---- state
    def on_joints(self, m: JointState):
        pos = dict(zip(m.name, m.position))
        if all(n in pos for n in JOINT_NAMES):
            self._q = [float(pos[n]) for n in JOINT_NAMES]

    def current_q(self, timeout=3.0):
        t0 = time.monotonic()
        while self._q is None and time.monotonic() - t0 < timeout:
            time.sleep(0.05)
        return list(self._q) if self._q is not None else None

    def publish_keepout(self):
        """The chassis keepout box (config/keepout.yaml) into the planning scene, once move_group answers."""
        if self._scene_done or not self.scene.wait_for_service(timeout_sec=0.1):
            return
        path = self.get_parameter("keepout_file").value or str(motions._share("config", "keepout.yaml"))
        try:
            objects = (yaml.safe_load(open(path)) or {}).get("objects") or []
        except OSError as e:
            self.get_logger().warn(f"no keepout file: {e}")
            self._scene_done = True
            return
        scene = PlanningScene(is_diff=True)
        for o in objects:
            co = CollisionObject()
            co.id = str(o["id"])
            co.header.frame_id = str(o.get("frame", "base_link"))
            box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in o["box"]])
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in o["center"])
            pose.orientation.w = 1.0
            co.primitives.append(box)
            co.primitive_poses.append(pose)
            co.operation = CollisionObject.ADD
            scene.world.collision_objects.append(co)
        res = self.scene.call(ApplyPlanningScene.Request(scene=scene))
        self._scene_done = bool(res.success)
        self.get_logger().info(f"planning scene: {[o['id'] for o in objects]} {'added' if res.success else 'NOT added'}")

    # ---- /arm/go_home: a MoveIt-planned move to vla_home (policies start there; the runner calls it)
    def on_go_home(self, req, res):
        with self._lock:
            if self._active is not None:
                res.success, res.message = False, "a wave is running"
                return res
        q = self.current_q()
        if q is None:
            res.success, res.message = False, "no joint states"
            return res
        ok, note = self.arm_ready()
        if not ok:
            res.success, res.message = False, note
            return res
        if not self.moveit.ready(timeout=5.0):
            res.success, res.message = False, "move_group or the trajectory controller is not up"
            return res
        home = self.poses["vla_home"]
        if motions.joint_distance(q, home) < float(self.get_parameter("start_tolerance").value):
            res.success, res.message = True, "already home"
            return res
        vel = float(self.get_parameter("approach_velocity_scale").value)
        traj = self.moveit.plan_to_joints(home, q_start=q, velocity_scale=vel, accel_scale=vel)
        if traj is None:
            res.success, res.message = False, self.moveit.message
            return res
        res.success, res.message = self.moveit.execute(traj.joint_trajectory)
        if res.success:
            res.message = "home"
        return res

    # ---- goal gating
    def on_goal(self, goal):
        with self._lock:
            if self._active is not None:
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def arm_ready(self) -> tuple[bool, str]:
        if not self.get_parameter("require_running").value:
            return True, ""
        if not self.power.wait_for_service(timeout_sec=2.0):
            return False, "arm_not_ready: no /arm/power service"
        st = self.power.call(ArmPower.Request(command=ArmPower.Request.STATUS))
        if st.robot_mode == "RUNNING" and st.safety_status == "NORMAL":
            return True, ""
        if not self.get_parameter("auto_power_up").value:
            return False, f"arm_not_ready: {st.robot_mode} / {st.safety_status}"
        up = self.power.call(ArmPower.Request(command=ArmPower.Request.POWER_ON, timeout_s=60.0))
        if not up.success or up.robot_mode != "RUNNING":
            return False, f"arm_not_ready: {up.message}"
        return True, up.message

    # ---- execution
    def execute(self, handle):
        with self._lock:
            self._active = handle
        t0 = time.monotonic()
        result = Wave.Result()
        try:
            ok, msg = self._run(handle, handle.request)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"error: {type(e).__name__}: {e}"
        finally:
            self._set_wiggle(False)
            with self._lock:
                self._active = None
        result.success = ok
        result.message = msg
        result.seconds = time.monotonic() - t0
        if handle.is_cancel_requested:
            handle.canceled()
        elif ok:
            handle.succeed()
        else:
            handle.abort()
        self.get_logger().info(f"wave '{handle.request.motion}': {msg} ({result.seconds:.1f}s)")
        return result

    def _set_wiggle(self, on: bool):
        if self.wiggle.wait_for_service(timeout_sec=0.5):
            self.wiggle.call_async(SetBool.Request(data=on))

    def _feedback(self, handle, phase, progress):
        fb = Wave.Feedback(phase=phase, progress=float(progress))
        handle.publish_feedback(fb)

    def _run(self, handle, goal) -> tuple[bool, str]:
        cancelled = lambda: handle.is_cancel_requested  # noqa: E731 - a property on the goal handle
        try:
            m = motions.load_motion(goal.motion or "wave")
        except (FileNotFoundError, ValueError) as e:
            return False, f"motion: {e}"
        q0 = self.current_q()
        if q0 is None:
            return False, "arm_not_ready: no joint states"
        ok, note = self.arm_ready()
        if not ok:
            return False, note
        if not self.moveit.ready(timeout=5.0):
            return False, "arm_not_ready: move_group or the trajectory controller is not up"
        vel = float(self.get_parameter("approach_velocity_scale").value)
        tol = float(self.get_parameter("start_tolerance").value)

        def plan_and_go(q_target, phase, frac):
            self._feedback(handle, phase, frac)
            here = self.current_q() or q0
            if motions.joint_distance(here, q_target) < tol:
                return True, "already there"
            traj = self.moveit.plan_to_joints(q_target, q_start=here, velocity_scale=vel, accel_scale=vel, cancelled=cancelled)
            if traj is None:
                return False, f"collision: {self.moveit.message}" if "failed" in self.moveit.message else self.moveit.message
            return self.moveit.execute(traj.joint_trajectory, cancelled=cancelled)

        # 1. approach
        ok, msg = plan_and_go(m["q"][0], "planning", 0.0)
        if not ok:
            return False, "cancelled" if cancelled() else msg
        # 2. the recording
        self._feedback(handle, "moving", 0.1)
        q, capped = motions.cap_steps(m["q"], float(self.get_parameter("max_dq").value))
        points = motions.resample(q, m["rate"], goal.speed_scale or 1.0)
        total = points[-1][0] if points else 0.0
        if goal.wiggle_fingers:
            self._set_wiggle(True)
        def on_progress(fb):
            t = fb.feedback.actual.time_from_start
            self._feedback(handle, "moving", 0.1 + 0.8 * min(1.0, (t.sec + t.nanosec * 1e-9) / max(total, 1e-3)))
        ok, msg = self.moveit.execute(trajectory_msg(points), cancelled=cancelled, progress=on_progress)
        self._set_wiggle(False)
        if not ok:
            return False, "cancelled" if cancelled() else msg
        # 3. return
        where = (goal.return_to or "home").lower()
        if where == "none":
            return True, f"done ({len(q)} samples{', ' + str(capped) + ' capped' if capped else ''}; no return)"
        target = q0 if where == "start" else self.poses.get("vla_home" if where == "home" else where)
        if target is None:
            return False, f"unknown return pose {where!r}"
        ok, msg = plan_and_go(target, "returning", 0.9)
        if not ok:
            return False, "cancelled" if cancelled() else msg
        self._feedback(handle, "returning", 1.0)
        return True, f"done ({len(q)} samples at {goal.speed_scale or 1.0:g}x{', ' + str(capped) + ' capped' if capped else ''}; returned to {where}{'; ' + note if note else ''})"


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = WaveServer()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
