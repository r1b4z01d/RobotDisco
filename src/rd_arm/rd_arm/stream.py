"""The streaming controller for policies and teleop (v1's servoJ thread, now on ros2_control).

  srv  /arm/stream           std_srvs/SetBool  on: forward_position_controller takes the arm (the trajectory
                                               controllers go inactive); off: back to the trajectory controller
  sub  /arm/target_pose      geometry_msgs/PoseStamped   Cartesian target (any TF frame); IK once per NEW target
  sub  /arm/target_joints    std_msgs/Float64MultiArray  joint target, 6 values
  pub  /ur5e/forward_position_controller/commands  at `rate_hz`: the sent joints follow the solution through a
                                               critically damped filter with speed and acceleration limits
  pub  /arm/stream_state     std_msgs/String   "on|off sends=.. ik=.. unreachable=.. rejected=.."

IK is MoveIt's /compute_ik (KDL in the rig; cuRobo through the same service on the Orin, phase 7). A solution
that jumps a joint by more than `max_jump` (an IK branch flip) is dropped and the previous target held.

Why a filter and not a step cap: the first run on the real arm (2026-11-10) ended 2.8 s in with the UR's
"C157 Collision detected by the Shoulder joint". Nothing was hit. The sent joints went from rest to 1.5 rad/s in
one 20 ms tick (the cap bounded the step, not its change), into a servoj loop the driver sets far stiffer than
v1's (gain 2000, lookahead 0.03 s; v1 ran 300 and 0.1 s and interpolated towards each target). The arm's own
torque check called that jerk a collision. Now what is sent starts from rest, accelerates at no more than
`max_accel`, and goes out often enough for the servoj loop to see a ramp and not a staircase.
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import threading
import time

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool

from .moveit_client import JOINT_NAMES, MoveItClient


def smooth_step(x, v, goal, dt: float, omega: float, vmax: float, amax: float):
    """One tick of the follower, per joint (pure, tested): a critically damped pull towards `goal` (natural
    frequency `omega`, rad/s) whose acceleration and speed are clamped. From rest it cannot jump: the first step
    is at most amax * dt * dt. Returns the new positions and velocities."""
    xs, vs = [], []
    for xi, vi, gi in zip(x, v, goal):
        a = omega * omega * (gi - xi) - 2.0 * omega * vi
        a = max(-amax, min(amax, a))
        vn = max(-vmax, min(vmax, vi + a * dt))
        xs.append(xi + vn * dt)
        vs.append(vn)
    return xs, vs


class Stream(Node):
    def __init__(self):
        super().__init__("arm_stream")
        self.declare_parameter("rate_hz", 125.0)        # what the servoj loop is fed; 50 Hz steps were a staircase to it
        self.declare_parameter("max_velocity", 1.0)     # rad/s per joint
        self.declare_parameter("max_accel", 5.0)        # rad/s^2 per joint: the bound whose absence tripped the UR's collision check
        self.declare_parameter("stiffness", 12.0)       # rad/s, the follower's natural frequency: a moving target is trailed by 2/stiffness s
        self.declare_parameter("max_jump", 0.5)         # rad: an IK solution further than this from the sent joints is a branch flip
        self.declare_parameter("controller_manager", "/ur5e/controller_manager")
        self.declare_parameter("stream_controller", "forward_position_controller")
        self.declare_parameter("trajectory_controllers", ["scaled_joint_trajectory_controller", "joint_trajectory_controller"])
        self.cb = ReentrantCallbackGroup()
        self.moveit = MoveItClient(self, callback_group=self.cb)
        cm = self.get_parameter("controller_manager").value
        self.switch = self.create_client(SwitchController, cm + "/switch_controller", callback_group=self.cb)
        self.list_controllers = self.create_client(ListControllers, cm + "/list_controllers", callback_group=self.cb)
        self.cmd_pub = self.create_publisher(Float64MultiArray, "/ur5e/" + self.get_parameter("stream_controller").value + "/commands", 10)
        self.state_pub = self.create_publisher(String, "/arm/stream_state", 10)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10, callback_group=self.cb)
        self.create_subscription(PoseStamped, "/arm/target_pose", self.on_pose, 10, callback_group=self.cb)
        self.create_subscription(Float64MultiArray, "/arm/target_joints", self.on_joint_target, 10, callback_group=self.cb)
        self.create_service(SetBool, "/arm/stream", self.on_enable, callback_group=self.cb)
        self._lock = threading.Lock()
        self.on = False
        self.actual = None
        self.sent = None
        self.velocity = [0.0] * 6
        self.goal = None
        self.sends = self.ik_calls = self.unreachable = self.rejected = 0
        self.dt = 1.0 / float(self.get_parameter("rate_hz").value)
        self.create_timer(self.dt, self.tick, callback_group=self.cb)
        self.create_timer(1.0, self.publish_state, callback_group=self.cb)

    def on_joints(self, m: JointState):
        pos = dict(zip(m.name, m.position))
        if all(n in pos for n in JOINT_NAMES):
            self.actual = [float(pos[n]) for n in JOINT_NAMES]

    # ---- enable / disable
    def on_enable(self, req, res):
        if not self.switch.wait_for_service(timeout_sec=3.0) or not self.list_controllers.wait_for_service(timeout_sec=3.0):
            res.success, res.message = False, "no controller manager"
            return res
        loaded = {c.name: c.state for c in self.list_controllers.call(ListControllers.Request()).controller}
        stream_ctl = self.get_parameter("stream_controller").value
        traj = [c for c in self.get_parameter("trajectory_controllers").value if c in loaded]
        if req.data:
            want, stop = [stream_ctl], [c for c in traj if loaded[c] == "active"]
        else:
            want = traj[:1]
            stop = [stream_ctl] if loaded.get(stream_ctl) == "active" else []
        if not want or want[0] not in loaded:
            res.success, res.message = False, f"controller {want[0] if want else '?'} is not loaded"
            return res
        r = self.switch.call(SwitchController.Request(activate_controllers=[c for c in want if loaded[c] != "active"],
                                                      deactivate_controllers=stop,
                                                      strictness=SwitchController.Request.STRICT, activate_asap=True))
        with self._lock:
            self.on = bool(req.data) and r.ok
            self.sent = None
            self.velocity = [0.0] * 6
            self.goal = None
        res.success = bool(r.ok)
        res.message = ("streaming on" if self.on else "streaming off") if r.ok else "controller switch failed"
        self.get_logger().info(res.message)
        return res

    # ---- targets
    def on_joint_target(self, m: Float64MultiArray):
        if len(m.data) != 6:
            return
        with self._lock:
            self.goal = [float(v) for v in m.data]

    def on_pose(self, p: PoseStamped):
        if not self.on or self.actual is None:
            return
        seed = self.sent or self.actual
        self.ik_calls += 1
        sol = self.moveit.solve_ik(p, seed)
        if sol is None:
            self.unreachable += 1
            self.get_logger().warn("target unreachable; holding", throttle_duration_sec=2.0)
            return
        if max(abs(a - b) for a, b in zip(sol, seed)) > float(self.get_parameter("max_jump").value):
            self.rejected += 1
            self.get_logger().warn("IK solution jumps (branch flip?); holding", throttle_duration_sec=2.0)
            return
        with self._lock:
            self.goal = sol

    # ---- the 50 Hz loop
    def tick(self):
        with self._lock:
            if not self.on or self.goal is None or self.actual is None:
                return
            p = lambda k: float(self.get_parameter(k).value)  # noqa: E731
            start = self.sent if self.sent is not None else self.actual
            nxt, self.velocity = smooth_step(start, self.velocity, self.goal, self.dt, p("stiffness"), p("max_velocity"), p("max_accel"))
            self.sent = nxt
        self.cmd_pub.publish(Float64MultiArray(data=nxt))
        self.sends += 1

    def publish_state(self):
        self.state_pub.publish(String(data=f"{'on' if self.on else 'off'} sends={self.sends} ik={self.ik_calls} "
                                          f"unreachable={self.unreachable} rejected={self.rejected}"))


def main():
    rclpy.init()
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    node = Stream()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
