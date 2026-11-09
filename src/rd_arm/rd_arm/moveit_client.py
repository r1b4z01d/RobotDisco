"""A small MoveIt client for the Wave server and the streamer: plan a joint-space goal with move_group
(collision-aware, no moveit_py), execute a trajectory on a FollowJointTrajectory controller, and IK via
/compute_ik. rclpy only."""
from __future__ import annotations

import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, PlanningOptions, RobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .motions import JOINTS

PREFIX = "ur_"
JOINT_NAMES = [PREFIX + j for j in JOINTS]
GROUP = "ur_manipulator"


def _wait(fut, spin_ok=lambda: True, timeout: float | None = None):
    t0 = time.monotonic()
    while not fut.done():
        if not spin_ok() or (timeout is not None and time.monotonic() - t0 > timeout):
            return None
        time.sleep(0.02)
    return fut.result()


def trajectory_msg(points: list[tuple[float, list[float]]]) -> JointTrajectory:
    """(time_from_start, q) list -> JointTrajectory over the UR joints."""
    jt = JointTrajectory()
    jt.joint_names = list(JOINT_NAMES)
    for t, q in points:
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in q]
        p.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
        jt.points.append(p)
    return jt


class MoveItClient:
    def __init__(self, node, controller: str = "/ur5e/joint_trajectory_controller", callback_group=None):
        self.node = node
        self.move_group = ActionClient(node, MoveGroup, "move_action", callback_group=callback_group)
        self.jtc = ActionClient(node, FollowJointTrajectory, controller + "/follow_joint_trajectory", callback_group=callback_group)
        self.ik = node.create_client(GetPositionIK, "compute_ik", callback_group=callback_group)
        self._goal = None

    def ready(self, timeout: float = 5.0) -> bool:
        return self.move_group.wait_for_server(timeout_sec=timeout) and self.jtc.wait_for_server(timeout_sec=timeout)

    # ---- planning
    def plan_to_joints(self, q_goal, q_start=None, velocity_scale=0.3, accel_scale=0.3, planning_time=5.0,
                       cancelled=lambda: False):
        """Plan (only) to a joint goal. Returns the moveit RobotTrajectory or None; `message` says why."""
        req = MotionPlanRequest()
        req.group_name = GROUP
        req.allowed_planning_time = float(planning_time)
        req.num_planning_attempts = 3
        req.max_velocity_scaling_factor = float(velocity_scale)
        req.max_acceleration_scaling_factor = float(accel_scale)
        if q_start is not None:
            rs = RobotState()
            rs.joint_state = JointState(name=list(JOINT_NAMES), position=[float(v) for v in q_start])
            req.start_state = rs
        else:
            req.start_state.is_diff = True
        c = Constraints()
        for name, v in zip(JOINT_NAMES, q_goal):
            jc = JointConstraint(joint_name=name, position=float(v), tolerance_above=0.005, tolerance_below=0.005, weight=1.0)
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        goal = MoveGroup.Goal(request=req, planning_options=PlanningOptions(plan_only=True))
        send = _wait(self.move_group.send_goal_async(goal), lambda: not cancelled(), 10.0)
        if send is None or not send.accepted:
            self.message = "move_group rejected the plan request"
            return None
        res = _wait(send.get_result_async(), lambda: not cancelled(), planning_time + 10.0)
        if res is None:
            send.cancel_goal_async()
            self.message = "planning cancelled"
            return None
        code = res.result.error_code.val
        if code != 1:
            self.message = f"planning failed (MoveIt error {code})"
            return None
        self.message = "planned"
        return res.result.planned_trajectory

    # ---- execution
    def execute(self, jt: JointTrajectory, cancelled=lambda: False, progress=lambda f: None) -> tuple[bool, str]:
        if jt.points:
            # The controller (JTC, allow_nonzero_velocity_at_trajectory_end false) rejects a trajectory that does not
            # end at rest. cuMotion's last interpolated sample keeps a residual 0.002..0.02 rad/s where MoveIt's time
            # parameterization ends OMPL's at exactly zero; the end of a motion is a rest point for every planner.
            last = jt.points[-1]
            last.velocities = [0.0] * len(last.velocities)
            last.accelerations = [0.0] * len(last.accelerations)
        goal = FollowJointTrajectory.Goal(trajectory=jt)
        send = _wait(self.jtc.send_goal_async(goal, feedback_callback=lambda fb: progress(fb)), lambda: not cancelled(), 10.0)
        if send is None or not send.accepted:
            return False, "controller rejected the trajectory"
        self._goal = send
        total = jt.points[-1].time_from_start.sec + jt.points[-1].time_from_start.nanosec * 1e-9 if jt.points else 0.0
        result_fut = send.get_result_async()
        t0 = time.monotonic()
        while not result_fut.done():
            if cancelled():
                send.cancel_goal_async()
                _wait(result_fut, timeout=5.0)
                return False, "cancelled"
            if time.monotonic() - t0 > total + 15.0:
                send.cancel_goal_async()
                return False, "controller did not finish in time"
            time.sleep(0.02)
        r = result_fut.result()
        if r.status != GoalStatus.STATUS_SUCCEEDED:
            return False, f"controller aborted (status {r.status}, error {r.result.error_code}: {r.result.error_string})"
        return True, "done"

    def cancel(self):
        if self._goal is not None:
            self._goal.cancel_goal_async()

    # ---- IK
    def solve_ik(self, pose: PoseStamped, seed, timeout: float = 0.05, avoid_collisions: bool = False):
        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP
        req.ik_request.robot_state.joint_state = JointState(name=list(JOINT_NAMES), position=[float(v) for v in seed])
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = avoid_collisions
        req.ik_request.timeout = Duration(sec=0, nanosec=int(timeout * 1e9))
        res = self.ik.call(req)
        if res.error_code.val != 1:
            return None
        js = res.solution.joint_state
        pos = dict(zip(js.name, js.position))
        return [float(pos[n]) for n in JOINT_NAMES]
