"""GoToLandmark action server: a landmark name -> Nav2 NavigateToPose -> result.

    ros2 run rd_nav go_to_landmark
    ros2 action send_goal /go_to_landmark rd_interfaces/action/GoToLandmark "{landmark: dining-room}" --feedback

One goal at a time (a second goal while one runs is rejected, as the interfaces README says). `map` in the
goal must be the active map or empty. `relocalize_first` calls the tag corrector's relocalize_now service
and waits for AMCL to settle before planning. Cancel cancels the Nav2 goal.
"""
from __future__ import annotations

import math
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from rd_interfaces.action import GoToLandmark
from rd_nav.landmarks import LandmarkStore
from rd_nav.maps_store import MapsStore


def _quat_z(yaw: float):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class GoToLandmarkServer(Node):
    def __init__(self):
        super().__init__("go_to_landmark")
        self.declare_parameter("landmarks_dir", "")
        self.declare_parameter("maps_dir", "")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("relocalize_service", "/tag_corrector/relocalize_now")
        self.declare_parameter("relocalize_settle_sec", 2.0)
        self.declare_parameter("require_localized", True)   # refuse when map->base is not available
        self.landmarks = LandmarkStore(self.get_parameter("landmarks_dir").value or None)
        self.maps = MapsStore(self.get_parameter("maps_dir").value or None)
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.cb = ReentrantCallbackGroup()
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose", callback_group=self.cb)
        self.reloc = self.create_client(Trigger, self.get_parameter("relocalize_service").value, callback_group=self.cb)
        self._lock = threading.Lock()
        self._active = None          # the running goal handle
        self._nav_handle = None
        self.server = ActionServer(
            self, GoToLandmark, "go_to_landmark", execute_callback=self.execute,
            goal_callback=self.on_goal, cancel_callback=self.on_cancel, callback_group=self.cb)
        self.get_logger().info(f"{len(self.landmarks.load())} landmarks from {self.landmarks.file}; "
                               f"active map {self.maps.active()}")

    # ---- goal gating
    def on_goal(self, goal: GoToLandmark.Goal):
        with self._lock:
            if self._active is not None:
                self.get_logger().warn(f"rejecting '{goal.landmark}': a goal is already running")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_cancel(self, _handle):
        return CancelResponse.ACCEPT

    def localized(self) -> bool:
        try:
            self.tf.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
            return True
        except TransformException:
            return False

    def current_pose(self):
        t = self.tf.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        p = PoseStamped()
        p.header.frame_id = self.map_frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = t.transform.translation.x
        p.pose.position.y = t.transform.translation.y
        p.pose.orientation = t.transform.rotation
        return p

    # ---- execution
    def execute(self, handle):
        goal: GoToLandmark.Goal = handle.request
        result = GoToLandmark.Result()
        with self._lock:
            self._active = handle
        try:
            return self._run(handle, goal, result)
        finally:
            with self._lock:
                self._active = None
                self._nav_handle = None

    def _finish(self, handle, result, ok: bool, message: str):
        result.success = ok
        result.message = message
        try:
            result.final_pose = self.current_pose()
        except TransformException:
            pass
        if handle.is_cancel_requested:
            handle.canceled()
        elif ok:
            handle.succeed()
        else:
            handle.abort()
        self.get_logger().info(f"'{handle.request.landmark}': {message}")
        return result

    def _run(self, handle, goal, result):
        active_map = self.maps.active()
        if goal.map and goal.map != active_map:
            return self._finish(handle, result, False, f"wrong_map: active map is {active_map}, goal says {goal.map}")
        lm = self.landmarks.find(goal.landmark, active_map)
        if lm is None:
            return self._finish(handle, result, False, "unknown_landmark")
        if self.get_parameter("require_localized").value and not self.localized():
            return self._finish(handle, result, False, "not_localized")

        if goal.relocalize_first and self.reloc.wait_for_service(timeout_sec=1.0):
            res = self.reloc.call(Trigger.Request())
            self.get_logger().info(f"relocalize_now: {res.message}")
            if res.success:
                time.sleep(float(self.get_parameter("relocalize_settle_sec").value))

        if not self.nav.wait_for_server(timeout_sec=5.0):
            return self._finish(handle, result, False, "nav2_unavailable")

        target = NavigateToPose.Goal()
        target.pose.header.frame_id = self.map_frame
        target.pose.header.stamp = self.get_clock().now().to_msg()
        target.pose.pose.position.x = float(lm["x"])
        target.pose.pose.position.y = float(lm["y"])
        z, w = _quat_z(float(lm.get("yaw", 0.0)))
        target.pose.pose.orientation.z = z
        target.pose.pose.orientation.w = w

        feedback = GoToLandmark.Feedback()

        def on_feedback(fb):
            feedback.distance_remaining_m = float(fb.feedback.distance_remaining)
            eta = fb.feedback.estimated_time_remaining
            feedback.eta_seconds = float(eta.sec + eta.nanosec * 1e-9)
            feedback.nav_state = "recovering" if fb.feedback.number_of_recoveries else "following"
            handle.publish_feedback(feedback)

        feedback.nav_state = "planning"
        handle.publish_feedback(feedback)
        send = self.nav.send_goal_async(target, feedback_callback=on_feedback)
        while not send.done():
            time.sleep(0.05)
        nav_handle = send.result()
        if not nav_handle.accepted:
            return self._finish(handle, result, False, "nav2_rejected")
        self._nav_handle = nav_handle
        done = nav_handle.get_result_async()
        while not done.done():
            if handle.is_cancel_requested:
                cancel = nav_handle.cancel_goal_async()
                while not cancel.done():
                    time.sleep(0.05)
                while not done.done():
                    time.sleep(0.05)
                return self._finish(handle, result, False, "cancelled")
            time.sleep(0.1)
        status = done.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            return self._finish(handle, result, True, "arrived")
        if status == GoalStatus.STATUS_CANCELED:
            return self._finish(handle, result, False, "cancelled")
        code = getattr(done.result().result, "error_code", 0)
        return self._finish(handle, result, False, f"aborted (nav2 error code {code})")


def main():
    rclpy.init()
    node = GoToLandmarkServer()
    ex = MultiThreadedExecutor(num_threads=4)
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
