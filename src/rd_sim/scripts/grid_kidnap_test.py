#!/usr/bin/env python3
"""Kidnap test for the Isaac grid localizer against Isaac Sim (needs the sim, the stack and
`ros2 launch rd_perception isaac.launch.py grid_localizer:=true use_sim_time:=true`).

Ground truth = the sim's odometry (a displacement from the spawn pose) composed with the spawn pose. Steps: read
truth, push AMCL to a wrong pose, trigger the localizer, compare its result with truth, then see where AMCL settles.
Run 2026-11-08 on the GPU box: pushed 6.8 m and 120 deg off, the localizer answered in 2 sim seconds within 1.7 cm
and 0.5 deg of truth and AMCL followed.

    python3 grid_kidnap_test.py [--wrong X Y YAW] [--spawn X Y YAW]"""
import argparse, math, sys, time
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_srvs.srv import Empty

ap = argparse.ArgumentParser(); ap.add_argument("--wrong", nargs=3, type=float, default=[7.78, 2.52, 1.5])
ap.add_argument("--spawn", nargs=3, type=float, default=[0.6957, 0.9208, 0.0548]); a = ap.parse_args()
def yaw(q): return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
def wrap(x): return (x + math.pi) % (2*math.pi) - math.pi
rclpy.init(); n = Node("kidnap_test"); box = {"init": []}
n.create_subscription(Odometry, "/diff_cont/odom", lambda m: box.__setitem__("odom", (m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation))), 10)
n.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", lambda m: box.__setitem__("amcl", (m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation))), QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
n.create_subscription(PoseWithCovarianceStamped, "/initialpose", lambda m: box["init"].append((time.time(), m)), 10)
pub = n.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
cli = n.create_client(Empty, "/trigger_grid_search_localization")
def spin(sec):
    t0 = time.time()
    while time.time() - t0 < sec: rclpy.spin_once(n, timeout_sec=0.05)
spin(1.5)
sx, sy, sa = a.spawn; ox, oy, oa = box["odom"]
truth = (sx + math.cos(sa)*ox - math.sin(sa)*oy, sy + math.sin(sa)*ox + math.cos(sa)*oy, wrap(sa + oa))
def err(p): return f"{math.hypot(p[0]-truth[0], p[1]-truth[1]):.3f} m, {math.degrees(wrap(p[2]-truth[2])):.1f} deg"
print(f"truth (sim odometry . spawn): x {truth[0]:.3f} y {truth[1]:.3f} yaw {truth[2]:.3f}")
print(f"AMCL before                 : x {box['amcl'][0]:.3f} y {box['amcl'][1]:.3f} yaw {box['amcl'][2]:.3f}  -> error {err(box['amcl'])}")
# kidnap: tell AMCL it is somewhere else
w = PoseWithCovarianceStamped(); w.header.frame_id = "map"; wx, wy, wa = a.wrong
w.pose.pose.position.x, w.pose.pose.position.y = wx, wy; w.pose.pose.orientation.z, w.pose.pose.orientation.w = math.sin(wa/2), math.cos(wa/2)
w.pose.covariance[0] = w.pose.covariance[7] = 0.05; w.pose.covariance[35] = 0.05
box["init"].clear(); pub.publish(w); spin(3.0)
print(f"AMCL after kidnap to ({wx}, {wy}, {wa}): x {box['amcl'][0]:.3f} y {box['amcl'][1]:.3f} yaw {box['amcl'][2]:.3f}  -> error {err(box['amcl'])}")
# the localizer's turn
box["init"].clear(); mine = [m for _, m in box["init"]]
if not cli.wait_for_service(timeout_sec=5.0): print("localizer service missing"); sys.exit(2)
t0 = time.time(); cli.call_async(Empty.Request()); got = None
while time.time() - t0 < 25 and got is None:
    rclpy.spin_once(n, timeout_sec=0.05)
    for t, m in box["init"]:
        if abs(m.pose.pose.position.x - wx) > 1e-6 or abs(m.pose.pose.position.y - wy) > 1e-6: got = (t, m)
if got is None: print("NO LOCALIZER RESULT in 25 s"); sys.exit(1)
m = got[1]; p = (m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation))
print(f"localizer result after {got[0]-t0:.2f} s: x {p[0]:.3f} y {p[1]:.3f} yaw {p[2]:.3f}  -> error vs truth {err(p)}")
spin(4.0)
print(f"AMCL 4 s later              : x {box['amcl'][0]:.3f} y {box['amcl'][1]:.3f} yaw {box['amcl'][2]:.3f}  -> error {err(box['amcl'])}")
rclpy.shutdown()
