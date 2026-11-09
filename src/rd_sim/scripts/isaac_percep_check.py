#!/usr/bin/env python3
"""Validate nvblox and cuVSLAM against Isaac Sim (rd_perception isaac.launch.py nvblox:=true visual_slam:=true
use_sim_time:=true on top of the sim stack): drive to a goal through the manager, log cuVSLAM's odometry against the
sim's exact odometry (both start at the odom origin, so a pose relative to each one's start is comparable), then
compare nvblox's ESDF slice with the lidar: the slice's zero-crossing cells against the nearest lidar return, and
the distance nvblox reports at the lidar returns (0 when it agrees the wall is there).

    python3 $(ros2 pkg prefix rd_sim)/share/rd_sim/scripts/isaac_percep_check.py [--goal 2.7 1.9] [--seconds 45] [--no-drive]

2026-11-09, GPU box: cuVSLAM 7.7 cm / 0.4 deg off ground truth after 2.1 m; nvblox surface cells a median 3.9 cm from
the lidar, 79% within 15 cm; nvblox's distance at the lidar returns median 0.0 m, 87% within 15 cm.
"""
import argparse, json, math, sys, time, urllib.request
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nvblox_msgs.msg import DistanceMapSlice
from tf2_ros import Buffer, TransformListener

ap = argparse.ArgumentParser(); ap.add_argument("--goal", nargs=2, type=float, default=[2.7, 1.9]); ap.add_argument("--seconds", type=float, default=45.0)
ap.add_argument("--no-drive", action="store_true"); a = ap.parse_args()

def yawq(q): return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
def wrap(x): return (x + math.pi) % (2*math.pi) - math.pi
rclpy.init(); n = rclpy.create_node("isaac_percep_check"); st = {}; buf = Buffer(); TransformListener(buf, n)
n.create_subscription(Odometry, "/diff_cont/odom", lambda m: st.update(truth=(m.pose.pose.position.x, m.pose.pose.position.y, yawq(m.pose.pose.orientation))), 10)
n.create_subscription(Odometry, "/visual_slam/tracking/odometry", lambda m: st.update(vo=(m.pose.pose.position.x, m.pose.pose.position.y, yawq(m.pose.pose.orientation)), vo_frame=(m.header.frame_id, m.child_frame_id)), 10)
n.create_subscription(LaserScan, "/rp_s2e_lidar/scan_filtered", lambda m: st.__setitem__("scan", m), QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
n.create_subscription(DistanceMapSlice, "/nvblox_node/static_map_slice", lambda m: st.__setitem__("slice", m), QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE))
def spin(sec):
    t = time.time()
    while time.time() - t < sec: rclpy.spin_once(n, timeout_sec=0.02)
spin(3.0)
for k in ("truth", "vo", "scan", "slice"):
    print(f"{k:6s} {'ok' if k in st else 'MISSING'}", flush=True)
if "truth" not in st or "vo" not in st: sys.exit("no odometry")
print("vo frames:", st["vo_frame"], flush=True)
# the two odometries start together at the origin only if cuVSLAM started before the robot moved: record the offset now
t0x, t0y, t0a = st["truth"]; v0x, v0y, v0a = st["vo"]
print(f"start: truth ({t0x:.3f}, {t0y:.3f}, {math.degrees(t0a):.1f} deg)  vo ({v0x:.3f}, {v0y:.3f}, {math.degrees(v0a):.1f} deg)", flush=True)
def rel(p, o):   # pose p expressed relative to start pose o (same frame)
    dx, dy = p[0]-o[0], p[1]-o[1]; c, s_ = math.cos(-o[2]), math.sin(-o[2])
    return (c*dx - s_*dy, s_*dx + c*dy, wrap(p[2]-o[2]))
if not a.no_drive:
    tx, ty, _ = st["truth"]; heading = math.atan2(a.goal[1]-ty, a.goal[0]-tx)
    r = urllib.request.Request("http://127.0.0.1:8092/api/publish", data=json.dumps({"type": "goal", "x": a.goal[0], "y": a.goal[1], "yaw": heading}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    print("goal:", json.loads(urllib.request.urlopen(r, timeout=5).read()), flush=True)
rows = []; t_start = time.time(); dist = 0.0; last = st["truth"]
while time.time() - t_start < a.seconds:
    spin(0.5)
    tr = rel(st["truth"], (t0x, t0y, t0a)); vo = rel(st["vo"], (v0x, v0y, v0a))
    dist += math.hypot(st["truth"][0]-last[0], st["truth"][1]-last[1]); last = st["truth"]
    err = math.hypot(tr[0]-vo[0], tr[1]-vo[1]); rows.append((time.time()-t_start, dist, err, math.degrees(wrap(tr[2]-vo[2]))))
    if len(rows) % 20 == 0: print(f"  t={rows[-1][0]:5.1f}s travelled {dist:5.2f} m  truth ({tr[0]:+.2f},{tr[1]:+.2f},{math.degrees(tr[2]):+.0f}deg)  vo ({vo[0]:+.2f},{vo[1]:+.2f},{math.degrees(vo[2]):+.0f}deg)  err {err:.3f} m / {rows[-1][3]:+.1f} deg", flush=True)
errs = [r[2] for r in rows]
print(f"cuVSLAM vs sim odometry over {dist:.2f} m: position error mean {np.mean(errs):.3f} max {max(errs):.3f} final {errs[-1]:.3f} m ({100*errs[-1]/max(dist,1e-6):.1f}% of distance); yaw error final {rows[-1][3]:+.1f} deg, mean |yaw| {np.mean([abs(r[3]) for r in rows]):.1f} deg", flush=True)
# nvblox: the ESDF slice against the lidar, both in the slice's frame, with the robot at rest (the drive is over).
# Inside an obstacle the ESDF goes negative, so "d <= 0" would pick the interior of thick walls: the surface is |d| ~ 0.
spin(2.0)
if "slice" in st and "scan" in st:
    sl = st["slice"]; d = np.array(sl.data, dtype=np.float32).reshape(sl.height, sl.width); known = d != sl.unknown_value
    occ = np.argwhere(known & (np.abs(d) <= 1.5 * sl.resolution))
    pts = np.c_[sl.origin.x + (occ[:, 1] + 0.5) * sl.resolution, sl.origin.y + (occ[:, 0] + 0.5) * sl.resolution]
    sc = st["scan"]; ang = sc.angle_min + np.arange(len(sc.ranges)) * sc.angle_increment; rg = np.array(sc.ranges, dtype=np.float32)
    ok = np.isfinite(rg) & (rg > sc.range_min) & (rg < sc.range_max)
    try:
        tf = buf.lookup_transform(sl.header.frame_id, sc.header.frame_id, rclpy.time.Time()); tt, q = tf.transform.translation, tf.transform.rotation; yaw = yawq(q)
        lx, ly = rg[ok]*np.cos(ang[ok]), rg[ok]*np.sin(ang[ok]); lp = np.c_[tt.x + math.cos(yaw)*lx - math.sin(yaw)*ly, tt.y + math.sin(yaw)*lx + math.cos(yaw)*ly]
        if len(pts) and len(lp):
            dmin = np.sqrt(((pts[:, None, :] - lp[None, :, :])**2).sum(-1)).min(1)
            print(f"nvblox slice ({sl.width}x{sl.height} @ {sl.resolution:.2f} m, frame {sl.header.frame_id}, z {sl.origin.z:.2f}, {known.sum()} known cells): {len(pts)} surface cells; "
                  f"to the nearest lidar return median {np.median(dmin):.3f} m, {100*np.mean(dmin < 0.15):.0f}% within 15 cm, {100*np.mean(dmin < 0.30):.0f}% within 30 cm", flush=True)
            ix = ((lp[:, 0] - sl.origin.x) / sl.resolution).astype(int); iy = ((lp[:, 1] - sl.origin.y) / sl.resolution).astype(int)
            inside = (ix >= 0) & (ix < sl.width) & (iy >= 0) & (iy < sl.height); vals = d[iy[inside], ix[inside]]; kn = vals != sl.unknown_value
            if kn.any():
                print(f"nvblox's distance at the {kn.sum()} lidar returns on known cells: median {np.median(vals[kn]):.3f} m, {100*np.mean(np.abs(vals[kn]) < 0.15):.0f}% within 15 cm of its surface", flush=True)
        else:
            print(f"nvblox slice: {len(pts)} surface cells, lidar points {len(lp)}", flush=True)
    except Exception as e:
        print("nvblox compare failed:", e)
else:
    print("nvblox slice / scan missing:", "slice" in st, "scan" in st)
rclpy.shutdown()
