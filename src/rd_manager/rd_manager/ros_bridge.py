"""The manager's ROS side: one rclpy node (own executor thread) + a TF listener, keeping the latest of
everything the pages draw and doing what the pages ask (v1 route_editor.py's ros thread, now the only one).

State for the pages: pose (map->base_footprint, fresh only), scans / clouds in the map frame, costmaps and
the nav map (latched grids), TF frames + joints for the 3D view, nav paths / goal / polygons / particles,
camera JPEGs (from rgb8 Image topics, encoded here), the face image, the Folks data (/face/people), the
hand state, the arm stream state, speaking.
Doing: drive pad (TwistStamped on /cmd_vel_web at 10 Hz, dropped when stale), HALT (cancel every nav goal,
zero velocity, streaming off, cancel the running RunPolicy / Wave / Say), nav goal (NavigateToPose) and
initial pose, GoToLandmark / Wave / RunPolicy / Say / ArmPower / go_home / stream, voice text, the Say volume
parameter, slam_toolbox save, the face topics, hand curls.
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Optional

import numpy as np

SCAN_TOPICS = {"s1": "/rp_s1_lidar/scan_filtered", "s2e": "/rp_s2e_lidar/scan_filtered", "velodyne_scan": "/velodyne/scan"}
CLOUD_TOPICS = {"velodyne": "/velodyne_points"}
COSTMAP_TOPICS = {"global": "/global_costmap/costmap", "local": "/local_costmap/costmap"}
NAV_TOPICS = {"plan": "/plan", "local_plan": "/local_plan"}
POLY_TOPICS = {"stop": "/polygon_stop", "slowdown": "/polygon_slowdown", "footprint": "/local_costmap/published_footprint"}
CAMERA_TOPICS = {"scene": "/cameras/scene/image", "wrist": "/cameras/wrist/image", "side": "/cameras/side/image",
                 "realsense": "/camera/camera/color/image_raw", "face": "/face/image"}
CAMERA_LABELS = {"scene": "Scene", "wrist": "Wrist", "side": "Side", "realsense": "RealSense", "face": "Face"}
POLY_MAX_POINTS = {"stop": 3, "slowdown": 3}
FRAME_FRESH_S, FRAME_HOLD_S = 2.0, 6.0
SCAN_MAX_POINTS, CLOUD_MAX_POINTS, CLOUD_BIN_MAX = 720, 4000, 60000
CLOUD_Z = (0.05, 2.0)
DRIVE_HZ, DRIVE_STALE_S, DRIVE_MAX_VX, DRIVE_MAX_WZ, DRIVE_TURBO = 10.0, 0.4, 0.25, 0.5, 4.0   # scales + LB turbo of rd_base joystick.yaml (1.0 m/s, 2.0 rad/s)


def _yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def encode_jpeg(rgb: np.ndarray, quality: int = 80) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class RosBridge:
    def __init__(self, log=print):
        self._log = log
        self.up = False
        self.status = "starting"
        self.lock = threading.Lock()
        self.pose = None
        self.pose_t = 0.0
        self.scans: dict = {}
        self.latest: dict = {}
        self.costmaps: dict = {}
        self.live_map: dict = {}
        self.frames: dict = {}
        self.frame_tf: dict = {}
        self.joints: dict = {}
        self.nav: dict = {}
        self.particles: list = []
        self.jpeg: dict = {}
        self.urdf = ""
        self.people: dict = {}
        self.hand_state: list = []
        self.stream_state = ""
        self.speaking = False
        self.arm_status: dict = {}
        self.voice: dict = {}
        self.want: dict = {"names": [], "t": 0.0}
        self.want3d: dict = {}
        self.drive = {"vx": 0.0, "wz": 0.0, "t": 0.0, "n": 0}
        self.published: dict = {}
        self.tf_debug: dict = {}
        self.nav_goal: dict = {"active": False, "status": "", "feedback": {}}
        self._nav_handle = None
        self._task_handle = None
        self._wave_handle = None
        self._say_handle = None
        self.policy_feedback: dict = {}
        self.nav2_active = False

    # ---- lifecycle
    def start(self) -> None:
        threading.Thread(target=self._main, daemon=True, name="ros-bridge").start()

    def _main(self):
        try:
            import rclpy
            from rclpy.node import Node
            from tf2_ros import Buffer, TransformListener
        except Exception as e:  # noqa: BLE001
            self.status = f"no rclpy ({type(e).__name__})"
            return
        try:
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        except (ImportError, TypeError):
            rclpy.init()
        self.rclpy = rclpy
        self.node = Node("rd_manager")
        tf_node = Node("rd_manager_tf")
        self.buf = Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        TransformListener(self.buf, tf_node, spin_thread=True)
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        self.cb = ReentrantCallbackGroup()
        self._subscribe()
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(self.node)
        threading.Thread(target=ex.spin, daemon=True, name="rd_manager-spin").start()
        threading.Thread(target=self._drive_loop, daemon=True, name="drive-pad").start()
        self.up = True
        self.status = "waiting for TF map->base_footprint"
        self._loop()

    def _subscribe(self):
        from action_msgs.srv import CancelGoal
        from geometry_msgs.msg import PolygonStamped, PoseStamped, PoseWithCovarianceStamped, TwistStamped
        from nav2_msgs.action import NavigateToPose, FollowWaypoints
        from nav_msgs.msg import OccupancyGrid, Path
        from rclpy.action import ActionClient
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState, LaserScan, PointCloud2
        from std_msgs.msg import Bool, Float32MultiArray, String
        from std_srvs.srv import SetBool, Trigger
        from rd_interfaces.action import GoToLandmark, RunPolicy, Say, Wave
        from rd_interfaces.srv import ArmPower
        n = self.node
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        best = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        keep = lambda name: (lambda m: self.latest.__setitem__(name, m))  # noqa: E731
        for name, topic in SCAN_TOPICS.items():
            n.create_subscription(LaserScan, topic, keep(name), qos_profile_sensor_data, callback_group=self.cb)
        for name, topic in CLOUD_TOPICS.items():
            n.create_subscription(PointCloud2, topic, keep(name), qos_profile_sensor_data, callback_group=self.cb)
        for name, topic in COSTMAP_TOPICS.items():
            n.create_subscription(OccupancyGrid, topic, (lambda nm: (lambda m: self.costmaps.__setitem__(nm, {"msg": m, "t": time.time()})))(name), latched, callback_group=self.cb)
        n.create_subscription(OccupancyGrid, "/map", lambda m: self.live_map.update(msg=m, t=time.time()), latched, callback_group=self.cb)
        n.create_subscription(String, "/robot_description", lambda m: setattr(self, "urdf", m.data), latched, callback_group=self.cb)
        n.create_subscription(JointState, "/joint_states", lambda m: self.joints.update({k: round(float(v), 5) for k, v in zip(m.name, m.position)}), 10, callback_group=self.cb)
        for name, topic in NAV_TOPICS.items():
            n.create_subscription(Path, topic, (lambda nm: (lambda m: self.nav.__setitem__(nm, {"frame": m.header.frame_id, "t": time.time(),
                                  "pts": [[round(p.pose.position.x, 3), round(p.pose.position.y, 3), round(p.pose.position.z, 3)] for p in m.poses[::max(1, len(m.poses) // 400)]]})))(name), 10, callback_group=self.cb)
        n.create_subscription(PoseStamped, "/goal_pose", lambda m: self.nav.__setitem__("goal", {"frame": m.header.frame_id, "x": round(m.pose.position.x, 3), "y": round(m.pose.position.y, 3), "yaw": round(_yaw(m.pose.orientation), 4), "t": time.time()}), 10, callback_group=self.cb)
        self.polys: dict = {}
        for name, topic in POLY_TOPICS.items():
            n.create_subscription(PolygonStamped, topic, (lambda nm: (lambda m: self.polys.__setitem__(nm, m)))(name), 10, callback_group=self.cb)
        try:
            from nav2_msgs.msg import ParticleCloud
            n.create_subscription(ParticleCloud, "/particle_cloud", keep("particles"), qos_profile_sensor_data, callback_group=self.cb)
        except Exception:  # noqa: BLE001
            pass
        self.cam_want: dict = {}
        for name, topic in CAMERA_TOPICS.items():
            n.create_subscription(Image, topic, (lambda nm: (lambda m: self._on_image(nm, m)))(name), best, callback_group=self.cb)
        n.create_subscription(String, "/face/people", lambda m: self._set_json("people", m.data), 10, callback_group=self.cb)
        n.create_subscription(Float32MultiArray, "/hand/state", lambda m: setattr(self, "hand_state", list(m.data)), 10, callback_group=self.cb)
        n.create_subscription(String, "/arm/stream_state", lambda m: setattr(self, "stream_state", m.data), 10, callback_group=self.cb)
        n.create_subscription(Bool, "/speaking", lambda m: setattr(self, "speaking", bool(m.data)), 10, callback_group=self.cb)
        for key in ("state", "heard", "command", "reply"):   # what it heard (speech or typed), took as the command, answered
            n.create_subscription(String, f"/voice/{key}", (lambda k: lambda m: self._voice_set(k, m.data))(key), 10, callback_group=self.cb)
        # publishers
        self.goal_pub = n.create_publisher(PoseStamped, "/goal_pose", latched)
        self.init_pub = n.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.drive_pub = n.create_publisher(TwistStamped, "/cmd_vel_web", 10)
        self.halt_pubs = [n.create_publisher(TwistStamped, "/cmd_vel_joy", 10), n.create_publisher(TwistStamped, "/diff_cont/cmd_vel", 10)]
        self.say_text_pub = n.create_publisher(String, "/say_text", 10)
        self.voice_text_pub = n.create_publisher(String, "/voice/text_in", 10)
        self.voice_task_pub = n.create_publisher(String, "/voice/run_task", 10)
        self.hand_curls_pub = n.create_publisher(Float32MultiArray, "/hand/curls", 10)
        self.hand_offsets_pub = n.create_publisher(Float32MultiArray, "/hand/offsets", 10)
        self.face_pubs = {k: n.create_publisher(String, f"/face/{k}", 10) for k in ("set_person_name", "forget_person", "reset_greeting")}
        # clients
        self.cancel_clients = {a: n.create_client(CancelGoal, f"/{a}/_action/cancel_goal", callback_group=self.cb)
                               for a in ("follow_waypoints", "navigate_to_pose", "navigate_through_poses", "go_to_landmark")}
        self.nav_client = ActionClient(n, NavigateToPose, "navigate_to_pose", callback_group=self.cb)
        self.waypoints_client = ActionClient(n, FollowWaypoints, "follow_waypoints", callback_group=self.cb)
        self.landmark_client = ActionClient(n, GoToLandmark, "go_to_landmark", callback_group=self.cb)
        self.wave_client = ActionClient(n, Wave, "wave", callback_group=self.cb)
        self.policy_client = ActionClient(n, RunPolicy, "run_policy", callback_group=self.cb)
        self.say_client = ActionClient(n, Say, "say", callback_group=self.cb)
        self.power_client = n.create_client(ArmPower, "/arm/power", callback_group=self.cb)
        self.stream_client = n.create_client(SetBool, "/arm/stream", callback_group=self.cb)
        self.home_client = n.create_client(Trigger, "/arm/go_home", callback_group=self.cb)
        self.hand_open = n.create_client(Trigger, "/hand/open", callback_group=self.cb)
        self.hand_close = n.create_client(Trigger, "/hand/close", callback_group=self.cb)
        self.wink_client = n.create_client(Trigger, "/face/wink", callback_group=self.cb)
        self.slam_clients: dict = {}
        self.param_client = None
        self.nav2_active_client = n.create_client(Trigger, "/lifecycle_manager_navigation/is_active", callback_group=self.cb)
        self.nav2_active = False
        self._nav2_fut = None

    def _set_json(self, key, text):
        try:
            setattr(self, key, json.loads(text))
        except ValueError:
            pass

    def _on_image(self, name, m):
        # encode only what is being watched (the pages ask for a JPEG / stream first) and at most 10 Hz
        now = time.time()
        if now - self.cam_want.get(name, 0.0) > 5.0 and name != "face":
            self.jpeg.setdefault(name, {"data": b"", "t": 0.0, "seen": 0.0})["seen"] = now
            return
        e = self.jpeg.get(name)
        if e and now - e["t"] < 0.1:
            return
        try:
            data = np.frombuffer(bytes(m.data), dtype=np.uint8)
            if m.encoding in ("rgb8", "bgr8"):
                a = data.reshape(m.height, m.step // 3, 3)[:, :m.width, :]
                rgb = a if m.encoding == "rgb8" else a[..., ::-1]
            else:
                return
            self.jpeg[name] = {"data": encode_jpeg(rgb), "t": now, "seen": now, "w": m.width, "h": m.height}
        except Exception as ex:  # noqa: BLE001
            self.status = f"{name}: {ex}"

    # ---- helpers
    def tf2d(self, frame: str):
        tf = self.buf.lookup_transform("map", frame, self.rclpy.time.Time())
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, t.z, _yaw(q)

    def _scan_to_map(self, m, full=False):
        try:
            tx, ty, _, yaw = self.tf2d(m.header.frame_id)
        except Exception:  # noqa: BLE001
            return None
        r = np.asarray(m.ranges, dtype=float)
        ang = m.angle_min + np.arange(len(r)) * m.angle_increment
        ok = np.isfinite(r) & (r > max(0.05, m.range_min)) & (r < min(30.0, m.range_max or 30.0))
        r, ang = r[ok], ang[ok]
        if not full and len(r) > SCAN_MAX_POINTS:
            step = int(math.ceil(len(r) / SCAN_MAX_POINTS))
            r, ang = r[::step], ang[::step]
        px, py = r * np.cos(ang), r * np.sin(ang)
        pts = np.stack([tx + px * math.cos(yaw) - py * math.sin(yaw), ty + px * math.sin(yaw) + py * math.cos(yaw)], 1)
        return pts if full else np.round(pts, 3).tolist()

    def _cloud_to_map(self, m):
        from sensor_msgs_py import point_cloud2 as pc2
        try:
            tx, ty, tz, yaw = self.tf2d(m.header.frame_id)
        except Exception:  # noqa: BLE001
            return None
        arr = pc2.read_points(m, field_names=("x", "y", "z"), skip_nans=True)
        if len(arr) == 0:
            return []
        pts = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(float)
        z = pts[:, 2] + tz
        pts = pts[(z > CLOUD_Z[0]) & (z < CLOUD_Z[1])]
        if len(pts) > CLOUD_MAX_POINTS:
            pts = pts[:: int(math.ceil(len(pts) / CLOUD_MAX_POINTS))]
        c, s = math.cos(yaw), math.sin(yaw)
        return np.round(np.stack([tx + pts[:, 0] * c - pts[:, 1] * s, ty + pts[:, 0] * s + pts[:, 1] * c], 1), 3).tolist()

    @staticmethod
    def inside_count(pts, poly) -> int:
        if pts is None or len(pts) == 0 or len(poly) < 3:
            return 0
        x, y = pts[:, 0], pts[:, 1]
        inside = np.zeros(len(pts), dtype=bool)
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            cond = (y1 > y) != (y2 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            inside ^= cond & (x < xi)
        return int(inside.sum())

    # ---- the loop (5 Hz scans / frames, 10 Hz pose)
    def _loop(self):
        rclpy = self.rclpy
        last_scan = 0.0
        frames_listed_t, frame_names = 0.0, []
        frame_fresh: dict = {}
        scan_full: dict = {}
        nav2_t = 0.0
        while True:
            time.sleep(0.05)
            now = time.time()
            if now - nav2_t > 2.0:                       # is Nav2 up? (the lifecycle manager's is_active)
                nav2_t = now
                if self._nav2_fut is not None and self._nav2_fut.done():
                    r = self._nav2_fut.result()
                    self.nav2_active = bool(r and r.success)
                    self._nav2_fut = None
                if self._nav2_fut is None:
                    from std_srvs.srv import Trigger
                    if self.nav2_active_client.service_is_ready():
                        self._nav2_fut = self.nav2_active_client.call_async(Trigger.Request())
                    else:
                        self.nav2_active = False
            if now - last_scan > 0.2:
                last_scan = now
                wanted = set(self.want["names"]) if now - self.want["t"] < 3.0 else set()
                scans = dict(self.scans)
                for name, m in list(self.latest.items()):
                    if name not in SCAN_TOPICS and name not in CLOUD_TOPICS:
                        continue
                    try:
                        if name in CLOUD_TOPICS:
                            if name not in wanted:
                                continue
                            pts, topic = self._cloud_to_map(m), CLOUD_TOPICS[name]
                        else:
                            pts, topic = self._scan_to_map(m), SCAN_TOPICS[name]
                            full = self._scan_to_map(m, full=True)
                            if full is not None:
                                scan_full[name] = full
                    except Exception as e:  # noqa: BLE001
                        self.status = f"{name}: {type(e).__name__}: {str(e)[:80]}"
                        continue
                    if pts is not None:
                        scans[name] = {"points": pts, "frame": m.header.frame_id, "topic": topic, "t": now}
                self.scans = scans
                for entry in list(self.costmaps.values()):
                    fr = entry["msg"].header.frame_id
                    if fr and fr != "map":
                        try:
                            tx, ty, _, yaw = self.tf2d(fr)
                            self.frame_tf[fr] = (tx, ty, yaw)
                        except Exception:  # noqa: BLE001
                            pass
                if now - frames_listed_t > 2.0:
                    frames_listed_t = now
                    try:
                        import yaml
                        frame_names = sorted((yaml.safe_load(self.buf.all_frames_as_yaml()) or {}).keys())
                    except Exception:  # noqa: BLE001
                        frame_names = []
                fr = {"map": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
                prev = self.frames
                ros_now = self.node.get_clock().now().nanoseconds * 1e-9   # ROS time (= sim time under use_sim_time): TF stamps are ROS time, never wall time
                dbg = {"frames_in_buffer": len(frame_names), "t": round(now, 1)}
                for a, b, key in (("odom", "base_footprint", "odom_base_age_s"), ("map", "odom", "map_odom_age_s")):
                    try:
                        tf = self.buf.lookup_transform(a, b, rclpy.time.Time())
                        dbg[key] = round(ros_now - (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9), 2)
                    except Exception as e:  # noqa: BLE001
                        dbg[key] = str(e)[:80]
                self.tf_debug = dbg
                for f in frame_names:
                    try:
                        tf = self.buf.lookup_transform("map", f, rclpy.time.Time())
                        if ros_now - (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9) > FRAME_FRESH_S:
                            raise RuntimeError("stale")
                        t, q = tf.transform.translation, tf.transform.rotation
                        fr[f] = [round(t.x, 4), round(t.y, 4), round(t.z, 4), round(q.x, 5), round(q.y, 5), round(q.z, 5), round(q.w, 5)]
                        frame_fresh[f] = now
                    except Exception:  # noqa: BLE001
                        if f in prev and now - frame_fresh.get(f, 0.0) < FRAME_HOLD_S:
                            fr[f] = prev[f]
                self.frames = fr
                polys_out = {}
                for pname, pm in list(self.polys.items()):
                    f2 = (0.0, 0.0, 0.0) if pm.header.frame_id in ("", "map") else self.frames.get(pm.header.frame_id)
                    if f2 is None:
                        continue
                    if len(f2) == 7:
                        f2 = (f2[0], f2[1], math.atan2(2 * (f2[6] * f2[5] + f2[3] * f2[4]), 1 - 2 * (f2[4] ** 2 + f2[5] ** 2)))
                    tx, ty, yaw = f2
                    c, s = math.cos(yaw), math.sin(yaw)
                    polys_out[pname] = {"frame": pm.header.frame_id,
                                        "pts": [[round(tx + p.x * c - p.y * s, 3), round(ty + p.x * s + p.y * c, 3)] for p in pm.polygon.points]}
                for pname, pg in polys_out.items():
                    if pname in POLY_MAX_POINTS:
                        cnt = sum(self.inside_count(pts, pg["pts"]) for pts in scan_full.values())
                        pg.update(inside=cnt, max_points=POLY_MAX_POINTS[pname], triggered=cnt > POLY_MAX_POINTS[pname])
                self.nav["polygons"] = polys_out
                pc = self.latest.get("particles")
                if pc is not None and now - self.want3d.get("particles", 0) < 3.0:
                    ps = pc.particles
                    self.particles = [[round(p.pose.position.x, 3), round(p.pose.position.y, 3), round(_yaw(p.pose.orientation), 3)]
                                      for p in ps[::max(1, len(ps) // 1500)]]
            try:
                tf = self.buf.lookup_transform("map", "base_footprint", rclpy.time.Time())
                ros_now = self.node.get_clock().now().nanoseconds * 1e-9       # ROS time: = sim time under use_sim_time (Isaac), wall time otherwise
                age = ros_now - (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9)
                if age > FRAME_FRESH_S:
                    raise RuntimeError(f"stale TF ({age:.0f} s old)")
                t, q = tf.transform.translation, tf.transform.rotation
                self.pose = {"x": round(t.x, 4), "y": round(t.y, 4), "yaw": round(_yaw(q), 4)}
                self.pose_t = now
                self.status = "ok"
            except Exception as e:  # noqa: BLE001
                if now - self.pose_t > FRAME_HOLD_S:
                    self.status = f"no live map->base_footprint: {e}" if "stale" in str(e) else "no map->base_footprint (localization not running or not initialised)"

    def pose_fresh(self):
        return self.pose if time.time() - self.pose_t < FRAME_HOLD_S else None

    # ---- the voice assistant
    def _voice_set(self, key: str, value: str):
        self.voice[key] = value
        self.voice[key + "_t"] = time.time()          # so a page can tell a new reply from the previous one

    def voice_text(self, text: str) -> dict:
        """The Live tab's text box: the text reaches rd_voice on /voice/text_in exactly as a Whisper transcript
        would (the wake phrase applies); listeners = 0 means the voice assistant is not running."""
        from std_msgs.msg import String
        self.voice_text_pub.publish(String(data=text))
        return {"ok": True, "text": text, "listeners": self.voice_text_pub.get_subscription_count()}

    # ---- the drive pad
    def _drive_loop(self):
        from geometry_msgs.msg import TwistStamped
        period, zeros_left = 1.0 / DRIVE_HZ, 0
        while True:
            d = self.drive
            fresh = d["t"] and (time.time() - d["t"]) <= DRIVE_STALE_S
            msg = TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            if fresh and (d["vx"] or d["wz"]):
                msg.twist.linear.x, msg.twist.angular.z = float(d["vx"]), float(d["wz"])
                self.drive_pub.publish(msg)
                zeros_left = int(DRIVE_HZ * 0.3)
            elif zeros_left > 0:
                self.drive_pub.publish(msg)
                zeros_left -= 1
            time.sleep(period)

    def set_drive(self, vx: float, wz: float, turbo: bool = False, stop: bool = False) -> dict:
        if vx != vx or wz != wz:
            raise ValueError("vx and wz must be finite")
        mx, mw = DRIVE_MAX_VX * (DRIVE_TURBO if turbo else 1.0), DRIVE_MAX_WZ * (DRIVE_TURBO if turbo else 1.0)
        vx, wz = max(-mx, min(mx, vx)), max(-mw, min(mw, wz))
        d = self.drive
        d["vx"], d["wz"] = (0.0, 0.0) if stop else (vx, wz)
        d["t"] = 0.0 if stop else time.time()
        d["n"] += 1
        return {"ok": True, "vx": d["vx"], "wz": d["wz"], "max": [mx, mw], "stop": stop}

    # ---- doing things
    def halt(self) -> list:
        from action_msgs.srv import CancelGoal
        from geometry_msgs.msg import TwistStamped
        from std_srvs.srv import SetBool
        did = []
        cancelled = [a for a, cl in self.cancel_clients.items() if cl.service_is_ready() and cl.call_async(CancelGoal.Request())]
        did.append(f"nav goals cancelled ({', '.join(cancelled) or 'none active'})")
        for h, label in ((self._task_handle, "policy run"), (self._wave_handle, "wave"), (self._say_handle, "speech")):
            if h is not None:
                try:
                    h.cancel_goal_async()
                    did.append(f"{label} cancelled")
                except Exception:  # noqa: BLE001
                    pass
        if self.stream_client.service_is_ready():
            self.stream_client.call_async(SetBool.Request(data=False))
            did.append("arm streaming off")
        zero = TwistStamped()
        zero.header.frame_id = "base_link"
        for _ in range(6):
            zero.header.stamp = self.node.get_clock().now().to_msg()
            for pub in self.halt_pubs:
                pub.publish(zero)
            time.sleep(0.05)
        did.append("zero velocity sent")
        self.drive.update(vx=0.0, wz=0.0, t=0.0)
        self.published = {"type": "halt", "did": did, "t": time.time()}
        return did

    def publish_goal(self, x: float, y: float, yaw: float) -> dict:
        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import NavigateToPose
        g = PoseStamped()
        g.header.frame_id = "map"
        g.header.stamp = self.node.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = float(x), float(y)
        g.pose.orientation.z, g.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_pub.publish(g)
        if self.nav_client.server_is_ready():
            fut = self.nav_client.send_goal_async(NavigateToPose.Goal(pose=g), feedback_callback=lambda fb: self.nav_goal.__setitem__(
                "feedback", {"distance_remaining": round(fb.feedback.distance_remaining, 2), "recoveries": fb.feedback.number_of_recoveries}))
            fut.add_done_callback(self._nav_sent)
        self.published = {"type": "goal", "x": x, "y": y, "yaw": yaw, "t": time.time()}
        return self.published

    def _nav_sent(self, fut):
        h = fut.result()
        self._nav_handle = h if h.accepted else None
        self.nav_goal.update(active=bool(h.accepted), status="accepted" if h.accepted else "rejected")
        if h.accepted:
            h.get_result_async().add_done_callback(lambda r: self.nav_goal.update(active=False, status=f"ended {r.result().status}"))

    def publish_initialpose(self, x: float, y: float, yaw: float) -> dict:
        from geometry_msgs.msg import PoseWithCovarianceStamped
        ip = PoseWithCovarianceStamped()
        ip.header.frame_id = "map"
        ip.pose.pose.position.x, ip.pose.pose.position.y = float(x), float(y)
        ip.pose.pose.orientation.z, ip.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.25
        cov[35] = 0.0685
        ip.pose.covariance = cov
        self.init_pub.publish(ip)
        self.published = {"type": "initialpose", "x": x, "y": y, "yaw": yaw, "t": time.time()}
        return self.published

    def _call_action(self, client, goal, timeout: float, keep: str, feedback=None) -> tuple[bool, str, object]:
        if not client.wait_for_server(timeout_sec=3.0):
            return False, "the server is not running", None
        fut = client.send_goal_async(goal, feedback_callback=feedback)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > 10:
                return False, "no answer from the server", None
            time.sleep(0.05)
        h = fut.result()
        if not h.accepted:
            return False, "rejected (busy)", None
        setattr(self, keep, h)
        try:
            res = h.get_result_async()
            while not res.done():
                if time.monotonic() - t0 > timeout:
                    h.cancel_goal_async()
                    return False, "timed out", None
                time.sleep(0.1)
            r = res.result()
            ok = r.status == 4 and bool(getattr(r.result, "success", True))
            return ok, str(getattr(r.result, "message", "")) or f"status {r.status}", r.result
        finally:
            setattr(self, keep, None)

    def go_to_landmark(self, landmark: str, timeout: float = 300.0) -> dict:
        from rd_interfaces.action import GoToLandmark
        ok, msg, res = self._call_action(self.landmark_client, GoToLandmark.Goal(landmark=landmark), timeout, "_nav_handle")
        return {"ok": ok, "message": msg}

    def follow_waypoints(self, poses, timeout: float = 900.0) -> tuple[bool, str]:
        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import FollowWaypoints
        goal = FollowWaypoints.Goal()
        for x, y, yaw in poses:
            p = PoseStamped()
            p.header.frame_id = "map"
            p.header.stamp = self.node.get_clock().now().to_msg()
            p.pose.position.x, p.pose.position.y = float(x), float(y)
            p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            goal.poses.append(p)
        ok, msg, res = self._call_action(self.waypoints_client, goal, timeout, "_nav_handle",
                                         feedback=lambda fb: self.nav_goal.__setitem__("feedback", {"waypoint": fb.feedback.current_waypoint}))
        if res is not None and getattr(res, "missed_waypoints", None):
            return False, f"missed waypoints {list(res.missed_waypoints)}"
        return ok, msg

    def run_policy(self, task: str, checkpoint: str = "", backend: str = "", max_seconds: float = 120.0, record: bool = False,
                   landmark: str = "", done_when: str = "", still_seconds: float = 0.0, home_first: bool = True, max_steps: int = 0) -> dict:
        from rd_interfaces.action import RunPolicy
        goal = RunPolicy.Goal(task=task, checkpoint=checkpoint, backend=backend, max_seconds=float(max_seconds), max_steps=int(max_steps),
                              home_first=home_first, record=record, landmark=landmark, done_when=done_when, still_seconds=float(still_seconds))
        ok, msg, res = self._call_action(self.policy_client, goal, max_seconds + 120.0, "_task_handle",
                                         feedback=lambda fb: setattr(self, "policy_feedback", {"step": fb.feedback.step, "seconds": round(fb.feedback.seconds, 1), "phase": fb.feedback.phase}))
        out = {"ok": ok, "message": msg}
        if res is not None:
            out.update(steps=int(getattr(res, "steps", 0)), seconds=round(float(getattr(res, "seconds", 0.0)), 1), run_dir=str(getattr(res, "run_dir", "")))
        return out

    def wave(self, motion: str = "wave", return_to: str = "home", speed_scale: float = 1.0) -> dict:
        from rd_interfaces.action import Wave
        ok, msg, _ = self._call_action(self.wave_client, Wave.Goal(motion=motion, return_to=return_to, speed_scale=float(speed_scale), wiggle_fingers=True), 120.0, "_wave_handle")
        return {"ok": ok, "message": msg}

    def say(self, text: str, wait: bool = False) -> dict:
        from std_msgs.msg import String
        from rd_interfaces.action import Say
        if not wait:
            self.say_text_pub.publish(String(data=text))
            return {"ok": True, "message": "queued"}
        ok, msg, _ = self._call_action(self.say_client, Say.Goal(text=text), 60.0, "_say_handle")
        return {"ok": ok, "message": msg}

    def arm_power(self, command: int, timeout_s: float = 60.0) -> dict:
        from rd_interfaces.srv import ArmPower
        if not self.power_client.wait_for_service(timeout_sec=2.0):
            return {"success": False, "message": "no /arm/power service"}
        r = self.power_client.call(ArmPower.Request(command=int(command), timeout_s=float(timeout_s)))
        out = {"success": r.success, "message": r.message, "robot_mode": r.robot_mode, "safety_status": r.safety_status,
               "remote_control": r.remote_control, "program_running": r.program_running}
        if command == 0:
            self.arm_status = {**out, "t": time.time()}
        return out

    def call_trigger(self, client, timeout=5.0) -> dict:
        if not client.wait_for_service(timeout_sec=timeout):
            return {"success": False, "message": f"no {client.srv_name} service"}
        r = client.call(type(client.srv_type.Request)())
        return {"success": r.success, "message": r.message}

    def set_stream(self, on: bool) -> dict:
        from std_srvs.srv import SetBool
        if not self.stream_client.wait_for_service(timeout_sec=2.0):
            return {"success": False, "message": "no /arm/stream service"}
        r = self.stream_client.call(SetBool.Request(data=bool(on)))
        return {"success": r.success, "message": r.message}

    def set_volume(self, level: float) -> dict:
        from rcl_interfaces.msg import Parameter, ParameterValue
        from rcl_interfaces.srv import SetParameters
        if self.param_client is None:
            self.param_client = self.node.create_client(SetParameters, "/say_server/set_parameters", callback_group=self.cb)
        if not self.param_client.wait_for_service(timeout_sec=2.0):
            return {"ok": False, "error": "no say_server"}
        pv = ParameterValue(type=3, double_value=float(level))
        r = self.param_client.call(SetParameters.Request(parameters=[Parameter(name="volume", value=pv)]))
        return {"ok": bool(r.results and r.results[0].successful)}

    def slam_save(self, path: str, timeout: float = 60.0) -> dict:
        from slam_toolbox.srv import SaveMap, SerializePoseGraph
        if not self.slam_clients:
            self.slam_clients["save"] = self.node.create_client(SaveMap, "/slam_toolbox/save_map", callback_group=self.cb)
            self.slam_clients["serialize"] = self.node.create_client(SerializePoseGraph, "/slam_toolbox/serialize_map", callback_group=self.cb)
        res: dict = {}
        for key, req in (("save_map", SaveMap.Request()), ("serialize", SerializePoseGraph.Request())):
            cl = self.slam_clients["save" if key == "save_map" else "serialize"]
            if not cl.wait_for_service(timeout_sec=3.0):
                return {"error": f"{key}: no server (is the SLAM unit running?)"}
            if key == "save_map":
                req.name.data = path
            else:
                req.filename = path
            fut = cl.call_async(req)
            t0 = time.time()
            while not fut.done():
                if time.time() - t0 > timeout:
                    return {"error": f"{key}: no answer within {timeout:.0f} s"}
                time.sleep(0.05)
            res[key] = int(getattr(fut.result(), "result", 0))
        if res.get("save_map") not in (0, 255):
            res["error"] = f"save_map returned {res['save_map']}"
        return res

    # ---- blobs
    def cloud_bin(self, name: str, max_points: int) -> Optional[bytes]:
        m = self.latest.get(name)
        if m is None:
            return None
        if name in CLOUD_TOPICS:
            from sensor_msgs_py import point_cloud2 as pc2
            names = [f.name for f in m.fields]
            want = ["x", "y", "z"] + (["intensity"] if "intensity" in names else [])
            arr = pc2.read_points(m, field_names=want, skip_nans=True)
            arr = arr[::max(1, int(math.ceil(len(arr) / max(1, max_points))))]
            out = np.empty((len(arr), 4), np.float32)
            out[:, 0], out[:, 1], out[:, 2] = arr["x"], arr["y"], arr["z"]
            out[:, 3] = arr["intensity"] if "intensity" in want else 0.0
        else:
            r = np.asarray(m.ranges, dtype=np.float32)
            ang = (m.angle_min + np.arange(len(r)) * m.angle_increment).astype(np.float32)
            ok = np.isfinite(r) & (r > max(0.02, m.range_min)) & (r < min(60.0, m.range_max or 60.0))
            inten = np.asarray(m.intensities, dtype=np.float32) if len(m.intensities) == len(r) else np.zeros(len(r), np.float32)
            r, ang, inten = r[ok], ang[ok], inten[ok]
            out = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(r), inten], 1).astype(np.float32)
        hdr = json.dumps({"name": name, "frame": m.header.frame_id, "n": int(len(out)), "fields": ["x", "y", "z", "i"],
                          "stamp": f"{m.header.stamp.sec}.{m.header.stamp.nanosec:09d}", "tf": self.frames.get(m.header.frame_id)}).encode()
        hdr += b" " * ((-(4 + len(hdr))) % 4)
        return len(hdr).to_bytes(4, "little") + hdr + out.tobytes()
