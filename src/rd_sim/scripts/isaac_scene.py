"""Build the Robot Disco scene in Isaac Sim 6.2 and bridge it to ROS 2 (Jazzy). Run with the Isaac Sim pip venv's python
(provision/gpu/provision_gpu.sh isaacsim), with the system Jazzy sourced first, RMW_IMPLEMENTATION=rmw_cyclonedds_cpp and
OMNI_KIT_ACCEPT_EULA=YES in the environment:

    ~/isaacsim-venv/bin/python ~/RobotDisco/src/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf [--headless] [--world zone_one]

What it does (Isaac Sim 6.2 / isaacsim.* API, every call checked against the 6.2 docs on 2026-11-07, NOT yet run - phase 7
validates it; the `# UNVERIFIED (6.2):` comments mark what the docs did not settle):
  1. imports the flat URDF (rd_sim export_urdf) with URDFImporter as a floating-base articulation, the drive wheels
     velocity-driven and the UR joints position-driven, and references the result into the stage; the hand is the baked visual
  2. builds the world: a ground plane and the active map's occupied cells, one 2.5 m tall box per cell (so the
     simulated lidars see exactly the walls the map has) from rd_nav's maps folder
  3. attaches sensors: two 2D RTX lidars (frames rp_s1_lidar / rp_s2e_lidar, 360 deg, 10 Hz, 0.1..20 m), a Velodyne-like
     3D RTX lidar (velodyne_frame), the RealSense on camera_link_optical (1280x720 colour + camera_info, depth + camera_info, the 848x480 stereo IR pair) and three RGB webcams (scene / wrist /
     side) - the PhysX lidar and isaacsim.sensors.camera are deprecated since 6.0
  4. wires the ROS 2 bridge (isaacsim.ros2.bridge): /clock, odom->base_link TF from the chassis pose, /joint_states
     (wheels + UR), /diff_cont/odom, the RAW scans on /rp_s1_lidar/scan and /rp_s2e_lidar/scan (rd_perception's body
     filter chains make the *_filtered topics, sim.launch.py runs them: the lidars see the robot's own body here too),
     /velodyne_points, the camera topics the rest of 5.0 subscribes to, and subscribers: /diff_cont/cmd_vel
     (TwistStamped, as rd_nav / rd_manager send it to the real base; relayed to the bridge's Twist subscriber and zeroed
     0.5 s after the last command) -> wheel velocities, and /ur5e/sim_joint_command (JointState from arm_mirror) -> UR drives.
     The robot's own frames come from robot_state_publisher (sim.launch.py), so the sim publishes no link TF itself.
  5. runs the simulation loop until closed

With this running, the 5.0 stack runs unchanged on the GPU box: rd_nav + rd_arm (sim_mode for the mock UR, mirrored into
the simulated arm by arm_mirror) + rd_vla + rd_manager. The base's own controller manager is NOT run (the bridge
integrates cmd_vel itself); the arm's controller manager runs on mock hardware and the bridge mirrors its commanded joints.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

WHEELS = ["left_wheel_joint", "right_wheel_joint"]
UR = ["ur_shoulder_pan_joint", "ur_shoulder_lift_joint", "ur_elbow_joint", "ur_wrist_1_joint", "ur_wrist_2_joint", "ur_wrist_3_joint"]
CMD_VEL_STAMPED = "/diff_cont/cmd_vel"               # what the stack publishes (geometry_msgs/TwistStamped)
CMD_VEL_TWIST = "diff_cont/cmd_vel_unstamped"        # what the bridge's ROS2SubscribeTwist (geometry_msgs/Twist) reads
ROBOT = "/World/RobotDisco"


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default="/tmp/robot_disco.urdf")
    ap.add_argument("--world", default="zone_one", help="a map from rd_nav/maps to build walls from ('' = empty)")
    ap.add_argument("--maps-dir", default="")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--x", type=float, default=0.6957)      # the Charger landmark (rd_nav landmarks.yaml), where the robot docks;
    ap.add_argument("--y", type=float, default=0.9208)      # sim.launch.py hands AMCL the same pose
    ap.add_argument("--yaw", type=float, default=0.0548)
    return ap.parse_args()


def map_boxes(maps_dir: Path, name: str, height: float = 2.5, merge: int = 1):
    """Occupied cells of a nav map as (x, y, size) blocks, merged merge x merge cells at a time. merge=1 is the map
    itself (zone_one: ~3500 cells of 0.05 m); 4 x 4 blocks fattened thin walls to 0.2 m and closed a doorway on the way
    to the Dining Room (Nav2 planned no path in the sim where the map has one)."""
    import numpy as np
    import yaml
    meta = yaml.safe_load((maps_dir / f"{name}.yaml").read_text())
    img = maps_dir / meta["image"]
    with open(img, "rb") as f:
        assert f.readline().strip() == b"P5"
        tok = []
        while len(tok) < 3:
            line = f.readline()
            if not line.startswith(b"#"):
                tok += line.split()
        w, h = int(tok[0]), int(tok[1])
        data = np.frombuffer(f.read(), dtype=np.uint8, count=w * h).reshape(h, w)
    occ = data < 50 if not meta.get("negate", 0) else data > 205
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    boxes = []
    for r in range(0, h, merge):
        for c in range(0, w, merge):
            block = occ[r:r + merge, c:c + merge]
            if block.mean() > 0.3:
                cx = ox + (c + merge / 2.0) * res
                cy = oy + (h - r - merge / 2.0) * res
                boxes.append((cx, cy, merge * res))
    return boxes, height


def arm_home():
    """rd_arm's vla_home (poses.yaml, the recorded home the policies start from); zeros if rd_arm is not installed."""
    try:
        import yaml
        from ament_index_python.packages import get_package_share_directory
        poses = yaml.safe_load(open(Path(get_package_share_directory("rd_arm")) / "config" / "poses.yaml"))["poses"]
        return [float(v) for v in poses["vla_home"]]
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: no rd_arm vla_home ({e}): the simulated UR starts at zero joints")
        return [0.0] * 6


def start_cmd_vel_relay(timeout: float = 0.5):
    """TwistStamped on /diff_cont/cmd_vel -> Twist on diff_cont/cmd_vel_unstamped for the bridge (its ROS2SubscribeTwist
    node only takes geometry_msgs/Twist), zero after `timeout` s without a command, as the base's diff_drive_controller
    does. rclpy next to the bridge is the pattern the 6.2 'ROS 2 Bridge in Standalone Workflow' page shows."""
    try:  # UNVERIFIED (6.2): the system Jazzy rclpy (Python 3.12) imported from the Isaac venv's python
        import rclpy
        from geometry_msgs.msg import Twist, TwistStamped
    except ImportError as e:  # noqa: BLE001
        print(f"cmd_vel relay OFF ({e}): the bridge subscribes Twist on {CMD_VEL_STAMPED} directly, TwistStamped senders will be ignored")
        return None
    rclpy.init()
    node = rclpy.create_node("rd_sim_cmd_vel_relay")
    pub = node.create_publisher(Twist, CMD_VEL_TWIST, 10)
    last = {"twist": Twist(), "t": 0.0}

    def on_cmd(m):
        last["twist"], last["t"] = m.twist, time.monotonic()

    node.create_subscription(TwistStamped, CMD_VEL_STAMPED, on_cmd, 10)
    node.create_timer(0.05, lambda: pub.publish(last["twist"] if time.monotonic() - last["t"] < timeout else Twist()))
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    return node


def main():
    args = parse()
    sys.stdout.reconfigure(line_buffering=True)        # the prints reach a log file while the sim runs
    from isaacsim import SimulationApp  # noqa: PLC0415 - Isaac Sim's python only; Omniverse imports must follow the app
    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720, "renderer": "RayTracedLighting"})
    import numpy as np
    import omni.graph.core as og
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.prim as prim_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from isaacsim.core.experimental.objects import Cube, GroundPlane
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, RtxCamera
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux

    for ext in ("isaacsim.ros2.bridge", "isaacsim.ros2.nodes", "isaacsim.core.nodes", "isaacsim.sensors.physics.nodes",
                "isaacsim.robot.wheeled_robots.nodes"):
        app_utils.enable_extension(ext)
    app.update()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    stage = stage_utils.get_current_stage()

    # 1. the robot: URDF -> USD file (next to the URDF) -> referenced at /World/RobotDisco
    wheels_v = {"left_wheel_joint": "velocity", "right_wheel_joint": "velocity"}
    # UNVERIFIED (6.2): the dict form maps joint-name regexes to values; match mode and precedence are not documented,
    # so the drives are re-asserted on the USD below
    cfg = URDFImporterConfig(urdf_path=str(Path(args.urdf).resolve()), fix_base=False, merge_fixed_joints=False,
                             joint_target_type={**wheels_v, "ur_.*": "position"},
                             override_joint_stiffness={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0, "ur_.*": 1e4},
                             override_joint_damping={"left_wheel_joint": 1e4, "right_wheel_joint": 1e4, "ur_.*": 1e3})
    usd_path = URDFImporter(cfg).import_urdf()
    robot = stage_utils.add_reference_to_stage(usd_path, ROBOT)
    assert robot and robot.GetChildren(), f"{usd_path}: nothing composed at {ROBOT} (no default prim?)"
    # The 6.2 converter (URDF USD Converter 0.3) writes the joints and bodies into a "Physics" variant set
    # (physx | physics | mujoco | none) and leaves it unselected: without this there are no joints at all.
    vs = robot.GetVariantSet("Physics")
    if vs and vs.GetVariantNames():
        choice = "physx" if "physx" in vs.GetVariantNames() else vs.GetVariantNames()[0]
        vs.SetVariantSelection(choice)
        print(f"Physics variant: {choice} (of {vs.GetVariantNames()})")
    xf = UsdGeom.Xformable(robot)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(args.x, args.y, 0.2))
    xf.AddRotateZOp().Set(math.degrees(args.yaw))

    def prim_named(name, predicate=lambda p: True):
        # the converter's layout: links nested under <robot>/Geometry/base_link/... as the kinematic tree, joints
        # flat under <robot>/Physics/<joint name> (a fixed joint between two links gets a sanitized name)
        p = next((p for p in Usd.PrimRange(robot) if p.GetName() == name and predicate(p)), None)
        if not p:
            near = sorted({q.GetName() for q in Usd.PrimRange(robot) if predicate(q)})[:40]
            raise AssertionError(f"{name}: not in the imported USD ({usd_path}); candidates: {near}")
        return p

    roots = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots, f"no articulation root in {usd_path}"
    if len(roots) > 1:   # a massless URDF root link makes every child body its own articulation: export_urdf gives it a mass
        print(f"WARNING: {len(roots)} articulation roots: {[str(p.GetPath()) for p in roots]}; using the first")
    root_path = str(roots[0].GetPath())
    home = arm_home()
    for name, stiffness, damping in [(w, 0.0, 1e4) for w in WHEELS] + [(j, 1e4, 1e3) for j in UR]:
        jp = prim_named(name, lambda p: p.IsA(UsdPhysics.Joint))
        drive = UsdPhysics.DriveAPI.Get(jp, "angular") or UsdPhysics.DriveAPI.Apply(jp, "angular")
        drive.CreateStiffnessAttr().Set(stiffness)          # 0 stiffness + damping = velocity drive (the wheels)
        drive.CreateDampingAttr().Set(damping)
        if name in UR:                                       # start folded at the recorded home, as the robot and the mock do:
            deg = math.degrees(home[UR.index(name)])         # at all-zero joints the UR lies stretched out sideways and tips the base
            drive.CreateTargetPositionAttr().Set(deg)
            for attr in ("state:angular:physics:position",):
                a = jp.GetAttribute(attr) if jp.HasAttribute(attr) else jp.CreateAttribute(attr, Sdf.ValueTypeNames.Float)
                a.Set(deg)

    # friction: the casters are fixed spheres here (on the robot they swivel and roll), so they must slide; the
    # tyres grip. PhysX combines two materials' friction by averaging unless one says "min".
    def physics_material(path, friction):
        mat = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr().Set(friction)
        api.CreateDynamicFrictionAttr().Set(friction)
        api.CreateRestitutionAttr().Set(0.0)
        try:
            from pxr import PhysxSchema
            PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr().Set("min" if friction == 0.0 else "average")
        except Exception as e:  # noqa: BLE001
            print(f"no PhysxMaterialAPI ({e}): caster friction averages with the floor's")
        return mat

    def bind_material(link_name, mat):
        n = 0
        for p in Usd.PrimRange(prim_named(link_name)):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
                n += 1
        assert n, f"{link_name}: no collision prim to bind a material to"
    slick, grip = physics_material("/World/PhysicsMaterials/caster", 0.0), physics_material("/World/PhysicsMaterials/tyre", 1.0)
    for name in ("caster_wheel_fl", "caster_wheel_fr", "caster_wheel_bl", "caster_wheel_br"):
        bind_material(name, slick)
    for name in ("left_wheel", "right_wheel"):
        bind_material(name, grip)

    # 2. the ground and the walls from the map
    GroundPlane("/World/GroundPlane")                        # collision plane + rendered mesh
    # light: a scene without a light renders every RGB camera black (the depth and the lidars do not care); a dome for
    # ambient light and a sun at an angle for shading, so walls, floor and the robot have edges a visual tracker can use
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 0.0))
    if args.world:
        maps_dir = Path(args.maps_dir) if args.maps_dir else Path(__import__("rd_nav.maps_store", fromlist=["default_maps_dir"]).default_maps_dir())
        boxes, height = map_boxes(maps_dir, args.world)
        if boxes:
            stage_utils.define_prim("/World/Walls", "Xform")
            paths = [f"/World/Walls/w{i}" for i in range(len(boxes))]
            Cube(paths, sizes=1.0, colors="gray", positions=np.array([[cx, cy, height / 2] for cx, cy, _ in boxes]),
                 scales=np.array([[size, size, height] for _, _, size in boxes]))
            for p in paths:                                  # static colliders (no rigid body = fixed)
                UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(p))
        print(f"{len(boxes)} wall blocks from {args.world}")

    # 3. sensors: RTX lidars (isaacsim.sensors.experimental.rtx) under the URDF frames, scan rate = tick rate = 10 Hz
    # Profiles shipped with 6.2 (isaacsim.sensors.rtx data/lidar_configs): RPLIDAR_S2E is the robot's own rear lidar and
    # stands in for the S1 too; no VLP-16 profile ships, so the Velodyne is a 32-channel 10 Hz Ouster OS1 stand-in
    # (Nav2's point cloud source only filters by height). The near/far range overrides sit on top of the profile.
    lidars = {}
    for frame, config, variant, near, far in (("rp_s1_lidar", "RPLIDAR_S2E", None, 0.1, 20.0), ("rp_s2e_lidar", "RPLIDAR_S2E", None, 0.1, 20.0),
                                              ("velodyne_frame", "OS1", "OS1_REV6_32ch10hz512res", 0.5, 50.0)):
        kw = {"variant": variant} if variant else {}      # a config is a sensor family (OS1); its variant picks channels / rate / resolution
        # outputFrameOfReference: the point cloud writer publishes the points as the sensor outputs them, and the
        # profiles default to WORLD coordinates, which the costmap then transforms a second time (phantom walls
        # across the corridor on 2026-11-07). SENSOR = points relative to the lidar, as a real one reports them.
        lidars[frame] = Lidar.create(path=f"{prim_named(frame).GetPath()}/lidar", config=config, tick_rate=10.0,
                                     attributes={"omni:sensor:Core:scanRateBaseHz": 10, "omni:sensor:Core:nearRangeM": near,
                                                 "omni:sensor:Core:farRangeM": far, "omni:sensor:Core:outputFrameOfReference": "SENSOR"}, **kw)
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    for frame, lidar in lidars.items():                      # what the profile left on the prim (the cloud frame question)
        lp = stage.GetPrimAtPath(lidar.paths[0])
        print(f"lidar {frame}: type {lp.GetTypeName()}, outputFrameOfReference="
              f"{lp.GetAttribute('omni:sensor:Core:outputFrameOfReference').Get() if lp.HasAttribute('omni:sensor:Core:outputFrameOfReference') else 'NO SUCH ATTRIBUTE'}, "
              f"attrs with Frame: {[a.GetName() for a in lp.GetAttributes() if 'rame' in a.GetName()]}")

    def laser_scan_metadata(prim_path):                      # from the 6.2 'RTX Lidar Sensors' ROS 2 tutorial
        rotation_rate = float(prim_utils.get_prim_attribute_value(prim_path, "omni:sensor:Core:scanRateBaseHz") or 0)
        near_range = float(prim_utils.get_prim_attribute_value(prim_path, "omni:sensor:Core:nearRangeM") or 0)
        far_range = float(prim_utils.get_prim_attribute_value(prim_path, "omni:sensor:Core:farRangeM") or 0)
        firing_rate = int(prim_utils.get_prim_attribute_value(prim_path, "omni:sensor:Core:patternFiringRateHz") or 0)
        if rotation_rate <= 0 or firing_rate <= 0:
            raise RuntimeError(f"{prim_path}: scanRateBaseHz or patternFiringRateHz is 0")
        return {"horizontalFov": 360.0, "horizontalResolution": 360.0 * rotation_rate / firing_rate,
                "depthRange": [near_range, far_range], "rotationRate": rotation_rate, "azimuthRange": [-180.0, 180.0]}

    sensors = []                                             # kept alive: a LidarSensor owns its render product
    for frame, topic in (("rp_s1_lidar", "rp_s1_lidar/scan"), ("rp_s2e_lidar", "rp_s2e_lidar/scan")):
        s = LidarSensor(lidars[frame], annotators=[])
        s.attach_writer("RtxLidarROS2PublishLaserScan", topicName=topic, frameId=frame, **laser_scan_metadata(lidars[frame].paths[0]))
        sensors.append(s)
    s = LidarSensor(lidars["velodyne_frame"], annotators=[])
    s.attach_writer("RtxLidarROS2PublishPointCloud", topicName="velodyne_points", frameId="velodyne_frame")
    sensors.append(s)

    # 4. the ROS 2 bridge OmniGraph (isaacsim.ros2.bridge.* node types; the 6.0+ publishers take their data from the
    # Isaac Read Joint State / Isaac Compute Odometry nodes)
    relay = start_cmd_vel_relay()
    keys = og.Controller.Keys
    ros_nodes = ("clock", "joints", "odom", "odom_tf", "cmd_vel", "arm_cmd")
    og.Controller.edit({"graph_path": "/World/ROS2Bridge", "evaluator_name": "execution"}, {
        keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("context", "isaacsim.ros2.bridge.ROS2Context"),
            ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("read_joints", "isaacsim.sensors.physics.IsaacReadJointState"),
            ("joints", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("odom_c", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("odom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ("odom_tf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ("cmd_vel", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
            ("lin", "omni.graph.nodes.BreakVector3"), ("ang", "omni.graph.nodes.BreakVector3"),
            ("diff", "isaacsim.robot.wheeled_robots.DifferentialController"),
            ("wheels", "isaacsim.core.nodes.IsaacArticulationController"),
            ("arm_cmd", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("arm", "isaacsim.core.nodes.IsaacArticulationController"),
        ],
        keys.SET_VALUES: [
            ("clock.inputs:topicName", "clock"),
            ("read_joints.inputs:prim", [Sdf.Path(root_path)]), ("joints.inputs:topicName", "joint_states"),
            ("odom_c.inputs:chassisPrim", [Sdf.Path(root_path)]),
            ("odom.inputs:odomFrameId", "odom"), ("odom.inputs:chassisFrameId", "base_link"), ("odom.inputs:topicName", "diff_cont/odom"),
            ("odom_tf.inputs:parentFrameId", "odom"), ("odom_tf.inputs:childFrameId", "base_link"),
            ("cmd_vel.inputs:topicName", CMD_VEL_TWIST if relay else CMD_VEL_STAMPED),
            ("diff.inputs:wheelRadius", 0.1592), ("diff.inputs:wheelDistance", 0.563), ("diff.inputs:maxLinearSpeed", 1.0),
            ("wheels.inputs:targetPrim", [Sdf.Path(root_path)]), ("wheels.inputs:jointNames", WHEELS),
            ("arm_cmd.inputs:topicName", "ur5e/sim_joint_command"),
            ("arm.inputs:targetPrim", [Sdf.Path(root_path)]),
        ],
        keys.CONNECT: [("context.outputs:context", f"{n}.inputs:context") for n in ros_nodes] + [
            ("tick.outputs:tick", "clock.inputs:execIn"), ("time.outputs:simulationTime", "clock.inputs:timeStamp"),
            ("tick.outputs:tick", "read_joints.inputs:execIn"), ("read_joints.outputs:execOut", "joints.inputs:execIn"),
            ("read_joints.outputs:jointNames", "joints.inputs:jointNames"), ("read_joints.outputs:jointPositions", "joints.inputs:jointPositions"),
            ("read_joints.outputs:jointVelocities", "joints.inputs:jointVelocities"), ("read_joints.outputs:jointEfforts", "joints.inputs:jointEfforts"),
            ("read_joints.outputs:jointDofTypes", "joints.inputs:jointDofTypes"), ("read_joints.outputs:stageMetersPerUnit", "joints.inputs:stageMetersPerUnit"),
            ("read_joints.outputs:sensorTime", "joints.inputs:sensorTime"), ("time.outputs:simulationTime", "joints.inputs:timeStamp"),
            ("tick.outputs:tick", "odom_c.inputs:execIn"), ("odom_c.outputs:execOut", "odom.inputs:execIn"), ("odom_c.outputs:execOut", "odom_tf.inputs:execIn"),
            ("time.outputs:simulationTime", "odom.inputs:timeStamp"), ("time.outputs:simulationTime", "odom_tf.inputs:timeStamp"),
            ("odom_c.outputs:position", "odom.inputs:position"), ("odom_c.outputs:orientation", "odom.inputs:orientation"),
            ("odom_c.outputs:linearVelocity", "odom.inputs:linearVelocity"), ("odom_c.outputs:angularVelocity", "odom.inputs:angularVelocity"),
            ("odom_c.outputs:position", "odom_tf.inputs:translation"), ("odom_c.outputs:orientation", "odom_tf.inputs:rotation"),
            ("tick.outputs:tick", "cmd_vel.inputs:execIn"), ("cmd_vel.outputs:execOut", "diff.inputs:execIn"), ("tick.outputs:deltaSeconds", "diff.inputs:dt"),
            ("cmd_vel.outputs:linearVelocity", "lin.inputs:tuple"), ("lin.outputs:x", "diff.inputs:linearVelocity"),
            ("cmd_vel.outputs:angularVelocity", "ang.inputs:tuple"), ("ang.outputs:z", "diff.inputs:angularVelocity"),
            ("diff.outputs:velocityCommand", "wheels.inputs:velocityCommand"), ("tick.outputs:tick", "wheels.inputs:execIn"),
            ("tick.outputs:tick", "arm_cmd.inputs:execIn"), ("tick.outputs:tick", "arm.inputs:execIn"),
            ("arm_cmd.outputs:jointNames", "arm.inputs:jointNames"), ("arm_cmd.outputs:positionCommand", "arm.inputs:positionCommand"),
        ]})
    # cameras: RtxCamera prims (tick_rate = publish rate) under the frames, each with a render product + ROS2CameraHelper.
    # A USD camera looks down its -Z with +Y up; the ROS optical frames look down +Z with -Y up, hence the 180 deg about X.
    # The RealSense publishes what realsense2_camera publishes on the robot: colour + its camera_info, the depth
    # (rendered from the colour camera, so it is the aligned depth) + camera_info, and the stereo IR pair (848x480,
    # 50 mm baseline, the D435's imagers; the right one sits +50 mm along optical x) for visual SLAM. `fov` is the
    # horizontal field of view in degrees (D435: 69 colour, 87 IR; webcams about 70), set through the USD aperture.
    cams = []

    def camera(name, frame, w, h, fov, publishers, offset=(0.0, 0.0, 0.0)):
        """An RtxCamera under `frame` with one graph: render product + a ROS2CameraHelper (kind rgb | depth) or a
        ROS2CameraInfoHelper (kind info) per (kind, topic) in `publishers`. One edit builds the whole graph: the
        node names in CONNECT only resolve for nodes created in the same edit."""
        path = f"{prim_named(frame).GetPath()}/cam_{name}"
        cams.append(RtxCamera(path, tick_rate=15.0))
        prim = stage.GetPrimAtPath(path)
        cxf = UsdGeom.Xformable(prim)   # UNVERIFIED (6.2): replacing the xform ops RtxCamera authored
        cxf.ClearXformOpOrder()
        cxf.AddTranslateOp().Set(Gf.Vec3d(*offset))
        cxf.AddRotateXOp().Set(180.0)
        usd_cam = UsdGeom.Camera(prim)
        usd_cam.GetFocalLengthAttr().Set(10.0)
        usd_cam.GetHorizontalApertureAttr().Set(2 * 10.0 * math.tan(math.radians(fov) / 2))
        usd_cam.GetVerticalApertureAttr().Set(2 * 10.0 * math.tan(math.radians(fov) / 2) * h / w)
        nodes = [("tick", "omni.graph.action.OnPlaybackTick"), ("context", "isaacsim.ros2.bridge.ROS2Context"),
                 ("rp", "isaacsim.core.nodes.IsaacCreateRenderProduct")]
        values = [("rp.inputs:cameraPrim", [Sdf.Path(path)]), ("rp.inputs:width", w), ("rp.inputs:height", h)]
        connects = [("tick.outputs:tick", "rp.inputs:execIn")]
        for i, (kind, topic) in enumerate(publishers):
            node = f"pub{i}"
            nodes.append((node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper" if kind == "info" else "isaacsim.ros2.bridge.ROS2CameraHelper"))
            values += [(f"{node}.inputs:topicName", topic), (f"{node}.inputs:frameId", frame)]
            if kind != "info":
                values.append((f"{node}.inputs:type", kind))
            connects += [("rp.outputs:execOut", f"{node}.inputs:execIn"), ("rp.outputs:renderProductPath", f"{node}.inputs:renderProductPath"),
                         ("context.outputs:context", f"{node}.inputs:context")]
        og.Controller.edit({"graph_path": f"/World/Cam_{name}", "evaluator_name": "execution"},
                           {keys.CREATE_NODES: nodes, keys.SET_VALUES: values, keys.CONNECT: connects})

    camera("realsense", "camera_link_optical", 1280, 720, 69.0,
           [("rgb", "/camera/camera/color/image_raw"), ("info", "/camera/camera/color/camera_info"),
            ("depth", "/camera/camera/depth/image_rect_raw"), ("info", "/camera/camera/depth/camera_info")])
    for name, frame, topic in (("scene", "camera_link_optical", "/cameras/scene/image"), ("wrist", "ur_tool0", "/cameras/wrist/image"),
                               ("side", "camera_link_optical", "/cameras/side/image")):
        camera(name, frame, 960, 540, 70.0, [("rgb", topic)])
    # the IR pair: both prims hang under camera_link_optical (the only RealSense frame in the URDF); their ROS frames
    # camera_infra{1,2}_optical_frame are static TFs from sim.launch.py (the robot's driver publishes its own)
    for name, offset in (("infra1", (0.0, 0.0, 0.0)), ("infra2", (0.05, 0.0, 0.0))):
        path = f"{prim_named('camera_link_optical').GetPath()}/cam_{name}"
        cams.append(RtxCamera(path, tick_rate=15.0))
        prim = stage.GetPrimAtPath(path)
        cxf = UsdGeom.Xformable(prim)
        cxf.ClearXformOpOrder()
        cxf.AddTranslateOp().Set(Gf.Vec3d(*offset))
        cxf.AddRotateXOp().Set(180.0)
        usd_cam = UsdGeom.Camera(prim)
        usd_cam.GetFocalLengthAttr().Set(10.0)
        usd_cam.GetHorizontalApertureAttr().Set(2 * 10.0 * math.tan(math.radians(87.0) / 2))
        usd_cam.GetVerticalApertureAttr().Set(2 * 10.0 * math.tan(math.radians(87.0) / 2) * 480 / 848)
    og.Controller.edit({"graph_path": "/World/Cam_infra", "evaluator_name": "execution"}, {
        keys.CREATE_NODES: [("tick", "omni.graph.action.OnPlaybackTick"), ("context", "isaacsim.ros2.bridge.ROS2Context"),
                            ("rp1", "isaacsim.core.nodes.IsaacCreateRenderProduct"), ("rp2", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                            ("img1", "isaacsim.ros2.bridge.ROS2CameraHelper"), ("img2", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                            ("info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper")],
        keys.SET_VALUES: [("rp1.inputs:cameraPrim", [Sdf.Path(f"{prim_named('camera_link_optical').GetPath()}/cam_infra1")]), ("rp1.inputs:width", 848), ("rp1.inputs:height", 480),
                          ("rp2.inputs:cameraPrim", [Sdf.Path(f"{prim_named('camera_link_optical').GetPath()}/cam_infra2")]), ("rp2.inputs:width", 848), ("rp2.inputs:height", 480),
                          ("img1.inputs:type", "rgb"), ("img1.inputs:topicName", "/camera/camera/infra1/image_rect_raw"), ("img1.inputs:frameId", "camera_infra1_optical_frame"),
                          ("img2.inputs:type", "rgb"), ("img2.inputs:topicName", "/camera/camera/infra2/image_rect_raw"), ("img2.inputs:frameId", "camera_infra2_optical_frame"),
                          ("info.inputs:topicName", "/camera/camera/infra1/camera_info"), ("info.inputs:frameId", "camera_infra1_optical_frame"),
                          ("info.inputs:topicNameRight", "/camera/camera/infra2/camera_info"), ("info.inputs:frameIdRight", "camera_infra2_optical_frame")],
        keys.CONNECT: [("tick.outputs:tick", "rp1.inputs:execIn"), ("tick.outputs:tick", "rp2.inputs:execIn"),
                       ("rp1.outputs:execOut", "img1.inputs:execIn"), ("rp1.outputs:renderProductPath", "img1.inputs:renderProductPath"), ("context.outputs:context", "img1.inputs:context"),
                       ("rp2.outputs:execOut", "img2.inputs:execIn"), ("rp2.outputs:renderProductPath", "img2.inputs:renderProductPath"), ("context.outputs:context", "img2.inputs:context"),
                       ("rp1.outputs:execOut", "info.inputs:execIn"), ("rp1.outputs:renderProductPath", "info.inputs:renderProductPath"),
                       ("rp2.outputs:renderProductPath", "info.inputs:renderProductPathRight"), ("context.outputs:context", "info.inputs:context")]})
    # 5. run (the timeline plays, every update renders a frame - the RTX sensors need that, headless or not)
    app_utils.play()
    print(f"Robot Disco in Isaac Sim: ROS 2 bridge up (clock, odom->base_link tf, joint_states, odom, scans, cloud, cameras; "
          f"{CMD_VEL_STAMPED} + ur5e/sim_joint_command in); articulation root {root_path}")
    while app.is_running():  # UNVERIFIED (6.2): the RTX sensors rendering under SimulationApp(headless=True)
        app.update()
    app_utils.stop()
    if relay:
        import rclpy
        relay.destroy_node()
        rclpy.try_shutdown()
    app.close()


if __name__ == "__main__":
    main()
