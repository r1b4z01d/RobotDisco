"""Build the Robot Disco scene in Isaac Sim 6.2 and bridge it to ROS 2 (Jazzy). Run INSIDE Isaac Sim's python:

    ~/isaacsim/python.sh ~/RobotDisco/src/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf [--headless] [--world zone_one]

What it does (Isaac Sim 6.2 / isaacsim.* API; written against the 6.2 docs, NOT yet run - phase 7 validates it):
  1. imports the flat URDF (rd_sim export_urdf) as an articulation with the drive wheels velocity-driven and the
     UR joints position-driven; the hand is the baked visual
  2. builds the world: a ground plane and the active map's occupied cells as 2.5 m boxes (so the simulated lidars
     see the walls the map has) from rd_nav's maps folder
  3. attaches sensors: two 2D PhysX lidars (frames rp_s1_lidar / rp_s2e_lidar, 360 deg, 1 deg, 0.1..20 m),
     a Velodyne-like 3D lidar (velodyne_frame), an RGB-D camera (camera_link_optical, 1280x720), three RGB
     cameras (scene / wrist / side)
  4. wires the ROS 2 bridge OmniGraph: /clock, TF (odom->base_link from the articulation's chassis pose,
     the robot's fixed frames), /joint_states (wheels + UR), /diff_cont/odom, the scans on
     /rp_s1_lidar/scan_filtered and /rp_s2e_lidar/scan_filtered, /velodyne_points, the camera topics the
     rest of 5.0 subscribes to, and subscribers: /diff_cont/cmd_vel (TwistStamped -> wheel velocities) and
     /ur5e/forward_position_controller/commands + /ur5e/joint_trajectory_controller (positions -> UR drives)
  5. runs the simulation loop until closed

With this running, the 5.0 stack runs unchanged on the GPU box: rd_nav + rd_arm (sim_mode for the mock UR, or
the UR driven by the bridge's joint command subscriber) + rd_vla + rd_manager. The base's own controller
manager is NOT run (the bridge integrates cmd_vel itself); the arm's controller manager runs on mock hardware
and the bridge mirrors its commanded joints.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default="/tmp/robot_disco.urdf")
    ap.add_argument("--world", default="zone_one", help="a map from rd_nav/maps to build walls from ('' = empty)")
    ap.add_argument("--maps-dir", default="")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--x", type=float, default=0.7)
    ap.add_argument("--y", type=float, default=0.9)
    ap.add_argument("--yaw", type=float, default=0.0)
    return ap.parse_args()


def map_boxes(maps_dir: Path, name: str, height: float = 2.5, merge: int = 4):
    """Occupied cells of a nav map as (x, y, size) blocks, merged merge x merge cells at a time."""
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


def main():
    args = parse()
    from isaacsim import SimulationApp  # noqa: PLC0415 - Isaac Sim's python only
    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
    import numpy as np
    import omni.graph.core as og
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.asset.importer.urdf import _urdf
    from pxr import Gf, UsdGeom, UsdPhysics

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # 1. the robot
    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints = False
    cfg.fix_base = False
    cfg.make_default_prim = False
    cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    cfg.default_drive_strength = 1e4
    cfg.default_position_drive_damping = 1e3
    cfg.distance_scale = 1.0
    ok, prim_path = _urdf.acquire_urdf_interface().import_robot(str(Path(args.urdf).parent), Path(args.urdf).name, cfg, "/World/RobotDisco")
    assert ok, "URDF import failed"
    stage = world.stage
    robot = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(robot)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(args.x, args.y, 0.2))
    xf.AddRotateZOp().Set(math.degrees(args.yaw))
    for wheel in ("left_wheel_joint", "right_wheel_joint"):          # the wheels are velocity driven
        drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{prim_path}/joints/{wheel}"), "angular")
        if drive:
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(1e4)

    # 2. the walls from the map
    if args.world:
        maps_dir = Path(args.maps_dir) if args.maps_dir else Path(__import__("rd_nav.maps_store", fromlist=["default_maps_dir"]).default_maps_dir())
        boxes, height = map_boxes(maps_dir, args.world)
        from isaacsim.core.api.objects import FixedCuboid
        for i, (cx, cy, size) in enumerate(boxes):
            world.scene.add(FixedCuboid(prim_path=f"/World/Walls/w{i}", name=f"wall_{i}", position=np.array([cx, cy, height / 2]),
                                        scale=np.array([size, size, height]), color=np.array([0.6, 0.6, 0.6])))
        print(f"{len(boxes)} wall blocks from {args.world}")

    # 3. sensors + 4. the ROS 2 bridge: one OmniGraph per sensor group (isaacsim.ros2.bridge nodes)
    import omni.kit.commands
    from isaacsim.core.utils import extensions
    extensions.enable_extension("isaacsim.ros2.bridge")
    extensions.enable_extension("isaacsim.sensors.physx")
    for frame, topic in (("rp_s1_lidar", "/rp_s1_lidar/scan_filtered"), ("rp_s2e_lidar", "/rp_s2e_lidar/scan_filtered")):
        omni.kit.commands.execute("RangeSensorCreateLidar", path=f"{prim_path}/{frame}/lidar", parent=None, min_range=0.1, max_range=20.0,
                                  draw_points=False, draw_lines=False, horizontal_fov=360.0, vertical_fov=1.0, horizontal_resolution=1.0,
                                  vertical_resolution=1.0, rotation_rate=10.0, high_lod=False, yaw_offset=0.0, enable_semantics=False)
    omni.kit.commands.execute("RangeSensorCreateLidar", path=f"{prim_path}/velodyne_frame/lidar", parent=None, min_range=0.5, max_range=50.0,
                              draw_points=False, draw_lines=False, horizontal_fov=360.0, vertical_fov=30.0, horizontal_resolution=0.4,
                              vertical_resolution=2.0, rotation_rate=10.0, high_lod=True, yaw_offset=0.0, enable_semantics=False)
    keys = og.Controller.Keys
    og.Controller.edit({"graph_path": "/World/ROS2Bridge", "evaluator_name": "execution"}, {
        keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("context", "isaacsim.ros2.bridge.ROS2Context"),
            ("joints", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("odom_c", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("odom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ("odom_tf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ("tf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ("cmd_vel", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
            ("diff", "isaacsim.robot.wheeled_robots.DifferentialController"),
            ("wheels", "isaacsim.core.nodes.IsaacArticulationController"),
            ("arm_cmd", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("arm", "isaacsim.core.nodes.IsaacArticulationController"),
            ("scan_s1", "isaacsim.ros2.bridge.ROS2PublishLaserScan"), ("read_s1", "isaacsim.sensors.physx.IsaacReadLidarBeams"),
            ("scan_s2e", "isaacsim.ros2.bridge.ROS2PublishLaserScan"), ("read_s2e", "isaacsim.sensors.physx.IsaacReadLidarBeams"),
            ("cloud", "isaacsim.ros2.bridge.ROS2PublishPointCloud"), ("read_v", "isaacsim.sensors.physx.IsaacReadLidarPointCloud"),
        ],
        keys.SET_VALUES: [
            ("clock.inputs:topicName", "clock"),
            ("joints.inputs:targetPrim", prim_path), ("joints.inputs:topicName", "joint_states"),
            ("odom_c.inputs:chassisPrim", prim_path),
            ("odom.inputs:odomFrameId", "odom"), ("odom.inputs:chassisFrameId", "base_link"), ("odom.inputs:topicName", "diff_cont/odom"),
            ("odom_tf.inputs:parentFrameId", "odom"), ("odom_tf.inputs:childFrameId", "base_link"),
            ("tf.inputs:parentPrim", prim_path), ("tf.inputs:targetPrims", [f"{prim_path}/rp_s1_lidar", f"{prim_path}/rp_s2e_lidar", f"{prim_path}/velodyne_frame",
                                                                             f"{prim_path}/camera_link_optical", f"{prim_path}/ur_tool0", f"{prim_path}/base_footprint"]),
            ("cmd_vel.inputs:topicName", "diff_cont/cmd_vel"),
            ("diff.inputs:wheelRadius", 0.1592), ("diff.inputs:wheelDistance", 0.563), ("diff.inputs:maxLinearSpeed", 1.0),
            ("wheels.inputs:targetPrim", prim_path), ("wheels.inputs:jointNames", ["left_wheel_joint", "right_wheel_joint"]),
            ("arm_cmd.inputs:topicName", "ur5e/sim_joint_command"),
            ("arm.inputs:targetPrim", prim_path),
            ("read_s1.inputs:lidarPrim", f"{prim_path}/rp_s1_lidar/lidar"), ("scan_s1.inputs:frameId", "rp_s1_lidar"), ("scan_s1.inputs:topicName", "rp_s1_lidar/scan_filtered"),
            ("read_s2e.inputs:lidarPrim", f"{prim_path}/rp_s2e_lidar/lidar"), ("scan_s2e.inputs:frameId", "rp_s2e_lidar"), ("scan_s2e.inputs:topicName", "rp_s2e_lidar/scan_filtered"),
            ("read_v.inputs:lidarPrim", f"{prim_path}/velodyne_frame/lidar"), ("cloud.inputs:frameId", "velodyne_frame"), ("cloud.inputs:topicName", "velodyne_points"),
        ],
        keys.CONNECT: [
            ("tick.outputs:tick", "clock.inputs:execIn"), ("time.outputs:simulationTime", "clock.inputs:timeStamp"),
            ("tick.outputs:tick", "joints.inputs:execIn"), ("time.outputs:simulationTime", "joints.inputs:timeStamp"),
            ("tick.outputs:tick", "odom_c.inputs:execIn"), ("odom_c.outputs:execOut", "odom.inputs:execIn"),
            ("time.outputs:simulationTime", "odom.inputs:timeStamp"),
            ("odom_c.outputs:position", "odom.inputs:position"), ("odom_c.outputs:orientation", "odom.inputs:orientation"),
            ("odom_c.outputs:linearVelocity", "odom.inputs:linearVelocity"), ("odom_c.outputs:angularVelocity", "odom.inputs:angularVelocity"),
            ("odom_c.outputs:execOut", "odom_tf.inputs:execIn"), ("time.outputs:simulationTime", "odom_tf.inputs:timeStamp"),
            ("odom_c.outputs:position", "odom_tf.inputs:translation"), ("odom_c.outputs:orientation", "odom_tf.inputs:rotation"),
            ("tick.outputs:tick", "tf.inputs:execIn"), ("time.outputs:simulationTime", "tf.inputs:timeStamp"),
            ("tick.outputs:tick", "cmd_vel.inputs:execIn"), ("cmd_vel.outputs:execOut", "diff.inputs:execIn"),
            ("cmd_vel.outputs:linearVelocity", "diff.inputs:linearVelocity"), ("cmd_vel.outputs:angularVelocity", "diff.inputs:angularVelocity"),
            ("diff.outputs:velocityCommand", "wheels.inputs:velocityCommand"), ("tick.outputs:tick", "wheels.inputs:execIn"),
            ("tick.outputs:tick", "arm_cmd.inputs:execIn"), ("arm_cmd.outputs:execOut", "arm.inputs:execIn"),
            ("arm_cmd.outputs:jointNames", "arm.inputs:jointNames"), ("arm_cmd.outputs:positionCommand", "arm.inputs:positionCommand"),
            ("tick.outputs:tick", "read_s1.inputs:execIn"), ("read_s1.outputs:execOut", "scan_s1.inputs:execIn"),
            ("read_s1.outputs:linearDepthData", "scan_s1.inputs:linearDepthData"), ("read_s1.outputs:numCols", "scan_s1.inputs:numCols"),
            ("read_s1.outputs:numRows", "scan_s1.inputs:numRows"), ("read_s1.outputs:horizontalFov", "scan_s1.inputs:horizontalFov"),
            ("read_s1.outputs:horizontalResolution", "scan_s1.inputs:horizontalResolution"), ("read_s1.outputs:depthRange", "scan_s1.inputs:depthRange"),
            ("read_s1.outputs:rotationRate", "scan_s1.inputs:rotationRate"), ("read_s1.outputs:intensitiesData", "scan_s1.inputs:intensitiesData"),
            ("time.outputs:simulationTime", "scan_s1.inputs:timeStamp"),
            ("tick.outputs:tick", "read_s2e.inputs:execIn"), ("read_s2e.outputs:execOut", "scan_s2e.inputs:execIn"),
            ("read_s2e.outputs:linearDepthData", "scan_s2e.inputs:linearDepthData"), ("read_s2e.outputs:numCols", "scan_s2e.inputs:numCols"),
            ("read_s2e.outputs:numRows", "scan_s2e.inputs:numRows"), ("read_s2e.outputs:horizontalFov", "scan_s2e.inputs:horizontalFov"),
            ("read_s2e.outputs:horizontalResolution", "scan_s2e.inputs:horizontalResolution"), ("read_s2e.outputs:depthRange", "scan_s2e.inputs:depthRange"),
            ("read_s2e.outputs:rotationRate", "scan_s2e.inputs:rotationRate"), ("read_s2e.outputs:intensitiesData", "scan_s2e.inputs:intensitiesData"),
            ("time.outputs:simulationTime", "scan_s2e.inputs:timeStamp"),
            ("tick.outputs:tick", "read_v.inputs:execIn"), ("read_v.outputs:execOut", "cloud.inputs:execIn"),
            ("read_v.outputs:data", "cloud.inputs:data"), ("time.outputs:simulationTime", "cloud.inputs:timeStamp"),
        ]})
    # cameras: the RGB-D on camera_link_optical and the three webcams, each through a ROS2CameraHelper
    from isaacsim.sensors.camera import Camera
    for name, frame, topic, size in (("realsense", "camera_link_optical", "/camera/camera/color/image_raw", (1280, 720)),
                                     ("scene", "camera_link_optical", "/cameras/scene/image", (960, 540)),
                                     ("wrist", "ur_tool0", "/cameras/wrist/image", (960, 540)),
                                     ("side", "camera_link_optical", "/cameras/side/image", (960, 540))):
        cam = Camera(prim_path=f"{prim_path}/{frame}/cam_{name}", frequency=15, resolution=size)
        cam.initialize()
        og.Controller.edit({"graph_path": f"/World/Cam_{name}", "evaluator_name": "execution"}, {
            keys.CREATE_NODES: [("tick", "omni.graph.action.OnPlaybackTick"), ("rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                                ("helper", "isaacsim.ros2.bridge.ROS2CameraHelper")],
            keys.SET_VALUES: [("rp.inputs:cameraPrim", f"{prim_path}/{frame}/cam_{name}"), ("rp.inputs:width", size[0]), ("rp.inputs:height", size[1]),
                              ("helper.inputs:type", "rgb"), ("helper.inputs:topicName", topic), ("helper.inputs:frameId", frame)],
            keys.CONNECT: [("tick.outputs:tick", "rp.inputs:execIn"), ("rp.outputs:execOut", "helper.inputs:execIn"),
                           ("rp.outputs:renderProductPath", "helper.inputs:renderProductPath")]})
    # 5. run
    world.reset()
    print("Robot Disco in Isaac Sim: ROS 2 bridge up (clock, tf, joint_states, odom, scans, cloud, cameras; cmd_vel + arm commands in)")
    while app.is_running():
        world.step(render=not args.headless)
    app.close()


if __name__ == "__main__":
    main()
