# rd_sim

Robot Disco in Isaac Sim 6.2 on the GPU box (phase 7): the robot from `rd_description`, walls from the active
nav map, two 2D lidars, a 3D lidar, the RealSense and the three webcams, and the ROS 2 bridge publishing the
same topics the robot does, so the whole 5.0 stack runs against it unchanged.

```
ros2 run rd_sim export_urdf /tmp/robot_disco.urdf                       # the flat URDF the importer takes
~/isaacsim-venv/bin/python $(ros2 pkg prefix rd_sim)/share/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf   # Jazzy sourced, Cyclone, EULA accepted
ros2 launch rd_sim sim.launch.py                                        # nav + arm + vla (fake) + manager, use_sim_time
```

`scripts/isaac_scene.py` was rewritten on 2026-11-07 against the Isaac Sim 6.2 docs (every API name it uses was
checked there; it has still not run - phase 7 does that). The pip install in a Python 3.12 venv
(`provision/gpu/provision_gpu.sh isaacsim`) runs it with `~/isaacsim-venv/bin/python`, not `python.sh`, with the system
Jazzy sourced, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and `OMNI_KIT_ACCEPT_EULA=YES`. Applied: the URDF goes through
`URDFImporter` / `URDFImporterConfig` (the 5.x `_urdf.ImportConfig` + `import_robot` path was removed in 6.0) and the
resulting USD is referenced into the stage; the deprecated `isaacsim.core.api` World / `FixedCuboid` gave way to
`isaacsim.core.experimental` (`GroundPlane`, `Cube`, the stage utils, `SimulationManager`); the lidars are RTX lidars
(`isaacsim.sensors.experimental.rtx` `Lidar` / `LidarSensor` with the `RtxLidarROS2PublishLaserScan` /
`RtxLidarROS2PublishPointCloud` writers) instead of the deprecated PhysX lidar, whose other replacement, the Physics
Raycast Sensor, has no ROS 2 publisher; the cameras are `RtxCamera` prims (the deprecated `isaacsim.sensors.camera` is
gone) behind `IsaacCreateRenderProduct` + `ROS2CameraHelper`; `/joint_states` is fed by `Isaac Read Joint State` and
`/diff_cont/odom` + the `odom->base_link` TF by `Isaac Compute Odometry` (the direct `targetPrim` / `targetPrims`
inputs are deprecated); the robot's link TF is left to robot_state_publisher (publishing it from the sim too gave tf2 a
second parent for frames like `ur_tool0`); `/diff_cont/cmd_vel` is TwistStamped, which the bridge's `ROS2SubscribeTwist`
cannot take, so a small rclpy relay inside the script turns it into Twist and zeroes it 0.5 s after the last command.
Still unverified until the first run (marked `# UNVERIFIED (6.2):` in the script): the importer's regex-dict joint
overrides and the layout of its USD (the script finds links and joints by name and re-asserts the wheel / UR drives on
the USD), `Example_Rotary` standing in for the Velodyne and the range overrides on the example lidar profiles, replacing
the camera prims' xform ops with the 180 deg optical-frame turn, rclpy from the system Jazzy inside the Isaac venv, and
RTX sensor rendering under `--headless`. First run on the GPU box (2026-11-07): the app comes up headless and the
URDF imports; the converter (URDF USD Converter 0.3) writes the joints into a `Physics` variant set that the script now
selects (`physx`), and a massless root link made every child body its own articulation with the wheels jointed to the
world, so `export_urdf` gives the root link a token mass. `arm_mirror` sends the mock UR's commanded joints to the simulated arm. Tests
cover what runs without Isaac Sim: the URDF export and the map-to-walls conversion. Replaces v1's `Scripts/sim_robot.py`
and the MuJoCo cell.
