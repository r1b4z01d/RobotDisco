# rd_sim

Robot Disco in Isaac Sim 6.2 on the GPU box (phase 7): the robot from `rd_description`, walls from the active
nav map, two 2D lidars, a 3D lidar, the RealSense and the three webcams, and the ROS 2 bridge publishing the
same topics the robot does, so the whole v2 stack runs against it unchanged.

```
ros2 run rd_sim export_urdf /tmp/robot_disco.urdf                       # the flat URDF the importer takes
~/isaacsim/python.sh $(ros2 pkg prefix rd_sim)/share/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf
ros2 launch rd_sim sim.launch.py                                        # nav + arm + vla (fake) + manager, use_sim_time
```

`scripts/isaac_scene.py` is written against the Isaac Sim 6.2 API and has not run yet. Known from the 6.2 docs
(read 2026-11-07) and still to apply on the first run: the pip install goes in a Python 3.12 venv
(`provision/gpu/provision_gpu.sh isaacsim`), the EULA needs `OMNI_KIT_ACCEPT_EULA=YES`, the ROS 2 bridge
extension is `isaacsim.ros2.bridge` with Jazzy recommended on 24.04 and Cyclone DDS supported; the transform
tree and joint state publishers now take their data from `Isaac Compute Transform Tree` and `Isaac Read Joint
State` nodes (the script's direct `targetPrims` / `targetPrim` inputs are deprecated but still accepted);
the PhysX lidar is deprecated in favour of the Physics Raycast Sensor / RTX lidar, so the three lidars will
move to that API. `arm_mirror` sends the mock UR's commanded joints to
the simulated arm. Tests cover what runs without Isaac Sim: the URDF export and the map-to-walls conversion.
Replaces v1's `Scripts/sim_robot.py` and the MuJoCo cell.
