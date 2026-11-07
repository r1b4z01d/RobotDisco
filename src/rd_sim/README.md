# rd_sim

Robot Disco in Isaac Sim 6.2 on the GPU box (phase 7): the robot from `rd_description`, walls from the active
nav map, two 2D lidars, a 3D lidar, the RealSense and the three webcams, and the ROS 2 bridge publishing the
same topics the robot does, so the whole v2 stack runs against it unchanged.

```
ros2 run rd_sim export_urdf /tmp/robot_disco.urdf                       # the flat URDF the importer takes
~/isaacsim/python.sh $(ros2 pkg prefix rd_sim)/share/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf
ros2 launch rd_sim sim.launch.py                                        # nav + arm + vla (fake) + manager, use_sim_time
```

`scripts/isaac_scene.py` is written against the Isaac Sim 6.2 API and has not run yet: the first run on the
4090 is expected to need node-name and attribute fixes. `arm_mirror` sends the mock UR's commanded joints to
the simulated arm. Tests cover what runs without Isaac Sim: the URDF export and the map-to-walls conversion.
Replaces v1's `Scripts/sim_robot.py` and the MuJoCo cell.
