# rd_perception

The sensors as ROS 2 topics, and the Isaac ROS layer behind launch arguments.

```
ros2 launch rd_perception perception.launch.py                      # robot: lidars + RealSense + webcams
ros2 launch rd_perception perception.launch.py isaac:=true apriltag:=true grid_localizer:=true nvblox:=true
ros2 launch rd_perception perception.launch.py sim_mode:=true       # rig: no drivers, the fake tag detector
```

| Topic | From |
|---|---|
| `/rp_s1_lidar/scan_filtered` | S1 driver → `scan_range_offset` (the faulty unit's constant, from robot.yaml) → body polygon + speckle filter |
| `/rp_s2e_lidar/scan_filtered` | S2E driver (UDP) → filter |
| `/velodyne_points`, `/velodyne/scan` | velodyne driver + transform + laserscan |
| `/camera/camera/color/image_raw`, `.../depth/color/points`, `.../infra*` | realsense2_camera (1280x720 @ 15 colour, 848x480 depth, IR only with `visual_slam`) |
| `/cameras/{scene,wrist,side}/image_raw`, `/image` (rotated upright, scene and side cut to v1's 0.40 / 0.30 squares) | usb_cam by USB port path + `image_rotate` (robot.yaml `rotate`, `square`); replaces v1's multicast camera service, whose square crop the policies were trained on |
| `/tags/detections` (`rd_interfaces/TagDetectionArray`) | `tag_adapter` from Isaac's `/tag_detections`, or `fake_tags` in the rig |

`isaac.launch.py` holds the Isaac ROS 4.6 pieces (image rectify + AprilTag, occupancy-grid localizer, nvblox,
visual SLAM), each off by default: the robot runs on lidars + AMCL and gains a layer when it is proven. It
cannot run in the rig; the GPU box (phase 7) validates it and the node / parameter names are expected to move.
Validated on the GPU box 2026-11-08 (Isaac ROS 4.6 debs, no camera): `ros2 run rd_perception fake_tag_image` renders a
tag36h11 on the RealSense colour topics and `isaac.launch.py apriltag:=true` detects it (id 3, centre, 0.571 m,
10 Hz through rectify, detector and adapter; the detector's pose orientation is a 180 degree turn about the optical
axis, which cancels in the corrector because tags are taught with the same detector). `grid_localizer:=true` loads
the LaserScan to FlatScan converter and the localizer wired scan_filtered -> /flatscan -> /initialpose, with
`/trigger_grid_search_localization` and is validated against Isaac Sim (`rd_sim/scripts/grid_kidnap_test.py`: AMCL pushed
6.8 m / 120 deg off, the localizer returned within 1.7 cm / 0.5 deg of the sim's ground truth in 2 sim seconds, AMCL
followed). The localizer has two scan inputs: `flatscan` is buffered for the trigger service, while every message on
`flatscan_localization` starts a localization and publishes /initialpose with zero covariance. Leave that one
unconnected: remapped onto /flatscan it re-seeded AMCL every 2 sim seconds on a 5 cm grid and defeated the kidnap
test. nvblox and visual SLAM are untested.

Tests: the range correction, the Isaac→ours conversion, the rotation and the configs (ROS-free); a launch test
of the offset node and the fake detector against a static TF. Hardware only: USB ports, the S1 constant, camera
rotations, the RealSense profiles on the Orin.
