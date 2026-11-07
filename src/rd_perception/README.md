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
| `/cameras/{scene,wrist,side}/image_raw`, `/image` (rotated upright) | usb_cam by USB port path + `image_rotate`; replaces v1's multicast camera service |
| `/tags/detections` (`rd_interfaces/TagDetectionArray`) | `tag_adapter` from Isaac's `/tag_detections`, or `fake_tags` in the rig |

`isaac.launch.py` holds the Isaac ROS 4.6 pieces (image rectify + AprilTag, occupancy-grid localizer, nvblox,
visual SLAM), each off by default: the robot runs on lidars + AMCL and gains a layer when it is proven. It
cannot run in the rig; the GPU box (phase 7) validates it and the node / parameter names are expected to move.

Tests: the range correction, the Isaac→ours conversion, the rotation and the configs (ROS-free); a launch test
of the offset node and the fake detector against a static TF. Hardware only: USB ports, the S1 constant, camera
rotations, the RealSense profiles on the Orin.
