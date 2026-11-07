# rig/ — the off-robot test rig

Everything in `src/` is built and tested here before it goes near the robot. The rig is a ROS 2 Jazzy
container (Docker via Colima on the Mac) with the same apt packages `provision/` installs, minus Isaac ROS
and CUDA. What each package's tests stand in for:

| Real thing | In the rig |
|---|---|
| RoboClaw base | ros2_control mock hardware (`rd_description` `sim_mode:=true`) |
| The world, lidars, odometry | `nav2_loopback_sim` (drives the TF tree from cmd_vel, fakes a scan from the map) |
| UR5e | `ur_robot_driver` with `use_mock_hardware:=true`; MoveIt plans with OMPL where the Orin uses cuMotion |
| AmazingHand ESP32 | a UDP loopback echo that speaks the firmware's line protocol |
| Isaac ROS AprilTag | a publisher of `apriltag_msgs/AprilTagDetectionArray` from a scripted scene |
| Isaac grid localizer, nvblox, visual SLAM | off (launch args); not tested here |
| Cameras | `usb_cam` on a test pattern, or image files published on the camera topics |
| Whisper, Kokoro, Claude / Ollama | fakes behind the same interfaces |
| VLA policy | a fake backend that returns scripted actions; SmolVLA itself on the GPU box |

```
rig/rig.sh image          # once, ~10 min
rig/rig.sh build          # colcon build of the whole workspace
rig/rig.sh test           # every package's tests
rig/rig.sh sh             # a shell with the workspace sourced
```

What only the GPU box can test (phase 7): Isaac Sim, every Isaac ROS node, SmolVLA inference speed.
What only the robot can test (phase 8): USB enumeration, the lidar range offset, the audio sink, the
8BitDo receiver, the real UR dashboard, PolyScope's remote-control mode, servo smoothness.
