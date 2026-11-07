# Robot Disco

The v2 workspace for Robot Disco: a differential-drive mobile robot (RoboClaw base, RPLiDAR S1 + S2E and a
Velodyne, a RealSense D435) carrying a UR5e with the 8-DOF AmazingHand, an animated face, a microphone and
a speaker. The robot PC is a Jetson AGX Orin on **JetPack 7.2.1 / Ubuntu 24.04**, the stack is **ROS 2
Jazzy + Isaac ROS 4.6**, and the whole robot is driven from one web console, `rd_manager`. A GPU box
(RTX 4090) trains the policies, runs Isaac Sim and mirrors the manager.

v2 is a rebuild of [rd_ws](https://github.com/r1b4z01d/rd_ws) (v1: Humble on JetPack 6, a patched
`apriltag_ros`, Foxglove, scripts spawning scripts). It keeps every v1 feature and drops Foxglove. The plan,
the feature-by-feature mapping, the phases and the risks are in the
[Robot Disco v2 Rebuild Plan](https://claude.ai/artifact/T2mFGJhZuEiFKGtLmpHvPM).

**Status 2026-11-06: phase 0.** The skeleton below exists; the robot still runs v1. Nothing here has run
on hardware.

## Layout

One repository, one colcon workspace, every piece of our code a ROS 2 package.

```
RobotDisco/
├── config/robot.yaml      every address and port, once (UR, hand, robot PC, GPU box, sensors, audio)
├── src/
│   ├── rd_interfaces/     RunPolicy, GoToLandmark, Wave, Say actions; ArmPower service        (done)
│   ├── rd_description/    URDF/xacro + meshes, ported from v1 with the same frames             (done)
│   ├── rd_base/           ros2_control RoboClaw diff drive, twist_mux, collision monitor, joystick
│   ├── rd_nav/            Nav2, maps + keepout, docking, AMCL + grid-localizer bootstrap, AprilTag corrector
│   ├── rd_perception/     Isaac ROS image pipeline, AprilTag, nvblox, optional visual SLAM; cameras as topics
│   ├── rd_arm/            UR driver (always on), MoveIt + cuMotion, hand node, Wave, ArmPower
│   ├── rd_face/           eyes + mouth, face ID, greeting, Folks data
│   ├── rd_voice/          ears, wake phrase, brain, skills, landmark tasks
│   ├── rd_audio/          the one sink watchdog, playback, Kokoro TTS, Say
│   ├── rd_vla/            recorder, inference server (SmolVLA, GR00T), RunPolicy, teleop
│   ├── rd_manager/        the web console (FastAPI + rclpy, WebSocket live data)
│   ├── rd_sim/            Isaac Sim scene + bridge
│   └── rd_bringup/        top-level launch, supervisor units, udev, netplan
├── vendor/                source-built ROS packages only where apt has no Jazzy binary (submodules)
├── firmware/AmazingHand/  the hand's CAD + ESP32 firmware (submodule, r1b4z01d/AmazingHand)
└── provision/             JetPack 7.2 provisioning phases, package lists, udev, netplan, DDS profile
```

Packages marked done build; the others hold a README saying what they will contain and where in v1 the
code comes from, plus a `COLCON_IGNORE` until they exist.

## Phases

| Phase | Delivers | Done when |
|---|---|---|
| 0 | v1 snapshot + NVMe copy on the GPU box, `rd_ws` tagged v1, this skeleton | the GPU box holds a verified copy of everything on the robot |
| 1 | JetPack 7.2 flash, Jazzy + Isaac ROS, rd_base, rd_nav, rd_perception, rd_bringup | localizes on boot, drives a v1 route, gamepad deadman works |
| 2 | rd_manager | every v1 page function with no second server |
| 3 | rd_arm | collision-aware wave; 50 Hz stream as smooth as v1's servoJ |
| 4 | rd_vla | "pick up tigger" from a landmark task end to end |
| 5 | rd_audio, rd_face, rd_voice | greets by name, waves, answers, runs a task by voice |
| 6 | rd_sim | the stack runs against Isaac Sim |
| 7 | optional: visual odometry, GR00T backend, docking | measured against SmolVLA |
| 8 | retire v1 | rd_ws archived, wiki rewritten |

## Build

```
cd ~/RobotDisco
rosdep install --from-paths src vendor --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch rd_description rsp.launch.py sim_mode:=true
```

## License

Our code is [CC BY-NC-SA 4.0](LICENSE). Submodules under `vendor/` and `firmware/` keep their own licenses.
