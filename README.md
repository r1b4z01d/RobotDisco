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

**Status 2026-11-07: every package written and passing its tests in the rig.** The robot still runs v1;
nothing here has run on hardware. Next: the GPU box (Isaac Sim, Isaac ROS), then the robot.

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

Every package builds and has tests; `rig/` holds the Docker test rig (`rig/rig.sh image|build|test|sh`) and
says what stands in for each piece of hardware. `rd_sim` is the one package still to write (Isaac Sim, on
the GPU box).

## Phases

| Phase | Delivers | Status |
|---|---|---|
| 0 | skeleton, `rd_ws` tagged v1 | done |
| 1 | the Docker rig (Jazzy, loopback sim, mock UR, fakes) | done |
| 2 | rd_bringup, rd_base, rd_nav, rd_perception | done, launch tests pass |
| 3 | rd_manager | done, launch test passes |
| 4 | rd_arm | done, launch test passes |
| 5 | rd_vla | done, launch test passes |
| 6 | rd_audio, rd_face, rd_voice | done, launch tests pass |
| 7 | rd_sim (Isaac Sim) + Isaac ROS validated on the GPU box | next |
| 8 | the robot: snapshot + `/mnt/data` copy (the gate), JetPack 7.2.1 flash, `provision.sh all`, hardware checks | after 7 |
| 9 | optional: visual odometry, GR00T, docking | |
| 10 | retire v1 | |

The detailed plan, risks and decisions live in the
[Robot Disco v2 Rebuild Plan](https://claude.ai/artifact/T2mFGJhZuEiFKGtLmpHvPM).

## Build and test

```
rig/rig.sh image && rig/rig.sh build && rig/rig.sh test      # on the Mac, in Docker
cd ~/RobotDisco && colcon build --symlink-install              # on the robot or the GPU box (Jazzy sourced)
ros2 launch rd_bringup robot.launch.py sim_mode:=true          # the base; each package has its own launch
```

On the robot every launch is a supervisor unit (`src/rd_bringup/services/`), and `provision/deploy_to_rd.sh`
pushes this tree there.

## License

Our code is [CC BY-NC-SA 4.0](LICENSE). Submodules under `vendor/` and `firmware/` keep their own licenses.
