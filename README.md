# Robot Disco

The 5.0 workspace for Robot Disco: a differential-drive mobile robot (RoboClaw base, RPLiDAR S1 + S2E and a
Velodyne, a RealSense D435) carrying a UR5e with the 8-DOF AmazingHand, an animated face, a microphone and
a speaker. The robot PC is a Jetson AGX Orin on **JetPack 7.2.1 / Ubuntu 24.04**, the stack is **ROS 2
Jazzy + Isaac ROS 4.6**, and the whole robot is driven from one web console, `rd_manager`. A GPU box
(RTX 4090) trains the policies, runs Isaac Sim and mirrors the manager.

5.0 is a rebuild of [rd_ws](https://github.com/r1b4z01d/rd_ws) (v1: Humble on JetPack 6, a patched
`apriltag_ros`, Foxglove, scripts spawning scripts). It keeps every v1 feature and drops Foxglove. The
[wiki](https://github.com/r1b4z01d/RobotDisco/wiki) explains the architecture, the hardware, setup and each subsystem.

**Status 2026-11-11: running on the robot.** The Jetson is flashed and provisioned, eleven supervisor units start at
boot, and every subsystem has run on the hardware (driving, mapping, Nav2 goals, AprilTag localisation, the arm and
hand, face recognition and greeting, voice, the console). Open: a policy run to the end on the real arm.

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
says what stands in for each piece of hardware. `rd_sim` runs the robot in Isaac Sim on the GPU box; see
*Run the sim* below.

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
| 7 | rd_sim (Isaac Sim) + Isaac ROS validated on the GPU box | done (2026-11-09) |
| 8 | the robot: snapshot + `/mnt/data` copy (the gate), JetPack 7.2.1 flash, `provision.sh all`, hardware checks | flashed and provisioned 2026-11-10; hardware checks under way |
| 9 | optional: visual odometry, GR00T, docking | |
| 10 | retire v1 | |

## Run the sim (GPU box)

The runbook is `provision/gpu/README.md` (*Manual session: the GUI and the sim*) with the sensor and bridge
details in `src/rd_sim/README.md`. The short form, from a shell with the workspace sourced:

```
ros2 run rd_sim export_urdf /tmp/robot_disco.urdf                        # after a pull + colcon build
~/isaacsim-venv/bin/python src/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf [--headless]
                                                                         # ~100 s, ends with "ROS 2 bridge up"
ros2 launch rd_sim sim.launch.py                                         # the robot's stack against it, in a second shell
```

Then `http://4090TI.local:8092/live`. Start the sim before the stack every time: the stack seeds AMCL at the
dock, so a robot that already drove somewhere needs a fresh sim first. Detached (survives an ssh drop):

```
(setsid nohup timeout 14400 ~/isaacsim-venv/bin/python src/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf --headless > ~/isaac.log 2>&1 < /dev/null & echo $! > ~/isaac.pid)
provision/tools/launch_bg.sh ~/sim-stack.pid ~/sim-stack.log ros2 launch rd_sim sim.launch.py
kill -INT $(cat ~/sim-stack.pid); kill -INT -- -$(cat ~/isaac.pid)     # stop: the stack by its launch pid, the sim by its group
```

`launch_bg.sh` starts the launch with default signal dispositions. From a non-interactive shell (an agent, a script, an ssh
command) a plain `&` inherits SIGINT ignored, and a launch started that way never reacts to `kill -INT`: that was the
"30 s hang with 38 processes alive" of 2026-11-09; started this way the same stack is down 10 s after SIGINT.

The `timeout 14400` is the sim's lifetime in seconds (4 h): the sim exits by itself when it runs out. A 3000 s limit ended a session
after 50 minutes on 2026-11-09, and everything read afterwards (a robot that no longer answered the drive commands) was a
dying sim, not a robot problem.

Stop the stack by signalling the `ros2 launch` process only, never its whole process group and never with
`-9`: a group signal kills the Nav2 nodes before the launch knows it is shutting down, so it respawns them
(seen 2026-11-07), and a hard-killed launch leaves its Python nodes running (`rd_bringup.process`).

## Build and test

Run the tests one package at a time (`colcon test --executor sequential`, which `rig/rig.sh test` does): the launch
tests share one ROS domain and the fakes' fixed ports (the UR dashboard on 29999, the hand on 8766), so two
packages tested in parallel fail each other.

```
rig/rig.sh image && rig/rig.sh build && rig/rig.sh test      # on the Mac, in Docker
cd ~/RobotDisco && colcon build --symlink-install              # on the robot or the GPU box (Jazzy sourced)
ros2 launch rd_bringup robot.launch.py sim_mode:=true          # the base; each package has its own launch
```

On the robot every launch is a supervisor unit (`src/rd_bringup/services/`), and `provision/deploy_to_rd.sh`
pushes this tree there.

## License

Our code is [CC BY-NC-SA 4.0](LICENSE). Submodules under `vendor/` and `firmware/` keep their own licenses.
