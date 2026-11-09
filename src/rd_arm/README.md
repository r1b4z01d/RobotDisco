# rd_arm

The UR5e and the AmazingHand: the driver's controller manager (always on, under `/ur5e`), MoveIt, the pose
and motion libraries, and rd_arm's servers.

```
ros2 launch rd_arm arm.launch.py                         # robot
ros2 launch rd_arm arm.launch.py sim_mode:=true          # rig: mock arm, fake dashboard, fake hand
ros2 launch rd_arm arm.launch.py sim_mode:=true cumotion:=true   # + Isaac ROS cuMotion as MoveIt's default planner (what sim.launch.py does)
ros2 service call /arm/power rd_interfaces/srv/ArmPower "{command: 1}"          # STATUS 0, POWER_ON 1, POWER_OFF 2, UNLOCK 3, SHUTDOWN 4
ros2 action send_goal /wave rd_interfaces/action/Wave "{motion: wave, return_to: home}" --feedback
ros2 service call /hand/close std_srvs/srv/Trigger ;  ros2 topic pub /hand/curls std_msgs/msg/Float32MultiArray "{data: [0.5, 0.5, 0.5, 0]}"
ros2 service call /arm/stream std_srvs/srv/SetBool "{data: true}"  # then /arm/target_pose or /arm/target_joints at ~10 Hz
```

| Piece | Does | From v1 |
|---|---|---|
| `arm.launch.py` | controller manager `/ur5e` on the shared URDF (`ur_control:=true base_control:=false`), spawners, the driver's dashboard / stopper / urscript nodes, MoveIt, the servers | `robot_disco_ur5e_driver` unit + three ur_rtde owners |
| `config/ur_controllers.yaml` | jsb, io/status, speed scaling, FT, scaled JTC (robot) / JTC (rig), `forward_position_controller` (inactive until streaming) | driver defaults, `ur_` prefix |
| `moveit.launch.py` + `config/robot.srdf`, `kinematics.yaml`, `ompl_planning.yaml`, `moveit_controllers.yaml` | move_group on the whole-robot model: chassis, lidars, platform and hand are obstacles; OMPL always, and with `cumotion:=true` Isaac ROS cuMotion as the default pipeline (a `cumotion_container` with the planner and its static planning scene server) | none (staged ramps by hand) |
| `config/ur5e_hand.xrdf` | cuMotion's robot description: NVIDIA's `ur5e.xrdf` with rd_description's `ur_` names plus the hand's spheres; its base frame is `base_link`, MoveIt's planning frame, because cuMotion takes the scene's collision objects in its own base frame as they come | |
| `config/keepout.yaml` | the box over the chassis the Wave server adds to the planning scene (v1's rule: nothing passes above the body) | `ur_motion.py` stage orders |
| `rd_arm/arm_power.py` | `ArmPower` service: dashboard power-up / protective-stop + fault recovery / power-off, then the driver's `resend_robot_program` and tool voltage | `ur_power.py`, `ur_dashboard.py`, eval_hw's startup |
| `rd_arm/wave_server.py` | `Wave` action: MoveIt plan to the recording's first sample, the samples as one trajectory at `speed_scale` with the finger wiggle, MoveIt plan back to `home` / `start` / `none` | `ur_motion.py` (servoJ) |
| `rd_arm/hand_node.py` + `hand_link.py` + `hand_protocol.py` | the hand over UDP with the liveness probe; `/hand/curls`, `/hand/offsets`, `/hand/state`, open / close / wiggle services | `amazing_hand_client.py`, `hand_wiggle.py` |
| `rd_arm/stream.py` | 50 Hz streaming on `forward_position_controller`: IK once per new target (`/compute_ik`), per-tick step cap, branch-flip rejection; `/arm/stream` switches controllers | `rtde_arm.py`'s servoJ thread |
| `rd_arm/motions.py`, `motions/wave.json`, `config/poses.yaml` | recorded motions (`$RD_MOTIONS_DIR` for the robot's own) and named poses; `vla_home` is the robot's recorded home (v1's `home_pose.json`, 2026-10-11) | `~/.robot_face/motions`, `VLA/outputs/home_pose.json` |
| `rd_arm/fake_dashboard.py`, `fake_hand.py` | the rig's UR dashboard and hand on loopback | |

cuMotion (`cumotion:=true`, needs the Isaac ROS 4.6 debs from `provision/isaac-packages.txt`): the Wave plans its two
moves through `/cumotion/move_group` (16 points in 0.06 s on the 4090) and plays the whole gesture on the mock arm
(2026-11-09). Two things the integration had to get right: cuMotion does not transform the planning scene, so the
XRDF's base frame must be MoveIt's planning frame (`base_link`; with `set_base_frame: ur_base_link` the keepout
box landed on the arm column and every plan failed `INVALID_INITIAL_CSPACE_POSITION`); and its last trajectory
sample keeps a residual velocity the controller rejects, so `MoveItClient.execute` ends every trajectory at rest.
The robot keeps OMPL (`cumotion` defaults to false) until the arm is validated on the Orin.

Tests (`rig/rig.sh test rd_arm`): protocol / link / dashboard / motion units against the fakes (incl. protective
stop and fault recovery), and a launch test on mock hardware: controllers up, `ArmPower` to RUNNING, hand
closed and reported back, the wave with MoveIt-planned moves to the transport pose and home, streaming at
50 Hz with a joint target and a Cartesian target through IK, controllers switched back.

Hardware only (phase 8): the real dashboard and PolyScope's remote-control mode, headless mode's program
resend, the tool voltage, speed scaling, servo smoothness against v1's servoJ (the plan's gate), the keepout
box against the CAD, cuMotion's trajectories on the scaled JTC (validated on the rig's JTC, below).
