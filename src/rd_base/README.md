# rd_base

The mobile base: ros2_control diff drive on the RoboClaw (mock hardware in `sim_mode`), twist_mux, the Nav2
collision monitor, the 8BitDo gamepad.

```
ros2 launch rd_base base.launch.py                                  # robot
ros2 launch rd_base base.launch.py sim_mode:=true joystick:=false   # rig
```

Velocity path: `cmd_vel_joy` (100) / `cmd_vel_web` (90) / `cmd_vel_nav` (10) → twist_mux → `cmd_vel_raw` →
collision_monitor (stop / slowdown polygons fed by `/rp_s1_lidar/scan_filtered` and `/rp_s2e_lidar/scan_filtered`)
→ `/diff_cont/cmd_vel` → diff_cont → wheels, `/diff_cont/odom`, `odom → base_link`. Every cmd_vel topic is
`geometry_msgs/TwistStamped`: Jazzy's twist_mux 4.5 speaks nothing else, so the controller, the collision monitor,
Nav2, teleop and the web pad are all configured stamped.

| File | From v1 |
|---|---|
| `config/controllers.yaml` | `my_controllers.yaml` (same wheel geometry and limits) |
| `config/twist_mux.yaml` | `twist_mux.yaml` |
| `config/collision_monitor.yaml` | `collision_monitor_params.yaml`, `max_points` → Jazzy's `min_points` |
| `config/joystick.yaml` | `joystick.yaml`; the pad name comes from `robot.yaml` at launch |
| `launch/base.launch.py` | `launch_robot.launch.py` minus the sensors (rd_perception) and the RTDE joint poller (the UR driver is always on in 5.0) |

Tests (`rig/rig.sh test rd_base`): the YAML contracts, and a launch test on mock hardware that drives the
robot through the whole velocity path and checks the collision monitor stops it for an object 0.35 m ahead.
Only hardware can verify: the RoboClaw serial port, wheel direction, the gamepad name and button indices.
