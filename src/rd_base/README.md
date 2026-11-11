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

The RoboClaw (2x15A, firmware 4.4.9) is on the Jetson's 40-pin header UART since 2026-11-08, not USB: pin 8
(TX) -> S1, pin 10 (RX) <- S2, pin 6 GND, `/dev/ttyTHS1` at 115200, address 128, with the board's 1 s serial
timeout as the watchdog (`robot.yaml sensors.roboclaw`; the launch passes port and baud to the xacro). The
plugin is `vendor/roboclaw_hardware_interface` with the UART patches from `provision/patches`; the UART itself
needs the PIO device tree from `provision.sh jetson`. See `provision/README.md`, *RoboClaw on the header UART*.
The gamepad goes through `joy_guard`: joy's reader republishes a pad's last sticks and buttons after the pad is gone
(Bluetooth drop, sleep) and keeps them until each control moves again, which after a reboot drove the base forward on
the first press of the deadman with the stick at rest. The guard sends a neutral pad while the device is absent and
holds a control at rest until it has really moved.

A lost reply no longer ends driving: the interface asks twice and rides out half a second of failed exchanges
(`provision/patches/roboclaw_hardware_interface-retry.patch`; before it, one late reply on a loaded Orin deactivated
the hardware for good, 2026-11-10). If the link stays down, `hardware_watchdog` ends the launch after 5 s and the
unit's autorestart starts a fresh one; nothing has to be restarted by hand.

`battery_monitor` listens for the battery board (`firmware/RD_Battery_Monitor`, an ESP32 with an INA228 on the pack)
on UDP 8889 and publishes `sensor_msgs/BatteryState` on `/battery` once a second. With the pack's voltages in
`robot.yaml battery` it also gives a percentage and says "my battery is low" once, then "please charge me now" every
five minutes while critical (held 10 s first, so a sag while the motors start does not count). The console's header
and the voice assistant ("how is your battery") read `/battery`.

Tests (`rig/rig.sh test rd_base`): the YAML contracts, and a launch test on mock hardware that drives the
robot through the whole velocity path and checks the collision monitor stops it for an object 0.35 m ahead.
Only hardware can verify: the RoboClaw link (ttyTHS1 in PIO mode, the probe tools in `provision/tools`),
wheel direction, the gamepad name and button indices.
