# rd_base

**Not started** (phase 1). Holds: ros2_control for the RoboClaw diff drive (`config/controllers.yaml`, v1's
`my_controllers.yaml`), twist_mux (nav 10 < web 90 < joystick 100), the Nav2 collision monitor, joy +
teleop_twist_joy with the 8BitDo deadman, and the base launch that brings them up under
`rd_description`'s robot_state_publisher.

Ported from v1: `rd_ws/src/robot_disco/config/{my_controllers,twist_mux,joystick,collision_monitor_params}.yaml`,
`launch/{launch_robot,joystick}.launch.py`. Hardware interface from `vendor/roboclaw_hardware_interface`.
The `COLCON_IGNORE` file goes away when the package exists.
