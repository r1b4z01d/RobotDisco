# rd_description

The robot model: one xacro tree, the meshes it needs, and a `robot_state_publisher` launch.

```
ros2 launch rd_description rsp.launch.py                 # real robot (RoboClaw ros2_control hardware)
ros2 launch rd_description rsp.launch.py sim_mode:=true  # mock hardware: dev stack, Isaac Sim
xacro urdf/robot.urdf.xacro > /tmp/robot_disco.urdf       # the flat URDF (Isaac Sim's importer takes this)
```

| File | Holds |
|---|---|
| `urdf/robot.urdf.xacro` | the entry point; includes everything below |
| `urdf/robot_core.xacro` | chassis, drive wheels, four casters, materials |
| `urdf/ros2_control.xacro` | the base's ros2_control system: RoboClaw (`roboclaw_port`, `roboclaw_baud` args: the header UART `/dev/ttyTHS1` at 115200 since 2026-11-08, from `robot.yaml` via rd_base's launch), or mock hardware in `sim_mode` (`base_control:=false` leaves it out) |
| `ur_control:=true` | adds the UR driver's ros2_control tag (`ur_robot_driver` macro, mock hardware in `sim_mode`) for rd_arm's controller manager under `/ur5e` |
| `urdf/rp_s1_lidar.xacro`, `rp_s2e_lidar.xacro`, `velodyne.xacro` | the three lidars and their frames |
| `urdf/realsense_d435.xacro` | the D435 and its optical frame |
| `urdf/ur_mount.xacro` | the UR5e (from `ur_description`, prefix `ur_`) on its platform over the back lidar |
| `urdf/amazing_hand.xacro` | the AmazingHand as a static visual at the open pose, meshes in `meshes/amazing_hand/` |

Ported from v1 (`rd_ws/src/robot_disco/description`) on 2026-11-06 with the same frames and offsets, so v1
maps, AprilTag landmark poses and the UR mount calibration carry over. The Gazebo sensor plugins are gone:
5.0 simulates in Isaac Sim (`rd_sim`), which imports the flat URDF.

The hand's finger meshes are the baked open pose. Re-bake with v1's
`src/robot_disco/scripts/bake_amazing_hand.py` if the CAD in `firmware/AmazingHand` changes.
