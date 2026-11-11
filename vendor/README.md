# vendor/

Source-built ROS packages, only where apt has no Jazzy binary or not the right one. Each is a git submodule pinned to a commit;
`provision/apply_patches.sh` (run by `provision.sh workspace`, `gpu/provision_gpu.sh workspace` and
`rig/rig.sh build`) applies `provision/patches/<name>-*.patch` on top, so every checkout rebuilds the same tree.

| Package | Pinned | Why here | Patches |
|---|---|---|---|
| `roboclaw_serial` (dumbotics) | `cb14207`, what v1 ran | no apt package | `roboclaw_serial-baud-rate.patch`: a baud rate on `SerialDevice`, a reply read until the line is quiet (a real UART delivers it in pieces) |
| `roboclaw_hardware_interface` (dumbotics) | `ae00980`, what v1 ran | no apt package; the base's ros2_control plugin | `roboclaw_hardware_interface-baud-rate.patch`: the `baud_rate` hardware parameter, a failed open fails `on_init` instead of aborting controller_manager |
| `rplidar_ros` (Slamtec) | `24cc9b6`, what v1 ran, unmodified | apt's `ros-jazzy-rplidar-ros` 2.1.0 is the old packaged driver: no `rplidar_node`, no Ethernet (the S2E is UDP). `lidars.launch.py` needs Slamtec's (`channel_type`, `topic_name`, `scan_frequency`) | none |

Both come from v1 (`rd_ws/src/`) with the patches v1 needed when the RoboClaw moved from USB to the Jetson's
header UART on 2026-11-08 (see `provision/README.md`, *RoboClaw on the header UART*). They build in the rig
(Jazzy, arm64) and on the GPU box; `roboclaw_serial`'s gtests run with the suite.

Checked against the Jazzy apt index and NOT vendored: `rplidar_ros`, `velodyne` (apt has both). Not coming
over from v1: apriltag / apriltag_ros (Isaac ROS AprilTag), foxglove_bridge (removed), AprilTag_Detection
(dead upstream), pyDHgripper (unused), roboclaw_ros + norlab_custom_interfaces (unused).
