# vendor/

Source-built ROS packages, only where apt has no Jazzy binary. Each is a git submodule pinned to a
commit; `provision/provision.sh workspace` inits them over https and applies `provision/patches/<name>-*.patch`.

Expected (phase 1 decides each after checking the Jazzy apt index):

| Package | v1 source | Why it may need to be here |
|---|---|---|
| roboclaw_hardware_interface + roboclaw_serial | dumbotics | no apt package |
| rplidar_ros | Slamtec, `ros2` branch | apt has it for Jazzy; vendor only if the S1 needs the branch |
| velodyne | ros-drivers | apt has it for Jazzy |

Not coming over from v1: apriltag / apriltag_ros (Isaac ROS AprilTag), foxglove_bridge (removed),
AprilTag_Detection (dead upstream), pyDHgripper (unused), roboclaw_ros + norlab_custom_interfaces (unused).
