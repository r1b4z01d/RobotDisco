# rd_perception

**Not started** (phase 1). Holds: the Isaac ROS launch set (image pipeline: decode / rectify / resize once
for every consumer; `isaac_ros_apriltag`; nvblox local costmap layer; optional `isaac_ros_visual_slam`),
the RealSense D435 launch, the three USB cameras as plain ROS 2 image topics (v1's multicast UDP camera
service goes away), the RPLiDAR S1 range-offset filter and the laser filters.

Ported from v1: `rd_ws/src/robot_disco/launch/{realsense,rplidar_*,velodyne}*.py`,
`scripts/scan_range_offset.py`, `VLA/ur5e_lerobot/sensors/`. Optional Isaac layers are launch args that
default off: the robot drives on lidars + AMCL alone.
