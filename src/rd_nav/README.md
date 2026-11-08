# rd_nav

Navigation: Nav2 (Jazzy) with v1's tuning, maps and no-go zones, SLAM, landmarks, the `GoToLandmark` action
and the AprilTag pose corrector.

```
ros2 launch rd_nav nav.launch.py                      # robot: localization (map server + AMCL) + navigation
ros2 launch rd_nav slam.launch.py                     # mapping instead of localization (Maps tab request)
ros2 launch rd_nav loopback_sim.launch.py             # rig: the simulator in place of base + lidars + AMCL
ros2 launch rd_nav navigation.launch.py use_sim_time:=true tag_corrector:=true
ros2 action send_goal /go_to_landmark rd_interfaces/action/GoToLandmark "{landmark: hallway}" --feedback
ros2 service call /tag_corrector/teach_landmarks std_srvs/srv/Trigger
rd_keepout                                            # rebuild maps/keepout_mask.* from keepout_zones.yaml
```

| Piece | From v1 | Notes |
|---|---|---|
| `config/nav2.yaml` | `nav2_params.yaml` | Jazzy names; every cmd_vel stamped; FollowPath = MPPI, v1's DWB kept as `FollowPathDWB`; global costmap at the map's 0.05 m |
| `config/slam.yaml` | `mapper_params_online_async.yaml` | |
| `maps/` | `robot_disco/maps` | `zone_one` (+ `zone_one.xcf`, the GIMP source of the pgm, not installed), `keepout_zones.yaml` → mask, `active_map` |
| `rd_nav/maps_store.py` | `Scripts/maps_store.py` | shared by the launches and rd_manager |
| `rd_nav/landmarks.py` | `Scripts/route_editor.py` + `robot_voice/tasks.py` | one store for places and their tasks |
| `rd_nav/go_to_landmark.py` | `path.py` + the voice assistant's own Nav2 client | the `GoToLandmark` server over `navigate_to_pose` |
| `rd_nav/tag_corrector.py` | `apriltag_relocalizer.py` | same maths; reads `rd_interfaces/TagDetectionArray` on `/tags/detections` |

Topics in: `/rp_s1_lidar/scan_filtered`, `/rp_s2e_lidar/scan_filtered`, `/velodyne_points` (local costmap only),
`/diff_cont/odom`, `/tags/detections`. Out: `/cmd_vel_nav` (TwistStamped) into rd_base's twist_mux, `/initialpose`.

Tests (`rig/rig.sh test rd_nav`): the stores and the keepout builder on temp copies, the corrector's maths,
and a launch test in Nav2's loopback simulator: Nav2 comes up on the v1 map, `GoToLandmark` drives to the
hallway landmark, rejects an unknown one, and a detection of a taught tag relocates the robot.
Only hardware can verify: AMCL on the real lidars, the keepout zones against the real room, MPPI's feel
against DWB, the grid-localizer bootstrap (Isaac, rd_perception).
