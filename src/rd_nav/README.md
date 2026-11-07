# rd_nav

**Not started** (phase 1). Holds: Nav2 params for Jazzy (MPPI controller, docking server for the charger),
maps + keepout masks + `make_keepout_mask.py`, slam_toolbox params, AMCL with the Isaac occupancy-grid
localizer as the boot-time bootstrap, the AprilTag pose corrector (v1's `apriltag_relocalizer.py`, now fed by
`isaac_ros_apriltag`), landmarks per map, and the `GoToLandmark` action server (v1's `path.py` route runner
becomes a client of it).

Ported from v1: `rd_ws/src/robot_disco/{config,maps,launch}` and `rd_ws/Scripts/{path,maps_store}.py`.
