# rd_manager

**Not started** (phase 2). The one web console: FastAPI + rclpy in one process, WebSocket live data (TF,
scans, clouds, nav state, cameras), the route editor / 2D map / 3D pages, landmarks, maps (SLAM new /
continue, activate), services (supervisor XML-RPC), HALT, Folks, volume, datasets / models / evals,
training. The GPU box runs the same package with its robot tabs pointed at the robot's instance.

Ported from v1: `rd_ws/Scripts/{route_editor.py,route_editor.html,map3d.html,static}` and
`rd_ws/VLA/webui`. Foxglove is not replaced; the 3D page is the viewer.
