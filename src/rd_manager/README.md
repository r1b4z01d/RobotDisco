# rd_manager

The one console: FastAPI + rclpy in one process. Every v1 page behind one port (8092 on the robot, 8081 on
the GPU box): the Live map with the drive pad and HALT, the 3D view, the route editor and runner, Maps
(SLAM new / continue, drawing editor, activate), Landmarks and their tasks, Folks, Panels (what the LED
matrices show), Services (supervisor), the arm / hand / policy page, datasets, models, evals, training.

```
manager --port 8092                  # the robot (ROS bridge on, supervisor on :9999)
manager --port 8081 --no-ros --robot-url http://192.168.11.80:8092     # the GPU box: data tabs here, robot tabs point at the robot
ros2 launch rd_manager manager.launch.py sim_mode:=true                # the rig
```

| Piece | Does | From v1 |
|---|---|---|
| `server.py` | every route; `/ws` streams pose / nav / scans at 5 Hz | `webui/server.py` + `route_editor.py`'s HTTP |
| `ros_bridge.py` | the live state (TF, scans, costmaps, cameras as JPEG, people, hand, speech) and the actions (drive, HALT, goals, GoToLandmark, Wave, RunPolicy, Say, ArmPower, volume, SLAM save) | `route_editor.py`'s ROS thread + proxies to five servers |
| `routes.py` | the route format and the runner over FollowWaypoints + RunPolicy / Say | `path.py` |
| `supervisor.py` | supervisord over XML-RPC, the ordered base restart, the activity log; a fake supervisord for the rig | `webui/server.py` |
| `grids.py`, `capture.py` | map / costmap PNGs, the stitched landmark photo | `route_editor.py` |
| `panels.py` | the Panels tab's server side: `/api/panels` (the WLED's matrix, segments and effects from its JSON API) and `/ws/panels`, one shared connection to the WLED's live pixel stream relayed to every open tab (the WLED serves a single live client); the address is robot.yaml `panels`, `--panels-url` overrides | `webui/server.py` (2026-11-09, v1's last change) |
| `datasets.py`, `models.py`, `ops.py`, `annotations.py` | LeRobot datasets, training runs, episode ops | `webui/` unchanged |
| `exposure.py` | the policy cameras' exposure and white balance through v4l2-ctl: v1's auto / − / + / lock keys, all cameras or one | `sensors/cameras.py` |
| `static/pages/robot/` | the VLA Teleop tab: v1's teleop page ported (REC / SAVE / DISCARD / HOME / +GRASP, hold-to-open and close, the three cameras, dataset and task, exposure, the last episode's stars and notes, the log). It talks to `/api/teleop/config`, `/api/teleop/cmd` and the `/api/teleop/events` stream, which the server composes from rd_vla's teleop node (`/teleop/state`, `/teleop/log`), the arm's status and the cameras. New: a TELEOP ON / OFF button (nothing moves while off). Not ported: freedrive, the hand's power-cycle, live crop sliders (the crops are robot.yaml's), brightness medians, the Stream Deck | `webui/static/robot`, `ur5e_lerobot/ui/webpanel.py` |
| `static/` | `index.html` + `app.js` (the manager), `pages/live.html` (2D), `pages/map3d.html` (3D, vendored three.js), `pages/arm.html`, `pages/folks.html` | the v1 pages; the Robot / Routes / Folks tabs now embed this server's own pages |

Not ported: the route tools (centre / densify from `record_path.py`), the Hand tab's firmware gestures and
raw lines (the hand speaks ROS now), the rerun episode viewer.

Tests: routes / runner / supervisor / grids (units); the HTTP API without ROS through the test client
(pages, maps incl. SLAM request and the drawing save, landmarks and tasks, routes, evals, services against
the fake supervisord); a launch test with the loopback simulator, Nav2, fake cameras and the Say server:
live pose, scans, map, camera JPEGs, a landmark captured with its photo, GoToLandmark from the API,
HALT stops the drive, the drive pad, speech, the WebSocket; the Panels tab against a fake WLED (its JSON and its
pixel stream on one loopback port: the info, two tabs sharing one upstream connection, the stream handed back when
the last tab leaves). Checked read-only against the real WLED on 2026-11-09: 64x48, segments Right and Left, a 32x24
thinned preview at 19 frames a second. The Live tab's text box sends typed text to the voice
assistant exactly as if Whisper had heard it (`POST /api/voice/text` -> `/voice/text_in`, so the wake phrase applies:
"Robot Disco, what time is it") and shows what it heard and answered (`/api/voice`, timestamped).
