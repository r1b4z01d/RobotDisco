# rd_vla

The policy side: the inference server with pluggable backends, the `RunPolicy` action, the recorder, gamepad
teleop, the checkpoint and eval registries. Observations and actions keep v1's contract (`schema.py`): a
10-D action (EE pose + 4 curls), a 16-D state, three cameras.

```
ros2 launch rd_vla vla.launch.py                       # robot: smolvla server (the LeRobot venv's python), runner, teleop
ros2 launch rd_vla vla.launch.py backend:=fake         # rig
ros2 action send_goal /run_policy rd_interfaces/action/RunPolicy "{task: 'pick up tigger', checkpoint: tigger_1_merged_smolvla_1, max_seconds: 60, record: true, done_when: closed}" --feedback
python3 -m rd_vla.infer_server --backend fake          # the server alone (ROS-free)
```

| Piece | Does | From v1 |
|---|---|---|
| `rd_vla/wire.py` + `infer_server.py` + `backends/` | obs → action RPC (JPEG images, pickle over TCP, loopback); backends `fake` (scripted, the rig), `smolvla` (LeRobot policy + chunk queue), `groot` (phase 9); `select` switches checkpoints | `remote.py`, `infer_server.py` |
| `rd_vla/observer.py` | cameras (rgb8 topics, no cv_bridge), `/joint_states`, TF `ur_base_link → ur_tool0` as a rotation vector, `/hand/state` → the 16-D state | `ur_amazinghand_robot.py`, `eval_hw.py observe()` |
| `rd_vla/policy_runner.py` | `RunPolicy`: select the checkpoint, `/arm/go_home`, streaming on, the loop at `fps` with the step and workspace clamps, `done_when` by grasp level, cancel, recording, the eval.json | `eval_hw.py`, `vla_action.py` |
| `rd_vla/recorder.py` | LeRobot v3 episode writer, with an npz fallback where lerobot is absent | `manual_panel.py`'s SimSession |
| `rd_vla/teleop.py` | `/joy` → `/arm/target_pose` + `/hand/curls` through the streamer | `gamepad.py` + the panel |
| `rd_vla/checkpoints.py`, `evals.py` | `run[@step]` resolution and the serve link; the eval registry rd_manager shows | `services.py`, `webui/evals.py` |

Tests (`rig/rig.sh test rd_vla`): schema / geometry / wire RPC against the fake server / checkpoints / registry /
recorder (npz, and LeRobot under `/opt/vla/bin/python`), and a launch test: fake server + runner + mock arm +
fake cameras, `RunPolicy` with `done_when: closed` ends `done` after the fake policy closes the hand, the
episode and the eval record exist; a second goal times out cleanly.

Not in the rig: SmolVLA inference itself (needs a checkpoint; GPU box, phase 7), the SpaceMouse and Stream
Deck inputs (phase 8), the camera exposure lock (phase 8).
