# rd_vla

The policy side: the inference server with pluggable backends, the `RunPolicy` action, the recorder, gamepad
teleop, the checkpoint and eval registries. Observations and actions keep v1's contract (`schema.py`): a
10-D action (EE pose + 4 curls), a 16-D state, three cameras.

```
ros2 launch rd_vla vla.launch.py                       # robot: smolvla server + runner + SpaceMouse teleop under the LeRobot venv's python
ros2 param set /vla_teleop dataset tigger_2 ; ros2 param set /vla_teleop task "Grab and hold tigger"     # recording a demonstration:
ros2 service call /teleop/enable std_srvs/srv/SetBool "{data: true}"      # the puck now moves the arm
ros2 service call /teleop/episode/start std_srvs/srv/Trigger ; ros2 service call /teleop/episode/save std_srvs/srv/Trigger
ros2 launch rd_sim sim.launch.py vla_backend:=smolvla vla_ckpt:=tigger_1_merged_smolvla_1   # GPU box: the real policy against Isaac Sim
ros2 launch rd_vla vla.launch.py backend:=fake         # rig
ros2 action send_goal /run_policy rd_interfaces/action/RunPolicy "{task: 'pick up tigger', checkpoint: tigger_1_merged_smolvla_1, max_seconds: 60, record: true, done_when: closed}" --feedback
python3 -m rd_vla.infer_server --backend fake          # the server alone (ROS-free)
```

| Piece | Does | From v1 |
|---|---|---|
| `rd_vla/wire.py` + `infer_server.py` + `backends/` | obs → action RPC (JPEG images, pickle over TCP, loopback); backends `fake` (scripted, the rig), `smolvla` (LeRobot 0.4 policy + chunk queue; `--n-action-steps` N executes N of each 50-step chunk per observation, v1 drove with 4..8), `groot` (phase 9); `select` switches checkpoints | `remote.py`, `infer_server.py` |
| `rd_vla/observer.py` | cameras (rgb8 topics, no cv_bridge), `/joint_states`, TF `ur_base → ur_tool0` as a rotation vector (the UR controller's base frame, see Frames), `/hand/state` → the 16-D state | `ur_amazinghand_robot.py`, `eval_hw.py observe()` |
| `rd_vla/policy_runner.py` | `RunPolicy`: select the checkpoint, `/arm/go_home`, streaming on, the loop at `fps` with the step and workspace clamps, `done_when` by grasp level, cancel, recording, the eval.json | `eval_hw.py`, `vla_action.py` |
| `rd_vla/recorder.py` | LeRobot v3 episode writer (each camera keeps its own frame size, as v1's sets: scene and side 540x540 squares, the wrist 540x960), with an npz fallback where lerobot is absent | `manual_panel.py`'s SimSession |
| `rd_vla/teleop.py` + `spacemouse.py` | the SpaceMouse (read from hidraw, no pip module) → `/arm/target_pose` + `/hand/curls` through the streamer, and the recording of demonstrations: `/teleop/enable`, `/teleop/home`, `/teleop/episode/start` / `save` / `discard`, parameters `dataset` and `task`, `/teleop/state`. Freedrive (hand-guiding, the recorded action is the pose the arm is at), the hand's power-cycle and the Stream Deck (`deck.py`, v1's keys; needs the `streamdeck` pip package in `/opt/vla`) are ported; the gamepad input is not | `spacemouse.py` + `manual_panel.py`'s SpaceMouse path |
| `rd_vla/checkpoints.py`, `evals.py` | `run[@step]` resolution under `$RD_DATA_DIR`/outputs/train (a run copied without its `last` symlink resolves to its newest step) and the serve link; the eval registry rd_manager shows | `services.py`, `webui/evals.py` |

Tests (`rig/rig.sh test rd_vla`): schema / geometry / wire RPC against the fake server / checkpoints / registry /
recorder (npz, and LeRobot under `/opt/vla/bin/python`), and a launch test: fake server + runner + mock arm +
fake cameras, `RunPolicy` with `done_when: closed` ends `done` after the fake policy closes the hand, the
episode and the eval record exist; a second goal times out cleanly.

Frames. Actions and the state's pose are in `ur_base`, the UR controller's base frame that v1's RTDE poses (and so every
v1 dataset and checkpoint) are in. The URDF's `ur_base_link` is that frame turned 180 deg about Z: at v1's recorded home
the TCP reads (-0.189, -0.107, 0.991) in `ur_base` and (0.205, 0.124, 0.98) in `ur_base_link`, and a policy fed the
wrong one steered 0.3 m off and sat in IK holds (2026-11-09). `ur_tool0` matches v1's TCP to 0.9 mm and 0.4 deg, so
there is no TCP offset to model. `runner.yaml` carries both frames.

Interpreter. The server and the runner run under `python` (`/opt/vla/bin/python`, the LeRobot venv; the rig passes
`python3`): the policy needs lerobot, and so does the recorder, which otherwise falls back to npz episodes that cannot
be trained on. The venv is `--system-site-packages` and must own its scipy (provisioning does `pip install
--ignore-installed --no-deps scipy`): Ubuntu's scipy satisfies the import but is built for numpy 1 and fails under the
venv's numpy 2 the moment transformers probes it.

Validated on the GPU box against Isaac Sim (2026-11-09, RTX 4090, `tigger_1_merged_smolvla_1@020000`, `n_action_steps`
8): the checkpoint loads in 7 s and takes 0.9 GB; a chunk takes 137 ms warm (361 ms cold); `RunPolicy` for 30 s ran 273
steps at 10.8 Hz with the observation 37 ms old at inference, the end effector following the policy's targets within
the 3 cm step cap over 5 m of path, and the episode saved as a LeRobot v3 dataset beside its eval record. The policy
does not find Tigger in the sim (the cameras show the sim room), so the run is a pipeline check, not a success rate.

On the robot's AGX Orin (2026-11-10, JetPack 7.2.1, `/opt/vla` with PyTorch's own cu130 arm64 build, MAXN): the same
checkpoint loads in 14 s, takes 0.9 GB of GPU memory and computes a chunk in 0.73 s (v1 on JetPack 6: 0.875 s). That is
longer than eight steps at 12 Hz, so the robot's unit serves with `--n-action-steps 24 --prefetch --prefetch-at 10`
(v1's setting): the next chunk is computed in a thread once eleven actions are left (`rd_vla/chunks.py`), and no step
waits for a forward pass. On the 4090 against a recorded episode (2026-11-11): a step costs under 1 ms instead of
107 ms at each boundary, the seams are those of the synchronous path, and `--rtc` blends them further. On the Orin the
prefetch and RTC's extra cost are not measured yet.

Not in the rig: SmolVLA inference itself (GPU box and Orin, above), the SpaceMouse and Stream Deck inputs (phase 8), the
camera exposure lock (phase 8).
