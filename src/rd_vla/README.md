# rd_vla

**Not started** (phase 4). Holds: the recorder (LeRobot v3 writer fed by ROS topics), the inference server
with pluggable backends (SmolVLA in a venv; GR00T N1.7 via its ZeroMQ server, phase 7), the `RunPolicy`
action server, teleop inputs (SpaceMouse, gamepad, Stream Deck), the eval registry.

Ported from v1: `rd_ws/VLA/ur5e_lerobot` and `VLA/scripts/{eval_hw,vla_action,hw_preflight,train_*}.py`.
Datasets stay LeRobot v3; GR00T's converter + a modality.json make them GR00T-ready.
