# rd_arm

**Not started** (phase 3). Holds: the UR5e driver launch (always on, owns the arm), MoveIt 2 + cuMotion
configs, cuRobo IK behind a small service, a 50 Hz streaming controller for the policy and teleop, the
hand driver node (UDP `J:` lines to `firmware/AmazingHand`), the recorded motion library, the `Wave`
action server and the `ArmPower` service.

Ported from v1: `rd_ws/Scripts/ur_power.py`, robot_face's wave recorder / replayer and staged-ramp logic,
`VLA/ur5e_lerobot/hand/amazing_hand_client.py`, `VLA/ur5e_lerobot/teleop/` (the servoJ streamer stays as
the fallback behind the same action if ros2_control streaming is not as smooth: plan, risks).
