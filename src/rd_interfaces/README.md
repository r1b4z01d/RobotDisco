# rd_interfaces

The contracts between features. Behaviours (voice, face, the route runner, the manager) call capabilities
(nav, arm, VLA, speech) only through these; nothing shells out to a script any more.

| Interface | Served by | Replaces in v1 |
|---|---|---|
| `action/RunPolicy` | rd_vla | `vla_action.py` → `eval_hw.py --auto`, spawned by robot_voice tasks and `path.py` |
| `action/GoToLandmark` | rd_nav | robot_voice driving Nav2 itself, `path.py` waypoint following |
| `action/Wave` | rd_arm | robot_face's wave replay over URScript / `ur_rtde` |
| `action/Say` | rd_audio | the TTS engines duplicated in robot_face and robot_voice |
| `srv/ArmPower` | rd_arm | `Scripts/ur_power.py`, the power-up in `eval_hw.py`, the face node's own |

Rules: a goal's empty string means "the server's configured default"; results always carry a short
`message` naming why the goal ended; every server rejects a goal while another of its own is active rather
than queueing it.

| `msg/TagDetection`, `msg/TagDetectionArray` | rd_perception | the detector's own type (apriltag_msgs in v1, Isaac's in 5.0): the corrector and the rig see only this one |

Everything else (hand, mouth, cameras, scans) stays on standard `sensor_msgs` / `std_msgs` topics.
