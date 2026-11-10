# rd_face

The face: eyes that follow whoever is in view, a mouth that moves with the speech (`/mouth/audio` from
rd_audio), idle gestures, face recognition with a gallery that learns strangers, and the greeting: hello by
name through the `Say` action plus a `Wave`, a wink for someone back within the hour, and for anyone it knows a
beating heart on the two WLED matrix panels.

```
ros2 launch rd_face face.launch.py                    # robot: the GL window on the portrait display, ArcFace / OpenCV face id
ros2 launch rd_face face.launch.py sim_mode:=true     # rig: headless renderer on /face/image, no face id
ros2 topic pub --once /face/set_person_name std_msgs/msg/String "{data: 'Bryan'}"       # name the person in view
ros2 topic echo /face/people                          # the Folks data (rd_manager shows it)
```

| Piece | From v1 |
|---|---|
| `rd_face/face_node.py` | `combined_face_hand_node.py` minus TTS (rd_audio), the wave (rd_arm), the Folks HTTP page (rd_manager; `/face/people` here), the hand relay |
| `face_id.py`, `faceid_worker.py`, `face_utils.py`, `mouth_gestures.py`, `gl_eye_renderer.py`, `eye_renderer.py`, `window_utils.py`, `models/` | unchanged |
| `headless_renderer.py` | new: the GL renderer's call shape on a numpy canvas, for the rig and the manager's preview |
| `wled_panels.py` | v1's module of 2026-11-09, its last change: one heart per matrix segment as WLED realtime UDP (DNRGB) for `wled_heart_seconds`, after which the WLED resumes its own effect by itself; the layout comes from the WLED's JSON API, the address from robot.yaml `panels`. `python3 wled_panels.py 192.168.11.218` shows it without ROS |

Tests: a launch test with the headless face: frames on `/face/image` at the render rate, the mouth reacts to
audio, a recognised person (the `/face/debug_person` stand-in) is greeted by name with a wave, the same person
again gets a wink, a stranger the generic hello; a fake WLED on loopback receives the heart for the first two
and nothing for the stranger. Units for the heart's geometry, the beat and the DNRGB packets. Hardware only: the
GL window on the robot's display and its focus handling, the ArcFace venv, real faces through the D435, the heart
on the real panels (64x48, two 32x48 segments).
