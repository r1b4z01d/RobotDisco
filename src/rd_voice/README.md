# rd_voice

"Robot Disco, ..." : Whisper ears (faster-whisper in its venv, `ears_worker.py` unchanged), the wake phrase,
the brain (rules first, then Claude or the local Ollama model with the skills as tools: weather, news, Home
Assistant, music, volume, clock, navigation, arm jobs), and landmark tasks that drive with `GoToLandmark` and
run the policy with `RunPolicy` instead of spawning scripts.

```
ros2 launch rd_voice voice.launch.py                  # robot
ros2 launch rd_voice voice.launch.py sim_mode:=true   # rig: no ears, rules only
ros2 topic pub --once /voice/text_in std_msgs/msg/String "{data: 'robot disco pick up tigger'}"
```

| Piece | From v1 |
|---|---|
| `wake.py`, `brain.py`, `skills.py`, `ears_worker.py` | unchanged (skills import the sink helpers from rd_audio) |
| `tasks.py` | the task state machine over callables (`navigate`, `run_policy`, `cancel`) instead of `vla_action.py` subprocesses |
| `voice_node.py` | `voice_assistant_node.py`: Say action instead of the face's `say` topic, GoToLandmark instead of its own Nav2 client, landmarks from rd_nav's store |

Tests: wake / matching / task runner / brain rules with fakes (ROS-free); a launch test with rd_audio's fake
voice and stub nav + policy servers: "robot disco, could you pick up tigger please" answers, drives to the
dining room, runs the tigger checkpoint with its task string. Hardware only: the microphone and Whisper venv,
Claude / Ollama (keys and the GPU box), music and Home Assistant.
