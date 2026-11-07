# rd_audio

The robot's one voice and speaker: the `Say` action (text → Kokoro / espeak / the rig's fake voice → the
speaker, with the samples streamed to `/mouth/audio` for the face), `/speaking`, the sink watchdog that keeps
the C-Media adapter the default sink, volume and mute.

```
ros2 launch rd_audio audio.launch.py [sim_mode:=true]
ros2 action send_goal /say rd_interfaces/action/Say "{text: 'hello there'}"
ros2 topic pub --once /say_text std_msgs/msg/String "{data: 'hi'}"
ros2 param set /say_server volume 0.6
```

From v1: `audio_sink.py`, `audio_utils.py`, `tts_engines.py` (+ `FakeTTS`), `kokoro_worker.py` (unchanged, runs
in the Kokoro venv), and the face node's `_speak` / `_play_wav_with_mouth` as `say_server.py`. The voice
node's duplicate copies are gone.

Tests: the fake voice and the chunking (units); a launch test that speaks through the fake voice with no
audio device and checks `/speaking`, the 25 Hz mouth feed, the duration and interruption. Hardware only:
Kokoro in its venv, aplay on the C-Media sink, the lead-in silence the adapter needs.
