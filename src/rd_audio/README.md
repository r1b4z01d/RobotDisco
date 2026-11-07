# rd_audio

**Not started** (phase 5). The one copy of: the PulseAudio sink watchdog, playback, the Kokoro TTS engine
(espeak fallback), the `Say` action server, the mouth waveform topic. rd_face and rd_voice import it.

Ported from v1: the audio code duplicated between robot_face and robot_voice (`audio_sink.py`, the TTS
engines, the sink unmute).
