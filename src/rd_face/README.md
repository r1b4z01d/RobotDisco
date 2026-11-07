# rd_face

**Not started** (phase 5). Holds: the eyes + mouth renderer with its gesture library, the ArcFace face-ID
worker and galleries, the greeting (name + `Wave` + `Say`), Folks data as services and topics. No HTTP
server of its own: rd_manager serves the Folks tab.

Ported from v1: `rd_ws/src/robot_face` minus the TTS engines (rd_audio) and the wave (rd_arm).
