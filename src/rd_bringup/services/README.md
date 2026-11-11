# Supervisor units

One unit per launch, all alike (`common.env` is sourced first: workspace, Cyclone DDS, the config file, offline Hub):

| Unit | Runs | Autostart |
|---|---|---|
| `rd_base` | `rd_base base.launch.py` (controller manager, twist_mux, collision monitor, gamepad) | yes |
| `rd_perception` | lidars, RealSense, webcams, Isaac AprilTag | yes |
| `rd_nav` | map server + AMCL + Nav2 + GoToLandmark + tag corrector | yes |
| `rd_slam` | slam_toolbox instead of AMCL (the Maps tab starts/stops it) | no |
| `rd_arm` | UR driver, MoveIt, ArmPower, Wave, hand, streamer | yes |
| `rd_vla` | the policy runner (the server is its own unit; the arm's gamepad teleop is off in the unit, its buttons collide with the base's deadman over Bluetooth) | yes |
| `vla_infer_server` | SmolVLA on the Orin, loopback :8777, serving `/mnt/data/outputs/serve_ckpt` | yes |
| `rd_audio`, `rd_face`, `rd_voice` | Say, the face (needs the GNOME display), the voice assistant | yes |
| `rd_manager` | the web console on :8092 | yes |
| `ollama` | the fallback LLM on loopback :11434 | yes |

`provision.sh services` installs them (`/etc/supervisor/conf.d/`), the web UI / XML-RPC sits on :9999.
Secrets (`ANTHROPIC_API_KEY`, `HASS_*`) live in `/home/rd/.robot_face/voice.env`, sourced by every unit.
