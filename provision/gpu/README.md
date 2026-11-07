# GPU box (4090TI)

Trains the policies, runs Isaac Sim and mirrors the manager. `provision_gpu.sh` sets it up (run it on the box
as bryan; it leaves v1's `~/VLA` and the `vla_webui` supervisor unit alone, 5.0 lives in `~/RobotDisco`,
`/opt/vla` and `~/isaacsim-venv`):

```
GIT_SSH_COMMAND="ssh -i ~/.ssh/github -o IdentitiesOnly=yes" git clone --recurse-submodules git@github.com:r1b4z01d/RobotDisco.git ~/RobotDisco   # private repo: the box's GitHub key
~/RobotDisco/provision/gpu/provision_gpu.sh all        # Jazzy packages, colcon build, the test suite, /opt/vla, Isaac Sim 6.2 (pip)
~/RobotDisco/provision/gpu/provision_gpu.sh isaacros   # Isaac ROS 4.6 host install: replaces the host OpenCV, read the phase first
```

Phases can run one at a time (`provision_gpu.sh ros workspace test`). Each is re-runnable.

## Found on the box 2026-11-07

Ubuntu 24.04.5, ROS 2 Jazzy at /opt/ros/jazzy, RTX 4090 24 GB, driver 595.99.02, 62 GB RAM, 32 cores, 1.5 TB
free, passwordless sudo, internet. No docker, no CUDA toolkit (nvcc), no Isaac Sim, no Isaac ROS. The v1
supervisor runs `vla_webui` on :8081; `ollama` and `vla_infer_server` are stopped. `~/VLA/outputs` holds the
v1 datasets (1.1 GB), train runs (4.5 GB), evals, and `home_pose.json`: the real VLA home pose, which replaces
the placeholder in `src/rd_arm/config/poses.yaml` (phase `workspace` copies it to `~/RobotDisco-data/`).

## Install facts the script relies on (docs read 2026-11-07)

- Isaac Sim 6.2.0: pip package `isaacsim[all,extscache,ros2]==6.2.0.0` from pypi.nvidia.com in a Python 3.12
  venv, torch 2.11 cu128 first; `OMNI_KIT_ACCEPT_EULA=YES`; driver 595 is the tested one; the ROS 2 bridge is
  `isaacsim.ros2.bridge` (Jazzy recommended on 24.04, Cyclone DDS supported). The standalone zip
  (`isaac-sim-standalone-6.2.0-linux-x86_64.zip`, `isaac-sim.sh` / `python.sh`) is the fallback if pip fails.
- Isaac ROS 4.6 on x86: apt repo `https://isaac.download.nvidia.com/isaac-ros/release-4.6 noble main` (key
  `/usr/share/keyrings/nvidia-isaac-ros.gpg`), needs CUDA 13.2+ (cuda-keyring repo) and NVIDIA's Jetson x86 repo
  (`repo.download.nvidia.com/jetson/x86_64/noble r39.2`), then `isaac-ros-cli` + `isaac-ros init venv`.
  NVIDIA recommends Docker; the host mode replaces the host OpenCV packages. Debs: isaac-ros-image-proc,
  -apriltag, -nvblox, -visual-slam; cuMotion has no 4.6 deb (source build).

## Done on the box 2026-11-07

- `ros workspace test` ran clean once two traps were out of the way: the box's `~/.local` holds setuptools 84 and
  OpenCV 5 (v1 tooling), so the script and the bashrc block set `PYTHONNOUSERSITE=1` (setuptools 84 drops
  `tests_require`, and colcon then silently runs unittest: every package "exited with code 5", 0 tests), and the
  packages also declare `extras_require['test']`. The whole suite passes here: 86 tests, the same as the rig.
- The system python got `python3-pandas` (apt) and `pyarrow<21` (pip, `--break-system-packages`: no deb on 24.04
  and pyarrow 21+ needs numpy 2 while the system numpy is 1.26): rd_manager's dataset reader needs both. The
  manager runs here with `ros2 run rd_manager manager -- --no-ros --data ~/RobotDisco-data --port 8092` and lists
  the mirrored v1 datasets (episode counts from the parquet metadata), models and evals; v1's `vla_webui` keeps :8081.
- `/opt/vla`: torch 2.10 cu128, lerobot 0.4.4, numpy 2.2 — its own consistent stack, leave numpy alone there (the
  venv's cv2 is its own 4.12; the system cv_bridge is never imported from it).
- Isaac Sim 6.2 (pip) runs headless. What the first runs taught, all applied in `src/rd_sim`: the URDF converter
  writes the joints into an unselected `Physics` variant set (the script selects `physx`); a massless URDF root
  link makes every child body its own articulation with the wheels jointed to the world (`export_urdf` gives the
  root a token mass); the description had the casters floating 0.10 m above the floor and a 0.5 kg chassis under a
  20 kg arm (fixed in `rd_description`); the fixed caster spheres need a frictionless material; at all-zero joints
  the UR lies stretched out sideways and tips the base (the sim starts it at `vla_home`); the 2D lidars use the
  shipped `RPLIDAR_S2E` profile, the Velodyne a 32-channel Ouster `OS1` variant (no VLP-16 profile ships); the
  sim publishes the RAW scans and `lidars.launch.py sim:=true` runs the robot's body filter chains on them.
- Running it detached, so an ssh drop does not kill it:
  `(setsid nohup timeout 3000 ~/isaacsim-venv/bin/python src/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf --headless > ~/isaac.log 2>&1 < /dev/null & echo $! > ~/isaac.pid)`,
  stop with `kill -INT -- -$(cat ~/isaac.pid)`. The first start takes ~100 s (extension cache); the log is
  line-buffered. Then `ros2 launch rd_sim sim.launch.py` for the stack. Never `pkill -f isaac_scene` from an ssh
  command that itself contains that text: it kills the shell running the command.

## After the script

- `ros2 run rd_sim export_urdf /tmp/robot_disco.urdf`, the detached Isaac scene run above, then
  `ros2 launch rd_sim sim.launch.py`; `src/rd_sim/README.md` has the sensor and bridge details.
- `ros2 launch rd_perception isaac.launch.py apriltag:=true` against `fake_cameras` once Isaac ROS is in.
- Training: `/opt/vla/bin/python -m lerobot.scripts.train ...` on the mirrored datasets, checkpoints under
  `~/RobotDisco-data/outputs/train`; the 5.0 manager serves them with `manager --no-ros --data ~/RobotDisco-data`.
