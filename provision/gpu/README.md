# GPU box (4090TI)

Trains the policies, runs Isaac Sim and mirrors the manager. `provision_gpu.sh` sets it up (run it on the box
as bryan; it leaves v1's `~/VLA` and the `vla_webui` supervisor unit alone, v2 lives in `~/RobotDisco`,
`/opt/vla` and `~/isaacsim-venv`):

```
git clone --recurse-submodules https://github.com/r1b4z01d/RobotDisco.git ~/RobotDisco
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

## After the script

- `ros2 run rd_sim export_urdf /tmp/robot_disco.urdf` then `~/isaacsim-venv/bin/python
  src/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf`: the first run of the Isaac scene, see
  `src/rd_sim/README.md` for the API changes already known.
- `ros2 launch rd_perception isaac.launch.py apriltag:=true` against `fake_cameras` once Isaac ROS is in.
- Training: `/opt/vla/bin/python -m lerobot.scripts.train ...` on the mirrored datasets, checkpoints under
  `~/RobotDisco-data/outputs/train`; the v2 manager serves them with `manager --no-ros --data ~/RobotDisco-data`.
