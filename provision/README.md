# Provisioning the v2 robot PC

Turns a Jetson AGX Orin flashed with **JetPack 7.2.1** (Jetson Linux 39.2.1, Ubuntu 24.04, CUDA 13.2, root
filesystem on the 512 GB NVMe) into the Robot Disco robot PC: user `rd`, hostname `robotdisco`,
`192.168.11.80` on the isolated robot subnet, ROS 2 Jazzy + Isaac ROS 4.6 as host packages, this repo
built at `~/RobotDisco`, everything under supervisor.

**Status 2026-11-06:** skeleton. `provision.sh` runs `preflight`, `base`, `ros`, `groups`, `udev`, `sysctl`
and `workspace`; the other phases print what they must do and skip. They get written on the freshly flashed
robot in phase 1 of the plan, ported from v1's `rd_ws/provision/` (which also keeps the USB port photos,
the xpad rebuild and the two-OpenCV patch that v2 no longer needs).

## Order of operations

1. **Phase 0, before the reflash** (the NVMe reflash destroys `/mnt/data`): on the workstation, with the v1
   robot still up, `rd_ws/provision/capture_rd_state.sh` for the snapshot, then copy `/mnt/data`
   (datasets, checkpoints, evals, galleries, maps) to the GPU box and verify it. Tag `rd_ws` as v1.
2. **Flash** with SDK Manager: JetPack 7.2.1, storage = NVMe, user `rd`, hostname `robotdisco`. Give the
   box an uplink with internet (second Ethernet or USB tether); the robot subnet has none.
3. `git clone https://github.com/r1b4z01d/RobotDisco.git ~/RobotDisco && ~/RobotDisco/provision/provision.sh all`
4. Reboot, then `provision.sh network` and move the cable to the robot subnet.
5. `SNAPSHOT=~/snapshot-YYYYMMDD provision.sh workspace audio` restores maps, galleries, motions and music.

| Phase | Does |
|---|---|
| `preflight` | user, sudo, Ubuntu 24.04 / aarch64 / JetPack present, root on NVMe |
| `jetson` | nvpmodel MAXN at boot, jetson_clocks service, xpad at boot (+ the rebuilt module if kernel 6.8 still lacks the 8BitDo ID), jtop |
| `base` | `apt-packages.txt`, ssh + supervisor + chrony, hostname, `/etc/hosts` for ur5e / amazinghand / gpu, `AUTHORIZED_KEY` |
| `ros` | ROS 2 Jazzy repo + `ros-packages.txt`, rosdep, `.bashrc`: Jazzy + workspace, Cyclone DDS with `cyclonedds.xml`, `ROS_DOMAIN_ID=0`, `RD_ROBOT_CONFIG` |
| `isaac` | NVIDIA's Isaac ROS 4.6 apt repo, `isaac-packages.txt` via the Isaac ROS CLI bare-metal mode |
| `groups` | dialout video input plugdev audio render docker |
| `udev` | `udev/99-robot-disco.rules`: SpaceMouse, RealSense, webcams, Stream Deck, gamepads, 8BitDo new_id fallback |
| `sysctl` | DDS receive buffers |
| `workspace` | clone / update `~/RobotDisco`, submodules, patches, rosdep, `colcon build --symlink-install` |
| `audio` | `pulse/default.pa` (C-Media adapter = default sink, no auto-switch), unmute, `alsactl store` |
| `services` | supervisor web UI / XML-RPC on `:9999`, the units from `src/rd_bringup/services/` |
| `llm` | Ollama under supervisor on loopback `:11434`, `qwen2.5:7b` |
| `display` | GNOME autologin, no blanking, the face window |
| `network` | static `192.168.11.80/24` (nmcli on the Jetson desktop, `netplan/01-robot-disco.yaml` otherwise) |

Files: `apt-packages.txt`, `ros-packages.txt`, `isaac-packages.txt`, `udev/`, `netplan/`,
`cyclonedds.xml` (the loopback-only DDS profile v1 settled on), `patches/` (empty until a vendored package
needs one). The GPU box's provisioning is `gpu/provision_gpu.sh` (see `gpu/README.md`); the day-to-day deploy script is `deploy_to_rd.sh`. Originally:
`rd_manager` exists to deploy.
