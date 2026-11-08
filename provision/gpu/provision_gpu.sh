#!/usr/bin/env bash
# Provision the GPU box (RTX 4090, Ubuntu 24.04, ROS 2 Jazzy already installed) for phase 7: the 5.0 workspace,
# its tests, the LeRobot venv, Isaac Sim 6.2 (pip) and Isaac ROS 4.6 (host venv mode). Run ON the box as bryan:
#
#   git clone --recurse-submodules https://github.com/r1b4z01d/RobotDisco.git ~/RobotDisco
#   ~/RobotDisco/provision/gpu/provision_gpu.sh all        # = ros workspace test vla isaacsim      (isaacros is NOT in "all")
#   ~/RobotDisco/provision/gpu/provision_gpu.sh isaacros   # Isaac ROS 4.6 host install: replaces the host OpenCV, read the phase first
#
# Nothing here touches ~/VLA, ~/rd_ws or the v1 supervisor units (vla_webui on :8081 keeps running): 5.0 lives in
# ~/RobotDisco, /opt/vla and ~/isaacsim-venv. Sources (read 2026-11-07):
#   Isaac Sim:  https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html  (6.2.0, Python 3.12, pip)
#   Isaac ROS:  https://nvidia-isaac-ros.github.io/v/release-4.6/getting_started/index.html         (apt repo, CLI, venv mode)
set -euo pipefail
# The box's ~/.local (v1 tooling) holds setuptools 84 and OpenCV 5: the first drops tests_require, so colcon
# silently runs unittest instead of pytest (0 tests); the second shadows the cv_bridge-matching system OpenCV.
# 5.0 runs on the system site-packages only, like the rig and the robot.
export PYTHONNOUSERSITE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WS="${WS:-$HOME/RobotDisco}"
ISAACSIM_VENV="${ISAACSIM_VENV:-$HOME/isaacsim-venv}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-6.2.0.0}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"              # cu128 or cu130 (driver 595 takes both)

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
pkglist() { sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$1" | tr '\n' ' '; }
apt_install() { sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$@"; }
# the ROS setup files reference unset variables: source them with -u off
src_ros() { set +u; source /opt/ros/jazzy/setup.bash; set -u; }
src_ws()  { set +u; source "$WS/install/setup.bash"; set -u; }

phase_preflight() {
  [ "$(uname -m)" = x86_64 ] || die "the GPU box script; this is $(uname -m)"
  . /etc/os-release; [ "${VERSION_ID:-}" = 24.04 ] || warn "expected Ubuntu 24.04, got ${PRETTY_NAME:-?}"
  sudo -n true 2>/dev/null || die "needs passwordless sudo"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || die "no NVIDIA driver"
  [ -d /opt/ros/jazzy ] || die "ROS 2 Jazzy is not installed (/opt/ros/jazzy): follow docs.ros.org Jazzy debs first"
  [ -d "$WS/.git" ] || die "clone the repo first: git clone --recurse-submodules https://github.com/r1b4z01d/RobotDisco.git $WS"
}

phase_ros() {
  # the same package lists the robot and the rig use (minus Jetson-only extras), plus the test tools
  log "Jazzy packages for the workspace"
  sudo apt-get update -qq
  apt_install $(pkglist "$ROOT/provision/ros-packages.txt") \
    ros-jazzy-nav2-loopback-sim ros-jazzy-launch-testing ros-jazzy-launch-testing-ros ros-jazzy-ament-cmake-pytest \
    ros-jazzy-ros2controlcli ros-jazzy-tf2-geometry-msgs ros-jazzy-rmw-cyclonedds-cpp \
    python3-pytest python3-pytest-timeout python3-yaml python3-numpy python3-opencv python3-websockets \
    python3-fastapi python3-uvicorn python3-httpx python3-pil python3-colcon-common-extensions python3-rosdep python3-venv python3-pip
  # rd_manager reads LeRobot parquet (datasets tab): no python3-pyarrow deb on 24.04, and pyarrow >= 21 needs numpy 2
  # while the system numpy is 1.26 -> the newest that fits, into /usr/local (nothing apt-managed is touched)
  sudo pip3 install -q --break-system-packages "pyarrow<21"
  sudo rosdep init 2>/dev/null || true
  rosdep update >/dev/null 2>&1 || warn "rosdep update failed (offline?)"
  grep -q "RobotDisco/install/setup.bash" "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<BASHRC

# Robot Disco 5.0 (provision_gpu.sh)
source /opt/ros/jazzy/setup.bash
[ -f $WS/install/setup.bash ] && source $WS/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$WS/provision/cyclonedds.xml"
export ROS_DOMAIN_ID=0
export RD_ROBOT_CONFIG=$WS/config/robot.yaml
export RD_DATA_DIR=$HOME/RobotDisco-data
export PYTHONNOUSERSITE=1    # ~/.local (setuptools 84, OpenCV 5) must not shadow the system packages
BASHRC
  # a bashrc block written by an earlier version of this script lacks the line above
  grep -q 'PYTHONNOUSERSITE' "$HOME/.bashrc" || echo 'export PYTHONNOUSERSITE=1    # Robot Disco 5.0: ~/.local must not shadow the system packages' >> "$HOME/.bashrc"
}

phase_workspace() {
  log "colcon build $WS"
  cd "$WS"
  git submodule update --init --recursive
  "$WS/provision/apply_patches.sh"
  src_ros
  colcon build --symlink-install --event-handlers console_cohesion+ 2>&1 | tail -3
  mkdir -p "$HOME/RobotDisco-data/outputs"/{datasets,train,evals}
  # the v1 data the 5.0 manager and trainer read: a mirror, not a move (v1's ~/VLA keeps working)
  if [ -d "$HOME/VLA/outputs" ]; then
    log "mirroring ~/VLA/outputs (datasets, train, evals) into ~/RobotDisco-data/outputs"
    rsync -a --ignore-existing "$HOME/VLA/outputs/datasets/" "$HOME/RobotDisco-data/outputs/datasets/" || true
    rsync -a --ignore-existing "$HOME/VLA/outputs/train/" "$HOME/RobotDisco-data/outputs/train/" || true
    rsync -a --ignore-existing "$HOME/VLA/outputs/evals/" "$HOME/RobotDisco-data/outputs/evals/" || true
    [ -f "$HOME/VLA/outputs/home_pose.json" ] && cp --update=none "$HOME/VLA/outputs/home_pose.json" "$HOME/RobotDisco-data/" || true
  fi
}

phase_test() {
  log "colcon test (every package, sequential: the launch tests share topics)"
  cd "$WS"
  src_ros && src_ws
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI="file://$WS/provision/cyclonedds.xml" ROS_DOMAIN_ID=0 RD_ROBOT_CONFIG="$WS/config/robot.yaml"
  colcon test --executor sequential --pytest-args --timeout=900 2>&1 | grep -E 'Finished|Failed' || true
  colcon test-result --verbose | tail -15
}

phase_vla() {
  # LeRobot with CUDA torch for training and the SmolVLA backend (the same /opt/vla path as the robot).
  # (pyarrow for the manager is the system one from phase ros; this venv resolves its own numpy 2 stack, keep it so)
  log "LeRobot venv /opt/vla"
  [ -x /opt/vla/bin/python ] || { sudo python3 -m venv --system-site-packages /opt/vla && sudo chown -R "$USER:$USER" /opt/vla; }
  /opt/vla/bin/pip install -q --upgrade pip
  /opt/vla/bin/pip install -q "lerobot>=0.3,<0.5" "transformers" "num2words"
  /opt/vla/bin/python -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'lerobot', lerobot.__version__)"
}

phase_isaacsim() {
  # Isaac Sim 6.2 as a pip package in a Python 3.12 venv (the docs' main path on Ubuntu 24.04; glibc 2.35+).
  log "Isaac Sim $ISAACSIM_VERSION in $ISAACSIM_VENV"
  [ -x "$ISAACSIM_VENV/bin/python" ] || python3.12 -m venv "$ISAACSIM_VENV"
  "$ISAACSIM_VENV/bin/pip" install -q --upgrade pip
  "$ISAACSIM_VENV/bin/pip" install -q torch==2.11.0 --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
  "$ISAACSIM_VENV/bin/pip" install -q "isaacsim[all,extscache,ros2]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com
  grep -q OMNI_KIT_ACCEPT_EULA "$HOME/.bashrc" || echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> "$HOME/.bashrc"
  export OMNI_KIT_ACCEPT_EULA=YES
  "$ISAACSIM_VENV/bin/python" -c "import isaacsim; print('isaacsim import ok')"
  cat <<NOTE
Isaac Sim installed. The first launch warms the extension cache (minutes). Run rd_sim's scene with the system Jazzy:
  source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp OMNI_KIT_ACCEPT_EULA=YES
  ros2 run rd_sim export_urdf /tmp/robot_disco.urdf
  $ISAACSIM_VENV/bin/python $WS/src/rd_sim/scripts/isaac_scene.py --urdf /tmp/robot_disco.urdf
The ROS 2 bridge extension is isaacsim.ros2.bridge; the docs recommend Jazzy on 24.04 and support Cyclone DDS
(set RMW_IMPLEMENTATION=rmw_cyclonedds_cpp in every terminal). Driver 595 is the version the 6.2 docs tested.
NOTE
}

phase_isaacros() {
  # Isaac ROS 4.6 host install, venv mode (Docker is NVIDIA's recommended mode; no Docker on this box).
  # READ FIRST: the host OpenCV packages are removed and replaced by NVIDIA's 4.6.0 build (the 2026-11 docs);
  # python3-opencv and ros-jazzy-cv-bridge are reinstalled afterwards from NVIDIA's/ROS's repos.
  log "Isaac ROS 4.6 (host, venv mode)"
  sudo apt-get update -qq && apt_install curl gnupg software-properties-common locales
  sudo locale-gen en_US en_US.UTF-8 >/dev/null && sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8
  sudo add-apt-repository -y universe >/dev/null
  # the CUDA apt repo (CUDA 13.2+ and TensorRT come through it)
  if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ] && ! dpkg -l cuda-keyring >/dev/null 2>&1; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb
  fi
  grep -q '/usr/local/cuda/bin' "$HOME/.bashrc" || echo 'export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/src/tensorrt/bin:${PATH}"' >> "$HOME/.bashrc"
  # NVIDIA's Jetson apt repo, which the x86 host modes also require (r39.2 = JetPack 7.2's)
  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | sudo gpg --dearmor --yes -o /etc/apt/keyrings/nvidia-jetson.gpg
  echo 'deb [signed-by=/etc/apt/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/x86_64/noble r39.2 main' | sudo tee /etc/apt/sources.list.d/nvidia-jetson-apt-source.list >/dev/null
  # the Isaac ROS 4.6 repo
  k=/usr/share/keyrings/nvidia-isaac-ros.gpg
  [ -f $k ] || curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo gpg --dearmor --yes -o $k
  s="deb [signed-by=$k] https://isaac.download.nvidia.com/isaac-ros/release-4.6 noble main"
  grep -qxF "$s" /etc/apt/sources.list.d/nvidia-isaac-ros.list 2>/dev/null || echo "$s" | sudo tee -a /etc/apt/sources.list.d/nvidia-isaac-ros.list >/dev/null
  sudo apt-get update -qq
  # rosdep's extra keys for Isaac ROS
  sudo mkdir -p /etc/ros/rosdep/sources.list.d
  sudo curl -fsSL -o /etc/ros/rosdep/sources.list.d/nvidia-isaac.yaml https://raw.githubusercontent.com/NVIDIA-ISAAC-ROS/isaac-ros-cli/release-4.6/docker/rosdep/extra_rosdeps.yaml
  echo "yaml file:///etc/ros/rosdep/sources.list.d/nvidia-isaac.yaml" | sudo tee /etc/ros/rosdep/sources.list.d/00-nvidia-isaac.list >/dev/null
  rosdep update >/dev/null 2>&1 || true
  apt_install isaac-ros-cli
  mkdir -p "$HOME/workspaces/isaac_ros-dev/src"
  grep -q ISAAC_ROS_WS "$HOME/.bashrc" || echo 'export ISAAC_ROS_WS="${ISAAC_ROS_WS:-${HOME}/workspaces/isaac_ros-dev/}"' >> "$HOME/.bashrc"
  warn "removing the host OpenCV packages now (NVIDIA's 4.6.0 build replaces them)"
  sudo apt-get remove -y -qq 'libopencv*' 'opencv*' >/dev/null 2>&1 || true
  sudo isaac-ros init venv --yes 2>&1 | tail -5 || sudo isaac-ros init venv
  apt_install $(pkglist "$ROOT/provision/isaac-packages.txt") python3-opencv ros-jazzy-cv-bridge || warn "some Isaac packages did not install: check provision/isaac-packages.txt against the 4.6 package pages"
  echo "Isaac ROS packages:"; dpkg -l | grep -E 'ros-jazzy-isaac-ros' | awk '{print "  " $2, $3}'
  cat <<NOTE
Validate rd_perception's Isaac launch against the fake cameras (no robot needed):
  ros2 run rd_perception fake_cameras &  ros2 launch rd_perception isaac.launch.py apriltag:=true
cuMotion has no 4.6 binary: it is a source build (isaac_ros_cumotion), done when MoveIt-on-cuMotion is validated here.
NOTE
}

PHASES="preflight ros workspace test vla isaacsim isaacros"
[ $# -gt 0 ] || { echo "usage: $0 all | <phase ...>   phases: $PHASES  (all = everything but isaacros)"; exit 2; }
phase_preflight
for phase in "$@"; do
  case "$phase" in
    all) for p in ros workspace test vla isaacsim; do "phase_$p"; done ;;
    preflight) ;;
    *) [[ " $PHASES " == *" $phase "* ]] || die "unknown phase '$phase' (phases: $PHASES)"; "phase_$phase" ;;
  esac
done
