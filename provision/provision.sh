#!/usr/bin/env bash
# Provision the Robot Disco robot PC: a Jetson AGX Orin flashed with JetPack 7.2.1 (Ubuntu 24.04,
# root filesystem on the NVMe), as user "rd". Run ON the robot, from any directory:
#
#   git clone https://github.com/r1b4z01d/RobotDisco.git ~/RobotDisco
#   ~/RobotDisco/provision/provision.sh all
#
# or one phase at a time (idempotent, safe to re-run). "all" runs everything except "network", which
# moves the box onto the internet-less robot subnet and so comes last, by hand.
#
#   provision.sh preflight jetson base ros isaac groups udev sysctl workspace audio services llm display network
#
# Knobs (all optional):
#   RD_USER=rd                   robot user (must be the user running this script)
#   HOSTNAME_NEW=robotdisco      hostname ("" = leave alone)
#   REPO=https://github.com/r1b4z01d/RobotDisco.git   REF=main
#   SNAPSHOT=/path               output of v1's capture_rd_state.sh (phase 0): maps, galleries, voice.env,
#                                motions and ~/Music are restored from it (existing files win)
#   NVPMODEL=0                   AGX Orin power mode id (0 = MAXN)
#   AUTHORIZED_KEY="ssh-ed25519 ..."
#   PROXY=socks5h://127.0.0.1:1080   apt/pip/git proxy when the box is already on the robot subnet
#   WITH_LLM=0 OLLAMA_MODEL=qwen2.5:7b
#   NET_IFACE= NET_GATEWAY=192.168.11.1 NET_DNS=8.8.8.8
#
# STATUS 2026-11-06: skeleton. Each phase states what it must do; the bodies are written and verified on
# the freshly flashed robot in phase 1 of the rebuild plan. v1's working implementation of the same
# phases is rd_ws/provision/{provision.sh,jetson/provision_jetson.sh}; port, do not rewrite.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RD_USER="${RD_USER:-rd}"
RD_HOME="$( { command -v getent >/dev/null && getent passwd "$RD_USER" | cut -d: -f6; } || true)"
RD_HOME="${RD_HOME:-/home/$RD_USER}"
WS="$RD_HOME/RobotDisco"
REPO="${REPO:-https://github.com/r1b4z01d/RobotDisco.git}"
REF="${REF:-main}"
HOSTNAME_NEW="${HOSTNAME_NEW:-robotdisco}"
SNAPSHOT="${SNAPSHOT:-}"
ROS_DISTRO=jazzy
NVPMODEL="${NVPMODEL:-0}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
todo() { warn "phase $1 is not implemented yet (port it from rd_ws/provision); skipping"; }
pkglist() { sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$1" | tr '\n' ' '; }
apt_install() { sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"; }

phase_preflight() {
  [ "$(id -un)" = "$RD_USER" ] || die "run as $RD_USER"
  sudo -n true 2>/dev/null || die "$RD_USER needs passwordless sudo for provisioning"
  . /etc/os-release
  [ "${VERSION_ID:-}" = "24.04" ] || warn "expected Ubuntu 24.04 (JetPack 7.2), got ${PRETTY_NAME:-?}"
  [ "$(uname -m)" = "aarch64" ] || warn "expected aarch64 (Jetson), got $(uname -m)"
  if [ -f /etc/nv_tegra_release ]; then log "Jetson Linux: $(head -1 /etc/nv_tegra_release)"; else warn "not a Jetson (no /etc/nv_tegra_release)"; fi
  findmnt -no SOURCE / | grep -q nvme || warn "root filesystem is not on the NVMe (plan: root on NVMe, eMMC as recovery)"
  [ -n "${PROXY:-}" ] && todo "preflight: proxy config (v1 setup_proxy)"
  true
}

phase_jetson() {
  # v1 phase_jetson: nvpmodel boot default + /var/lib/nvpmodel/status, jetson-clocks.service, nvidia docker
  # runtime, xpad at boot (provision/xpad rebuild still needed on kernel 6.8: 8BitDo 2dc8:310a lands in 6.10), jtop.
  todo jetson
}

phase_base() {
  log "apt packages"
  sudo apt-get update
  apt_install $(pkglist "$HERE/apt-packages.txt")
  sudo systemctl enable --now ssh supervisor chrony
  if [ -n "$HOSTNAME_NEW" ] && [ "$(hostname)" != "$HOSTNAME_NEW" ]; then sudo hostnamectl set-hostname "$HOSTNAME_NEW"; fi
  for entry in "192.168.11.21 ur5e" "192.168.11.117 amazinghand" "192.168.11.119 gpu"; do
    grep -qE "^${entry%% *}[[:space:]]" /etc/hosts || echo "$entry" | sudo tee -a /etc/hosts >/dev/null
  done
  if [ -n "${AUTHORIZED_KEY:-}" ]; then
    mkdir -p "$RD_HOME/.ssh" && chmod 700 "$RD_HOME/.ssh"
    grep -qF "$AUTHORIZED_KEY" "$RD_HOME/.ssh/authorized_keys" 2>/dev/null || echo "$AUTHORIZED_KEY" >> "$RD_HOME/.ssh/authorized_keys"
    chmod 600 "$RD_HOME/.ssh/authorized_keys"
  fi
}

phase_ros() {
  log "ROS 2 $ROS_DISTRO"
  if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
    sudo apt-get update
  fi
  apt_install $(pkglist "$HERE/ros-packages.txt")
  sudo rosdep init 2>/dev/null || true
  rosdep update
  if ! grep -q "RobotDisco/install/setup.bash" "$RD_HOME/.bashrc"; then
    cat >> "$RD_HOME/.bashrc" <<BASHRC

# Robot Disco (provision.sh phase ros)
source /opt/ros/$ROS_DISTRO/setup.bash
[ -f $WS/install/setup.bash ] && source $WS/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$WS/provision/cyclonedds.xml"
export ROS_DOMAIN_ID=0
export RD_ROBOT_CONFIG=$WS/config/robot.yaml
BASHRC
  fi
}

phase_isaac() {
  # Isaac ROS 4.6 host install for JetPack 7.2 / Jazzy: NVIDIA's apt repo + isaac-packages.txt via the Isaac ROS
  # CLI in bare-metal mode (https://nvidia-isaac-ros.github.io/v/release-4.6/getting_started/index.html).
  todo isaac
}

phase_groups() {
  for g in dialout video input plugdev audio render docker; do getent group "$g" >/dev/null && sudo usermod -aG "$g" "$RD_USER"; done
  log "groups: $(id -nG "$RD_USER")"
}

phase_udev() {
  sudo install -m 644 "$HERE/udev/99-robot-disco.rules" /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
}

phase_sysctl() {
  # v1 raised UDP receive buffers for the multicast camera stream; v2 cameras are ROS topics. Cyclone DDS still
  # wants a larger buffer for point clouds: net.core.rmem_max.
  printf 'net.core.rmem_max=8388608\nnet.core.rmem_default=8388608\n' | sudo tee /etc/sysctl.d/60-robot-disco.conf >/dev/null
  sudo sysctl --system >/dev/null
}

phase_workspace() {
  log "workspace $WS"
  if [ ! -d "$WS/.git" ]; then git clone "$REPO" "$WS"; fi
  git -C "$WS" fetch --all --tags && git -C "$WS" checkout "$REF" && git -C "$WS" pull --ff-only || true
  git -C "$WS" submodule update --init --recursive || warn "a submodule failed; check vendor/ and firmware/"
  for p in "$HERE"/patches/*.patch; do [ -f "$p" ] && todo "workspace: apply $(basename "$p")"; done
  cd "$WS"
  rosdep install --from-paths src vendor --ignore-src -r -y || warn "rosdep left something unresolved"
  colcon build --symlink-install
  [ -n "$SNAPSHOT" ] && todo "workspace: restore maps from $SNAPSHOT"
  true
}

phase_audio()    { todo audio; }     # v1: pulse/default.pa, unmute the C-Media sink, alsactl store
phase_services() { todo services; }  # v1: supervisor web UI :9999, install src/rd_bringup/services/*.conf
phase_llm()      { todo llm; }       # v1: Ollama under supervisor on loopback :11434, pull $OLLAMA_MODEL
phase_display()  { todo display; }   # v1: GNOME autologin for rd, no screen blanking, the face window on the portrait display
phase_network()  { todo network; }   # v1: static 192.168.11.80/24 via nmcli (Jetson desktop) or netplan/01-robot-disco.yaml

PHASES="preflight jetson base ros isaac groups udev sysctl workspace audio services llm display network"
phase_all() { for p in $PHASES; do [ "$p" = network ] && continue; "phase_$p"; done; log "done; reboot, then run: provision.sh network"; }

[ $# -gt 0 ] || { echo "usage: provision.sh all | <phase ...>   phases: $PHASES"; exit 2; }
phase_preflight
for phase in "$@"; do
  case "$phase" in
    all) phase_all ;;
    preflight) ;;
    *) [[ " $PHASES " == *" $phase "* ]] || die "unknown phase '$phase' (phases: $PHASES)"; "phase_$phase" ;;
  esac
done
