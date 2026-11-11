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
#   provision.sh preflight jetson base isaac ros isaacpkgs groups udev sysctl data workspace venvs audio services llm display network
#
# Knobs (all optional):
#   RD_USER=rd                   robot user (must be the user running this script)
#   HOSTNAME_NEW=robotdisco      hostname ("" = leave alone)
#   REPO=https://github.com/r1b4z01d/RobotDisco.git   REF=main
#   SNAPSHOT=/path               output of v1's capture_rd_state.sh (phase 0): maps, galleries, voice.env,
#                                motions and ~/Music are restored from it (existing files win)
#   NVPMODEL=0                   AGX Orin power mode id (0 = MAXN)
#   UART1_PIO=1                  boot a copy of the device tree without ttyTHS1's DMA (v1's remedy on JetPack 6 for
#                                zeroed RoboClaw reads; not needed on 7.2.1, see the jetson phase)
#   AUTHORIZED_KEY="ssh-ed25519 ..."
#   PROXY=socks5h://127.0.0.1:1080   apt/pip/git proxy when the box is already on the robot subnet
#   WITH_LLM=0 OLLAMA_MODEL=qwen2.5:7b
#   NET_IFACE= NET_GATEWAY=192.168.11.1 NET_DNS=8.8.8.8
#
# STATUS 2026-11-07: every phase is written (ported from v1's provision.sh + jetson/provision_jetson.sh) and
# parses; none has run on JetPack 7.2 yet. Expect the Isaac apt repo line, the cu132 torch wheel and the
# kernel-6.8 xpad to need attention on the first run (the plan's risks).
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
  if [ -n "${PROXY:-}" ]; then
    printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' "$PROXY" "$PROXY" | sudo tee /etc/apt/apt.conf.d/99proxy >/dev/null
    git config --global http.proxy "$PROXY"; export PIP_PROXY="$PROXY"
  fi
}

phase_jetson() {
  # v1 jetson/provision_jetson.sh phase_jetson: boot into MAXN, pin the clocks, nvidia docker runtime, xpad, jtop,
  # and the header UART the RoboClaw hangs off (uart1_pio_dtb).
  [ -f "$HERE/jetson/apt-packages-jetson.txt" ] && apt_install $(pkglist "$HERE/jetson/apt-packages-jetson.txt") || true
  log "Jetson power mode and clocks"
  if [ -f /etc/nvpmodel.conf ]; then
    sudo sed -i -E "s/^< PM_CONFIG DEFAULT=[0-9]+ >/< PM_CONFIG DEFAULT=$NVPMODEL >/" /etc/nvpmodel.conf
    printf 'pmode:%04d\n' "$NVPMODEL" | sudo tee /var/lib/nvpmodel/status >/dev/null      # nvpmodel restores the LAST mode, not the default
  fi
  if command -v jetson_clocks >/dev/null && [ ! -f /etc/systemd/system/jetson-clocks.service ]; then
    sudo tee /etc/systemd/system/jetson-clocks.service >/dev/null <<'UNIT'
[Unit]
Description=Pin Jetson clocks to max (the robot runs inference at a fixed control rate)
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
    sudo systemctl daemon-reload && sudo systemctl enable --now jetson-clocks.service
  fi
  if command -v nvidia-ctk >/dev/null && command -v docker >/dev/null; then
    sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && sudo systemctl restart docker || true
  fi
  # the 8BitDo receiver needs xpad (and, before kernel 6.10, the rebuilt module with its ID: provision/xpad)
  echo xpad | sudo tee /etc/modules-load.d/xpad.conf >/dev/null
  sudo modprobe xpad || warn "xpad module not available on this kernel"
  if [ "$(uname -r | cut -d. -f1-2 | tr -d .)" -lt 610 ] 2>/dev/null; then
    warn "kernel $(uname -r) predates xpad support for the 8BitDo 2C (2dc8:310a): build provision/xpad (see its README) or rely on the udev new_id rule"
  fi
  sudo python3 -m pip install -U --break-system-packages jetson-stats 2>/dev/null || warn "jetson-stats install failed"
  # Not needed on JetPack 7.2.1: on the stock device tree (DMA still bound to serial@3100000) the RoboClaw answered
  # 3200 CRC-checked reads at 100 Hz without one error (tools/roboclaw_crc_soak.py, 2026-11-10). UART1_PIO=1 keeps
  # v1's remedy at hand should a kernel update bring the zeroed reads back.
  if [ "${UART1_PIO:-0}" = 1 ]; then
    log "ttyTHS1 (header UART, the RoboClaw link) in PIO mode"
    uart1_pio_dtb
  else
    log "ttyTHS1 on the stock device tree (UART1_PIO=1 switches it to PIO mode; check with tools/roboclaw_crc_soak.py)"
  fi
  # a login console on the RoboClaw's UART would eat its replies; the stock units target other ports, so this acts only if one does not
  if systemctl is-enabled nvgetty.service >/dev/null 2>&1 && systemctl cat nvgetty.service 2>/dev/null | grep -q ttyTHS1; then
    sudo systemctl disable --now nvgetty.service && log "nvgetty disabled: it was a login console on ttyTHS1"
  fi
}

# The RoboClaw's packet-serial link is the 40-pin header UART (pins 8 TX / 10 RX / 6 GND = /dev/ttyTHS1 =
# serial@3100000) since 2026-11-08 (v1: its USB jack is bad and its USB ground reset the Jetson). On JetPack 6
# that UART's receive DMA handed back zeroed bytes (NVIDIA forum "DMA on /dev/ttyTHS1 corrupts receiving data");
# ttyTHS2 ships without DMA bindings and was fine, so v1 boots a copy of the stock device tree with the two DMA
# properties removed, through an FDT line in extlinux.conf (the mechanism jetson-io uses). Whether kernel 6.8
# (JetPack 7.2) still has the bug is unknown: this only acts when the stock DTB binds DMA to that UART, and
# `dmesg | grep 3100000.serial` must say "RX in PIO mode" after the reboot. Idempotent: rebuilds the PIO copy when
# the stock DTB is newer (kernel/dtb package update), leaves an extlinux.conf that already has an FDT line alone,
# keeps the stock entry as a boot-menu fallback.
uart1_pio_dtb() {
  local base pio conf=/boot/extlinux/extlinux.conf node=/bus@0/serial@3100000
  command -v fdtget >/dev/null || apt_install device-tree-compiler
  base=$(ls /boot/dtb/kernel_tegra234-*.dtb /boot/dtb/tegra234-*.dtb 2>/dev/null | grep -v -- '-uart1-pio' | head -1)
  [ -n "$base" ] || { warn "no tegra234 DTB under /boot/dtb: ttyTHS1 PIO device tree skipped; check 'dmesg | grep 3100000.serial' for 'RX in PIO mode'"; return 0; }
  if ! fdtget "$base" "$node" dmas >/dev/null 2>&1; then log "$base binds no DMA to $node: ttyTHS1 already runs in PIO mode"; return 0; fi
  pio="${base%.dtb}-uart1-pio.dtb"
  if [ ! -f "$pio" ] || [ "$base" -nt "$pio" ]; then
    sudo cp "$base" "$pio"
    sudo fdtput -d "$pio" "$node" dma-names
    sudo fdtput -d "$pio" "$node" dmas
    log "built $pio (stock tree minus $node dmas/dma-names)"
  fi
  [ -f "$conf" ] || { warn "no $conf: boot $pio by hand (the boot layout changed?)"; return 0; }
  if grep -q "FDT $pio" "$conf"; then
    log "extlinux.conf already boots $pio"
  elif grep -qE '^\s*FDT ' "$conf"; then
    warn "extlinux.conf already has an FDT line (jetson-io?): not touched; drop dmas/dma-names from $node in that DTB instead"
  else
    sudo cp -n "$conf" "$conf.pre-uart1-pio"
    sudo python3 - "$conf" "$pio" <<'PY' || warn "extlinux.conf not edited: add 'FDT $pio' under the primary LINUX line by hand"
import re, sys
conf, pio = sys.argv[1], sys.argv[2]
s = open(conf).read()
m = re.search(r"^(\s*)LINUX /boot/Image\n", s, re.M)
a = re.search(r"^\s*APPEND .*$", s, re.M)
if not m or not a:
    sys.exit("no 'LINUX /boot/Image' + APPEND pair found")
s = s[:m.end()] + m.group(1) + "FDT " + pio + "\n" + s[m.end():]
s = s.rstrip("\n") + "\n\nLABEL stock\n      MENU LABEL stock device tree (ttyTHS1 with DMA)\n      LINUX /boot/Image\n      INITRD /boot/initrd\n" + a.group(0) + "\n"
open(conf, "w").write(s)
PY
    log "extlinux.conf: primary entry boots $pio ('stock' entry kept, backup $conf.pre-uart1-pio); reboot to apply"
  fi
}

phase_data() {
  # The 512 GB NVMe. With the root filesystem ON the NVMe (the plan), /mnt/data is a directory on the same
  # disk; the paths every package uses (robot.yaml robot.data) stay valid either way. A separate data drive
  # labelled "$DATA_LABEL" is mounted there instead (v1's layout: a 512 GB NVMe beside the 64 GB eMMC root).
  DATA_MOUNT="${DATA_MOUNT:-/mnt/data}"; DATA_LABEL="${DATA_LABEL:-data}"
  log "Data dir $DATA_MOUNT"
  if ! findmnt -rn "$DATA_MOUNT" >/dev/null 2>&1; then
    local dev; dev="$(sudo blkid -L "$DATA_LABEL" 2>/dev/null || true)"
    if [ -n "$dev" ]; then
      local uuid; uuid="$(sudo blkid -s UUID -o value "$dev")"
      sudo mkdir -p "$DATA_MOUNT"
      findmnt --fstab -rn "$DATA_MOUNT" >/dev/null 2>&1 || echo "UUID=$uuid $DATA_MOUNT ext4 defaults,nofail,x-systemd.device-timeout=10 0 2" | sudo tee -a /etc/fstab >/dev/null
      sudo systemctl daemon-reload && sudo mount "$DATA_MOUNT"
      sudo mkdir -p /etc/systemd/system/supervisor.service.d
      printf '[Unit]\nRequiresMountsFor=%s\n' "$DATA_MOUNT" | sudo tee /etc/systemd/system/supervisor.service.d/nvme-data.conf >/dev/null
      sudo systemctl daemon-reload
    else
      log "no drive labelled '$DATA_LABEL': $DATA_MOUNT is a directory on the root disk"
      sudo mkdir -p "$DATA_MOUNT"
    fi
  fi
  sudo chown "$RD_USER:$RD_USER" "$DATA_MOUNT" && sudo chmod 755 "$DATA_MOUNT"
  mkdir -p "$DATA_MOUNT/outputs/datasets" "$DATA_MOUNT/outputs/train" "$DATA_MOUNT/outputs/evals" "$DATA_MOUNT/cache" "$RD_HOME/.robot_face"
  # the HF cache (SmolVLA's VLM, Arrow copies) and the robot's own state live on the data disk
  for pair in ".cache/huggingface:cache/huggingface" ".robot_face/faces:faces" ".robot_face/motions:motions"; do
    local src="$RD_HOME/${pair%%:*}" dst="$DATA_MOUNT/${pair##*:}"
    if [ ! -e "$src" ]; then mkdir -p "$dst" "$(dirname "$src")" && ln -s "$dst" "$src"; fi
  done
  if [ -n "$SNAPSHOT" ]; then
    log "restoring robot state from $SNAPSHOT"
    for item in people.json people.arcface.json voice.env; do [ -f "$SNAPSHOT/$item" ] && [ ! -f "$RD_HOME/.robot_face/$item" ] && cp "$SNAPSHOT/$item" "$RD_HOME/.robot_face/" || true; done
    for d in faces motions Music; do [ -d "$SNAPSHOT/$d" ] && rsync -a --ignore-existing "$SNAPSHOT/$d/" "$RD_HOME/$( [ "$d" = Music ] && echo Music || echo .robot_face/$d )/" || true; done
    [ -d "$SNAPSHOT/maps" ] && rsync -a --ignore-existing "$SNAPSHOT/maps/" "$WS/src/rd_nav/maps/" || true
    [ -d "$SNAPSHOT/landmarks" ] && rsync -a --ignore-existing "$SNAPSHOT/landmarks/" "$WS/src/rd_nav/landmarks/" || true
  fi
  df -h "$DATA_MOUNT" | sed 's/^/  /'
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
  # NVIDIA's Jetson suite (noble-jetpack, pinned at 600 by the Isaac CLI) carries its own UR driver stack, versioned
  # 99.2.0 so it outranks the ROS repo's 3.9.0. Its ur_ros2_control macro has an older interface (no ur_type), and
  # rd_description is written, rig-tested and GPU-box-tested against 3.9.0: with 99.2.0 the robot's URDF does not
  # expand and the arm stack never starts (2026-11-10). No Isaac package depends on NVIDIA's copy, so the UR stack
  # comes from the ROS repo.
  printf 'Package: ros-jazzy-ur ros-jazzy-ur-*\nPin: origin packages.ros.org\nPin-Priority: 1001\n' | sudo tee /etc/apt/preferences.d/50-robot-disco-ur-from-ros.pref >/dev/null
  apt_install --allow-downgrades $(pkglist "$HERE/ros-packages.txt")
  # rd_manager reads LeRobot parquet (datasets tab): no python3-pyarrow deb on 24.04, and pyarrow >= 21 needs numpy 2
  # while the system numpy is 1.26 -> the newest that fits, into /usr/local (nothing apt-managed is touched)
  sudo pip3 install -q --break-system-packages "pyarrow<21"
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
  # Isaac ROS 4.6 as host debs, NVIDIA's "Virtual Environment" mode (debs on the host through apt, the CLI's pip shims
  # in its own venv, never the system python): the mode the GPU box runs. From the 4.6 Getting Started page's Jetson
  # (JetPack 7.2) steps and the first run on the robot, 2026-11-09:
  #  - the Jetson suite of the apt repo is `noble-jetpack` (157 ros-jazzy-isaac-ros-* packages for arm64); `noble` is
  #    the x86 suite and carries NO Isaac package for arm64, only the CLI
  #  - the isaac-ros-cli deb drops apt pins: OpenCV 4.6.0, CUDA 13.2, TensorRT 10.16
  #  - nvidia-jetpack brings OpenCV 4.8.0, which NVIDIA has removed here; Ubuntu's 4.6.0 then comes back with the
  #    packages that need it. This phase runs BEFORE `ros` for that reason: nothing else depends on OpenCV yet, so
  #    the removal takes OpenCV alone (after `ros` it would take cv_bridge, Nav2 and friends with it)
  log "Isaac ROS 4.6: apt repo, CLI, OpenCV 4.8 -> 4.6"
  apt_install curl gnupg
  local k=/usr/share/keyrings/nvidia-isaac-ros.gpg suite=noble
  if [ "$(dpkg --print-architecture)" = arm64 ] && [ -f /etc/nv_tegra_release ]; then suite=noble-jetpack; fi
  [ -f $k ] || curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo gpg --dearmor --yes -o $k
  echo "deb [signed-by=$k] https://isaac.download.nvidia.com/isaac-ros/release-4.6 $suite main" | sudo tee /etc/apt/sources.list.d/nvidia-isaac-ros.list >/dev/null
  sudo apt-get update
  sudo mkdir -p /etc/ros/rosdep/sources.list.d      # read by `rosdep update` in the ros phase
  sudo curl -fsSL -o /etc/ros/rosdep/sources.list.d/nvidia-isaac.yaml https://raw.githubusercontent.com/NVIDIA-ISAAC-ROS/isaac-ros-cli/release-4.6/docker/rosdep/extra_rosdeps.yaml || true
  echo "yaml file:///etc/ros/rosdep/sources.list.d/nvidia-isaac.yaml" | sudo tee /etc/ros/rosdep/sources.list.d/00-nvidia-isaac.list >/dev/null
  apt_install isaac-ros-cli
  mkdir -p "$RD_HOME/workspaces/isaac_ros-dev/src"
  grep -q ISAAC_ROS_WS "$RD_HOME/.bashrc" || echo 'export ISAAC_ROS_WS="${ISAAC_ROS_WS:-${HOME}/workspaces/isaac_ros-dev/}"' >> "$RD_HOME/.bashrc"
  if dpkg-query -W -f='${Version}\n' 'libopencv*' 2>/dev/null | grep -q '^4\.8'; then
    warn "removing JetPack's OpenCV 4.8 (NVIDIA's Isaac ROS step); 4.6.0 is pinned and returns with the ROS and Isaac packages"
    # the removal takes the nvidia-jetpack metapackages with it (they depend on nvidia-opencv): mark what they pulled
    # in as wanted, or a later `apt autoremove` would take CUDA, cuDNN, TensorRT and VPI
    sudo apt-mark manual $(apt-cache depends nvidia-jetpack-runtime nvidia-jetpack-dev 2>/dev/null | awk '/Depends:/{print $2}' | grep -v -i -e opencv -e '^<' | sort -u) >/dev/null
    sudo apt-get remove -y 'libopencv*' 'opencv*'
  fi
  sudo isaac-ros init venv --yes 2>&1 | tail -5
}

phase_isaacpkgs() {
  # after `ros`: the Isaac packages depend on ros-jazzy-* from the ROS repo. One at a time, so a package that cannot
  # resolve does not hold back the rest. On the Jetson that is nvblox (2026-11-09): it needs isaac_ros_triton ->
  # triton-server -> datacenter-gpu-manager-4-core + libnccl2, which only NVIDIA's CUDA server repo (ubuntu2404/sbsa)
  # carries; the Jetson and Isaac repos do not.
  log "Isaac ROS 4.6 packages"
  local p failed=""
  for p in $(pkglist "$HERE/isaac-packages.txt") python3-opencv ros-jazzy-cv-bridge; do
    apt_install "$p" >/dev/null 2>&1 || failed="$failed $p"
  done
  [ -z "$failed" ] || warn "not installable here:$failed"
  dpkg -l | grep -E '^ii +ros-jazzy-isaac-ros' | awk '{print "  " $2, $3}' || true
  dpkg-query -W -f='  ${Package} ${Version}\n' 'libopencv-core*' 2>/dev/null || true
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
  # v1 raised UDP receive buffers for the multicast camera stream; 5.0 cameras are ROS topics. Cyclone DDS still
  # wants a larger buffer for point clouds: net.core.rmem_max.
  printf 'net.core.rmem_max=8388608\nnet.core.rmem_default=8388608\n' | sudo tee /etc/sysctl.d/60-robot-disco.conf >/dev/null
  sudo sysctl --system >/dev/null
}

phase_workspace() {
  log "workspace $WS"
  if [ ! -d "$WS/.git" ]; then git clone "$REPO" "$WS"; fi
  git -C "$WS" fetch --all --tags && git -C "$WS" checkout "$REF" && git -C "$WS" pull --ff-only || true
  git -C "$WS" submodule update --init --recursive || warn "a submodule failed; check vendor/ and firmware/"
  # roboclaw_serial-baud-rate.patch + roboclaw_hardware_interface-baud-rate.patch: the RoboClaw is on a real UART
  # (header ttyTHS1): a `baud_rate` hardware parameter sets the tty speed, a failed open fails on_init instead of
  # aborting controller_manager, and a reply is read until the line goes quiet (a UART delivers it in pieces).
  "$WS/provision/apply_patches.sh" || warn "a vendor patch did not apply: the base's hardware interface may not build"
  cd "$WS"
  # the shell running this has not sourced ROS (the ros phase only wrote .bashrc; ssh sessions never read it):
  # without it rosdep has no distro and colcon finds no ament
  set +u; . "/opt/ros/$ROS_DISTRO/setup.bash"; set -u
  rosdep install --from-paths src vendor --ignore-src -r -y || warn "rosdep left something unresolved"
  colcon build --symlink-install
  true
}

phase_venvs() {
  # The three Python environments that do not mix with the ROS python: the ArcFace face-ID worker, the
  # Whisper ears, Kokoro TTS (v1's setup scripts, paths updated), and the LeRobot venv for rd_vla.
  log "venvs: face id, voice, kokoro, lerobot"
  # through bash, and a failure is named: setup_voice.sh sat in the repo without its execute bit and `|| true` hid the
  # "Permission denied", so the first run on the robot left no voice venv and said nothing
  [ "${WITH_FACEID:-1}" != 1 ] || bash "$HERE/setup_faceid.sh" || warn "setup_faceid.sh failed: no ArcFace venv (the face falls back to OpenCV YuNet + SFace)"
  [ "${WITH_VOICE:-1}" != 1 ] || bash "$HERE/setup_voice.sh" "${WHISPER_MODEL:-base.en}" || warn "setup_voice.sh failed: no voice venv (no ears)"
  [ "${WITH_KOKORO:-1}" != 1 ] || bash "$HERE/setup_kokoro.sh" || warn "setup_kokoro.sh failed: no Kokoro venv (speech falls back to espeak)"
  if [ ! -x /opt/vla/bin/python ]; then
    sudo python3 -m venv --system-site-packages /opt/vla && sudo chown -R "$RD_USER:$RD_USER" /opt/vla
  fi
  # torch for the Orin on JetPack 7.2.1 (tested 2026-11-09): PyTorch's own CUDA 13.0 build for arm64, which bundles its
  # CUDA libraries, sees the GPU ("Orin", cuDNN 9.15) and runs matmul, conv and torchvision ops on it. LeRobot 0.4 wants
  # torch < 2.11, and the cu132 tree starts at 2.12, hence cu130. The Jetson AI Lab build (sbsa/cu130) does NOT import
  # here: it links NVPL and cuDSS, which only NVIDIA's CUDA server repo carries.
  local torch_index="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
  /opt/vla/bin/pip install -q --upgrade pip
  /opt/vla/bin/pip install -q "torch==2.10.0" "torchvision==0.25.0" --index-url "$torch_index" --extra-index-url https://pypi.org/simple || warn "torch install failed (index $torch_index)"
  /opt/vla/bin/pip install -q --extra-index-url "$torch_index" "lerobot>=0.3,<0.5" "transformers" "num2words" || warn "lerobot install failed"
  # The venv sees the system site-packages (it needs rclpy), and Ubuntu's scipy, pandas and matplotlib are built for
  # numpy 1: they satisfy pip, so it installs nothing, and then fail to import under the venv's numpy 2 (transformers
  # probes scipy, datasets imports pandas). Give the venv its own copy of whichever resolves outside it.
  for m in scipy pandas matplotlib; do
    /opt/vla/bin/python -c "import importlib.util as u, sys; s = u.find_spec('$m'); sys.exit(0 if s and s.origin and s.origin.startswith('/opt/vla/') else 1)" 2>/dev/null \
      || /opt/vla/bin/pip install -q --ignore-installed --no-deps "$m"
  done
  /opt/vla/bin/python -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'lerobot', lerobot.__version__)" || true
}

phase_audio() {
  # Speaker = the dock's C-Media USB adapter, microphone = the TONOR. v1 learned: module-switch-on-connect made a
  # newly plugged device the default; the adapter enumerates after pulse starts; alsactl restore re-applied a
  # hardware mute at every boot. The user-level default.pa settles the first two, unmute + alsactl store the third.
  local sink; sink="$(python3 -c "import yaml;print(yaml.safe_load(open('$WS/config/robot.yaml'))['audio']['sink'])")"
  log "Audio: PulseAudio policy -> $sink, unmute, alsactl store"
  mkdir -p "$RD_HOME/.config/pulse"
  sed "s|^set-default-sink .*|set-default-sink $sink|" "$HERE/pulse/default.pa" > "$RD_HOME/.config/pulse/default.pa"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if ! pactl info >/dev/null 2>&1; then warn "no PulseAudio session yet (desktop not logged in); the policy applies at the next login"; return 0; fi
  if ! pactl list short sinks | awk '{print $2}' | grep -qx "$sink"; then warn "sink $sink not present (adapter unplugged?)"; return 0; fi
  pactl set-default-sink "$sink" || true
  pactl set-sink-mute "$sink" 0 || true
  local vol; vol="$(pactl get-sink-volume "$sink" | grep -oE '[0-9]+%' | head -1 | tr -d %)"
  [ "${vol:-0}" -ge 10 ] || pactl set-sink-volume "$sink" 60%
  sudo alsactl store 2>/dev/null || warn "alsactl store failed (the adapter may come back muted after a reboot)"
}

phase_services() {
  log "Supervisor: web UI / XML-RPC on :9999 and the units in src/rd_bringup/services"
  if ! grep -rqs "inet_http_server" /etc/supervisor/; then
    printf '; supervisord web UI / XML-RPC (no auth: trusted robot subnet)\n[inet_http_server]\nport=0.0.0.0:9999\n' | sudo tee /etc/supervisor/conf.d/00-inet-http.conf >/dev/null
    sudo systemctl restart supervisor
  fi
  mkdir -p "$WS/log"
  # The units must not start before the clock is set. After a total power loss the Jetson's clock comes up at 1970
  # (no RTC battery on the devkit) and jumps 56 years when chrony first syncs; started before that, Nav2's bringup
  # saw its timeouts expire at once and aborted (the robot, 2026-11-10, when the battery died). Supervisor waits for
  # chrony for at most a minute; without a network it then starts with the clock it has.
  # fake-hwclock keeps the last known time on disk (hourly and at shutdown) and puts it back early in the boot, so
  # without a network the clock is hours behind, not at 1970, and chrony's later step is the time spent off. The
  # second boot showed the minute's wait alone is not enough: the router came up 16 s after it gave up.
  apt_install fake-hwclock || warn "fake-hwclock not installed: after a power loss the clock starts at 1970"
  sudo tee /etc/systemd/system/rd-wait-clock.service >/dev/null <<'UNIT'
[Unit]
Description=Robot Disco: wait (at most a minute) for the clock to be set before the units start
After=chrony.service network-online.target
Wants=chrony.service network-online.target
Before=supervisor.service

[Service]
Type=oneshot
ExecStart=-/usr/bin/chronyc waitsync 12 0 0 5
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
  sudo mkdir -p /etc/systemd/system/supervisor.service.d
  printf '[Unit]\nAfter=rd-wait-clock.service\nWants=rd-wait-clock.service\n' | sudo tee /etc/systemd/system/supervisor.service.d/10-wait-clock.conf >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable rd-wait-clock.service >/dev/null 2>&1 || warn "could not enable rd-wait-clock.service"
  sudo cp "$WS"/src/rd_bringup/services/*.conf /etc/supervisor/conf.d/
  sudo supervisorctl reread >/dev/null && sudo supervisorctl update >/dev/null
  sudo supervisorctl status || true
}

phase_llm() {
  # Ollama under supervisor on loopback :11434 (the GPU box's is asked first). Its own systemd unit is disabled.
  [ "${WITH_LLM:-1}" = 1 ] || { log "WITH_LLM=0: skipping"; return 0; }
  log "Local LLM: Ollama (unit 'ollama') with $OLLAMA_MODEL"
  command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  if systemctl list-unit-files ollama.service 2>/dev/null | grep -q '^ollama.service'; then sudo systemctl disable --now ollama.service; fi
  sudo mkdir -p /mnt/data/ollama && sudo chown ollama:ollama /mnt/data/ollama 2>/dev/null || true
  sudo cp "$WS/src/rd_bringup/services/ollama.conf" /etc/supervisor/conf.d/ && sudo supervisorctl reread >/dev/null && sudo supervisorctl update >/dev/null
  sudo supervisorctl status ollama | grep -qE 'RUNNING|STARTING' || sudo supervisorctl start ollama
  local i; for i in $(seq 1 30); do curl -fsS 127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
  if ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL}"; then echo "model $OLLAMA_MODEL present"
  elif curl -fsI -m 8 https://ollama.com >/dev/null 2>&1; then ollama pull "$OLLAMA_MODEL" || warn "pull failed"
  else warn "no internet: ollama pull $OLLAMA_MODEL later"; fi
}

phase_display() {
  # The face is a GL window on the robot's own GNOME session: autologin, never blank or lock.
  log "GNOME autologin for $RD_USER, no blanking"
  # No crash-report dialogs on the face's screen: apport shows "System program problem detected" at every login while
  # a report sits in /var/crash (the installer's and Firefox's from the flash day did, 2026-11-10).
  if [ -f /etc/default/apport ]; then
    sudo sed -i 's/^enabled=.*/enabled=0/' /etc/default/apport
    sudo systemctl disable --now apport.service >/dev/null 2>&1 || true
    sudo mkdir -p /var/crash/old && sudo find /var/crash -maxdepth 1 -name '*.crash' -exec mv {} /var/crash/old/ \;
  fi
  if [ -f /etc/gdm3/custom.conf ]; then
    sudo sed -i -E -e "s/^#?\s*AutomaticLoginEnable\s*=.*/AutomaticLoginEnable = true/" -e "s/^#?\s*AutomaticLogin\s*=.*/AutomaticLogin = $RD_USER/" /etc/gdm3/custom.conf
    grep -q "^AutomaticLoginEnable" /etc/gdm3/custom.conf || sudo sed -i "/^\[daemon\]/a AutomaticLoginEnable = true\nAutomaticLogin = $RD_USER" /etc/gdm3/custom.conf
  else
    warn "/etc/gdm3/custom.conf not found: the face needs a desktop session"
  fi
  if command -v gsettings >/dev/null; then
    dbus-launch gsettings set org.gnome.desktop.session idle-delay 0 || true
    dbus-launch gsettings set org.gnome.desktop.screensaver lock-enabled false || true
    dbus-launch gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type nothing || true
  fi
}

phase_network() {
  # Static address on the isolated robot subnet, from robot.yaml. NetworkManager (the Jetson desktop): change the
  # connection in place; otherwise netplan. Last, by hand: this SSH session drops.
  local ip gw dns iface="${NET_IFACE:-}"
  ip="$(python3 -c "import yaml;c=yaml.safe_load(open('$WS/config/robot.yaml'));print(c['robot']['ip'])")"
  gw="${NET_GATEWAY:-$(python3 -c "import yaml;c=yaml.safe_load(open('$WS/config/robot.yaml'));print(c['network'].get('gateway',''))")}"
  dns="${NET_DNS:-$(python3 -c "import yaml;c=yaml.safe_load(open('$WS/config/robot.yaml'));print(c['network'].get('dns','8.8.8.8'))")}"
  [ -n "$iface" ] || iface="$(ip -o link show | awk -F': ' '$2 ~ /^(en|eth)/ {print $2; exit}')"
  [ -n "$iface" ] || die "no wired interface found; set NET_IFACE"
  log "static $ip/24 on $iface"
  if command -v nmcli >/dev/null && nmcli -t -f DEVICE,STATE dev status 2>/dev/null | grep -q "^$iface:connected"; then
    local con; con="$(nmcli -t -f DEVICE,CONNECTION dev status | awk -F: -v d="$iface" '$1==d {print $2}')"
    nohup bash -c "sleep 2; sudo -n nmcli con mod '$con' ipv4.method manual ipv4.addresses $ip/24 ipv4.gateway '$gw' ipv4.dns '$dns' connection.autoconnect yes && sudo -n nmcli con up '$con'" >/tmp/netswitch.log 2>&1 </dev/null &
    log "address change in 2 s (this session drops); reconnect to $ip"
    return 0
  fi
  sed -e "s/__IFACE__/$iface/" -e "s|192.168.11.80/24|$ip/24|" "$HERE/netplan/01-robot-disco.yaml" | sudo tee /etc/netplan/01-robot-disco.yaml >/dev/null
  sudo chmod 600 /etc/netplan/01-robot-disco.yaml
  for f in /etc/netplan/*.yaml; do [ "$f" = /etc/netplan/01-robot-disco.yaml ] && continue; grep -q "$iface" "$f" && sudo mv "$f" "$f.disabled-by-provision" || true; done
  sudo netplan generate && sudo netplan apply
  ip -br addr show "$iface"
}

PHASES="preflight jetson base isaac ros isaacpkgs groups udev sysctl data workspace venvs audio services llm display network"
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
