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
#   provision.sh preflight jetson base ros isaac groups udev sysctl data workspace venvs audio services llm display network
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
  # v1 jetson/provision_jetson.sh phase_jetson: boot into MAXN, pin the clocks, nvidia docker runtime, xpad, jtop.
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
  [ -f "$HERE/jetson/apt-packages-jetson.txt" ] && apt_install $(pkglist "$HERE/jetson/apt-packages-jetson.txt") || true
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
  # Isaac ROS 4.6 as host debs for JetPack 7.2 / Jazzy (https://nvidia-isaac-ros.github.io/v/release-4.6/getting_started/index.html):
  # NVIDIA's apt repo, then isaac-packages.txt. VERIFY the repo line against the 4.6 docs on the first run.
  log "Isaac ROS 4.6"
  if [ ! -f /etc/apt/sources.list.d/isaac-ros.list ]; then
    sudo curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key -o /usr/share/keyrings/isaac-ros-keyring.gpg || die "could not fetch the Isaac ROS repo key (internet?)"
    echo "deb [signed-by=/usr/share/keyrings/isaac-ros-keyring.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.6 $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      | sudo tee /etc/apt/sources.list.d/isaac-ros.list >/dev/null
    sudo apt-get update
  fi
  apt_install $(pkglist "$HERE/isaac-packages.txt") || warn "some Isaac packages did not install: check the 4.6 package names"
  # cuRobo / cuMotion for MoveIt and the VLA streamer's IK come with isaac-ros-cumotion; nothing else here
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
  for p in "$HERE"/patches/*.patch; do
    [ -f "$p" ] || continue
    local sub="${p##*/}"; sub="${sub%%-*}"
    if [ -d "$WS/vendor/$sub" ] && ! git -C "$WS/vendor/$sub" apply --check --reverse "$p" >/dev/null 2>&1; then
      git -C "$WS/vendor/$sub" apply "$p" && log "patched vendor/$sub with $(basename "$p")" || warn "patch $(basename "$p") did not apply"
    fi
  done
  cd "$WS"
  rosdep install --from-paths src vendor --ignore-src -r -y || warn "rosdep left something unresolved"
  colcon build --symlink-install
  true
}

phase_venvs() {
  # The three Python environments that do not mix with the ROS python: the ArcFace face-ID worker, the
  # Whisper ears, Kokoro TTS (v1's setup scripts, paths updated), and the LeRobot venv for rd_vla.
  log "venvs: face id, voice, kokoro, lerobot"
  [ "${WITH_FACEID:-1}" = 1 ] && "$HERE/setup_faceid.sh" || true
  [ "${WITH_VOICE:-1}" = 1 ] && WHISPER_MODEL="${WHISPER_MODEL:-base.en}" "$HERE/setup_voice.sh" "${WHISPER_MODEL:-base.en}" || true
  [ "${WITH_KOKORO:-1}" = 1 ] && "$HERE/setup_kokoro.sh" || true
  if [ ! -x /opt/vla/bin/python ]; then
    sudo python3 -m venv --system-site-packages /opt/vla && sudo chown -R "$RD_USER:$RD_USER" /opt/vla
  fi
  # torch for CUDA 13.2 on JetPack 7.2: NVIDIA's wheel index (the plan's risk #3: pin what works on the flashed box)
  /opt/vla/bin/pip install -q --upgrade pip
  /opt/vla/bin/pip install -q ${TORCH_INDEX:+--extra-index-url "$TORCH_INDEX"} "lerobot>=0.3,<0.5" || warn "lerobot/torch install failed: see the plan's risk about cu132 wheels (set TORCH_INDEX)"
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

PHASES="preflight jetson base ros isaac groups udev sysctl data workspace venvs audio services llm display network"
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
