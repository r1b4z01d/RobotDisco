# Provisioning the 5.0 robot PC

Turns a Jetson AGX Orin flashed with **JetPack 7.2.1** (Jetson Linux 39.2.1, Ubuntu 24.04, CUDA 13.2, root
filesystem on the 512 GB NVMe) into the Robot Disco robot PC: user `rd`, hostname `robotdisco`,
`192.168.11.80` on the isolated robot subnet, ROS 2 Jazzy + Isaac ROS 4.6 as host packages, this repo
built at `~/RobotDisco`, everything under supervisor.

**Status 2026-11-06:** skeleton. `provision.sh` runs `preflight`, `base`, `ros`, `groups`, `udev`, `sysctl`
and `workspace`; the other phases print what they must do and skip. They get written on the freshly flashed
robot in phase 1 of the plan, ported from v1's `rd_ws/provision/` (which also keeps the USB port photos,
the xpad rebuild and the two-OpenCV patch that 5.0 no longer needs).

## Order of operations

1. **Phase 0, before the reflash** (the NVMe reflash destroys `/mnt/data`): on the workstation, with the v1
   robot still up, `rd_ws/provision/capture_rd_state.sh` for the snapshot, then copy `/mnt/data`
   (datasets, checkpoints, evals, galleries, maps) to the GPU box and verify it. Tag `rd_ws` as v1.
2. **Flash** with NVIDIA's Jetson ISO, their recommended path for the AGX Orin devkit since JetPack 7.2 (no host PC;
   procedure from the devkit user guide's Quick Start, read 2026-11-09; the robot is on firmware 36.5.2, above the
   r35.5 the installer needs):
   1. Write `jetsoninstaller-r39.2.1-2026-08-07-18-30-47-arm64.iso` (5.0 GB, linked from the Quick Start) to a USB
      stick of 8 GB or more with Balena Etcher ("Flash from file"); a plain file copy does not make it bootable.
   2. Display and keyboard straight into the robot (no KVM), stick in, power on. The boot order is NVMe, then eMMC,
      so v1 comes up unless the stick is picked in the UEFI boot manager (the boot screen names the key, Esc).
   3. Answer `Y` to the QSPI capsule update; it runs twice. Do not cut power during it.
   4. In the installer's menu: `Install Jetson ISO r39.2.1`, then `Install on NVMe`. This erases the NVMe
      (`/mnt/data`). Several minutes of scrolling text, then it reboots: pull the stick.
   5. First-boot setup: user `rd`, computer name `robotdisco` (provision.sh refuses any other user).
   6. Check: `cat /etc/nv_tegra_release` says `R39 (release), REVISION: 2.1`, and `findmnt -no SOURCE /` names the NVMe.

   The JetPack components (CUDA, cuDNN, TensorRT) are not in the image: `sudo apt update && sudo apt dist-upgrade`,
   reboot, `sudo apt install nvidia-jetpack` (about an hour). Do not count on the eMMC's JetPack 6 as a fallback
   afterwards: the firmware moves to r39, and NVIDIA does not say what the NVMe install leaves on the eMMC. The
   backup is the fallback. If the installer loops (two forum reports on other units), flash from an x86 Ubuntu
   host instead (NVIDIA names 20.04 and 22.04; SDK Manager also lists 24.04): Force Recovery (hold the middle
   button, press and release Power; `lsusb` shows `0955:7023`), a cable from the host to the USB-C port beside the
   40-pin header, and in the release's `Linux_for_Tegra`, after `tools/l4t_flash_prerequisites.sh` and
   `apply_binaries.sh`: `sudo ./l4t_initrd_flash.sh --external-device nvme0n1p1 -c tools/kernel_flash/flash_l4t_t234_nvme.xml --erase-all jetson-agx-orin-devkit external`.
   Give the box an uplink with internet (second Ethernet or USB tether); the robot subnet has none.
3. `git clone https://github.com/r1b4z01d/RobotDisco.git ~/RobotDisco && ~/RobotDisco/provision/provision.sh all`
4. Reboot, then `provision.sh network` and move the cable to the robot subnet.
5. `SNAPSHOT=~/snapshot-YYYYMMDD provision.sh workspace audio` restores maps, galleries, motions and music.

| Phase | Does |
|---|---|
| `preflight` | user, sudo, Ubuntu 24.04 / aarch64 / JetPack present, root on NVMe |
| `jetson` | `jetson/apt-packages-jetson.txt`, nvpmodel MAXN at boot, jetson_clocks service, xpad at boot (+ the rebuilt module if kernel 6.8 still lacks the 8BitDo ID), jtop, the ttyTHS1 PIO device tree for the RoboClaw link (`uart1_pio_dtb`, see *RoboClaw on the header UART*) |
| `base` | `apt-packages.txt`, ssh + supervisor + chrony, hostname, `/etc/hosts` for ur5e / amazinghand / gpu, `AUTHORIZED_KEY` |
| `ros` | ROS 2 Jazzy repo + `ros-packages.txt`, rosdep, `.bashrc`: Jazzy + workspace, Cyclone DDS with `cyclonedds.xml`, `ROS_DOMAIN_ID=0`, `RD_ROBOT_CONFIG` |
| `isaac` | NVIDIA's Isaac ROS 4.6 apt repo, `isaac-packages.txt` via the Isaac ROS CLI bare-metal mode |
| `groups` | dialout video input plugdev audio render docker |
| `udev` | `udev/99-robot-disco.rules`: SpaceMouse, RealSense, webcams, Stream Deck, gamepads, 8BitDo new_id fallback |
| `sysctl` | DDS receive buffers |
| `workspace` | clone / update `~/RobotDisco`, submodules, `apply_patches.sh` (the RoboClaw driver's UART patches), rosdep, `colcon build --symlink-install` |
| `audio` | `pulse/default.pa` (C-Media adapter = default sink, no auto-switch), unmute, `alsactl store` |
| `services` | supervisor web UI / XML-RPC on `:9999`, the units from `src/rd_bringup/services/` |
| `llm` | Ollama under supervisor on loopback `:11434`, `qwen2.5:7b` |
| `display` | GNOME autologin, no blanking, the face window |
| `network` | static `192.168.11.80/24` (nmcli on the Jetson desktop, `netplan/01-robot-disco.yaml` otherwise) |

Files: `apt-packages.txt`, `ros-packages.txt`, `isaac-packages.txt`, `udev/`, `netplan/`,
`cyclonedds.xml` (the loopback-only DDS profile v1 settled on), `patches/` (applied to `vendor/` submodules by
`apply_patches.sh`, which the rig and the GPU box run too), `tools/` (RoboClaw and UART probes). The GPU box's
provisioning is `gpu/provision_gpu.sh` (see `gpu/README.md`); the day-to-day deploy script is `deploy_to_rd.sh`.

## RoboClaw on the header UART (since 2026-11-08, v1 change carried over)

The base's RoboClaw 2x15A (firmware 4.4.9) no longer hangs off USB: its USB jack is mechanically bad (enumerates
only when pressed; "device descriptor read/64, error -32" on every Jetson port), and once the board shared a
DC-DC converter with the audio amp, plugging that USB in reset the Jetson twice (ground offset through the
cable). It talks packet serial on the Jetson's 40-pin header: **pin 8 (TX) -> S1, pin 10 (RX) <- S2, pin 6 GND**,
3.3 V levels, `/dev/ttyTHS1` (`serial@3100000`, `uarta`). `robot.yaml sensors.roboclaw` holds port, baud and
address; rd_base's launch hands them to the xacro. Three things make it work:

1. **RoboClaw config**: packet serial mode (7 = address 0x80), 115200, and a 1 s serial timeout as a watchdog
   (the board otherwise keeps the last speed when the link dies). Set and saved with **Motion Studio over USB**
   (which also took the firmware from 4.3.6 to 4.4.9; verified to survive a power cycle), or with the buttons
   (MODE -> SET to 7 -> LIPO, then SET menu option 6 -> LIPO). Over packet serial the settings only changed in
   RAM on 4.3.6 (save command 94 never acknowledged), so `tools/roboclaw_config.py` is for reading the config and
   RAM-only changes. `tools/roboclaw_probe_uart.py /dev/ttyTHS1 115200` is the link test (`tools/uart_capture.py`
   shows the raw reply, `tools/uart_loopback.py` checks pins 8/10 with a jumper).
2. **Jetson device tree** (phase `jetson`, `uart1_pio_dtb`): on JetPack 6 the UART's receive DMA returned zeroed
   bytes (right length and framing, DMA-read bytes 0x00; NVIDIA forum "DMA on /dev/ttyTHS1 corrupts receiving
   data"). `ttyTHS2` ships without DMA bindings and is fine, so the stock `/boot/dtb/kernel_tegra234-*.dtb` is
   copied to `...-uart1-pio.dtb` with `dmas`/`dma-names` removed from `serial@3100000` and booted via an `FDT`
   line in `/boot/extlinux/extlinux.conf` (what jetson-io does; a `stock` entry stays as the fallback). The
   function only acts when the stock DTB binds DMA to that UART, so on JetPack 7.2 (kernel 6.8) it may find
   nothing to do. Either way `dmesg | grep 3100000.serial` must say `RX in PIO mode`, and the phase re-runs
   after a kernel/dtb package update (it rebuilds the copy when the stock file is newer).
3. **Driver patches** (`patches/roboclaw_serial-baud-rate.patch`, `patches/roboclaw_hardware_interface-baud-rate.patch`,
   applied by `apply_patches.sh`): a `baud_rate` hardware parameter (the upstream driver never set the tty speed,
   USB CDC did not care), a failed open fails `on_init` instead of aborting controller_manager on the first read,
   and a reply is read until the line goes quiet (USB delivered a reply in one `read()`, a UART delivers it in
   pieces, which the CRC check then rejected).

Traps: never open a `ttyTHS*` below 9600 baud. The Tegra driver enables the FIFO before it sets the speed on the
next open, and a 2400-baud leftover (uart clock 38400 Hz) makes every open fail with `Failed to enable FIFO mode:
-110` until `echo 1843200 | sudo tee /sys/kernel/debug/bpmp/debug/clk/uarta/rate` (no reboot needed). After a
RoboClaw link loss the controller manager still has to be restarted by hand (`sudo supervisorctl restart rd_base`);
`base.launch.py` deliberately does not respawn it.
