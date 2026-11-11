#!/usr/bin/env python3
"""Soak the RoboClaw's packet-serial link with READ commands only and verify every reply's CRC.

    roboclaw_crc_soak.py /dev/ttyTHS1 115200 [count=1000] [hz=100] [address=128]

Answers the question roboclaw_probe_uart.py cannot: its battery reads only count bytes, and the JetPack 6 fault on the
header UART (receive DMA handing back zeroed bytes with the right length and framing) passes a length check. Here each
reply's CRC16 is checked and all-zero payloads are counted separately. Commands used: 21 version, 24 main battery,
78 both encoder counts, 49 motor currents, 90 status. Nothing is written to the controller's settings or motors.

2026-11-10, robot on JetPack 7.2.1 (kernel 6.8.12), STOCK device tree (DMA still bound to serial@3100000):
3200 replies at 100 rounds a second, 0 CRC errors, 0 short, 0 zeroed. The PIO device tree v1 needed is not needed here.
"""
import os, select, struct, sys, termios, time

dev, baud = sys.argv[1], int(sys.argv[2])
count = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
hz = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
addr = int(sys.argv[5]) if len(sys.argv) > 5 else 128
fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
a = termios.tcgetattr(fd)
a[0] = 0; a[1] = 0; a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL; a[3] = 0
a[4] = a[5] = getattr(termios, "B%d" % baud)
termios.tcsetattr(fd, termios.TCSANOW, a)


def crc16(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def ask(cmd, nbytes, timeout=0.2):
    """Send [addr, cmd], read nbytes (payload + 2 CRC bytes), or up to the first NUL + 2 for the version string."""
    termios.tcflush(fd, termios.TCIOFLUSH)
    os.write(fd, bytes([addr, cmd]))
    buf, t0 = b"", time.monotonic()
    while time.monotonic() - t0 < timeout:
        if nbytes is not None and len(buf) >= nbytes:
            break
        if nbytes is None and b"\x00" in buf and len(buf) >= buf.index(b"\x00") + 3:
            buf = buf[:buf.index(b"\x00") + 3]
            break
        if select.select([fd], [], [], 0.02)[0]:
            try:
                buf += os.read(fd, 256)
            except BlockingIOError:
                pass
    return buf


CMDS = {21: ("version", None), 24: ("main battery", 4), 78: ("encoders", 10), 49: ("currents", 6), 90: ("status", 6)}
stats = {c: {"ok": 0, "crc": 0, "short": 0, "zero": 0} for c in CMDS}
first_bad = []
period = 1.0 / hz
t_start = time.monotonic()
for i in range(count):
    for cmd, (name, n) in CMDS.items():
        if cmd in (21, 90) and i % 10:        # the long version string and the status every tenth round
            continue
        r = ask(cmd, n)
        s = stats[cmd]
        if (n is not None and len(r) < n) or len(r) < 3:
            s["short"] += 1
            if len(first_bad) < 5: first_bad.append((i, name, "short", r.hex()))
            continue
        payload, got = r[:-2], struct.unpack(">H", r[-2:])[0]
        if crc16(bytes([addr, cmd]) + payload) != got:
            s["crc"] += 1
            if not any(r): s["zero"] += 1
            if len(first_bad) < 5: first_bad.append((i, name, "crc", r.hex()))
        else:
            s["ok"] += 1
    delay = t_start + (i + 1) * period - time.monotonic()
    if delay > 0:
        time.sleep(delay)
dt = time.monotonic() - t_start
total = sum(sum(v.values()) - v["zero"] for v in stats.values())
bad = sum(v["crc"] + v["short"] for v in stats.values())
for cmd, (name, _) in CMDS.items():
    v = stats[cmd]
    print(f"  {name:13s} ok {v['ok']:5d}  crc errors {v['crc']}  short {v['short']}  all-zero {v['zero']}")
v = ask(21, None)
print(f"{total} replies in {dt:.1f} s ({count / dt:.0f} rounds/s): {bad} bad;", "version", repr(v[:-3].decode(errors='replace')))
for b in first_bad:
    print("  first failures:", b)
sys.exit(1 if bad else 0)
