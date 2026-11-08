"""Process helpers every node shares: die with the launch that started us.

Why: a `ros2 launch` that is killed hard (SIGKILL, a dead ssh session, a supervisor `killasgroup` that
missed a child) closes its children's stdout pipes. A C++ node dies of SIGPIPE at its next log line, but
Python ignores SIGPIPE, so a quiet Python node outlives the launch and keeps its name, its topics and
its UDP ports. Eight stopped sim stacks on the GPU box left 23 such nodes behind on 2026-11-07 (hand,
description_publisher, go_to_landmark, scan_relay), and the next launch then had nine hand nodes.

`exit_with_parent()` asks the kernel to deliver a signal when the parent exits (Linux `PR_SET_PDEATHSIG`),
so the node ends the moment its launch does, whatever killed the launch. Call it first thing in `main()`.
"""
from __future__ import annotations

import ctypes
import os
import signal
import sys

PR_SET_PDEATHSIG = 1  # linux/prctl.h


def exit_with_parent(sig: int = signal.SIGTERM) -> bool:
    """Deliver `sig` to this process when its parent exits. Returns True when the kernel accepted it.

    SIGTERM, not SIGKILL: rclpy's handler turns it into a clean shutdown (ExternalShutdownException out of
    spin, the node's `finally:` runs). No-op off Linux. If the parent is already gone (we were orphaned
    between fork and here) the signal is sent at once, since the kernel only fires it on a later exit.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, int(sig), 0, 0, 0) != 0:
            return False
    except (OSError, AttributeError):
        return False
    if os.getppid() == 1:
        os.kill(os.getpid(), sig)
    return True
