"""The robot's URDF with the UR driver's servo settings made ours.

    python3 -m rd_arm.robot_urdf --servoj-gain 300 --servoj-lookahead 0.1 <xacro file> [xacro args...]

ur_robot_driver's ros2_control macro hard-codes servoj_gain 2000 and servoj_lookahead_time 0.03 for the streamed
position interface. Those are the stiffest and shortest the UR accepts: every 8 ms step of a streamed target is
chased at once, and with a policy's targets (a new one every 83 ms, jittering by centimetres) the arm was loud
enough to sound broken (the robot, 2026-11-13). v1 streamed with gain 300 and lookahead 0.1 s. This expands the
xacro and rewrites the two parameters; arm.launch.py uses it in place of a bare `xacro`.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys


def with_servoj(urdf: str, gain: float, lookahead: float) -> tuple[str, int]:
    """(the URDF with both parameters set, how many were replaced)."""
    urdf, a = re.subn(r'(<param name="servoj_gain">)[^<]*(</param>)', rf"\g<1>{gain:g}\g<2>", urdf)
    urdf, b = re.subn(r'(<param name="servoj_lookahead_time">)[^<]*(</param>)', rf"\g<1>{lookahead:g}\g<2>", urdf)
    return urdf, a + b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servoj-gain", type=float, default=300.0)
    ap.add_argument("--servoj-lookahead", type=float, default=0.1)
    ap.add_argument("xacro", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if not 100.0 <= args.servoj_gain <= 2000.0 or not 0.03 <= args.servoj_lookahead <= 0.2:
        print("servoj gain must be 100..2000 and lookahead 0.03..0.2 s (the UR's own limits)", file=sys.stderr)
        return 2
    r = subprocess.run(["xacro", *args.xacro], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        return r.returncode
    urdf, n = with_servoj(r.stdout, args.servoj_gain, args.servoj_lookahead)
    if n == 0 and "ur_robot_driver" in r.stdout and "mock_components" not in r.stdout:
        print("robot_urdf: the UR hardware has no servoj parameters to set (did the driver's macro change?)", file=sys.stderr)
    sys.stdout.write(urdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
