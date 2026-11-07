"""Write the flat URDF Isaac Sim's importer takes (package:// mesh URIs resolved to absolute paths), from
rd_description's xacro.    export_urdf [out.urdf] [--sim-mode]"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def package_path(pkg: str) -> str:
    from ament_index_python.packages import get_package_share_directory
    return get_package_share_directory(pkg)


def flat_urdf(sim_mode: bool = True) -> str:
    xacro = os.path.join(package_path("rd_description"), "urdf", "robot.urdf.xacro")
    urdf = subprocess.run(["xacro", xacro, f"sim_mode:={'true' if sim_mode else 'false'}", "base_control:=false", "ur_control:=false"],
                          check=True, capture_output=True, text=True).stdout
    cache: dict = {}

    def repl(m):
        pkg, rel = m.group(1), m.group(2)
        if pkg not in cache:
            try:
                cache[pkg] = package_path(pkg)
            except Exception:  # noqa: BLE001
                cache[pkg] = None
        return f'filename="{cache[pkg]}/{rel}"' if cache[pkg] else m.group(0)
    return re.sub(r'filename="package://([^/"]+)/([^"]+)"', repl, urdf)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = next((a for a in argv if not a.startswith("--")), "/tmp/robot_disco.urdf")
    text = flat_urdf(sim_mode="--real" not in argv)
    with open(out, "w") as f:
        f.write(text)
    print(f"{out}: {len(text)} bytes, {text.count('<link ')} links, {text.count('package://')} unresolved package URIs")
    return 0
