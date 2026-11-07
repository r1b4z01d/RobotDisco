"""Write the flat URDF Isaac Sim's importer takes (package:// mesh URIs resolved to absolute paths, a token mass on
the massless root link), from rd_description's xacro.    export_urdf [out.urdf] [--real]"""
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
    return with_root_mass(re.sub(r'filename="package://([^/"]+)/([^"]+)"', repl, urdf))


ROOT_INERTIAL = ('<inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.1"/>'
                 '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>')


def with_root_mass(urdf: str) -> str:
    """Isaac Sim's URDF converter (6.2) treats a massless root link as a world attachment: every child body becomes
    its own articulation and the wheel joints end up jointed to the world (seen on the first import of
    rd_description, whose base_link is an empty frame). A token mass on the root link keeps one articulation
    rooted at base_link, which is also the frame the odometry reports."""
    links = re.findall(r'<link\s+name="([^"]+)"', urdf)
    children = set(re.findall(r'<child\s+link="([^"]+)"', urdf))
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        return urdf
    m = re.search(rf'<link\s+name="{re.escape(roots[0])}"\s*(/>|>)', urdf)
    if not m:
        return urdf
    if m.group(1) == "/>":
        return urdf[:m.start()] + f'<link name="{roots[0]}">{ROOT_INERTIAL}</link>' + urdf[m.end():]
    end = urdf.index("</link>", m.end())
    if "<inertial" in urdf[m.end():end]:
        return urdf
    return urdf[:m.end()] + ROOT_INERTIAL + urdf[m.end():]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = next((a for a in argv if not a.startswith("--")), "/tmp/robot_disco.urdf")
    text = flat_urdf(sim_mode="--real" not in argv)
    with open(out, "w") as f:
        f.write(text)
    print(f"{out}: {len(text)} bytes, {text.count('<link ')} links, {text.count('package://')} unresolved package URIs")
    return 0
