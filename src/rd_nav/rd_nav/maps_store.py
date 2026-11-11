"""The robot's nav maps as files (v1 Scripts/maps_store.py, now the one copy rd_nav's launches and rd_manager share).

A map is `<name>.yaml` + its image (nav2 map_server format) in the maps folder; a map made or re-mapped with
slam_toolbox also has `<name>.posegraph` + `<name>.data`, which "edit via SLAM" needs. Two small files next
to them carry state between the manager, the launch files and supervisor:

    active_map          one line, the map's name: what localization.launch.py loads and what landmarks and
                        routes are tagged with
    slam_request.yaml   {mode: new | continue, name, requested, start_pose} - what slam.launch.py starts
                        slam_toolbox with

Nothing here talks to ROS.
"""
from __future__ import annotations

import io
import os
import re
import time
from pathlib import Path

import yaml

RESERVED = {"keepout_mask", "keepout_zones", "slam_request"}   # yaml files in the folder that are not nav maps
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_png_cache: dict = {}


def default_maps_dir() -> Path:
    """$RD_MAPS_DIR, else the package's maps folder: the REAL folder behind the symlink install, so files
    written later (active_map, new maps) are seen by every process."""
    env = os.environ.get("RD_MAPS_DIR")
    if env:
        return Path(env)
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("rd_nav")) / "maps"
    except Exception:  # noqa: BLE001 - no ROS: the source tree
        share = Path(__file__).resolve().parents[1] / "maps"
    for probe in ("demo.yaml", "keepout_mask.yaml", "active_map"):
        p = share / probe
        if p.is_symlink():
            return p.resolve().parent
    return share


def valid_name(name: str) -> bool:
    return bool(name) and bool(_NAME.match(name)) and name not in RESERVED


def _pgm_size(path: Path) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as f:
            if f.readline().strip() not in (b"P5", b"P2"):
                return None
            tok: list = []
            while len(tok) < 2:
                line = f.readline()
                if not line:
                    return None
                if not line.startswith(b"#"):
                    tok += line.split()
            return int(tok[0]), int(tok[1])
    except Exception:  # noqa: BLE001
        return None


class MapsStore:
    def __init__(self, maps_dir: Path | str | None = None):
        self.dir = Path(maps_dir) if maps_dir else default_maps_dir()
        self.active_file = self.dir / "active_map"
        self.slam_file = self.dir / "slam_request.yaml"

    # ---- naming / listing
    def yaml_path(self, name: str) -> Path:
        return self.dir / f"{name}.yaml"

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.yaml") if p.stem not in RESERVED)

    def active(self) -> str | None:
        try:
            n = self.active_file.read_text().strip()
            if n and self.yaml_path(n).exists():
                return n
        except FileNotFoundError:
            pass
        names = self.names()
        return "demo" if "demo" in names else (names[0] if names else None)

    def active_yaml(self) -> Path | None:
        a = self.active()
        return self.yaml_path(a) if a else None

    def set_active(self, name: str) -> None:
        if not self.yaml_path(name).exists():
            raise FileNotFoundError(f"no map {name!r}")
        self.active_file.write_text(name + "\n")

    def meta(self, name: str) -> dict:
        return yaml.safe_load(self.yaml_path(name).read_text()) or {}

    def info(self, name: str, active: str | None) -> dict:
        y = self.yaml_path(name)
        m = self.meta(name)
        img = self.dir / str(m.get("image", f"{name}.pgm"))
        size = _pgm_size(img) if img.suffix.lower() == ".pgm" else None
        if size is None and img.exists():
            try:
                from PIL import Image
                size = Image.open(img).size
            except Exception:  # noqa: BLE001
                size = None
        res = float(m.get("resolution", 0.05))
        files = [y] + [p for p in (img, self.dir / f"{name}.posegraph", self.dir / f"{name}.data") if p.exists()]
        return {"name": name, "active": name == active, "resolution": res,
                "origin": [float(v) for v in (m.get("origin") or [0, 0, 0])[:2]],
                "width": size[0] if size else None, "height": size[1] if size else None,
                "metres": [round(size[0] * res, 1), round(size[1] * res, 1)] if size else None,
                "image": img.name, "posegraph": (self.dir / f"{name}.posegraph").exists(),
                "bytes": sum(p.stat().st_size for p in files), "mtime": max(p.stat().st_mtime for p in files),
                "png": f"/api/maps/{name}.png"}

    def list(self) -> dict:
        active = self.active()
        out = []
        for n in self.names():
            try:
                out.append(self.info(n, active))
            except Exception as e:  # noqa: BLE001 - a broken yaml must not hide the others
                out.append({"name": n, "active": n == active, "error": f"{type(e).__name__}: {e}"})
        return {"maps": out, "active": active, "dir": str(self.dir)}

    # ---- images
    def png(self, name: str) -> bytes:
        from PIL import Image

        m = self.meta(name)
        img = self.dir / str(m.get("image", f"{name}.pgm"))
        key = (name, img.stat().st_mtime)
        if _png_cache.get("key") != key:
            buf = io.BytesIO()
            Image.open(img).convert("L").save(buf, "PNG")
            _png_cache.clear()
            _png_cache.update(key=key, png=buf.getvalue())
        return _png_cache["png"]

    def save_drawn(self, name: str, png_bytes: bytes, save_as: str | None = None) -> dict:
        """The drawing editor's result: a greyscale PNG the size of the map. Dark = occupied, light = free,
        anything in between = unknown (nav2's trinary convention). Written as <target>.pgm + .yaml."""
        import numpy as np
        from PIL import Image

        src = self.meta(name)
        target = save_as or name
        if not valid_name(target):
            raise ValueError(f"bad map name {target!r}")
        im = Image.open(io.BytesIO(png_bytes)).convert("L")
        a = np.asarray(im, dtype=np.uint8)
        src_img = self.dir / str(src.get("image", f"{name}.pgm"))
        size = _pgm_size(src_img)
        if size and (a.shape[1], a.shape[0]) != size:
            raise ValueError(f"image is {a.shape[1]}x{a.shape[0]}, the map is {size[0]}x{size[1]}")
        out = np.full(a.shape, 205, np.uint8)
        out[a <= 100] = 0
        out[a >= 200] = 254
        pgm = self.dir / f"{target}.pgm"
        with open(pgm, "wb") as f:
            f.write(f"P5\n# {target}: drawn with the manager's map editor on {time.strftime('%Y-%m-%d %H:%M')}\n"
                    f"{a.shape[1]} {a.shape[0]}\n255\n".encode())
            f.write(out.tobytes())
        meta = dict(src)
        meta["image"] = pgm.name
        meta.setdefault("mode", "trinary")
        meta.setdefault("negate", 0)
        meta.setdefault("occupied_thresh", 0.65)
        meta.setdefault("free_thresh", 0.25)
        self.yaml_path(target).write_text(yaml.safe_dump(meta, sort_keys=False))
        _png_cache.clear()
        return self.info(target, self.active())

    # ---- delete
    def delete(self, name: str) -> list[str]:
        if name == self.active():
            raise PermissionError(f"{name!r} is the active map; activate another one first")
        y = self.yaml_path(name)
        if not y.exists():
            raise FileNotFoundError(f"no map {name!r}")
        m = self.meta(name)
        gone = []
        for p in (self.dir / str(m.get("image", f"{name}.pgm")), self.dir / f"{name}.posegraph",
                  self.dir / f"{name}.data", y):
            if p.exists() and p.stem not in RESERVED:
                p.unlink()
                gone.append(p.name)
        _png_cache.clear()
        return gone

    # ---- SLAM request (read by slam.launch.py when the unit starts)
    def slam_request(self) -> dict | None:
        try:
            return yaml.safe_load(self.slam_file.read_text()) or None
        except FileNotFoundError:
            return None

    def set_slam_request(self, mode: str, name: str, start_pose=None) -> dict:
        """`start_pose` (continue only) is [x, y, yaw] in the map's own frame: where the robot stands now.
        Without it slam_toolbox resumes at the posegraph's dock, and unless the robot is parked there every
        new scan merges in at an offset."""
        if mode not in ("new", "continue"):
            raise ValueError("mode must be new or continue")
        if not valid_name(name):
            raise ValueError(f"bad map name {name!r}")
        if mode == "continue" and not (self.dir / f"{name}.posegraph").exists():
            raise FileNotFoundError(f"{name!r} has no posegraph to continue from")
        req = {"mode": mode, "name": name, "requested": time.strftime("%Y-%m-%d %H:%M:%S")}
        if mode == "continue" and start_pose is not None:
            try:
                x, y, yaw = (float(v) for v in start_pose)
            except (TypeError, ValueError):
                raise ValueError(f"start_pose must be [x, y, yaw], got {start_pose!r}") from None
            req["start_pose"] = [round(x, 4), round(y, 4), round(yaw, 4)]
        self.slam_file.write_text("# written by rd_manager (Maps tab); slam.launch.py reads it at start\n"
                                  + yaml.safe_dump(req, sort_keys=False))
        return req

    def clear_slam_request(self) -> None:
        try:
            self.slam_file.unlink()
        except FileNotFoundError:
            pass

    def slam_overrides(self) -> dict:
        """slam_toolbox parameters for the pending request: a fresh map, or continuing a posegraph."""
        out: dict = {"mode": "mapping", "map_file_name": "", "map_start_at_dock": False}
        req = self.slam_request()
        if req and req.get("mode") == "continue" and req.get("name") \
                and (self.dir / f"{req['name']}.posegraph").exists():
            out = {"mode": "mapping", "map_file_name": str(self.dir / req["name"])}
            sp = req.get("start_pose")
            if isinstance(sp, (list, tuple)) and len(sp) == 3:
                out["map_start_pose"] = [float(v) for v in sp]
            else:
                out["map_start_at_dock"] = True
        return out
