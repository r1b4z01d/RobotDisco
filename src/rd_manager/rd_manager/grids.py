"""Maps and costmaps as PNGs for the pages (v1 route_editor.py load_map / serve_grid / costmap_png)."""
from __future__ import annotations

import io
import math
import time
from pathlib import Path

import numpy as np
import yaml

COSTMAP_COLORS = {"global": "ff3caa", "local": "8c32ff"}


def _hex_rgb(h):
    h = (h or "").strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return None


class MapView:
    """The map the pages draw: a file map (the active one) or slam_toolbox's live grid while mapping."""

    def __init__(self):
        self.png = b""
        self.meta = None
        self.rev = 0
        self._cm_cache: dict = {}

    def load_file(self, yaml_path: Path) -> dict:
        from PIL import Image
        meta = yaml.safe_load(Path(yaml_path).read_text())
        im = Image.open(Path(yaml_path).parent / meta["image"]).convert("L")
        buf = io.BytesIO()
        im.save(buf, "PNG")
        self.png = buf.getvalue()
        self.rev += 1
        self.meta = {"resolution": float(meta["resolution"]), "origin": [float(meta["origin"][0]), float(meta["origin"][1])],
                     "width": im.size[0], "height": im.size[1], "name": Path(yaml_path).name, "image": "/map.png", "rev": self.rev}
        return self.meta

    def load_grid(self, msg, name: str) -> dict:
        from PIL import Image
        a = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        px = np.full(a.shape, 205, np.uint8)
        px[a == 0] = 254
        px[a >= 65] = 0
        mid = (a > 0) & (a < 65)
        px[mid] = (254 - a[mid] * 2.5).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(np.flipud(px), "L").save(buf, "PNG")
        self.png = buf.getvalue()
        self.rev += 1
        self.meta = {"resolution": float(msg.info.resolution), "origin": [float(msg.info.origin.position.x), float(msg.info.origin.position.y)],
                     "width": int(msg.info.width), "height": int(msg.info.height), "name": name, "image": "/map.png", "rev": self.rev, "live": True}
        return self.meta

    def costmap_png(self, name: str, entry: dict, frame_tf: dict, topic: str, color=None):
        """(png, georeferenced meta) of a latched costmap message; rendered once per message and colour."""
        from PIL import Image
        m = entry["msg"]
        rgb = _hex_rgb(color) or _hex_rgb(COSTMAP_COLORS.get(name, "ff3caa"))
        chex = "%02x%02x%02x" % rgb
        st = (m.header.stamp.sec, m.header.stamp.nanosec, m.info.width, m.info.height, chex)
        c = self._cm_cache.get(name)
        if not c or c["stamp"] != st:
            w, h = m.info.width, m.info.height
            d = np.array(m.data, dtype=np.int16).reshape(h, w)
            rgba = np.zeros((h, w, 4), np.uint8)
            g = (d > 0) & (d < 99)
            t = (d[g] / 98.0)[:, None]
            pale = 0.25 * np.array(rgb, float) + 0.75 * 255.0
            rgba[g, :3] = (pale[None, :] * (1 - t) + np.array(rgb, float)[None, :] * t).clip(0, 255).astype(np.uint8)
            rgba[g, 3] = 255
            rgba[d == 99] = (*rgb, 255)
            rgba[d == 100] = (*(int(v * 0.6) for v in rgb), 255)
            buf = io.BytesIO()
            Image.fromarray(rgba[::-1]).save(buf, "PNG")
            q = m.info.origin.orientation
            base = {"name": name, "topic": topic, "resolution": float(m.info.resolution), "frame": m.header.frame_id or "map",
                    "origin_frame": [float(m.info.origin.position.x), float(m.info.origin.position.y)],
                    "origin_yaw": math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                    "width": w, "height": h, "stamp": f"{st[0]}.{st[1]:09d}",
                    "image": f"/costmap/{name}.png?v={st[0]}.{st[1]}&color={chex}", "color": chex}
            c = self._cm_cache[name] = {"stamp": st, "png": buf.getvalue(), "meta": base}
        meta = dict(c["meta"])
        meta["age_s"] = round(time.time() - entry["t"], 1)
        ox, oy = meta["origin_frame"]
        if meta["frame"] == "map":
            meta["origin"], meta["yaw"] = [ox, oy], meta["origin_yaw"]
        else:
            tf = frame_tf.get(meta["frame"])
            if tf is None:
                meta["error"] = f"no TF map->{meta['frame']} yet (localization not initialised?)"
            else:
                tx, ty, tyaw = tf
                cs, sn = math.cos(tyaw), math.sin(tyaw)
                meta["origin"] = [round(tx + ox * cs - oy * sn, 4), round(ty + ox * sn + oy * cs, 4)]
                meta["yaw"] = meta["origin_yaw"] + tyaw
        return c["png"], meta
