"""Build the Nav2 keepout mask (keepout_mask.pgm + .yaml) from keepout_zones.yaml.

The mask has the SAME size, resolution and origin as the navigation map it names, so the KeepoutFilter can
look cells up 1:1. Cells inside a zone are black (occupied -> lethal cost), everything else white (free).

    rd_keepout [maps_dir]        # default: the package's maps folder (the real one behind the symlink install)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


def pgm_header(path: Path) -> tuple[int, int]:
    """(width, height) of a P5 PGM."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5", "map image must be binary PGM"
        tok: list[bytes] = []
        while len(tok) < 3:
            line = f.readline()
            if not line.startswith(b"#"):
                tok += line.split()
        return int(tok[0]), int(tok[1])


def inside(x: float, y: float, poly: list) -> bool:
    r = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            r = not r
    return r


def build_mask(maps_dir: Path, zones_file: str = "keepout_zones.yaml") -> dict:
    """Write keepout_mask.pgm/.yaml next to the zones file. Returns per-zone cell counts."""
    zones = yaml.safe_load((maps_dir / zones_file).read_text()) or {}
    map_yaml = maps_dir / str(zones.get("map", "demo.yaml"))
    meta = yaml.safe_load(map_yaml.read_text())
    res = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"]]
    w, h = pgm_header(maps_dir / str(meta["image"]))
    free, occ = 254, 0
    img = bytearray([free]) * (w * h)
    counts: dict[str, int] = {}
    for z in zones.get("zones") or []:
        poly = [(float(p[0]), float(p[1])) for p in z["polygon"]]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        c0, c1 = max(0, int((min(xs) - origin[0]) / res)), min(w - 1, int((max(xs) - origin[0]) / res))
        r0, r1 = max(0, int((min(ys) - origin[1]) / res)), min(h - 1, int((max(ys) - origin[1]) / res))
        n = 0
        for r in range(r0, r1 + 1):            # r counts up from the bottom row (map y); PGM rows go top-down
            y = origin[1] + (r + 0.5) * res
            for c in range(c0, c1 + 1):
                x = origin[0] + (c + 0.5) * res
                if inside(x, y, poly):
                    img[(h - 1 - r) * w + c] = occ
                    n += 1
        counts[str(z.get("name", "?"))] = n
    with open(maps_dir / "keepout_mask.pgm", "wb") as f:
        f.write(f"P5\n# keepout mask for {meta['image']}: black = no-go\n{w} {h}\n255\n".encode())
        f.write(bytes(img))
    (maps_dir / "keepout_mask.yaml").write_text(
        f"image: keepout_mask.pgm\nmode: trinary\nresolution: {res}\norigin: [{origin[0]}, {origin[1]}, 0]\n"
        f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return counts


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        d = Path(argv[0])
    else:
        from .maps_store import default_maps_dir
        d = default_maps_dir()
    counts = build_mask(d)
    for name, n in counts.items():
        print(f"{name}: {n} cells")
    print(f"wrote {d / 'keepout_mask.pgm'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
