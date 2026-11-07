"""A landmark's photo: the scene and side cameras stitched side by side (and the RealSense frame), from the
bridge's JPEG cache (v1 route_editor.py stitch_jpegs / capture_landmark)."""
from __future__ import annotations

import io
from pathlib import Path

LANDMARK_CAMS = ("scene", "side", "realsense")


def _label_font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stitch_jpegs(parts: list) -> bytes | None:
    from PIL import Image, ImageDraw
    ims = []
    for label, data in parts:
        if data:
            try:
                ims.append((label, Image.open(io.BytesIO(data)).convert("RGB")))
            except Exception:  # noqa: BLE001
                pass
    if not ims:
        return None
    h = max(im.size[1] for _, im in ims)
    ims = [(lb, im if im.size[1] == h else im.resize((round(im.size[0] * h / im.size[1]), h))) for lb, im in ims]
    out = Image.new("RGB", (sum(im.size[0] for _, im in ims), h))
    draw, font, x = ImageDraw.Draw(out), _label_font(max(14, h // 22)), 0
    for lb, im in ims:
        out.paste(im, (x, 0))
        draw.text((x + 10, 10), lb, fill=(0, 0, 0), font=font)
        draw.text((x + 8, 8), lb, fill=(255, 255, 255), font=font)
        x += im.size[0]
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def capture_landmark(lid: str, img_dir: Path, pose: dict | None, jpegs: dict, fresh_s: float = 3.0, want_pose=True, want_images=True) -> dict:
    """What a landmark takes from the robot right now. LookupError = no pose."""
    import time
    out: dict = {}
    if want_pose:
        if pose is None:
            raise LookupError("no robot pose")
        out.update(pose)
    if want_images:
        now = time.time()
        frames = {n: (jpegs.get(n) or {}).get("data") if now - (jpegs.get(n) or {}).get("t", 0) < fresh_s else None for n in LANDMARK_CAMS}
        img_dir.mkdir(parents=True, exist_ok=True)
        images = {"stitched": None, "realsense": None}
        stitched = stitch_jpegs([("scene", frames["scene"]), ("side", frames["side"])])
        if stitched:
            (img_dir / f"{lid}_stitched.jpg").write_bytes(stitched)
            images["stitched"] = f"{lid}_stitched.jpg"
        if frames["realsense"]:
            (img_dir / f"{lid}_realsense.jpg").write_bytes(frames["realsense"])
            images["realsense"] = f"{lid}_realsense.jpg"
        out["images"] = images
        out["cameras"] = [n for n in LANDMARK_CAMS if frames[n]]
        out["missing"] = [n for n in LANDMARK_CAMS if not frames[n]]
    return out
