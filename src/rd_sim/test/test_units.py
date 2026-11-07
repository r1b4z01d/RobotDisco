"""What can be tested without Isaac Sim: the flat URDF export and the map-to-walls conversion."""
import importlib.util
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def test_export_urdf_resolves_meshes(tmp_path):
    from rd_sim.export_urdf import flat_urdf
    text = flat_urdf(sim_mode=True)
    assert text.count("<link ") >= 20 and "package://" not in text
    assert 'filename="/' in text and "amazing_hand" in text


def test_map_boxes_from_zone_one():
    spec = importlib.util.spec_from_file_location("isaac_scene", PKG / "scripts" / "isaac_scene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    maps = PKG.parent / "rd_nav" / "maps"
    boxes, height = mod.map_boxes(maps, "zone_one")
    assert height == 2.5 and 50 < len(boxes) < 20000
    xs = [b[0] for b in boxes]
    assert min(xs) > -1.0 and max(xs) < 15.0
