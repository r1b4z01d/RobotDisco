"""ROS-free tests: maps store, landmark store, keepout mask builder."""
import shutil
from pathlib import Path

import pytest
import yaml

from rd_nav.keepout import build_mask, pgm_header
from rd_nav.landmarks import LandmarkStore, task_from_body
from rd_nav.maps_store import MapsStore, valid_name

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture
def maps(tmp_path):
    d = tmp_path / "maps"
    shutil.copytree(PKG / "maps", d)
    return MapsStore(d)


def test_maps_list_and_active(maps):
    assert maps.names() == ["zone_one"]
    assert maps.active() == "zone_one"
    info = maps.list()["maps"][0]
    assert info["active"] and info["width"] and info["height"] and info["resolution"] == 0.05
    assert maps.active_yaml().name == "zone_one.yaml"
    with pytest.raises(FileNotFoundError):
        maps.set_active("nope")


def test_maps_slam_request(maps):
    req = maps.set_slam_request("new", "kitchen")
    assert req["mode"] == "new" and maps.slam_request()["name"] == "kitchen"
    assert maps.slam_overrides() == {"mode": "mapping", "map_file_name": "", "map_start_at_dock": False}
    with pytest.raises(FileNotFoundError):
        maps.set_slam_request("continue", "zone_one")        # no posegraph in the rig copy
    (maps.dir / "zone_one.posegraph").write_bytes(b"x")
    maps.set_slam_request("continue", "zone_one", [1.0, 2.0, 0.5])
    o = maps.slam_overrides()
    assert o["map_file_name"].endswith("zone_one") and o["map_start_pose"] == [1.0, 2.0, 0.5]
    maps.clear_slam_request()
    assert maps.slam_request() is None
    assert not valid_name("keepout_mask") and valid_name("zone-2") and not valid_name("../x")


def test_maps_png_and_drawn(maps):
    png = maps.png("zone_one")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    out = maps.save_drawn("zone_one", png, save_as="zone_copy")
    assert out["name"] == "zone_copy" and (maps.dir / "zone_copy.pgm").exists()
    assert pgm_header(maps.dir / "zone_copy.pgm") == pgm_header(maps.dir / "zone_one.pgm")
    with pytest.raises(PermissionError):
        maps.delete("zone_one")
    assert "zone_copy.pgm" in maps.delete("zone_copy")


def test_keepout_mask(maps):
    counts = build_mask(maps.dir)
    assert set(counts) == {"top_box", "right_box", "bottom_L"} and all(n > 100 for n in counts.values())
    assert pgm_header(maps.dir / "keepout_mask.pgm") == pgm_header(maps.dir / "zone_one.pgm")
    meta = yaml.safe_load((maps.dir / "keepout_mask.yaml").read_text())
    assert meta["origin"][:2] == [-0.797, -9.6] and meta["resolution"] == 0.05


@pytest.fixture
def landmarks(tmp_path):
    d = tmp_path / "landmarks"
    shutil.copytree(PKG / "landmarks", d)
    return LandmarkStore(d)


def test_landmarks_find(landmarks):
    assert {i["id"] for i in landmarks.all()} == {"charger", "dining-room", "hallway"}
    assert landmarks.find("Dining Room")["id"] == "dining-room"
    assert landmarks.find("dining rom")["id"] == "dining-room"          # fuzzy
    assert landmarks.find("charger", map_name="zone_one")["id"] == "charger"
    assert landmarks.find("charger", map_name="other_map") is None
    assert landmarks.find("mars") is None


def test_landmarks_tasks(landmarks):
    lm, t = landmarks.find_task("robot disco, pick up tigger please")
    assert lm["id"] == "dining-room" and t["model"] == "tigger_1_merged_smolvla_1"
    assert landmarks.find_task("pick up tiger")[1]["id"] == "pick-up-tigger"   # fuzzy
    assert landmarks.find_task("make coffee") is None


def test_landmarks_edit_roundtrip(landmarks):
    new = landmarks.upsert({"name": "Left Bookshelf", "map": "zone_one", "x": 1, "y": 2, "yaw": 0.1})
    assert new["id"] == "left-bookshelf"
    again = landmarks.upsert({"name": "Left Bookshelf", "map": "zone_one", "x": 1, "y": 2, "yaw": 0.1})
    assert again["id"] == "left-bookshelf-2"
    t = landmarks.set_task("left-bookshelf", {"name": "Grab the cup.", "model": "cups_v3", "task": "grab the cup", "seconds": "45"})
    assert t == {"name": "grab the cup", "model": "cups_v3", "task": "grab the cup", "seconds": 45.0, "id": "grab-the-cup"}
    t2 = landmarks.set_task("left-bookshelf", {"keep_grasp": "true"}, task_id="grab-the-cup")
    assert t2["keep_grasp"] is True and t2["seconds"] == 45.0
    fresh = LandmarkStore(landmarks.dir)                  # re-read from disk
    assert fresh.find_task("grab the cup")[0]["id"] == "left-bookshelf"
    assert landmarks.delete_task("left-bookshelf", "grab-the-cup") and not landmarks.delete_task("left-bookshelf", "x")
    assert landmarks.delete("left-bookshelf-2") and landmarks.get("left-bookshelf-2") is None
    with pytest.raises(ValueError):
        task_from_body({"name": "x", "model": "m", "task": "t", "done_when": "maybe"})
