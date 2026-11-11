"""The face gallery without a camera: synthetic embeddings, enrol / match / rename-merge / forget / greeting
bookkeeping, and the JSON round trip."""
import json
import time

import numpy as np
import pytest

from rd_face.face_id import FaceGallery


def unit(seed, dim=128):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def gallery(tmp_path):
    return FaceGallery(str(tmp_path / "people.json"), max_embeddings_per_person=3, expected_dim=128, backend="test",
                       thumb_dir=str(tmp_path / "faces"))


def test_enroll_match_and_persist(gallery, tmp_path):
    a = gallery.enroll(unit(1))
    b = gallery.enroll(unit(2), name="Bryan")
    assert a.person_id != b.person_id and b.name == "Bryan" and a.label == a.person_id
    person, score = gallery.best_match(unit(2) + 0.01 * unit(9))
    assert person.person_id == b.person_id and score > 0.9
    person, score = gallery.best_match(unit(3))
    assert score < 0.5
    gallery.add_embedding(b, unit(4))
    gallery.add_embedding(b, unit(5))
    gallery.add_embedding(b, unit(6))
    assert len(b.embeddings) == 3                          # capped per person
    gallery.save()
    again = FaceGallery(str(tmp_path / "people.json"), expected_dim=128, backend="test")
    assert {p.person_id for p in again.people.values()} == {a.person_id, b.person_id}
    assert again.find("Bryan").person_id == b.person_id
    data = json.loads((tmp_path / "people.json").read_text())
    assert data


def test_rename_merges_and_forget(gallery):
    a = gallery.enroll(unit(1))
    b = gallery.enroll(unit(2), name="Bryan")
    merged = gallery.rename(a.person_id, "Bryan")
    assert merged.person_id == b.person_id and len(gallery.people) == 1
    assert gallery.find(a.person_id) is None
    assert gallery.rename("nobody", "x") is None
    assert gallery.forget("Bryan") and not gallery.people
    assert not gallery.forget("Bryan")


def test_greeting_bookkeeping(gallery):
    p = gallery.enroll(unit(1), name="Zoe")
    assert p.last_greeted == 0.0
    gallery.mark_greeted(p.person_id)
    assert time.time() - gallery.find("Zoe").last_greeted < 2.0
    assert gallery.reset_greeting(p.person_id) and gallery.find("Zoe").last_greeted == 0.0
    assert not gallery.reset_greeting("nope")
    assert any("Zoe" in line for line in gallery.summary())


def _scene(face_z, behind_z, tilt=0.0, w=640, h=480, box=(280, 180, 80, 100)):
    """A depth image: a face-sized patch at face_z; around it a surface at behind_z, tilted by `tilt` m per pixel."""
    import numpy as np
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    d = behind_z + tilt * (xs - (box[0] + box[2] / 2.0))
    x, y, bw, bh = box
    d[y - 10:y + bh + 10, x - 6:x + bw + 6] = face_z + tilt * (xs[y - 10:y + bh + 10, x - 6:x + bw + 6] - (x + bw / 2.0))
    return d.astype(np.float32), box


def test_liveness_tells_a_head_from_a_screen():
    import numpy as np
    from rd_face.liveness import measure, refusal
    fx = 640 / (2 * np.tan(np.radians(69.0) / 2))          # the D435's colour camera
    d, box = _scene(1.0, 2.5)                              # a head 1 m away, the wall 1.5 m behind it: 17 cm wide
    why, m = refusal(d, box, fx)
    assert why is None and 0.15 < m["width_m"] < 0.19 and m["left"] > 1.0 and m["right"] > 1.0
    d, box = _scene(1.0, 1.0)                              # the same box on a flat surface: a television
    assert "flat" in refusal(d, box, fx)[0]
    d, box = _scene(1.0, 1.0, tilt=0.002)                  # the television seen at an angle: one side nearer, one farther
    m = measure(d, box, fx)
    assert m["left"] < -0.05 and m["right"] > 0.05 and "flat" in refusal(d, box, fx)[0]
    d, box = _scene(3.0, 3.0)                              # a close-up on a big screen across the room: half a metre of face
    assert "cm wide" in refusal(d, box, fx)[0]
    d, box = _scene(1.0, 0.0)                              # nothing behind the head within range: a head
    assert refusal(d, box, fx)[0] is None
    assert refusal(np.zeros((480, 640), np.float32), box, fx) == (None, None)       # no depth at all: no verdict
