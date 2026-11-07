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
