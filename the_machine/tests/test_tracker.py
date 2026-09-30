"""Tracker tests (§56): association, smoothing, session reset."""
from __future__ import annotations

from the_machine.vision.tracker import FaceTracker
from the_machine.vision.types import FaceDetection


def _det(x: int, y: int, w: int = 100, h: int = 100) -> FaceDetection:
    return FaceDetection(x=x, y=y, w=w, h=h, confidence=0.8)


def test_single_face_gets_stable_id() -> None:
    tr = FaceTracker()
    a = tr.update([_det(10, 10)], dt=0.1)
    b = tr.update([_det(14, 12)], dt=0.1)   # moved slightly -> same track
    assert len(a) == 1 and len(b) == 1
    assert a[0].track_id == b[0].track_id == 1


def test_two_faces_get_distinct_ids() -> None:
    tr = FaceTracker()
    res = tr.update([_det(0, 0), _det(400, 300)], dt=0.1)
    ids = {t.track_id for t in res}
    assert ids == {1, 2}


def test_smoothing_reduces_jitter() -> None:
    tr = FaceTracker()
    tr.update([_det(100, 100)], dt=0.1)
    out = tr.update([_det(140, 100)], dt=0.1)   # sudden 40px jump
    x = out[0].box[0]
    assert 100 < x < 140                          # EMA, not raw measurement


def test_lost_face_expires_and_ids_not_reused_in_session() -> None:
    tr = FaceTracker()
    tr.update([_det(0, 0)], dt=0.1)
    for _ in range(20):                           # 20 * 0.1s > KEEP_ALIVE*3
        tr.update([], dt=0.1)
    assert tr.active_tracks == []
    new = tr.update([_det(500, 500)], dt=0.1)     # different location -> new id
    assert new[0].track_id == 2                   # never reuses #001 this session


def test_reset_kills_session_ids() -> None:
    tr = FaceTracker()
    tr.update([_det(0, 0)], dt=0.1)
    tr.reset()                                    # camera disconnect (§10)
    again = tr.update([_det(0, 0)], dt=0.1)
    assert again[0].track_id == 1                 # fresh numbering after reset
