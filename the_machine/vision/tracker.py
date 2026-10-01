"""Session-scoped face tracker (§10).

Associates detections across frames with temporary IDs (PERSON #001...),
performs IoU matching, and smooths boxes so the UI never shows jitter
(§82: "yüz kutusu titriyorsa" = failure). IDs are valid only for the
current camera session — explicitly NOT identities.
"""
from __future__ import annotations

from dataclasses import dataclass

from the_machine.vision.types import FaceDetection


@dataclass
class TrackedFace:
    track_id: int
    box: tuple[float, float, float, float]   # smoothed x,y,w,h
    confidence: float
    identity: str
    age_s: float          # seconds since first seen
    lost_for: float       # seconds since last matched detection
    similarity: float = 0.0   # recognition match score (0 => UNKNOWN)


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _ema(old: tuple[float, ...], new: tuple[float, ...],
         alpha: float) -> tuple[float, float, float, float]:
    return tuple(round(o + (n - o) * alpha, 2)  # type: ignore[return-value]
                 for o, n in zip(old, new))


class FaceTracker:
    """Greedy IoU matcher + exponential smoothing. O(n·m), n,m ≤ ~10."""

    IOU_MATCH = 0.25        # threshold to consider same face
    KEEP_ALIVE_S = 1.2      # keep tracking through brief detection dropouts
    SMOOTH_ALPHA = 0.35     # EMA factor toward new measurement (higher=snappier)

    def __init__(self) -> None:
        self._tracks: dict[int, TrackedFace] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Called on camera disconnect — session IDs die here (§10)."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_tracks(self) -> list[TrackedFace]:
        return [t for t in self._tracks.values() if t.lost_for < self.KEEP_ALIVE_S]

    def update(self, detections: list[FaceDetection], dt: float) -> list[TrackedFace]:
        used: set[int] = set()

        # sort detections largest-first so main subject claims tracks first
        ordered = sorted(detections, key=lambda f: f.w * f.h, reverse=True)
        for det in ordered:
            det_box = (float(det.x), float(det.y), float(det.w), float(det.h))
            best_id, best_iou = None, 0.0
            for tid, tr in self._tracks.items():
                if tid in used or tr.lost_for >= self.KEEP_ALIVE_S:
                    continue
                iou = _iou(tr.box, det_box)
                if iou > best_iou:
                    best_id, best_iou = tid, iou

            if best_id is not None and best_iou >= self.IOU_MATCH:
                tr = self._tracks[best_id]
                tr.box = _ema(tr.box, det_box, self.SMOOTH_ALPHA)
                tr.confidence = det.confidence
                tr.identity = det.identity
                tr.similarity = det.similarity
                tr.lost_for = 0.0
                used.add(best_id)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = TrackedFace(
                    track_id=tid, box=det_box, confidence=det.confidence,
                    identity=det.identity, age_s=0.0, lost_for=0.0,
                    similarity=det.similarity,
                )
                used.add(tid)

        # age unmatched tracks; garbage-collect long-lost ones
        for tid, tr in list(self._tracks.items()):
            if tid not in used:
                tr.lost_for += dt
                if tr.lost_for > self.KEEP_ALIVE_S * 3:
                    del self._tracks[tid]
            else:
                tr.age_s += dt

        result = [t for t in self._tracks.values() if t.lost_for < self.KEEP_ALIVE_S]
        result.sort(key=lambda t: t.track_id)
        return result
