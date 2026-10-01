"""Face detection (§7: DETECTION only — "is there a face?").

Backend is pluggable via `BaseFaceDetector`:
  - HaarCascadeDetector : zero-download, always available (Phase 1 default)
  - ONNX detectors (YuNet/SCRFD) can be added in Phase 2 without touching
    any other module.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from the_machine.vision.types import FaceDetection, VisionResult

logger = logging.getLogger("machine.vision.face")


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        """Return faces for one frame. Must not raise on empty/odd frames."""


class HaarCascadeDetector(BaseFaceDetector):
    """OpenCV Haar cascade detector. Runs entirely local (§40)."""

    def __init__(self, model_path: Path, min_face_size_px: int = 60) -> None:
        self._cascade_path = str(model_path / "haarcascade_frontalface_default.xml")
        self._min_face = max(24, int(min_face_size_px))
        self._cascade: cv2.CascadeClassifier | None = None

    def _ensure_loaded(self) -> bool:
        if self._cascade is not None:
            return True
        cascade = cv2.CascadeClassifier(self._cascade_path)
        if cascade.empty():
            logger.error("Haar cascade not loadable at %s", self._cascade_path)
            return False
        self._cascade = cascade
        return True

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        if frame_bgr is None or frame_bgr.size == 0 or not self._ensure_loaded():
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(self._min_face, self._min_face),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        faces: list[FaceDetection] = []
        for (x, y, w, h) in rects:
            faces.append(FaceDetection(
                x=int(x), y=int(y), w=int(w), h=int(h),
                confidence=_heuristic_confidence(int(w), int(h), frame_bgr.shape),
            ))
        # largest first — main subject gets the most attention
        faces.sort(key=lambda f: f.w * f.h, reverse=True)
        return faces


def _heuristic_confidence(w: int, h: int, shape: tuple[int, ...]) -> float:
    """Haar gives no probability; derive a *display* confidence from face
    size relative to the frame. Honest range 0.55–0.95 — never 1.0 (§38)."""
    frame_h = shape[0] if len(shape) else 1
    ratio = min(1.0, max(w, h) / max(1.0, frame_h * 0.5))
    return round(0.55 + 0.40 * ratio, 3)


class FaceDetectorEngine:
    """Thin wrapper: throttling is handled by the pipeline; this class just
    exposes detect() with timing for debug mode."""

    def __init__(self, detector: BaseFaceDetector) -> None:
        self._detector = detector

    def run(self, frame_bgr: np.ndarray) -> VisionResult:
        t0 = time.perf_counter()
        faces = self._detector.detect(frame_bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        h, w = frame_bgr.shape[:2]
        return VisionResult(faces=faces, frame_w=w, frame_h=h, detect_ms=round(ms, 2))


def build_detector(detector_name: str, models_dir: Path,
                   min_face_size_px: int) -> BaseFaceDetector:
    """Factory used by the pipeline; unknown names fall back to Haar."""
    if detector_name == "haar":
        return HaarCascadeDetector(models_dir, min_face_size_px)
    logger.warning("detector '%s' not implemented yet — using haar", detector_name)
    return HaarCascadeDetector(models_dir, min_face_size_px)
