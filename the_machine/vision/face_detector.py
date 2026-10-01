"""Face detection (§7: DETECTION only — "is there a face?").

Backend is pluggable via `BaseFaceDetector`:
  - YuNetDetector      : OpenCV built-in ONNX face detector (real confidence
                         scores). Model (~230 KB) is downloaded once to
                         data/models/ on first run; if the machine is offline
                         it falls back to Haar automatically.
  - HaarCascadeDetector: zero-download fallback, always available. Works with
                         both OpenCV <5 (cv2.CascadeClassifier) and OpenCV 5+
                         (objdetect module).

Selection order (config `vision.detector`):
  auto   -> YuNet if model present/downloadable, else Haar
  onnx   -> YuNet only
  haar   -> Haar only
"""
from __future__ import annotations

import logging
import time
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from the_machine.vision.types import FaceDetection, VisionResult

logger = logging.getLogger("machine.vision.face")

# YuNet face detector ONNX (OpenCV Zoo, MIT-licensed, ~230 KB) — local inference.
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_FILE = "face_detection_yunet_2023mar.onnx"


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        """Return faces for one frame. Must not raise on empty/odd frames."""


# --------------------------------------------------------------- Haar fallback
def _make_cascade(path: str):
    """Create a Haar cascade classifier across OpenCV versions (objdetect in 5+)."""
    objdetect = getattr(cv2, "objdetect", None)
    if objdetect is not None and hasattr(objdetect, "CascadeClassifier"):
        return objdetect.CascadeClassifier(path)
    if hasattr(cv2, "CascadeClassifier"):
        return cv2.CascadeClassifier(path)
    return None


class HaarCascadeDetector(BaseFaceDetector):
    """OpenCV Haar cascade detector. Runs entirely local (§40)."""

    def __init__(self, model_path: Path, min_face_size_px: int = 60) -> None:
        self._cascade_path = str(model_path / "haarcascade_frontalface_default.xml")
        self._min_face = max(24, int(min_face_size_px))
        self._cascade = None
        self._failed = False

    def _ensure_loaded(self) -> bool:
        if self._cascade is not None:
            return True
        if self._failed:
            return False
        cascade = _make_cascade(self._cascade_path)
        if cascade is None or cascade.empty():
            logger.error("Haar cascade not loadable at %s", self._cascade_path)
            self._failed = True
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


# ------------------------------------------------------------------ YuNet ONNX
def ensure_yunet_model(models_dir: Path) -> Path | None:
    """Return path to the YuNet ONNX file, downloading once if missing.
    Returns None when unavailable (offline first run) — caller falls back."""
    target = models_dir / YUNET_FILE
    if target.exists() and target.stat().st_size > 100_000:
        return target
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading YuNet face model (~230 KB, one-time)…")
        tmp = target.with_suffix(".tmp")
        urllib.request.urlretrieve(YUNET_URL, tmp)          # noqa: S310 (https)
        tmp.replace(target)
        logger.info("YuNet model ready: %s", target)
        return target
    except Exception as exc:                                  # network/disk
        logger.warning("YuNet model download failed (%s) — using Haar fallback",
                       exc)
        return None


class YuNetDetector(BaseFaceDetector):
    """OpenCV YuNet ONNX face detector — real confidence scores, CPU-fast."""

    INPUT_SIZES = [(320, 320), (640, 640)]

    def __init__(self, model_path: Path, min_face_size_px: int = 60,
                 score_threshold: float = 0.6) -> None:
        self._model = str(Path(model_path) / YUNET_FILE)
        self._min_face = max(24, int(min_face_size_px))
        self._score_thr = float(score_threshold)
        self._detectors: dict[tuple[int, int], object] = {}
        self._failed = False

    def _detector_for(self, w: int, h: int):
        """YuNet needs a fixed input size; pick nearest supported size ≥ frame."""
        best = next((s for s in self.INPUT_SIZES if s[0] >= w and s[1] >= h),
                    self.INPUT_SIZES[-1])
        if best not in self._detectors:
            # OpenCV 5 requires explicit top_k (int); pass it positionally so
            # the call also works on OpenCV 4.x. input_size as tuple/list is
            # accepted by both.
            det = None
            for size_arg in ((best[0], best[1]), [best[0], best[1]]):
                try:
                    det = cv2.FaceDetectorYN_create(
                        self._model, "", size_arg,
                        self._score_thr, 0.3, 500)
                    break
                except Exception:
                    continue
            if det is None:
                raise RuntimeError("FaceDetectorYN_create failed")
            self._detectors[best] = det
        return best, self._detectors[best]

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        if frame_bgr is None or frame_bgr.size == 0 or self._failed:
            return []
        try:
            h, w = frame_bgr.shape[:2]
            (iw, ih), det = self._detector_for(w, h)
            canvas = frame_bgr if (w, h) == (iw, ih) else \
                cv2.resize(frame_bgr, (iw, ih))
            _, out = det.detect(canvas)
        except Exception as exc:                              # corrupt model etc.
            logger.warning("YuNet detection failed (%s) — disabling", exc)
            self._failed = True
            return []
        faces: list[FaceDetection] = []
        if out is None:
            return faces
        sx, sy = w / iw, h / ih
        for row in out:
            x, y, fw, fh, score = row[0], row[1], row[2], row[3], row[14]
            fx, fy, ff, fhh = int(x * sx), int(y * sy), int(fw * sx), int(fh * sy)
            if ff < self._min_face or fh < self._min_face:
                continue
            # YuNet landmark rows 4..13: right/left eye, nose, mouth corners.
            lm = np.array([[row[4 + i * 2] * sx, row[5 + i * 2] * sy]
                           for i in range(5)], dtype=np.float32)
            faces.append(FaceDetection(x=fx, y=fy, w=ff, h=fhh,
                                       confidence=round(float(score), 3),
                                       landmarks=lm))
        faces.sort(key=lambda f: f.w * f.h, reverse=True)
        return faces


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
    """Factory used by the pipeline. 'auto'/'onnx' prefer YuNet (with honest
    error-handled fallback to Haar); unknown names fall back to Haar."""
    if detector_name in ("auto", "onnx", "yunet"):
        model = ensure_yunet_model(models_dir)
        if model is not None:
            return YuNetDetector(models_dir, min_face_size_px)
        if detector_name != "auto":
            logger.warning("detector '%s' unavailable — falling back to Haar",
                           detector_name)
    return HaarCascadeDetector(models_dir, min_face_size_px)
