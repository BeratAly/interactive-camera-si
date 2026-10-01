"""Face embedding extractor (§7 RECOGNITION half — "whose face is this?").

Zero new pip dependencies: OpenCV ships its own ONNX engine (cv2.dnn), so we
run a small ArcFace ONNX model (~12 MB, one-time download to data/models/)
entirely on the local CPU.

Embeddings are stored ONLY in the local SQLite database (privacy §8/§78):
raw faces are never persisted; a 512-d vector alone cannot reconstruct an
image. Recognition compares vectors by cosine similarity with an honest
threshold — below it, the person stays UNKNOWN (§38).
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("machine.vision.embedder")

# InsightFace ArcFace residual net (arcfaceresnet100-8), ONNX export.
# ~12 MB, runs comfortably on CPU (~10-20 ms per aligned 112x112 crop).
ARCFACE_URL = ("https://huggingface.co/DobbyLibraries/arcface_onnx/resolve/"
               "main/arcfaceresnet100-8.onnx")
ARCFACE_FILE = "arcfaceresnet100-8.onnx"

# Standard 112x112 alignment reference landmarks (InsightFace convention).
_REF_LANDMARKS = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                           [56.0252, 71.7366], [41.5493, 92.3655],
                           [70.7299, 92.2041]], dtype=np.float32)


def ensure_arcface_model(models_dir: Path) -> Path | None:
    """Return path to the ArcFace ONNX file, downloading once if missing."""
    target = models_dir / ARCFACE_FILE
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading ArcFace embedding model (~12 MB, one-time)…")
        tmp = target.with_suffix(".tmp")
        urllib.request.urlretrieve(ARCFACE_URL, tmp)          # noqa: S310 (https)
        tmp.replace(target)
        logger.info("ArcFace model ready: %s", target)
        return target
    except Exception as exc:
        logger.warning("ArcFace download failed (%s) — recognition disabled", exc)
        return None


class _DnnRunner:
    """Thin wrapper over cv2.dnn readNet (present in OpenCV 4.x and 5.x)."""

    def __init__(self, model_path: str) -> None:
        dnn = getattr(cv2, "dnn", None)
        read_net = getattr(dnn, "readNet", None) if dnn else None
        if read_net is None:
            raise RuntimeError("cv2.dnn unavailable in this OpenCV build")
        self._net = read_net(model_path, "", "")
        if self._net.empty():
            raise RuntimeError("could not load embedding model")

    def run(self, blob: np.ndarray) -> np.ndarray:
        self._net.setInput(blob)
        return self._net.forward()


class FaceEmbedder:
    """Compute a normalized 512-d embedding for each detected face crop.

    YuNet returns 5 landmark points per face; we use them for affine
    alignment (much better accuracy than raw crops). Without landmarks we
    fall back to a simple resized crop — still usable, slightly less accurate.
    """

    def __init__(self, model_path: Path) -> None:
        self._net: _DnnRunner | None = None
        self._failed = False
        try:
            self._net = _DnnRunner(str(model_path))
        except Exception as exc:
            logger.warning("embedding model unusable (%s)", exc)
            self._failed = True

    @property
    def available(self) -> bool:
        return self._net is not None and not self._failed

    def embed(self, frame_bgr: np.ndarray, box: tuple[int, int, int, int],
              landmarks: np.ndarray | None = None) -> np.ndarray | None:
        """Return float32 (512,) L2-normalized vector or None on failure."""
        if not self.available:
            return None
        try:
            aligned = self._align(frame_bgr, box, landmarks)
            blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112),
                                         (127.5, 127.5, 127.5), swapRB=True)
            out = self._net.run(blob)                         # type: ignore[union-attr]
            vec = np.asarray(out).flatten().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-6:
                return None
            return vec / norm
        except Exception as exc:
            logger.debug("embed failed: %s", exc)
            return None

    @staticmethod
    def _align(frame: np.ndarray, box: tuple[int, int, int, int],
               landmarks: np.ndarray | None) -> np.ndarray:
        if landmarks is not None and getattr(landmarks, "shape", (0,)) == (5, 2):
            m = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32),
                                            _REF_LANDMARKS)[0]
            if m is not None:
                return cv2.warpAffine(frame, m, (112, 112),
                                      borderMode=cv2.BORDER_REFLECT)
        x, y, w, h = box
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            crop = np.zeros((1, 1, 3), np.uint8)
        return cv2.resize(crop, (112, 112))


def compare(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-normalized vectors, clipped to [0, 1]."""
    sim = float(np.dot(a, b))
    return min(max(sim, 0.0), 1.0)
