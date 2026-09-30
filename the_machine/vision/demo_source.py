"""DEMO MODE (§57): synthetic camera source — works with NO webcam.

Generates 640x480 frames at the configured FPS containing two moving
frontal-face-like patterns that OpenCV's Haar cascade actually detects
(symmetric dark eye-regions over a lighter face oval). Lets you test the
full pipeline (detection → tracking → animated boxes → events) anywhere,
including headless CI via --screenshot.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np


class SyntheticSource:
    """Duck-typed replacement for cv2.VideoCapture (read/isOpened only)."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = max(5, min(60, fps))
        self._t0 = time.monotonic()
        self._frame_interval = 1.0 / self.fps

    def isOpened(self) -> bool:  # noqa: N802 (duck-type cv2 API)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        # pace the generator so measured FPS is honest
        elapsed = time.monotonic() - self._t0
        target = round(elapsed * self.fps)
        now = time.monotonic()
        if now - self._t0 < (target * self._frame_interval) - self._frame_interval:
            time.sleep(0.002)
        t = time.monotonic() - self._t0
        frame = self._render(t)
        return True, frame

    # ------------------------------------------------------------------ render
    def _render(self, t: float) -> np.ndarray:
        w, h = self.width, self.height
        img = np.full((h, w, 3), 24, dtype=np.uint8)
        # subtle noise background so it feels like a real feed
        rng = np.random.default_rng(int(t * self.fps) & 0xFFFF)
        noise = rng.integers(0, 14, size=(h // 4, w // 4, 3), dtype=np.uint16)
        base = cv2.resize(noise.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_LINEAR)
        img = cv2.add(img, base)

        faces = [
            (0.34 + 0.16 * math.sin(t * 0.6), 0.46 + 0.10 * math.sin(t * 0.9),
             0.16 + 0.02 * math.sin(t * 1.7)),
            (0.66 + 0.10 * math.cos(t * 0.45), 0.55 + 0.08 * math.cos(t * 0.7),
             0.12 + 0.015 * math.cos(t * 1.3)),
        ]
        for fx, fy, fs in faces:
            self._draw_face(img, fx * w, fy * h, fs * min(w, h) * 2.2)
        return img

    @staticmethod
    def _draw_face(img: np.ndarray, cx: float, cy: float, size: float) -> None:
        """Haar-friendly pattern: light oval, darker brow/eye band, bright
        cheek band below eyes — the classic Haar feature layout."""
        cx, cy, s = int(cx), int(cy), max(60, int(size))
        # face oval (light skin tone on gray scene)
        cv2.ellipse(img, (cx, cy), (s // 2, int(s * 0.62)), 0, 0, 360,
                    (178, 168, 158), -1)
        # eye-band: darker rectangle across brow line
        band_h = max(6, s // 7)
        y_band = cy - s // 4
        cv2.rectangle(img, (cx - s // 2 + s // 8, y_band),
                      (cx + s // 2 - s // 8, y_band + band_h), (92, 86, 80), -1)
        # two very dark eye blobs inside the band
        er = max(3, s // 12)
        cv2.circle(img, (cx - s // 4, y_band + band_h // 2), er, (40, 38, 36), -1)
        cv2.circle(img, (cx + s // 4, y_band + band_h // 2), er, (40, 38, 36), -1)
        # bright cheek band right under the eyes (the "below-eyes" Haar feature)
        ch = max(6, s // 7)
        cv2.rectangle(img, (cx - s // 2 + s // 8, y_band + band_h),
                      (cx + s // 2 - s // 8, y_band + band_h + ch),
                      (210, 200, 190), -1)
        # mouth hint
        cv2.ellipse(img, (cx, cy + s // 3), (s // 6, s // 14), 0, 0, 360,
                    (120, 100, 100), -1)
