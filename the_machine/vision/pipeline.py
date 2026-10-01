"""Vision pipeline (§29/§30): detection in its OWN worker thread.

Architecture:
    camera thread --(latest frame)--> VisionPipeline thread --> EventBus + shared state
Camera capture keeps full FPS; face detection runs at vision.detect_fps;
the tracker smooths boxes between detections so overlay never jitters.

The pipeline exposes `snapshot()` (lock-protected) that the GUI reads once
per paint — no cross-thread Qt object usage, no blocking.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from the_machine.config.settings import Settings
from the_machine.core.event_bus import (
    CAMERA_DISCONNECTED,
    FACE_DETECTED,
    FACE_LOST,
    EventPriority,
    EventBus,
)
from the_machine.vision.camera import CameraWorker
from the_machine.vision.face_detector import FaceDetectorEngine, build_detector
from the_machine.vision.tracker import FaceTracker, TrackedFace

logger = logging.getLogger("machine.vision.pipeline")


@dataclass
class PipelineSnapshot:
    """Immutable view of latest vision state, safe to read from GUI thread."""
    tracks: list[TrackedFace] = field(default_factory=list)
    frame_size: tuple[int, int] = (0, 0)   # source frame w,h (for box scaling)
    detect_ms: float = 0.0
    detect_fps: float = 0.0
    queue_drop_count: int = 0


class VisionPipeline(threading.Thread):
    def __init__(self, settings: Settings, bus: EventBus, camera: CameraWorker) -> None:
        super().__init__(name="vision-pipeline", daemon=True)
        self._settings = settings
        self._bus = bus
        self._camera = camera
        detector = build_detector(settings.vision.detector,
                                  settings.paths.models_dir,
                                  settings.vision.min_face_size_px)
        self._engine = FaceDetectorEngine(detector)
        self._tracker = FaceTracker()
        self._lock = threading.Lock()
        self._snap = PipelineSnapshot()
        self._running = threading.Event()
        self._last_seq = -1
        self._known_track_ids: set[int] = set()
        self._detect_times: list[float] = []      # rolling window for detect fps
        self._drops = 0

    # ---------------------------------------------------------------- control
    def start_pipeline(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self.start()
        self._bus.subscribe(CAMERA_DISCONNECTED, self._on_camera_gone)

    def close(self) -> None:
        self._running.clear()
        if self.is_alive():
            self.join(timeout=2.0)

    def snapshot(self) -> PipelineSnapshot:
        with self._lock:
            return PipelineSnapshot(
                tracks=list(self._snap.tracks),
                frame_size=self._snap.frame_size,
                detect_ms=self._snap.detect_ms,
                detect_fps=self._snap.detect_fps,
                queue_drop_count=self._snap.queue_drop_count,
            )

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        interval = 1.0 / max(1, self._settings.vision.detect_fps)
        last_detect = 0.0
        last_frame_time = time.monotonic()
        while self._running.is_set():
            now = time.monotonic()
            frame, seq = self._camera.pop_latest_frame()
            if frame is None or seq == self._last_seq:
                time.sleep(0.005)
                continue
            dt_frame = now - last_frame_time
            last_frame_time = now

            # throttle inference (§30) — but always keep tracking fresh
            if now - last_detect >= interval:
                last_detect = now
                result = self._engine.run(frame)
                dt = min(max(dt_frame, 0.001), 0.5)
                tracks = self._tracker.update(result.faces, dt)
                self._publish_face_events(tracks)
                self._record_timing(now, result.detect_ms)
                with self._lock:
                    self._snap = PipelineSnapshot(
                        tracks=tracks,
                        frame_size=(result.frame_w, result.frame_h),
                        detect_ms=result.detect_ms,
                        detect_fps=self._current_detect_fps(),
                        queue_drop_count=self._drops,
                    )
            else:
                # interpolate: advance existing tracks through tiny motion
                dt = min(max(dt_frame, 0.001), 0.5)
                tracks = self._tracker.update([], dt)
                with self._lock:
                    self._snap.tracks = tracks
            self._last_seq = seq

    # ------------------------------------------------------------- internals
    def _record_timing(self, now: float, ms: float) -> None:
        self._detect_times.append(now)
        cutoff = now - 2.0
        self._detect_times = [t for t in self._detect_times if t >= cutoff]

    def _current_detect_fps(self) -> float:
        ts = self._detect_times
        if len(ts) < 2:
            return 0.0
        span = ts[-1] - ts[0]
        return round((len(ts) - 1) / span, 1) if span > 0 else 0.0

    def _publish_face_events(self, tracks: list[TrackedFace]) -> None:
        current = {t.track_id for t in tracks}
        new_ids = current - self._known_track_ids
        gone_ids = self._known_track_ids - current
        for tid in sorted(new_ids):
            tr = next(t for t in tracks if t.track_id == tid)
            self._bus.emit(FACE_DETECTED, priority=EventPriority.NORMAL,
                           track_id=tid, confidence=tr.confidence)
            logger.info("face detected track=%d conf=%.2f", tid, tr.confidence)
        for tid in sorted(gone_ids):
            self._bus.emit(FACE_LOST, track_id=tid)
        self._known_track_ids = current

    def _on_camera_gone(self, event) -> None:
        """Session-scoped IDs die with the camera session (§10)."""
        self._tracker.reset()
        self._known_track_ids.clear()
        with self._lock:
            self._snap = PipelineSnapshot()
