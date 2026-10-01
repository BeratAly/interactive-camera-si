"""Camera capture (§5) — runs in its OWN worker thread; GUI never blocks.

Responsibilities:
  - open webcam (or DEMO MODE video file)
  - expose real resolution / FPS
  - drop-oldest single-frame buffer (UI always gets the freshest frame)
  - detect disconnection, auto-reconnect with interval
  - user pause/resume WITHOUT killing the reconnect state machine
  - publish CAMERA_CONNECTED / _DISCONNECTED / CAMERA_ERROR events
No frames are ever written to disk (privacy.save_frames is enforced upstream;
this module has no save path at all — safe by construction).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from the_machine.config.settings import CameraConfig
from the_machine.core.event_bus import (
    CAMERA_CONNECTED,
    CAMERA_DISCONNECTED,
    CAMERA_ERROR,
    CAMERA_PAUSED,
    CAMERA_RESUMED,
    EventPriority,
    EventBus,
)

logger = logging.getLogger("machine.vision.camera")


@dataclass
class CameraInfo:
    source: str
    width: int
    height: int
    fps: float
    demo_mode: bool


class CameraWorker(threading.Thread):
    """Capture loop in a dedicated thread. Exposes latest_frame() for UI."""

    def __init__(self, cfg: CameraConfig, bus: EventBus) -> None:
        super().__init__(name="camera-worker", daemon=True)
        self._cfg = cfg
        self._bus = bus
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._frame_seq = 0
        self._running = threading.Event()      # thread alive?
        self._enabled = threading.Event()      # capture wanted? (pause toggle)
        self._info: CameraInfo | None = None
        # true measured FPS over a sliding second
        self._timestamps: deque[float] = deque(maxlen=64)

    # ---------------------------------------------------------------- control
    def start_capture(self) -> None:
        """Resume/enable capture. Safe to call repeatedly; the worker THREAD
        is never killed here, so reopening after a pause or disconnect always
        works (previous bug: stop() joined and terminated the thread)."""
        self._enabled.set()
        if not self._running.is_set():
            self._running.set()
            self.start()

    def stop_capture(self) -> None:
        """Pause: release the device but keep the state machine alive."""
        self._enabled.clear()
        if self._cap is not None:
            self._release_cap()
            self._bus.emit(CAMERA_PAUSED, priority=EventPriority.NORMAL,
                           source=self._info.source if self._info else "")
            logger.info("camera paused by user (device released)")

    def close(self) -> None:
        """Clean shutdown — app must never crash when camera closes (§82)."""
        self._enabled.clear()
        self._running.clear()
        if self.is_alive():
            self.join(timeout=2.0)
        self._release_cap()

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    # ------------------------------------------------------------------- data
    def pop_latest_frame(self) -> tuple[np.ndarray | None, int]:
        """Return (frame, sequence). Frame is a private copy-safe reference."""
        with self._lock:
            return self._latest, self._seq_snapshot()

    def _seq_snapshot(self) -> int:
        return self._frame_seq

    @property
    def info(self) -> CameraInfo | None:
        return self._info

    @property
    def is_online(self) -> bool:
        cap = self._cap
        if cap is None:
            return False
        try:
            return bool(cap.isOpened())
        except Exception:
            return False

    def measured_fps(self) -> float:
        ts = list(self._timestamps)
        if len(ts) < 2:
            return 0.0
        span = ts[-1] - ts[0]
        return (len(ts) - 1) / span if span > 0 else 0.0

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        last_connect_attempt = 0.0
        was_online = False
        while self._running.is_set():
            if not self._enabled.is_set():
                # paused by user: idle cheaply, thread stays alive
                time.sleep(0.1)
                continue

            if not self.is_online:
                now = time.monotonic()
                if now - last_connect_attempt >= self._cfg.reconnect_interval_s \
                        or last_connect_attempt == 0.0:
                    last_connect_attempt = now
                    was_online = self._try_open(was_online)
                time.sleep(0.1)
                continue

            ok, frame = self._cap.read()  # type: ignore[union-attr]
            if not ok or frame is None or frame.size == 0:
                logger.warning("frame read failed — treating as disconnect")
                self._handle_disconnect()
                continue

            now = time.monotonic()
            self._timestamps.append(now)
            with self._lock:
                self._latest = frame
                self._frame_seq += 1

    def _try_open(self, was_online: bool) -> bool:
        source = self._resolve_source()
        if source == "__synthetic__":
            return self._open_synthetic(was_online)
        try:
            cap = cv2.VideoCapture(source)
        except Exception as exc:
            logger.error("VideoCapture raised: %s", exc)
            self._bus.emit(CAMERA_ERROR, priority=EventPriority.HIGH,
                           message=f"Camera access error: {exc}")
            return False
        if not cap.isOpened():
            if was_online:
                self._bus.emit(CAMERA_DISCONNECTED, priority=EventPriority.HIGH,
                               source=str(source))
            self._bus.emit(CAMERA_ERROR, priority=EventPriority.HIGH,
                           message=f"Unable to access camera ({source}). "
                                   f"Retrying in {self._cfg.reconnect_interval_s:.0f}s.")
            logger.error("could not open camera source=%s", source)
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or float(self._cfg.fps)
        self._cap = cap
        self._info = CameraInfo(
            source=("DEMO VIDEO" if isinstance(source, str) else f"DEVICE {source}"),
            width=w, height=h, fps=round(fps, 1),
            demo_mode=isinstance(source, str),
        )
        self._bus.emit(CAMERA_CONNECTED, source=self._info.source,
                        width=w, height=h, fps=self._info.fps,
                        demo_mode=self._info.demo_mode)
        logger.info("camera connected: %s %dx%d @%.1f", self._info.source, w, h, fps)
        return True

    def _resolve_source(self) -> int | str:
        if self._cfg.demo_video:
            return self._cfg.demo_video          # DEMO MODE (§57)
        if isinstance(self._cfg.device, str) and not self._cfg.device.isdigit():
            return self._cfg.device
        return int(self._cfg.device)

    def _open_synthetic(self, was_online: bool) -> bool:
        """DEMO MODE without any file: synthetic frame generator (§57)."""
        from the_machine.vision.demo_source import SyntheticSource
        src = SyntheticSource(width=self._cfg.width, height=self._cfg.height,
                              fps=self._cfg.fps)
        self._cap = src
        self._info = CameraInfo(source="SYNTHETIC DEMO", width=src.width,
                                height=src.height, fps=float(src.fps),
                                demo_mode=True)
        self._bus.emit(CAMERA_CONNECTED, source=self._info.source,
                       width=src.width, height=src.height, fps=float(src.fps),
                       demo_mode=True)
        logger.info("synthetic demo source started")
        return True

    def _handle_disconnect(self) -> None:
        self._bus.emit(CAMERA_DISCONNECTED, priority=EventPriority.HIGH,
                       source=(self._info.source if self._info else "unknown"))
        logger.warning("camera disconnected")
        self._release_cap()
        # tracker IDs die with the session (§10) — pipeline listens to event

    def _release_cap(self) -> None:
        if self._cap is not None:
            try:
                # SyntheticSource has no release(); duck-typed safely.
                release = getattr(self._cap, "release", None)
                if callable(release):
                    release()
            except Exception:
                logger.exception("camera release failed")
            self._cap = None
            self._info = None
            with self._lock:
                self._latest = None
