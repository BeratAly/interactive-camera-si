"""Main window (§20/§29): layout + thread-safe wiring between workers and GUI.

Threading contract:
  - CameraWorker / VisionPipeline run in their own threads and NEVER touch Qt.
  - A GUI-thread QTimer (frame pump) pulls the latest frame by sequence
    number and converts it to QPixmap here — exactly one copy per paint tick.
  - EventBus handlers fire on worker threads; they only set lightweight
    Python flags / append to a queue drained by the same timer.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from the_machine.config.settings import Settings
from the_machine.core.event_bus import (
    CAMERA_CONNECTED,
    CAMERA_DISCONNECTED,
    CAMERA_ERROR,
    FACE_DETECTED,
    FACE_LOST,
    STATE_CHANGED,
    SYSTEM_ERROR,
    EventBus,
    MachineEvent,
)
from the_machine.core.state_manager import StateManager, SystemState
from the_machine.ui.camera_view import CameraView
from the_machine.ui.panels import EventLogPanel, SystemPanel, VisionPanel
from the_machine.ui.themes import Theme, build_qss
from the_machine.vision.camera import CameraWorker
from the_machine.vision.pipeline import VisionPipeline

logger = logging.getLogger("machine.ui")


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, bus: EventBus,
                 camera: CameraWorker, pipeline: VisionPipeline,
                 state: StateManager) -> None:
        super().__init__()
        self._settings = settings
        self._bus = bus
        self._camera = camera
        self._pipeline = pipeline
        self._state = state
        self._theme = Theme.by_name(settings.ui.theme)

        self.setWindowTitle("THE MACHINE")
        self.setStyleSheet(build_qss(self._theme))
        self.resize(1280, 760)
        if settings.ui.fullscreen:
            self.showFullScreen()

        # ---- pending-event drain queue (worker threads -> GUI thread)
        self._event_queue: list[MachineEvent] = []
        self._last_frame_seq = -1
        self._cam_running = True   # started by main.py before window shows

        self._build_layout()
        self._wire_events()

        # frame pump: ~30 Hz, drains frames + events + stats. Never blocks.
        self._pump = QTimer(self)
        self._pump.timeout.connect(self._tick)
        self._pump.start(33)

    # ------------------------------------------------------------------ layout
    def _build_layout(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(10)

        # header
        header = QHBoxLayout()
        title = QLabel("THE MACHINE")
        title.setObjectName("Title")
        status = QLabel("SYSTEM STARTING")
        status.setObjectName("Subtle")
        self._status_label = status
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(status)
        root.addLayout(header)

        # body: camera (left, expanding) + sidebar (right, fixed)
        body = QHBoxLayout()
        body.setSpacing(10)
        self.camera_view = CameraView(self._theme,
                                      show_scanlines=self._settings.ui.scanlines)
        body.addWidget(self.camera_view, stretch=1)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(10)
        self.system_panel = SystemPanel(self._theme)
        self.vision_panel = VisionPanel(self._theme)
        sidebar.addWidget(self.system_panel)
        sidebar.addWidget(self.vision_panel)
        sidebar.addStretch(1)

        controls = QHBoxLayout()
        self.btn_camera = QPushButton("CAMERA: ON")
        self.btn_camera.clicked.connect(self._toggle_camera)
        controls.addWidget(self.btn_camera)
        sidebar.addLayout(controls)
        body.addLayout(sidebar, stretch=0)
        root.addLayout(body, stretch=1)

        # event log across bottom
        self.event_panel = EventLogPanel(self._theme)
        self.event_panel.setFixedHeight(150)
        root.addWidget(self.event_panel)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "LOCAL PROCESSING ONLY  |  FRAMES NOT SAVED  |  CLOUD OFF")

    # --------------------------------------------------------------- wiring
    def _wire_events(self) -> None:
        """Bus handlers run on worker threads: enqueue only, never touch Qt."""
        for name in (CAMERA_CONNECTED, CAMERA_DISCONNECTED, CAMERA_ERROR,
                     FACE_DETECTED, FACE_LOST, STATE_CHANGED, SYSTEM_ERROR):
            self._bus.subscribe(name, self._enqueue)

    def _enqueue(self, event: MachineEvent) -> None:
        self._event_queue.append(event)   # list.append is atomic under GIL

    # ------------------------------------------------------------------ tick
    def _tick(self) -> None:
        # 1) newest frame (drop-oldest semantics: we only ever take latest)
        frame, seq = self._camera.pop_latest_frame()
        if frame is not None and seq != self._last_frame_seq:
            self._last_frame_seq = seq
            self.camera_view.on_frame(frame.copy())  # private copy for Qt

        # 2) vision overlay data
        snap = self._pipeline.snapshot()
        self.camera_view.update_tracks(snap.tracks, snap.frame_size)
        self.vision_panel.update_stats(
            faces=len(snap.tracks),
            capture_fps=self._camera.measured_fps(),
            detect_fps=snap.detect_fps,
            detect_ms=snap.detect_ms,
            debug=self._settings.ui.debug_mode,
        )

        # 3) drain queued events on GUI thread
        events, self._event_queue = self._event_queue, []
        for ev in events:
            self.event_panel.append_event(ev)
            self._apply_event(ev)

    def _apply_event(self, ev: MachineEvent) -> None:
        p = ev.payload
        if ev.name == CAMERA_CONNECTED:
            demo = p.get("demo_mode", False)
            self.camera_view.set_online(True, "DEMO MODE — SYNTHETIC SOURCE"
                                        if demo else "")
            self.system_panel.set_status("CAMERA", "ONLINE", True)
            self.system_panel.set_status("VISION", "ONLINE", True)
            self._status_label.setText("SYSTEM ONLINE")
        elif ev.name == CAMERA_DISCONNECTED:
            self.camera_view.set_online(False, "CAMERA SIGNAL LOST — RECONNECTING…")
            self.system_panel.set_status("CAMERA", "OFFLINE", False)
            self.system_panel.set_status("VISION", "OFFLINE", False)
            self.vision_panel.set_offline()
        elif ev.name == CAMERA_ERROR:
            self.camera_view.set_message(p.get("message", "CAMERA ERROR"))
            self.system_panel.set_status("CAMERA", "ERROR", False)
        elif ev.name == STATE_CHANGED:
            new = p.get("new", "")
            self.system_panel.set_status("STATE", new,
                                         new not in ("OFFLINE", "ERROR"))
            if new == SystemState.READY.value:
                self._status_label.setText("SYSTEM ONLINE")
            elif new == SystemState.ERROR.value:
                self._status_label.setText("SYSTEM ERROR")
        elif ev.name == SYSTEM_ERROR:
            self.system_panel.set_status("STATE", "ERROR", False)

    # -------------------------------------------------------------- controls
    def _toggle_camera(self) -> None:
        """User-visible camera on/off (§8). Worker threads keep running;
        capture thread simply releases/reopens the device."""
        if self._cam_running:
            self._camera.stop_capture()
            self._camera._release_cap()  # ensure device is actually freed
            self._cam_running = False
            self.btn_camera.setText("CAMERA: OFF")
            self.camera_view.set_online(False, "CAMERA DISABLED BY USER")
            self.system_panel.set_status("CAMERA", "OFFLINE", False)
            self.system_panel.set_status("VISION", "OFFLINE", False)
            self.vision_panel.set_offline()
        else:
            self._camera.start_capture()
            self._cam_running = True
            self.btn_camera.setText("CAMERA: ON")

    # ------------------------------------------------------------- lifecycle
    def closeEvent(self, event) -> None:  # noqa: N802
        """Clean shutdown order: pump -> pipeline -> camera. Never crash (§82)."""
        logger.info("shutting down")
        self._pump.stop()
        try:
            self._pipeline.close()
        except Exception:
            logger.exception("pipeline shutdown error")
        try:
            self._camera.close()
        except Exception:
            logger.exception("camera shutdown error")
        self._state.set(SystemState.OFFLINE, reason="application exit")
        super().closeEvent(event)
