"""Side panels (§21): SYSTEM status, VISION stats, EVENT LOG, AI CHAT.

All updates arrive via Qt signals from the main window (GUI thread only).
Event log shows REAL events fed by the EventBus — no fake terminal text (§65).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from the_machine.core.event_bus import MachineEvent
from the_machine.ui.themes import Theme


def _panel(title: str, theme: Theme) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Panel")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(4)
    header = QLabel(title)
    header.setObjectName("SectionHeader")
    lay.addWidget(header)
    return frame, lay


def _kv_row(key: str, value_label: QLabel) -> QHBoxLayout:
    row = QHBoxLayout()
    k = QLabel(key)
    k.setObjectName("Subtle")
    row.addWidget(k)
    row.addStretch(1)
    row.addWidget(value_label)
    return row


class SystemPanel(QFrame):
    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        frame, lay = _panel("SYSTEM", theme)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        self.camera_val = self._val("OFFLINE")
        self.vision_val = self._val("OFFLINE")
        self.ai_val = self._val("STANDBY")
        self.state_val = self._val("OFFLINE")
        lay.addLayout(_kv_row("CAMERA", self.camera_val))
        lay.addLayout(_kv_row("VISION", self.vision_val))
        lay.addLayout(_kv_row("AI CORE", self.ai_val))
        lay.addLayout(_kv_row("STATE", self.state_val))

    def _val(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {self._theme.text_dim}; font-weight: bold;")
        return lbl

    def set_status(self, name: str, text: str, online: bool) -> None:
        mapping = {"CAMERA": self.camera_val, "VISION": self.vision_val,
                   "AI": self.ai_val, "STATE": self.state_val}
        lbl = mapping.get(name)
        if lbl is None:
            return
        color = self._theme.ok if online else (
            self._theme.err if "ERROR" in text.upper() or "OFFLINE" in text.upper()
            else self._theme.text_dim)
        dot = "●" if online else "○"
        lbl.setText(f"{dot} {text}")
        lbl.setStyleSheet(f"color: {color.name() if hasattr(color,'name') else color}; font-weight: bold;")


class ChatPanel(QFrame):
    """AI conversation panel (§45). Emits user_text on Enter — the main
    window forwards it to AICore.process_command(); replies arrive via
    add_message(). The GUI thread never waits for the AI."""

    user_text = Signal(str)

    MAX_ROWS = 200

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        frame, lay = _panel("AI", theme)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        self.log = QListWidget()
        self.log.setObjectName("EventLog")
        self.log.setFont(QFont("Consolas", 9))
        lay.addWidget(self.log, stretch=1)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText(
            'Ask the Machine…  ("What do you see?" / "Ne görüyorsun?")')
        self.input.returnPressed.connect(self._submit)
        row.addWidget(self.input, stretch=1)
        lay.addLayout(row)

        self.add_message("machine",
                         "Machine online. State a query or command.", dim=True)

    def _submit(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.user_text.emit(text)

    def add_message(self, role: str, content: str, dim: bool = False) -> None:
        prefix = "USER  > " if role == "user" else "MACHINE < "
        item = QListWidgetItem(prefix + content)
        if role == "user":
            item.setForeground(QColor(self._theme.warn))
        elif dim:
            item.setForeground(QColor(self._theme.text_dim))
        else:
            item.setForeground(QColor(self._theme.accent))
        self.log.addItem(item)
        while self.log.count() > self.MAX_ROWS:
            self.log.takeItem(0)
        self.log.scrollToBottom()

    def set_thinking(self, thinking: bool) -> None:
        pass  # state indicator lives in SYSTEM panel; kept for API clarity


class VisionPanel(QFrame):
    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        frame, lay = _panel("VISION", theme)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        self.faces_val = QLabel("0")
        self.objects_val = QLabel("—")          # phase 4
        self.fps_val = QLabel("0.0")
        self.dfps_val = QLabel("0.0")
        self.infer_val = QLabel("—")
        lay.addLayout(_kv_row("FACES", self.faces_val))
        lay.addLayout(_kv_row("OBJECTS", self.objects_val))
        lay.addLayout(_kv_row("CAPTURE FPS", self.fps_val))
        lay.addLayout(_kv_row("DETECT FPS", self.dfps_val))
        lay.addLayout(_kv_row("INFERENCE", self.infer_val))
        for v in (self.faces_val, self.objects_val, self.fps_val,
                  self.dfps_val, self.infer_val):
            v.setStyleSheet(f"color: {theme.accent}; font-weight: bold;")

        self._debug_visible = False
        self.debug_frame = QFrame()
        dlay = QVBoxLayout(self.debug_frame)
        dlay.setContentsMargins(0, 0, 0, 0)
        self.debug_label = QLabel("")
        self.debug_label.setObjectName("Subtle")
        dlay.addWidget(self.debug_label)
        lay.addWidget(self.debug_frame)
        self.debug_frame.setVisible(False)

    def update_stats(self, faces: int, capture_fps: float, detect_fps: float,
                     detect_ms: float, debug: bool) -> None:
        self.faces_val.setText(str(faces))
        self.fps_val.setText(f"{capture_fps:.1f}")
        self.dfps_val.setText(f"{detect_fps:.1f}")
        self.infer_val.setText(f"{detect_ms:.0f} ms")
        self._debug_visible = debug
        self.debug_frame.setVisible(debug)
        if debug:
            self.debug_label.setText(
                f"infer {detect_ms:.1f} ms | det {detect_fps:.1f} fps | cap {capture_fps:.1f} fps")

    def set_offline(self) -> None:
        self.faces_val.setText("0")
        self.fps_val.setText("0.0")
        self.dfps_val.setText("0.0")
        self.infer_val.setText("—")


class EventLogPanel(QFrame):
    MAX_ROWS = 300

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        frame, lay = _panel("EVENT LOG", theme)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("EventLog")
        self.list_widget.setFont(QFont("Consolas", 9))
        lay.addWidget(self.list_widget)

    def append_event(self, event: MachineEvent) -> None:
        text = self._format(event)
        item = QListWidgetItem(text)
        color_map = {
            "CRITICAL": self._theme.err,
            "HIGH": self._theme.warn,
        }
        pen = color_map.get(event.priority.name)
        if pen is not None:
            item.setForeground(QColor(pen))
        self.list_widget.addItem(item)
        while self.list_widget.count() > self.MAX_ROWS:
            self.list_widget.takeItem(0)
        self.list_widget.scrollToBottom()

    @staticmethod
    def _format(e: MachineEvent) -> str:
        p = e.payload
        detail = ""
        if e.name == "FACE_DETECTED":
            detail = f"ID #{p.get('track_id', '?'):0>3} CONF {p.get('confidence', 0)*100:.1f}%"
        elif e.name == "FACE_LOST":
            detail = f"ID #{p.get('track_id', '?'):0>3}"
        elif e.name == "CAMERA_CONNECTED":
            detail = f"{p.get('source','')} {p.get('width')}x{p.get('height')} @{p.get('fps')}FPS"
        elif e.name in ("CAMERA_DISCONNECTED", "CAMERA_ERROR"):
            detail = p.get("message") or p.get("source") or ""
        elif e.name == "STATE_CHANGED":
            detail = f"{p.get('old')} -> {p.get('new')}"
        head = e.name.replace("_", " ")
        line = f"{e.local_time()}  {head}"
        return f"{line}  {detail}".rstrip()
