"""Camera view widget (§5/§6/§22) — the core of Phase 1.

- Receives frames via Qt signal (queued, thread-safe) from CameraWorker.
- Draws overlay in paintEvent: animated PoI-style face boxes with
  fade-in/fade-out, corner brackets, crosshair, tracking IDs, confidence.
- All animation is time-based inside paintEvent — no per-frame CPU effects,
  no extra widgets, GUI stays smooth (§92).
- Privacy indicator "● CAMERA ACTIVE" always visible while streaming (§8).
"""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from the_machine.ui.themes import Theme
from the_machine.vision.tracker import TrackedFace

FADE_IN_S = 0.35
FADE_OUT_S = 0.6


class CameraView(QWidget):
    """Live camera feed + vision HUD overlay."""

    # emitted by CameraWorker via queued connection — safe cross-thread
    frame_ready = Signal(object)   # np.ndarray (BGR copy)

    def __init__(self, theme: Theme, show_scanlines: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._show_scanlines = show_scanlines
        self.setMinimumSize(480, 270)
        # Enum accessed via the class (PySide6 6.5+ removed instance-level enum
        # attribute access; works on all supported versions this way).
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._pixmap: QPixmap | None = None
        self._frame_size = (0, 0)          # source frame w,h
        self._tracks: list[TrackedFace] = []
        # track_id -> first_seen / last_seen monotonic times (for fades)
        self._seen: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        self._online = False
        self._message = ""                 # camera error / waiting text

        # ~30 Hz repaint for smooth animation; cheap when offline
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)

    # ------------------------------------------------------------- public API
    def set_online(self, online: bool, message: str = "") -> None:
        self._online = online
        self._message = message
        if not online:
            self._tracks = []
            self._seen.clear()
            self._last_seen.clear()

    def set_message(self, message: str) -> None:
        self._message = message

    @Slot(object)
    def on_frame(self, frame: object) -> None:
        """Queued slot: convert BGR ndarray -> QPixmap (GUI thread only)."""
        arr = np.ascontiguousarray(frame)  # type: ignore[arg-type]
        h, w = arr.shape[:2]
        rgb = arr[:, :, ::-1].copy()       # BGR->RGB
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._frame_size = (w, h)

    def update_tracks(self, tracks: list[TrackedFace],
                      frame_size: tuple[int, int]) -> None:
        now = time.monotonic()
        self._tracks = tracks
        self._frame_size = frame_size
        ids = {t.track_id for t in tracks}
        for t in tracks:
            self._seen.setdefault(t.track_id, now)
            self._last_seen[t.track_id] = now
        # mark disappeared tracks for fade-out
        for tid in list(self._seen.keys()):
            if tid not in ids and now - self._last_seen.get(tid, now) > FADE_OUT_S:
                del self._seen[tid]
                self._last_seen.pop(tid, None)

    # --------------------------------------------------------------- painting
    def _map_rect(self, box: tuple[float, float, float, float],
                  img_rect) -> tuple[float, float, float, float]:
        fw, fh = self._frame_size
        if fw <= 0 or fh <= 0:
            return (0, 0, 0, 0)
        x, y, w, h = box
        sx = img_rect.width() / fw
        sy = img_rect.height() / fh
        return (img_rect.x() + x * sx, img_rect.y() + y * sy,
                w * sx, h * sy)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        t = self._theme
        now = time.monotonic()

        rect = self.rect()
        painter.fillRect(rect, QColor(t.bg))

        # --- image (letterboxed, aspect-preserving)
        img_rect = rect
        if self._pixmap and not self._pixmap.isNull():
            pw, ph = self._pixmap.width(), self._pixmap.height()
            scale = min(rect.width() / pw, rect.height() / ph)
            dw, dh = int(pw * scale), int(ph * scale)
            x0 = rect.x() + (rect.width() - dw) // 2
            y0 = rect.y() + (rect.height() - dh) // 2
            img_rect = QRect(x0, y0, dw, dh)
            painter.drawPixmap(img_rect, self._pixmap)
        else:
            self._paint_idle_screen(painter, rect, now)

        if self._online and self._pixmap:
            self._paint_hud_base(painter, img_rect, now)
            for tr in self._tracks:
                self._paint_face_box(painter, tr, img_rect, now)
            if self._show_scanlines:
                self._paint_scanline(painter, img_rect, now)
            self._paint_privacy_indicator(painter, rect)

        painter.end()

    # ------------------------------------------------------------ sub-painters
    def _paint_idle_screen(self, p: QPainter, rect, now: float) -> None:
        t = self._theme
        grid = QColor(t.panel_border)
        p.setPen(QPen(grid, 1))
        step = 48
        for gx in range(rect.x(), rect.right(), step):
            p.drawLine(gx, rect.y(), gx, rect.bottom())
        for gy in range(rect.y(), rect.bottom(), step):
            p.drawLine(rect.x(), gy, rect.right(), gy)
        # pulsing "eye" (§93)
        pulse = 0.5 + 0.5 * math.sin(now * 2.0)
        cx, cy = rect.center().x(), rect.center().y()
        eye = QColor(t.accent)
        eye.setAlphaF(0.25 + 0.45 * pulse)
        p.setPen(QPen(eye, 2))
        p.drawEllipse(cx - 26, cy - 26, 52, 52)
        p.drawPoint(cx, cy)
        p.setPen(QColor(t.text_dim))
        msg = self._message or "AWAITING SIGNAL — CAMERA OFFLINE"
        p.drawText(rect.adjusted(0, 40, 0, 0), Qt.AlignHCenter | Qt.AlignTop, msg)

    def _paint_hud_base(self, p: QPainter, img_rect, now: float) -> None:
        t = self._theme
        dim = QColor(t.accent)
        dim.setAlpha(40)
        p.setPen(QPen(dim, 1))
        p.drawRect(img_rect)                       # frame border
        # corner ticks
        p.setPen(QPen(QColor(t.accent), 2))
        c = 18
        for (cx, cy, dx, dy) in [(img_rect.left(), img_rect.top(), 1, 1),
                                 (img_rect.right(), img_rect.top(), -1, 1),
                                 (img_rect.left(), img_rect.bottom(), 1, -1),
                                 (img_rect.right(), img_rect.bottom(), -1, -1)]:
            p.drawLine(cx, cy, cx + c * dx, cy)
            p.drawLine(cx, cy, cx, cy + c * dy)
        # technical labels (§22)
        p.setPen(QColor(t.text_dim))
        stamp = time.strftime("%H:%M:%S")
        p.drawText(img_rect.adjusted(8, 6, -8, 0),
                   Qt.AlignLeft | Qt.AlignTop, f"{stamp}  REC:{'●' if self._online else '○'}")
        p.drawText(img_rect.adjusted(8, 0, -8, -6),
                   Qt.AlignRight | Qt.AlignBottom,
                   f"{self._frame_size[0]}x{self._frame_size[1]}")

    def _paint_face_box(self, p: QPainter, tr: TrackedFace, img_rect,
                        now: float) -> None:
        t = self._theme
        x, y, w, h = self._map_rect(tr.box, img_rect)
        if w <= 2 or h <= 2:
            return
        first = self._seen.get(tr.track_id, now)
        last = self._last_seen.get(tr.track_id, now)
        alpha_in = min(1.0, (now - first) / FADE_IN_S)
        # active tracks fade in only; vanished ones linger via _last_seen and fade out
        alpha_out = 1.0 if tr.track_id in {t.track_id for t in self._tracks} \
            else max(0.0, 1.0 - (now - last) / FADE_OUT_S)
        alpha = max(0.0, min(alpha_in, alpha_out))
        if alpha <= 0.01:
            return

        col = QColor(t.accent)
        col.setAlphaF(alpha)
        # bracket corners (PoI style)
        pen = QPen(col, 2)
        p.setPen(pen)
        c = max(10.0, min(w, h) * 0.22)
        for (bx, by, dx, dy) in [(x, y, 1, 1), (x + w, y, -1, 1),
                                 (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
            p.drawLine(int(bx), int(by), int(bx + c * dx), int(by))
            p.drawLine(int(bx), int(by), int(bx), int(by + c * dy))
        # faint full box
        soft = QColor(t.accent)
        soft.setAlphaF(alpha * 0.28)
        p.setPen(QPen(soft, 1))
        p.drawRect(int(x), int(y), int(w), int(h))
        # crosshair center
        ccx, ccy = x + w / 2, y + h / 2
        p.setPen(QPen(col, 1))
        p.drawLine(int(ccx) - 6, int(ccy), int(ccx) + 6, int(ccy))
        p.drawLine(int(ccx), int(ccy) - 6, int(ccx), int(ccy) + 6)

        # label block above box
        label = tr.identity if tr.identity != "UNKNOWN" else "UNKNOWN"
        conf = f"CONFIDENCE {tr.confidence * 100:.1f}%"
        tid = f"PERSON #{tr.track_id:03d}"
        p.setFont(_mono_font(11, bold=True))
        fm_w = max(_text_w(p, label), _text_w(p, conf), _text_w(p, tid))
        ly = y - 54 if y > 60 else y + h + 6
        bg = QColor(t.panel)
        bg.setAlphaF(alpha * 0.85)
        p.fillRect(int(x), int(ly), int(fm_w + 16), 52, bg)
        p.setPen(QPen(col, 1))
        p.drawRect(int(x), int(ly), int(fm_w + 16), 52)
        text_col = QColor(t.accent)
        text_col.setAlphaF(alpha)
        p.setPen(text_col)
        p.drawText(int(x) + 8, int(ly) + 16, label)
        dim_col = QColor(t.text)
        dim_col.setAlphaF(alpha * 0.8)
        p.setPen(dim_col)
        p.setFont(_mono_font(10))
        p.drawText(int(x) + 8, int(ly) + 32, conf)
        p.drawText(int(x) + 8, int(ly) + 46, tid)

    def _paint_scanline(self, p: QPainter, img_rect, now: float) -> None:
        """Single moving line — one drawLine per frame, effectively free (§92)."""
        period = 4.0
        frac = (now % period) / period
        y = img_rect.top() + frac * img_rect.height()
        col = QColor(self._theme.accent)
        col.setAlpha(36)
        p.setPen(QPen(col, 1))
        p.drawLine(img_rect.left(), int(y), img_rect.right(), int(y))

    def _paint_privacy_indicator(self, p: QPainter, rect) -> None:
        t = self._theme
        txt = "● CAMERA ACTIVE — LOCAL PROCESSING ONLY"
        p.setFont(_mono_font(10, bold=True))
        w = _text_w(p, txt) + 16
        box_x = rect.right() - w - 10
        box_y = rect.top() + 8
        p.fillRect(box_x, box_y, w, 22, QColor(t.panel))
        p.setPen(QPen(QColor(t.warn), 1))
        p.drawRect(box_x, box_y, w, 22)
        p.setPen(QColor(t.warn))
        p.drawText(box_x + 8, box_y + 15, txt)


# ---------------------------------------------------------------- text helpers
_font_cache: dict[tuple[int, bool], object] = {}


def _mono_font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _font_cache:
        from PySide6.QtGui import QFont
        f = QFont("Consolas" if _is_windows() else "DejaVu Sans Mono", size)
        f.setBold(bold)
        _font_cache[key] = f
    return _font_cache[key]


def _is_windows() -> bool:
    import platform
    return platform.system() == "Windows"


def _text_w(painter: QPainter, text: str) -> int:
    return int(painter.fontMetrics().horizontalAdvance(text))
