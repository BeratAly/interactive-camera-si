"""Vision data types shared across vision modules and UI."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FaceDetection:
    """One detected face in source-frame coordinates.

    identity stays 'UNKNOWN' until face recognition (phase 6) is enabled;
    detection and recognition are deliberately separate concepts (§7).
    """
    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0          # 0..1 — never presented as certainty (§38)
    identity: str = "UNKNOWN"
    track_id: int | None = None      # valid only for this camera session (§10)
    landmarks: object = None         # optional (5,2) float32 points from YuNet
    similarity: float = 0.0          # recognition match score when identified

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class VisionResult:
    """Output of one detection pass, tagged with its source frame size."""
    faces: list[FaceDetection] = field(default_factory=list)
    frame_w: int = 0
    frame_h: int = 0
    detect_ms: float = 0.0           # inference latency (debug mode)
