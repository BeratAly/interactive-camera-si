"""Configuration loader: YAML defaults + user overrides + .env secrets.

Secrets (API keys) are NEVER read from YAML; they come only from the
environment / .env file (privacy & security model, ARCHITECTURE.md §F).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../the_machine


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Safe nested lookup: _get(cfg, 'camera.width', 1280)."""
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass(frozen=True)
class CameraConfig:
    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    demo_video: str = ""      # empty -> real webcam; set -> DEMO MODE
    reconnect_interval_s: float = 3.0


@dataclass(frozen=True)
class VisionConfig:
    face_detection: bool = True
    detect_fps: int = 10          # inference runs slower than capture (§I)
    min_face_size_px: int = 60
    detector: str = "auto"         # auto | onnx (YuNet) | haar
    object_detection: bool = False  # phase 4
    ocr: bool = False               # phase 5


@dataclass(frozen=True)
class UiConfig:
    theme: str = "green"          # green | amber | cyan
    fullscreen: bool = False
    debug_mode: bool = False
    scanlines: bool = True


@dataclass(frozen=True)
class PrivacyConfig:
    save_frames: bool = False     # hard default OFF
    save_audio: bool = False      # hard default OFF
    cloud_ai: bool = False        # hard default OFF
    screen_analysis: bool = False # hard default OFF


@dataclass(frozen=True)
class PathsConfig:
    root: Path = PROJECT_ROOT
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "models")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "logs")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")


@dataclass(frozen=True)
class Secrets:
    """Read lazily from environment; never logged, never serialized."""
    ai_api_key: str = ""
    ai_base_url: str = ""
    ai_model: str = ""


@dataclass(frozen=True)
class Settings:
    camera: CameraConfig
    vision: VisionConfig
    ui: UiConfig
    privacy: PrivacyConfig
    paths: PathsConfig
    secrets: Secrets


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines). Avoids extra dependency."""
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_settings(config_dir: Path | None = None) -> Settings:
    config_dir = config_dir or (PROJECT_ROOT / "config")
    raw: dict[str, Any] = {}
    default_file = config_dir / "config.yaml"
    user_file = config_dir / "config.local.yaml"
    for f in (default_file, user_file):  # local overrides default
        if f.exists():
            loaded = yaml.safe_load(f.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw.update(_deep_merge(raw, loaded))

    load_dotenv()

    return Settings(
        camera=CameraConfig(
            device=_get(raw, "camera.device", 0),
            width=int(_get(raw, "camera.width", 1280)),
            height=int(_get(raw, "camera.height", 720)),
            fps=int(_get(raw, "camera.fps", 30)),
            demo_video=str(_get(raw, "camera.demo_video", "")),
            reconnect_interval_s=float(_get(raw, "camera.reconnect_interval_s", 3.0)),
        ),
        vision=VisionConfig(
            face_detection=bool(_get(raw, "vision.face_detection", True)),
            detect_fps=int(_get(raw, "vision.detect_fps", 10)),
            min_face_size_px=int(_get(raw, "vision.min_face_size_px", 60)),
            detector=str(_get(raw, "vision.detector", "auto")),
            object_detection=bool(_get(raw, "vision.object_detection", False)),
            ocr=bool(_get(raw, "vision.ocr", False)),
        ),
        ui=UiConfig(
            theme=str(_get(raw, "ui.theme", "green")),
            fullscreen=bool(_get(raw, "ui.fullscreen", False)),
            debug_mode=bool(_get(raw, "ui.debug_mode", False)),
            scanlines=bool(_get(raw, "ui.scanlines", True)),
        ),
        privacy=PrivacyConfig(
            save_frames=bool(_get(raw, "privacy.save_frames", False)),
            save_audio=bool(_get(raw, "privacy.save_audio", False)),
            cloud_ai=bool(_get(raw, "privacy.cloud_ai", False)),
            screen_analysis=bool(_get(raw, "privacy.screen_analysis", False)),
        ),
        paths=PathsConfig(),
        secrets=Secrets(
            ai_api_key=os.environ.get("AI_API_KEY", ""),
            ai_base_url=os.environ.get("AI_BASE_URL", ""),
            ai_model=os.environ.get("AI_MODEL", ""),
        ),
    )


def _deep_merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
