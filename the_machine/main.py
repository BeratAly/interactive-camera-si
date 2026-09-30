"""THE MACHINE — entry point (Phase 1).

Usage:
    python -m the_machine.main              # normal start (webcam)
    python -m the_machine.main --demo       # DEMO MODE, synthetic video, no camera
    python -m the_machine.main --video X.mp4# DEMO MODE with a real video file
    python -m the_machine.main --theme amber|cyan|green
    python -m the_machine.main --debug      # developer mode stats (§58)
    python -m the_machine.main --fs         # fullscreen

Startup sequence (§60): config -> logging -> database-less (phase 9) ->
camera -> vision -> UI. Each step is fault-tolerant; a failing module logs
and degrades instead of crashing the app (§32).
"""
from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from the_machine.config.settings import Settings, load_settings
from the_machine.core.context_engine import ContextEngine
from the_machine.core.event_bus import EventBus
from the_machine.core.logger import setup_logging
from the_machine.core.state_manager import StateManager, SystemState
from the_machine.ai.ai_core import AICore
from the_machine.ai.providers import build_provider
from the_machine.ui.main_window import MainWindow
from the_machine.vision.camera import CameraWorker
from the_machine.vision.pipeline import VisionPipeline


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="the-machine", description="The Machine — Phase 1")
    p.add_argument("--demo", action="store_true",
                   help="DEMO MODE: generate synthetic video, no camera needed")
    p.add_argument("--video", type=str, default="",
                   help="play a video file as camera source (DEMO MODE)")
    p.add_argument("--device", type=int, default=None, help="camera device index")
    p.add_argument("--theme", choices=["green", "amber", "cyan"], default=None)
    p.add_argument("--debug", action="store_true", help="developer mode")
    p.add_argument("--fs", action="store_true", help="fullscreen")
    p.add_argument("--screenshot", type=str, default="",
                   help="render N seconds headless and save a PNG (CI / preview)")
    p.add_argument("--seconds", type=float, default=4.0,
                   help="duration for --screenshot capture")
    return p.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """CLI overrides > config.local.yaml > config.yaml > defaults."""
    from dataclasses import replace
    cam = settings.camera
    if args.demo:
        cam = replace(cam, demo_video="__synthetic__")
    elif args.video:
        cam = replace(cam, demo_video=args.video)
    if args.device is not None:
        cam = replace(cam, device=args.device)

    ui = settings.ui
    if args.theme:
        ui = replace(ui, theme=args.theme)
    if args.debug:
        ui = replace(ui, debug_mode=True)
    if args.fs:
        ui = replace(ui, fullscreen=True)
    return replace(settings, camera=cam, ui=ui)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # First-run wizard: ask for AI provider / API key in the terminal,
    # save to gitignored .env. Skips silently when already configured or
    # when stdin is not a TTY (CI). Use --no-ask to force-skip.
    from the_machine.config.bootstrap import run_first_time_setup
    run_first_time_setup()

    settings = apply_overrides(load_settings(), args)

    # screenshot mode implies DEMO MODE (fully deterministic, no hardware)
    if args.screenshot and not settings.camera.demo_video:
        from dataclasses import replace
        settings = replace(settings, camera=replace(settings.camera,
                                                    demo_video="__synthetic__"))

    setup_logging(settings.paths.logs_dir)
    import logging
    log = logging.getLogger("machine.startup")
    log.info("INITIALIZING MACHINE…")

    bus = EventBus()
    state = StateManager(bus)
    state.set(SystemState.STARTING, reason="boot")

    # Synthetic DEMO source handled inside CameraWorker via special path.
    camera = CameraWorker(settings.camera, bus)

    pipeline = VisionPipeline(settings, bus, camera)
    camera.start_capture()
    pipeline.start_pipeline()
    log.info("[OK] Configuration / Camera / Vision Engine")

    # AI Core (§15): provider from .env; NullProvider => offline local answers.
    context = ContextEngine(bus, pipeline, state)
    context.set_fps_source(camera)
    provider = build_provider(settings.secrets)
    ai_core = AICore(bus, state, context, provider)
    ai_core.start_core()
    if provider.name == "null":
        log.info("[OK] AI Core (offline local mode — vision commands work; "
                 "configure a provider in .env for open conversation)")
    else:
        log.info("[OK] AI Core (provider: %s)", provider.name)

    app = QApplication(sys.argv)
    window = MainWindow(settings, bus, camera, pipeline, state, ai_core=ai_core)

    if args.screenshot:
        # Headless verification mode: force DEMO source, render N seconds,
        # save PNG of the full UI (overlay + panels + event log).
        from PySide6.QtCore import QTimer
        window.resize(1280, 760)
        window.show()
        state.set(SystemState.READY, reason="system ready")

        def _grab() -> None:
            pix = window.grab()
            ok = pix.save(args.screenshot)
            log.info("screenshot saved=%s -> %s", ok, args.screenshot)
            window.close()
            app.quit()

        QTimer.singleShot(int(args.seconds * 1000), _grab)
        return app.exec()

    window.show()
    state.set(SystemState.READY, reason="system ready")
    log.info("SYSTEM READY")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
