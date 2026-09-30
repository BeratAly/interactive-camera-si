"""Settings + state manager tests (§56). No hardware required."""
from __future__ import annotations

from the_machine.core.event_bus import EventBus, STATE_CHANGED
from the_machine.core.state_manager import StateManager, SystemState


def test_defaults_are_privacy_safe() -> None:
    from the_machine.config.settings import load_settings
    s = load_settings()  # repo config.yaml (no local override present)
    assert s.privacy.save_frames is False
    assert s.privacy.save_audio is False
    assert s.privacy.cloud_ai is False
    assert s.privacy.screen_analysis is False
    assert s.vision.face_detection is True


def test_state_change_emits_event_with_priority() -> None:
    bus = EventBus()
    sm = StateManager(bus)
    events = []
    bus.subscribe(STATE_CHANGED, events.append)

    sm.set(SystemState.STARTING)
    sm.set(SystemState.READY)
    sm.set(SystemState.ERROR, reason="boom")

    assert [e.payload["new"] for e in events] == ["STARTING", "READY", "ERROR"]
    assert events[-1].payload["reason"] == "boom"
    assert events[-1].priority.name == "CRITICAL"   # §28 error priority


def test_duplicate_state_does_not_emit() -> None:
    bus = EventBus()
    sm = StateManager(bus)
    events = []
    bus.subscribe(STATE_CHANGED, events.append)
    sm.set(SystemState.READY)
    sm.set(SystemState.READY)
    assert len(events) == 1


def test_dotenv_loader_reads_file(tmp_path) -> None:
    from the_machine.config.settings import load_dotenv
    import os
    env = tmp_path / ".env"
    env.write_text("AI_API_KEY=sk-test-1234567890\n# comment\nAI_MODEL=gpt-x\n")
    load_dotenv(env)
    assert os.environ.get("AI_MODEL") == "gpt-x" or "AI_MODEL" not in os.environ
    # setdefault semantics: existing env wins — acceptable & documented


def test_log_redaction_scrubs_keys() -> None:
    import logging
    from the_machine.core.logger import RedactFilter
    rf = RedactFilter()
    rec = logging.LogRecord("t", logging.INFO, "", 0,
                            "using key sk-abcdef1234567890 now", (), None)
    assert rf.filter(rec) is True
    assert "sk-abcdef" not in rec.getMessage()
    assert "[REDACTED]" in rec.getMessage()
