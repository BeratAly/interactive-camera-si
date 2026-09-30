"""Tests for the AI brain: command router, providers, context engine (§56).

No network, no camera, no Qt required — pure logic with fakes.
"""
from __future__ import annotations

import pytest

from the_machine.ai.command_router import route_command
from the_machine.ai.providers import (
    NullProvider,
    OllamaProvider,
    OpenAICompatProvider,
    ProviderError,
    build_provider,
)
from the_machine.core.event_bus import EventBus
from the_machine.core.state_manager import StateManager


# --------------------------------------------------------------------- fixtures
def make_vctx(people=1, camera_online=True, identity="UNKNOWN", conf=0.97):
    return {
        "people": people,
        "camera_online": camera_online,
        "faces": [{"identity": identity, "confidence": conf, "track_id": 1}]
                  if people else [],
        "scene": "test",
        "capture_fps": 29.5,
        "detect_fps": 10.0,
    }


class FakePipeline:
    def snapshot(self):
        from types import SimpleNamespace
        return SimpleNamespace(tracks=[], frame_size=(640, 480),
                               detect_ms=12.0, detect_fps=10.0,
                               queue_drop_count=0)


# ------------------------------------------------------------------- router TR
def test_ne_goruyorsun_tr():
    r = route_command("Ne görüyorsun?", make_vctx(), [], "READY", [])
    assert r is not None and r.action == "WHAT_DO_YOU_SEE"
    assert "kisi var" in r.answer.lower() or "person" in r.answer.lower()


def test_kac_kisi_var_tr():
    r = route_command("Kaç kişi var?", make_vctx(people=2), [], "READY", [])
    assert r.action == "PEOPLE_COUNT"
    assert "2" in r.answer


def test_ben_kimim_unknown():
    r = route_command("Ben kimim?", make_vctx(), [], "READY", [])
    assert r.action == "WHO_AM_I"
    assert "UNKNOWN" in r.answer


def test_ben_kimim_recognized():
    vctx = make_vctx(identity="Berat")
    r = route_command("Ben kimim?", vctx, [], "READY", [])
    assert "Berat" in r.answer


def test_camera_offline_degrades():
    r = route_command("Ne görüyorsun?", make_vctx(camera_online=False),
                      [], "READY", [])
    assert "kamera" in r.answer.lower() or "camera" in r.answer.lower()


def test_fps_query():
    r = route_command("FPS kaç?", make_vctx(), [], "READY", [])
    assert r.action == "FPS_QUERY"
    assert "29.5" in r.answer


def test_temizle_command():
    r = route_command("Konuşmayı temizle", make_vctx(), [], "READY", [])
    assert r.clear_conversation is True


def test_hatirla_memory():
    r = route_command("Hatırla: laptopumun adı Orion", make_vctx(),
                      [], "READY", [])
    assert r.remember and "Orion" in r.remember


# ------------------------------------------------------------------- router EN
def test_what_do_you_see_en():
    r = route_command("What do you see?", make_vctx(), [], "READY", [])
    assert r.action == "WHAT_DO_YOU_SEE"


def test_how_many_people_en():
    r = route_command("How many people are there?", make_vctx(people=1),
                      [], "READY", [])
    assert r.action == "PEOPLE_COUNT"


def test_who_am_i_en():
    r = route_command("Who am I?", make_vctx(), [], "READY", [])
    assert r.action == "WHO_AM_I"


def test_recent_events_en():
    events = ["12:00:01 face detected (id 1, conf 97%)"]
    r = route_command("What happened recently?", make_vctx(), events,
                      "READY", [])
    assert r.action == "EVENT_HISTORY"
    assert "face detected" in r.answer


def test_remember_en():
    r = route_command("Remember that my laptop is called Orion",
                      make_vctx(), [], "READY", [])
    assert r.remember and "Orion" in r.remember


def test_unknown_falls_through_to_llm():
    assert route_command("explain quantum physics", make_vctx(), [],
                         "READY", []) is None


def test_punctuation_and_case_tolerant():
    r = route_command("NE GÖRÜYORSUN!!", make_vctx(), [], "READY", [])
    assert r is not None


# -------------------------------------------------------------------- providers
def test_null_provider_raises():
    with pytest.raises(ProviderError):
        NullProvider().generate([{"role": "user", "content": "hi"}])


def test_build_provider_none(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "none")
    from the_machine.config.settings import Secrets
    p = build_provider(Secrets())
    assert isinstance(p, NullProvider)


def test_build_provider_ollama(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    from the_machine.config.settings import Secrets
    p = build_provider(Secrets(ai_base_url="http://localhost:11434",
                               ai_model="qwen2.5"))
    assert isinstance(p, OllamaProvider)
    assert p.available()


def test_build_provider_openai(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    from the_machine.config.settings import Secrets
    p = build_provider(Secrets(ai_api_key="sk-test",
                               ai_base_url="https://api.example.com/v1",
                               ai_model="gpt-4o-mini"))
    assert isinstance(p, OpenAICompatProvider)


def test_build_provider_default_null(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    from the_machine.config.settings import Secrets
    assert isinstance(build_provider(Secrets()), NullProvider)


def test_http_provider_unreachable_raises_provider_error():
    # port 1 is reliably refused; must raise ProviderError, never OSError
    p = OpenAICompatProvider("http://127.0.0.1:1", "k", "m", timeout_s=1)
    with pytest.raises(ProviderError):
        p.generate([{"role": "user", "content": "x"}])


# --------------------------------------------------------------- context engine
def test_context_engine_budget_and_flow():
    from the_machine.core.context_engine import ContextEngine
    bus = EventBus()
    state = StateManager(bus)
    ctx = ContextEngine(bus, FakePipeline(), state)
    for i in range(50):
        bus.emit("FACE_DETECTED", track_id=i, confidence=0.9)
    recent = ctx.recent_events()
    assert len(recent) <= ContextEngine.MAX_RECENT_EVENTS   # §76 budget

    ctx.remember("laptop name is Orion")
    msgs = ctx.build_messages("What is my laptop name?")
    assert msgs[0]["role"] == "system"
    assert "Orion" in msgs[0]["content"]
    assert msgs[-1]["content"].startswith("What is my laptop")
    assert "CURRENT OBSERVATION" in msgs[0]["content"]

    ctx.add_turn("user", "hello")
    ctx.add_turn("assistant", "hi")
    msgs2 = ctx.build_messages("again")
    assert {"role": "user", "content": "hello"} in msgs2
    ctx.clear_conversation()
    msgs3 = ctx.build_messages("fresh")
    assert msgs3[1]["role"] == "user" and msgs3[1]["content"] == "fresh"


def test_context_engine_no_frames_in_payload():
    """Privacy contract: context is text-only, tiny (§16/§8)."""
    import json
    from the_machine.core.context_engine import ContextEngine
    bus = EventBus()
    state = StateManager(bus)
    ctx = ContextEngine(bus, FakePipeline(), state)
    blob = json.dumps(ctx.vision_context())
    assert len(blob) < 2000          # structured summary, never pixels
    assert all(isinstance(v, (int, str, bool, list, dict))
               for v in ctx.vision_context().values())
