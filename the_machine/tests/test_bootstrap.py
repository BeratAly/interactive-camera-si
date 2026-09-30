"""Tests for the first-run terminal setup wizard (bootstrap)."""
from __future__ import annotations

import pytest

from the_machine.config import bootstrap


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setattr(bootstrap, "ENV_PATH", env)
    return env


def test_skip_when_already_configured(tmp_env, monkeypatch, capsys):
    monkeypatch.setenv("AI_PROVIDER", "nvidia")
    monkeypatch.setenv("AI_API_KEY", "nvapi-x")
    bootstrap.run_first_time_setup(interactive=True)
    assert not tmp_env.exists()          # never nags / rewrites
    assert "INITIAL SETUP" not in capsys.readouterr().out


def test_offline_default_writes_none(tmp_env, monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.setattr(bootstrap, "_input", lambda prompt="": "")  # Enter => 1
    bootstrap.run_first_time_setup(interactive=True)
    content = tmp_env.read_text(encoding="utf-8")
    assert "AI_PROVIDER=none" in content
    assert "nvapi" not in content


def test_nvidia_preset_saved(tmp_env, monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    answers = iter(["2", "nvapi-SECRET123", ""])   # choice, key, default model
    monkeypatch.setattr(bootstrap, "_input",
                        lambda prompt="": next(answers))
    bootstrap.run_first_time_setup(interactive=True)
    content = tmp_env.read_text(encoding="utf-8")
    assert "AI_PROVIDER=nvidia" in content
    assert "AI_API_KEY=nvapi-SECRET123" in content
    assert "integrate.api.nvidia.com" in content
    assert "nemotron-nano-8b" in content
    # env is updated for the current process too
    import os
    assert os.environ["AI_PROVIDER"] == "nvidia"
    os.environ.pop("AI_PROVIDER", None)
    os.environ.pop("AI_API_KEY", None)
    os.environ.pop("AI_BASE_URL", None)
    os.environ.pop("AI_MODEL", None)


def test_openrouter_key_skipped_stays_offline(tmp_env, monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    answers = iter(["3", ""])   # chose openrouter but pressed Enter at key
    monkeypatch.setattr(bootstrap, "_input",
                        lambda prompt="": next(answers))
    bootstrap.run_first_time_setup(interactive=True)
    content = tmp_env.read_text(encoding="utf-8")
    assert "AI_PROVIDER=none" in content


def test_non_interactive_does_not_write(tmp_env, monkeypatch, capsys):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    bootstrap.run_first_time_setup(interactive=False)
    assert not tmp_env.exists()
    assert "Non-interactive" in capsys.readouterr().out


def test_write_preserves_comments(tmp_env, monkeypatch):
    tmp_env.write_text("# keep me\nAI_PROVIDER=none\n", encoding="utf-8")
    bootstrap._write_env({"AI_PROVIDER": "nvidia", "AI_API_KEY": "k"}, tmp_env)
    content = tmp_env.read_text(encoding="utf-8")
    assert "# keep me" in content
    assert "AI_PROVIDER=nvidia" in content
    assert "AI_API_KEY=k" in content
