"""Shared pytest fixtures: isolate every test from ambient env vars.

CI machines (and the developer's shell) may export AI_PROVIDER / AI_API_KEY
etc.; tests must never depend on that (§56).
"""
from __future__ import annotations

import os

import pytest

AI_VARS = ("AI_PROVIDER", "AI_API_KEY", "AI_BASE_URL", "AI_MODEL", "OLLAMA_HOST")


@pytest.fixture(autouse=True)
def _clean_ai_env(monkeypatch):
    """Remove AI_* variables from the process environment for each test."""
    for var in AI_VARS:
        monkeypatch.delenv(var, raising=False)
