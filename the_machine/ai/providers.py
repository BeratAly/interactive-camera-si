"""AI provider abstraction (§15, §49). One interface, swappable backends.

    NullProvider     — default. Fully offline; deterministic answers from the
                       command router + real vision context. No install needed.
    OpenAICompat     — any OpenAI-compatible HTTP API (OpenAI, OpenRouter,
                       Together…). Key comes ONLY from .env / environment.
    OllamaProvider   — local LLM via Ollama's OpenAI-compatible endpoint.

Secrets are never logged or serialized. All providers return plain text and
raise ProviderError on failure so the brain can fall back gracefully (§32).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

logger = logging.getLogger("machine.ai.providers")


class ProviderError(RuntimeError):
    """Any recoverable provider failure (network, auth, timeout, bad model)."""


class AIProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Return a complete text answer. Raise ProviderError on failure."""

    def available(self) -> bool:
        return True


class NullProvider(AIProvider):
    """Offline fallback marker. The brain answers vision/state questions
    deterministically from REAL structured data when this is active."""

    name = "null"

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise ProviderError("no AI provider configured")


class _HttpChatProvider(AIProvider):
    """Shared OpenAI-compatible /chat/completions client (stdlib only)."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout_s: float = 30.0, temperature: float = 0.4) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._model = model
        self._timeout = timeout_s
        self._temperature = temperature

    def available(self) -> bool:
        return bool(self._base and self._model)

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.available():
            raise ProviderError(f"{self.name}: base_url/model not configured")
        body = json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": 400,
            "stream": False,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(f"{self._base}/chat/completions",
                                     data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # read error body but NEVER log request headers (contain the key)
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:200]
            except Exception:
                pass
            raise ProviderError(f"{self.name}: HTTP {e.code} {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise ProviderError(f"{self.name}: unreachable ({reason})") from e
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"{self.name}: malformed response") from e
        return str(text).strip()


class OpenAICompatProvider(_HttpChatProvider):
    name = "openai_compat"


class OllamaProvider(_HttpChatProvider):
    name = "ollama"

    def __init__(self, host: str, model: str, **kw) -> None:
        super().__init__(base_url=f"{host.rstrip('/')}/v1", api_key="",
                         model=model, **kw)


def build_provider(secrets) -> AIProvider:
    """Select provider from environment (.env). Nothing is sent anywhere
    unless the user explicitly configured a provider (§40 local-first).

    Precedence: AI_PROVIDER env var; otherwise ollama if AI_BASE_URL points
    at localhost, openai_compat if an API key exists, else NullProvider.
    """
    kind = os.environ.get("AI_PROVIDER", "").strip().lower()
    if kind == "none":
        return NullProvider()
    base = secrets.ai_base_url
    model = secrets.ai_model

    if kind == "ollama":
        host = base or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaProvider(host, model or "llama3.2")
    if kind in ("openai_compat", "openai"):
        return OpenAICompatProvider(base or "https://api.openai.com/v1",
                                    secrets.ai_api_key,
                                    model or "gpt-4o-mini")
    # auto-detect
    if "localhost" in base.lower() or "127.0.0.1" in base:
        return OllamaProvider(base, model or "llama3.2")
    if secrets.ai_api_key and base:
        return OpenAICompatProvider(base, secrets.ai_api_key,
                                    model or "gpt-4o-mini")
    return NullProvider()
