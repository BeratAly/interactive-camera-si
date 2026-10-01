"""AI Core / brain (§15, §29, §61): UI-agnostic request handler.

    ai_core.process_command(text)   # callable from ANY thread (GUI safe)

Flow per request:
  1. THINKING state + USER_SPEECH event
  2. command router fast-path (deterministic, offline, real data)
  3. otherwise provider.generate() with structured context (§16)
  4. provider failure => graceful fallback message; vision keeps working (§32)
  5. AI_RESPONSE event + conversation history update

All heavy/network work happens on this daemon thread — never the GUI thread.
"""
from __future__ import annotations

import logging
import queue
import threading

from the_machine.ai.command_router import route_command
from the_machine.ai.providers import AIProvider, ProviderError
from the_machine.core.context_engine import ContextEngine
from the_machine.core.event_bus import (
    AI_RESPONSE,
    EventPriority,
    EventBus,
    USER_SPEECH,
)
from the_machine.core.state_manager import StateManager, SystemState

logger = logging.getLogger("machine.ai.core")


class AICore(threading.Thread):
    def __init__(self, bus: EventBus, state: StateManager,
                 context: ContextEngine, provider: AIProvider) -> None:
        super().__init__(name="ai-core", daemon=True)
        self._bus = bus
        self._state = state
        self._ctx = context
        self._provider = provider
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=8)
        self._running = threading.Event()
        self._last_latency_s = 0.0

    # ------------------------------------------------------------ lifecycle
    def start_core(self) -> None:
        if not self._running.is_set():
            self._running.set()
            self.start()

    def close(self) -> None:
        self._running.clear()
        if self.is_alive():
            self.join(timeout=2.0)

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def is_llm_active(self) -> bool:
        return self._provider.available() and self._provider.name != "null"

    # ------------------------------------------------------------- public API
    def process_command(self, text: str) -> None:
        """Non-blocking: enqueue for the worker thread. GUI never waits."""
        clean = text.strip()
        if not clean:
            return
        try:
            self._q.put_nowait(clean)
        except queue.Full:
            logger.warning("ai queue full, dropping request")

    # ---------------------------------------------------------------- worker
    def run(self) -> None:
        while self._running.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle(text)

    def _handle(self, user_text: str) -> None:
        prev = self._state.state
        self._state.set(SystemState.THINKING, reason="user query")
        self._bus.emit(USER_SPEECH, priority=EventPriority.NORMAL,
                       text=user_text[:200])
        try:
            answer, source = self._answer(user_text)
        except Exception as e:  # brain must never crash the app (§32)
            logger.exception("ai core failure")
            answer = ("AI CORE OFFLINE — goruntu ozellikleri calismaya devam "
                      "ediyor. / AI core error, vision features remain "
                      "available.")
            source = "error"
            self._bus.emit("SYSTEM_ERROR", priority=EventPriority.HIGH,
                           module="ai_core", message=str(e)[:200])
        finally:
            if self._state.state is SystemState.THINKING:
                self._state.set(prev if prev in
                                (SystemState.READY, SystemState.LISTENING)
                                else SystemState.READY, reason="answer ready")

        self._ctx.add_turn("user", user_text)
        self._ctx.add_turn("assistant", answer)
        self._bus.emit(AI_RESPONSE, priority=EventPriority.NORMAL,
                       text=answer[:600], source=source,
                       latency_s=round(self._last_latency_s, 2))
        logger.info("ai answered (%s): %s", source, answer[:120])

    # -------------------------------------------------------------- reasoning
    def _answer(self, user_text: str) -> tuple[str, str]:
        import time
        t0 = time.monotonic()
        vctx = self._ctx.vision_context()
        fps = self._ctx.fps_data()          # real capture/detect rates
        vctx["capture_fps"] = fps["capture_fps"]
        vctx["detect_fps"] = fps["detect_fps"]

        result = route_command(user_text, vctx, self._ctx.recent_events(),
                               self._state.state.value,
                               self._ctx.memories_snapshot())
        if result is not None:
            if result.clear_conversation:
                self._ctx.clear_conversation()
            if result.remember:
                self._ctx.remember(result.remember)
            self._last_latency_s = time.monotonic() - t0
            return (result.answer or "Yapildi. / Done.", "router")

        # LLM path
        messages = self._ctx.build_messages(user_text)
        try:
            text_out = self._provider.generate(messages)
            self._last_latency_s = time.monotonic() - t0
            return text_out, self._provider.name
        except ProviderError as e:
            logger.warning("provider unavailable: %s", e)
            self._last_latency_s = time.monotonic() - t0
            people = vctx.get("people", 0)
            fallback = (
                "LLM saglayicisi baglanamadi. Yerel yanit: gorusumde "
                f"{people} kisi var. / No LLM provider reachable — locally: "
                f"{people} person(s) in view. Configure a provider in .env "
                "for conversational answers."
            )
            return fallback, "fallback"
