"""Central event bus (§19). Modules communicate only through events.

Thread-safety contract: publish() may be called from ANY thread.
Subscribers that touch the GUI must use subscribe_gui() (Qt queued signal)
or emit via Qt signals themselves; plain subscribe() callbacks run on the
publishing thread.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable

logger = logging.getLogger("machine.event")


class EventPriority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# Canonical event names (§19) — phases add more as needed.
FACE_DETECTED = "FACE_DETECTED"
FACE_LOST = "FACE_LOST"
FACE_RECOGNIZED = "FACE_RECOGNIZED"       # phase 6
OBJECT_DETECTED = "OBJECT_DETECTED"       # phase 4
CAMERA_CONNECTED = "CAMERA_CONNECTED"
CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
CAMERA_ERROR = "CAMERA_ERROR"
WAKE_WORD = "WAKE_WORD"                   # phase 8
USER_SPEECH = "USER_SPEECH"               # phase 8
AI_RESPONSE = "AI_RESPONSE"               # phase 7
SYSTEM_ERROR = "SYSTEM_ERROR"
STATE_CHANGED = "STATE_CHANGED"
CAMERA_PAUSED = "CAMERA_PAUSED"           # user toggle (not a failure)
CAMERA_RESUMED = "CAMERA_RESUMED"
PROFILE_LEARNED = "PROFILE_LEARNED"       # enrollment completed (§35)
DATA_CLEARED = "DATA_CLEARED"             # §55 right-to-be-forgotten


@dataclass(frozen=True)
class MachineEvent:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def local_time(self) -> str:
        return self.timestamp.astimezone().strftime("%H:%M:%S")


EventHandler = Callable[[MachineEvent], None]


class EventBus:
    def __init__(self, history_size: int = 500) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._lock = threading.RLock()
        self.history: deque[MachineEvent] = deque(maxlen=history_size)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            self._handlers.setdefault(event_name, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Receive every event (used by event log / history panel)."""
        with self._lock:
            self._global_handlers.append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_name, [])
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: MachineEvent) -> None:
        with self._lock:
            self.history.append(event)
            handlers = list(self._handlers.get(event.name, [])) + list(self._global_handlers)
        for h in handlers:
            try:
                h(event)
            except Exception:  # a broken handler must never crash the bus
                logger.exception("event handler failed for %s", event.name)

    def emit(self, name: str, priority: EventPriority = EventPriority.NORMAL,
             **payload: Any) -> None:
        self.publish(MachineEvent(name=name, payload=payload, priority=priority))
