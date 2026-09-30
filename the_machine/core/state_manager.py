"""Central system state machine (§61)."""
from __future__ import annotations

import threading
from enum import Enum

from the_machine.core.event_bus import (
    EventPriority,
    EventBus,
    STATE_CHANGED,
)


class SystemState(str, Enum):
    OFFLINE = "OFFLINE"
    STARTING = "STARTING"
    READY = "READY"
    LISTENING = "LISTENING"   # phase 8
    THINKING = "THINKING"     # phase 7
    SPEAKING = "SPEAKING"     # phase 8
    ERROR = "ERROR"


# States that mean "vision is working" for UI purposes.
_ACTIVE = {SystemState.STARTING, SystemState.READY, SystemState.LISTENING,
           SystemState.THINKING, SystemState.SPEAKING}


class StateManager:
    def __init__(self, bus: EventBus) -> None:
        self._state = SystemState.OFFLINE
        self._lock = threading.Lock()
        self._bus = bus

    @property
    def state(self) -> SystemState:
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        return self.state in _ACTIVE

    def set(self, new_state: SystemState, reason: str = "") -> None:
        with self._lock:
            if new_state == self._state:
                return
            old = self._state
            self._state = new_state
        priority = (EventPriority.CRITICAL if new_state is SystemState.ERROR
                    else EventPriority.NORMAL)
        self._bus.emit(STATE_CHANGED, priority=priority,
                       old=old.value, new=new_state.value, reason=reason)
