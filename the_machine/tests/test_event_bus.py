"""Unit tests for the event bus (§56). No Qt, no camera — pure logic."""
from __future__ import annotations

import threading

import pytest

from the_machine.core.event_bus import (
    EventPriority,
    EventBus,
    FACE_DETECTED,
    MachineEvent,
)


def test_publish_subscribe_roundtrip() -> None:
    bus = EventBus()
    got: list[MachineEvent] = []
    bus.subscribe(FACE_DETECTED, got.append)
    bus.emit(FACE_DETECTED, track_id=1, confidence=0.9)
    assert len(got) == 1
    assert got[0].payload["track_id"] == 1


def test_history_records_all_events() -> None:
    bus = EventBus(history_size=3)
    for i in range(5):
        bus.emit("X", i=i)
    assert len(bus.history) == 3          # bounded retention
    assert [e.payload["i"] for e in bus.history] == [2, 3, 4]


def test_handler_exception_does_not_break_bus() -> None:
    bus = EventBus()
    calls: list[int] = []

    def boom(_e: MachineEvent) -> None:
        raise RuntimeError("bad handler")

    bus.subscribe("E", boom)
    bus.subscribe("E", lambda e: calls.append(1))
    bus.emit("E")
    assert calls == [1]                    # second handler still ran


def test_unsubscribe() -> None:
    bus = EventBus()
    seen: list[int] = []
    h = lambda e: seen.append(1)  # noqa: E731
    bus.subscribe("E", h)
    bus.emit("E")
    bus.unsubscribe("E", h)
    bus.emit("E")
    assert len(seen) == 1


def test_threadsafe_publish_from_many_threads() -> None:
    bus = EventBus()
    count = 0
    lock = threading.Lock()

    def handler(_e: MachineEvent) -> None:
        nonlocal count
        with lock:
            count += 1

    bus.subscribe("T", handler)
    threads = [threading.Thread(target=lambda: [bus.emit("T") for _ in range(50)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count == 200


def test_priority_levels_exist() -> None:
    assert EventPriority.LOW < EventPriority.NORMAL < EventPriority.HIGH \
        < EventPriority.CRITICAL
