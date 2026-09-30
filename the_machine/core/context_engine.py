"""Context engine (§16, §75, §76): builds the STRUCTURED text context for AI.

Privacy contract: raw camera frames are NEVER part of this context. Only
lightweight structured data (face count/confidence, scene description,
recent events, conversation tail, user-marked memories) flows to a provider.
Context budget is deliberately small (§76).
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any


class ContextEngine:
    MAX_RECENT_EVENTS = 10      # §76 — never ship hundreds of events
    MAX_MEMORY_ITEMS = 15
    MAX_CONVERSATION_TURNS = 12

    def __init__(self, bus, pipeline, state) -> None:
        self._bus = bus
        self._pipeline = pipeline
        self._state = state
        self._camera = None
        self._lock = threading.Lock()
        self._memories: deque[str] = deque(maxlen=self.MAX_MEMORY_ITEMS)
        self._conversation: deque[dict[str, str]] = deque(
            maxlen=self.MAX_CONVERSATION_TURNS * 2)
        self._scene_description: str = ""

        bus.subscribe("OBJECT_DETECTED", self._on_object)
        bus.subscribe("FACE_RECOGNIZED", self._on_face_recognized)

    # ------------------------------------------------------------- ingestion
    def _on_object(self, event) -> None:
        name = str(event.payload.get("name", ""))[:40]
        if name:
            self._scene_description = f"{name} visible in view"

    def _on_face_recognized(self, event) -> None:
        ident = str(event.payload.get("identity", ""))[:40]
        if ident and ident != "UNKNOWN":
            self._scene_description = f"recognized person present: {ident}"

    def remember(self, text: str) -> None:
        """Long-term memory ONLY when the user explicitly asks (§17/§53)."""
        clean = " ".join(text.split())[:200]
        if clean:
            with self._lock:
                self._memories.append(clean)

    def add_turn(self, role: str, content: str) -> None:
        with self._lock:
            self._conversation.append({"role": role, "content": content[:800]})

    def clear_conversation(self) -> None:
        with self._lock:
            self._conversation.clear()

    def forget_all(self) -> None:
        with self._lock:
            self._memories.clear()
            self._conversation.clear()
            self._scene_description = ""

    def memories_snapshot(self) -> list[str]:
        with self._lock:
            return list(self._memories)

    def set_fps_source(self, camera) -> None:
        """Optional wiring so FPS queries report real capture rate."""
        self._camera = camera

    def fps_data(self) -> dict[str, float]:
        cam_fps = 0.0
        if getattr(self, "_camera", None) is not None:
            try:
                cam_fps = float(self._camera.measured_fps())
            except Exception:
                cam_fps = 0.0
        return {"capture_fps": cam_fps,
                "detect_fps": float(self._pipeline.snapshot().detect_fps)}

    # ------------------------------------------------------------- snapshot
    def vision_context(self) -> dict[str, Any]:
        """Structured observation from the vision engine (§16 JSON shape)."""
        snap = self._pipeline.snapshot()
        faces = [
            {
                "identity": t.identity,
                "confidence": round(float(t.confidence), 3),
                "track_id": int(t.track_id),
            }
            for t in snap.tracks
        ]
        return {
            "people": len(faces),
            "faces": faces,
            "scene": self._scene_description or "no additional scene info",
            "camera_online": bool(snap.frame_size[0]),
        }

    def recent_events(self) -> list[str]:
        out: list[str] = []
        for ev in list(self._bus.history)[-40:]:
            p = ev.payload
            if ev.name == "FACE_DETECTED":
                out.append(f"{ev.local_time()} face detected "
                           f"(id {p.get('track_id')}, conf "
                           f"{float(p.get('confidence', 0)):.0%})")
            elif ev.name == "FACE_LOST":
                out.append(f"{ev.local_time()} face lost (id {p.get('track_id')})")
            elif ev.name == "OBJECT_DETECTED":
                out.append(f"{ev.local_time()} object detected: "
                           f"{p.get('name')} {float(p.get('confidence', 0)):.0%}")
            elif ev.name == "CAMERA_CONNECTED":
                out.append(f"{ev.local_time()} camera connected")
            elif ev.name in ("CAMERA_DISCONNECTED", "CAMERA_ERROR"):
                out.append(f"{ev.local_time()} camera unavailable")
            elif ev.name == "USER_SPEECH":
                out.append(f"{ev.local_time()} user asked a question")
            elif ev.name == "AI_RESPONSE":
                out.append(f"{ev.local_time()} AI responded")
        return out[-self.MAX_RECENT_EVENTS:]

    def build_messages(self, user_text: str) -> list[dict[str, str]]:
        """System + memory + events + conversation tail + current question."""
        import json
        ctx = self.vision_context()
        with self._lock:
            memories = list(self._memories)
            convo = list(self._conversation)

        system = (
            "You are The Machine, a personal desktop AI. Personality: calm, "
            "professional, concise, technical. Answer in the language the user "
            "writes in (Turkish or English). Keep replies short (1-3 sentences) "
            "unless asked for detail.\n"
            "You receive a structured OBSERVATION block from your vision engine. "
            "Only describe what appears there. If the observation is empty or "
            "offline, say you cannot see anything right now — never invent "
            "objects or people. Present confidence as likelihood, never as "
            "certainty.\n"
            f"CURRENT OBSERVATION (JSON): {json.dumps(ctx, ensure_ascii=False)}\n"
            f"RECENT EVENTS: {json.dumps(self.recent_events(), ensure_ascii=False)}\n"
            f"STORED MEMORIES (user-approved): {json.dumps(memories, ensure_ascii=False)}\n"
            f"SYSTEM STATE: {self._state.state.value}"
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(convo)
        messages.append({"role": "user", "content": user_text[:1500]})
        return messages
