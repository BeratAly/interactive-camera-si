# THE MACHINE

A local-first desktop AI system inspired visually by *Person of Interest*:
camera perception (faces/objects/tracking), an AI brain with chat, a
pluggable LLM provider (offline / NVIDIA NIM free tier / OpenRouter / Ollama),
and a futuristic dark UI.

**Current status: PHASE 1 + AI CORE — working build.** Vision pipeline,
animated face boxes, event log, chat panel and command router are live.

## Quick start (GitHub → 3 clicks)

```text
1. git clone https://github.com/<you>/the-machine.git
2. cd the-machine
3. Windows: double-click run.bat        Linux/macOS: ./run.sh
```

The scripts create a virtual environment and install dependencies on first
run. On **first launch the terminal asks** how the AI should connect:

```text
[1] OFFLINE / LOCAL MODE   — no key needed, vision commands work (default)
[2] NVIDIA NIM             — FREE cloud models (paste nvapi-... key)
[3] OpenRouter             — FREE cloud models (paste sk-or-... key)
[4] Ollama                 — local LLM on your PC
[5] Other OpenAI-compatible API
```

Your answers are saved to `.env`, which is **gitignored — the key never
enters the repository or leaves your PC unless you pick a cloud option**.
You can change it later by deleting `.env` and restarting, or editing the
file directly (template: `.env.example`).

Manual alternative:

```text
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m the_machine.main            # add --demo to run without a camera
```

Free NVIDIA key: sign in at <https://build.nvidia.com> (free account) →
pick any model → "Get API Key" → paste into the setup prompt.

## Using the AI

Type into the **AI chat panel** (or just talk once the voice phase lands).
Works fully offline; Turkish and English:

```text
Ne görüyorsun?          What do you see?
Kaç kişi var?           How many people?
Ben kimim?              Who am I?
FPS kaç?                Sistem durumu / system status
Son olaylar neler?      What happened recently?
Konuşmayı temizle       Clear conversation
Hatırla: ...            Remember that ...
```

With a cloud/local LLM configured, open-ended questions are answered too —
always grounded in the structured vision context (never raw frames).

## Principles (non-negotiable)

1. **Modular** — nothing lives in `main.py` except bootstrap.
2. **Local-first** — vision runs offline; frames/audio are never saved or uploaded by default.
3. **Thread discipline** — GUI thread only paints; capture/inference/LLM run in workers.
4. **Privacy visible** — camera/mic active state always shown in UI.
5. **No arbitrary execution** — system commands go through a whitelist + permissions + confirmations.
6. **Face recognition ethics** — only matches user-created local profiles; strangers stay `UNKNOWN`.
7. **Graceful degradation** — any module can be OFFLINE without crashing the app.

Full design: `docs/ARCHITECTURE.md`.

## Development

```text
pip install pytest
python -m pytest the_machine/tests -q          # unit tests (no hardware needed)
python -m the_machine.main --demo              # synthetic camera, no webcam
python -m the_machine.main --demo --screenshot preview.png   # headless UI render
```

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 0 | Architecture | done |
| 1 | Camera + HUD + face boxes + event log | done |
| 7 | AI Core (chat, command router, providers incl. NVIDIA free tier) | done |
| 2–6 | ONNX detectors, object detection, tracking, profiles | next |
| 8 | Voice (wake word "Machine", STT/TTS) | planned |
| 9 | Memory database (SQLite) | planned |
| 10–14 | System commands, screen analysis, plugins, packaging | planned |
