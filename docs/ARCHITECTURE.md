# THE MACHINE — Architecture & Development Plan (v0.1)

> Status: **PHASE 0 — Architecture**. No production code until design approval.
> Inspired by *Person of Interest* visually; engineered as a real, local-first desktop AI system.

---

## A. General Architecture

Layered, event-driven, thread-isolated. Modules never call each other directly for
cross-cutting concerns; they communicate through a central **Event Bus** and read/write
shared state through a **State Manager** (single source of truth).

```
        ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
sensors │   CAMERA    │   │  MICROPHONE  │   │ SCREEN (opt) │
        └──────┬──────┘   └──────┬───────┘   └──────┬───────┘
               ▼                 ▼                  │
        ┌──────────────────────────────┐            │
        │  VISION ENGINE (worker thr.) │            │
        │  detect → recognize → track  │            │
        │  OCR / scene analysis (late) │            │
        └──────────────┬───────────────┘            │
                       ▼                            ▼
        ┌──────────────────────────────┐   ┌────────────────┐
        │  CONTEXT ENGINE              │◄──│ MEMORY SYSTEM  │
        │  structured snapshot (JSON)  │   │ SQLite + opt.  │
        └──────────────┬───────────────┘   │ embeddings     │
                       ▼                   └────────┬───────┘
        ┌──────────────────────────────┐            │
        │  AI CORE (provider-agnostic) │────────────┘
        │  command router / reasoning  │
        └───┬──────────────┬───────────┘
            ▼              ▼
     ┌────────────┐  ┌──────────────┐     ┌──────────────────┐
     │ UI (Qt thr)│  │ AUDIO OUT/TTS│     │ COMMAND MANAGER  │
     │ HUD+panels │  │ WAKE WORD    │     │ whitelist+perm.  │
     └────────────┘  └──────────────┘     └──────────────────┘

        Everything publishes/consumes on the EVENT BUS.
```

**Key invariants**
1. GUI thread only renders. All heavy work runs in worker threads with bounded queues
   (drop-oldest policy — vision must never lag behind the camera).
2. Vision output is **structured context**, never raw frames, sent to the AI.
3. Every capability is a module that can fail independently (`MODULE OFFLINE` states),
   never crashing the app.
4. Privacy defaults are safe-by-default: no frames/audio saved, no cloud, face DB local only.

---

## B. Technology Stack (recommended)

| Concern | Choice | Why | Alternative considered |
|---|---|---|---|
| Language | Python 3.11+ | ecosystem, typing | — |
| GUI | **PySide6** (LGPL) | native perf, QPainter overlays, threads | Tkinter (too limited), Electron (heavy, non-local feel) |
| Camera | **OpenCV** `VideoCapture` | cross-platform, DSHOW/MSMF/V4L2 backends | DirectShow wrappers (Win-only) |
| Face detection | **ONNX Runtime + SCRFD/YuNet** | fast CPU, GPU-ready, no torch needed | MediaPipe (pipeline lock-in), MTCNN (slow) |
| Face recognition | **ONNX ArcFace embeddings** + cosine matching | 128-d vectors, local-only | face_recognition/dlib (heavy build on Windows) |
| Object detection | **YOLOv8n/YOLO11n via ultralytics or plain ONNX** | COCO classes, tracker-friendly | YOLO-NAS, RT-DETR (heavier) |
| Tracking | **ByteTrack** (via ultralytics) or lightweight IoU/centroid fallback | session IDs, anti-jitter | DeepSORT (needs extra models) |
| OCR (later phase) | **RapidOCR / PaddleOCR-onnx** | offline, light | Tesseract (weak on photos) |
| STT | **faster-whisper (small/base)** | offline, tr+en | Vosk (lighter, less accurate) |
| Wake word | **openWakeWord** ("Machine") or Porcupine (commercial, best accuracy) | fully local | always-on Whisper (expensive) |
| TTS | **Piper** (local) → SAPI/win32 fallback | offline natural voice | Cloud TTS (privacy trade-off, optional) |
| Database | **SQLite** (WAL mode) + schema_version migrations | zero-config, local | LMDB (overkill now) |
| Config | YAML (`config.yaml`) + `.env` (secrets only) + dataclass settings | separation of secrets vs prefs | pydantic-settings (adopt inside settings.py) |
| LLM | Provider abstraction: Ollama / OpenAI-compatible / none | swappable, offline-capable | hard-coded SDK (forbidden) |
| Packaging | PyInstaller later (Phase 14) | single .exe | Nuitka |

Dependencies kept minimal; each one justified in `requirements.txt` comments.

---

## C. Hardware Requirements

**Minimum (CPU-only, "basic features" tier):**
- 4-core x86-64 (Intel 8th gen / Ryzen 2000+), 8 GB RAM
- Webcam 720p, mic (any), ~3 GB disk
- Expectation: camera 30 FPS, face det ~15–20 FPS, obj det ~8–12 FPS (nano models),
  LLM via cloud (small local model not recommended at this tier)

**Recommended:**
- 6+ cores, 16 GB RAM, NVIDIA GTX 1660/RTX (CUDA via onnxruntime-gpu / ultralytics)
- Expectation: all vision ≥20 FPS concurrent, whisper-small near real-time, Piper TTS instant

**Not required:** GPU, internet (system degrades gracefully; vision works fully offline).

---

## D. Roadmap (Phase 0 → Final)

Each phase ends with a runnable, tested build. Version tags v0.x.

| Phase | Deliverable | Fail criteria guarded (§82) |
|---|---|---|
| 0 | This architecture + repo skeleton, config, logging, event bus | — |
| 1 | **Camera in PySide6 GUI**: device/resolution pick, live feed, true FPS, animated face box (UNKNOWN), event log, camera status, disconnect/retry, DEMO MODE (video file) | no GUI freeze, no blocking, stable boxes |
| 2 | Face detection hardened: smoothing/interpolation, fade in/out, confidence | jitter |
| 3 | Futuristic UI pass: HUD overlays, cheap scanlines, status panels, theme | readability, CPU cost |
| 4 | Object detection (COCO subset) + list panel | frame budget |
| 5 | ByteTrack session IDs (PERSON #001…), lost/recover logic | ID flicker |
| 6 | Face profiles: enrollment wizard (10 samples), local embedding DB, recognition, Guest/Unknown ethics | privacy |
| 7 | AI Core: provider abstraction, context engine, chat panel, streaming, commands ("what do you see", "who am I") | hallucination guard |
| 8 | Voice: wake word → listen → STT → AI → TTS state machine, TR/EN | latency |
| 9 | Memory: short-term/session/long-term, "remember that…", delete/export data | retention |
| 10 | System commands whitelist (open app, screenshot, volume…) + permissions + confirmations | arbitrary exec |
| 11 | Screen analysis (default OFF) + OCR question answering | privacy |
| 12 | Plugin system + manifests + permissions | sandboxing |
| 13 | Optimization: profiling, inference decoupling (cam 30 / det 10 / ui 60), debug panel | regressions |
| 14 | Packaging: run.bat/sh, PyInstaller, README install guide | startup UX |

---

## E. File Structure

As proposed in the brief, with two additions: `context/` (context engine is its own concern)
and `utils/`:

```
the_machine/
├── main.py                  # bootstrap only: init order, shutdown, crash handler
├── config/{config.yaml, settings.py}
├── core/{ai_core.py, event_bus.py, state_manager.py, command_manager.py, scheduler.py}
├── context/context_engine.py
├── vision/{camera.py, face_detector.py, face_recognition.py, object_detector.py,
│           tracker.py, ocr.py, scene_analyzer.py, pipeline.py}
├── audio/{microphone.py, speech_to_text.py, text_to_speech.py, wake_word.py}
├── memory/{database.py, short_term.py, long_term.py, embeddings.py}
├── ui/{main_window.py, camera_view.py, hud.py, widgets/, themes/}
├── security/{permissions.py, encryption.py, privacy.py}
├── plugins/
├── data/{profiles/, memory/, logs/, models/}   # gitignored
├── tests/
├── requirements.txt / README.md / .env.example / .gitignore
```

---

## F. Security & Privacy Model

- **Local-first**: frames/audio never leave the machine unless Cloud AI explicitly enabled.
- **Defaults**: `save_frames=false`, `save_audio=false`, `screen_sharing=false`, `cloud_ai=off`.
- **Visible capture state**: persistent `● CAMERA ACTIVE` / `● MIC LISTENING` indicators; wake-word
  engine is local only (no continuous cloud streaming).
- **No identity lookup online**: unknown faces remain `UNKNOWN` (ethical rule §78).
- **Secrets**: `.env` only (gitignored); keys never logged; log redaction filter for tokens/PII.
- **Commands**: whitelist registry (`OPEN_APPLICATION`, `TAKE_SCREENSHOT`, …) → permission check →
  validation → user confirmation for risky ops → execute. **Shell strings are never constructed.**
- **Plugins**: manifest declares permissions; denied unless granted by user.
- **Data lifecycle**: DELETE PROFILE/MEMORY/SESSION/ALL DATA + JSON export (self-service deletion).
- **Future remote access (§96–97)**: off by default; would require auth + TLS + rate limiting.

---

## G. AI Architecture

- `AIProvider` interface: `generate()`, `stream()`, `vision()` (optional), `is_available()`.
- Implementations: `OllamaProvider` (local), `OpenAICompatProvider` (any base URL), `NullProvider`
  (offline fallback — deterministic command answers still work without any LLM).
- **Context Engine** builds a bounded JSON snapshot (≤ ~1 KB): people count, identities+confidence,
  top objects, last OCR text, scene summary, recent ≤10 events, relevant memories, time, profile.
  Raw frames go to a model only if it is vision-capable, explicitly configured, AND the user asked.
- Anti-hallucination prompt contract: answer only from `observed:` fields; hedge with
  "appears/probably"; confidence bands (high ≥0.9, medium 0.7–0.9, low <0.7); never claim certainty.
- Command Router runs **before** the LLM: local intents ("fps kaç", "kamerayı kapat", "who am I")
  resolve instantly without network/cost; unmatched input falls through to LLM with context.
- Personality: calm, concise, technical (config-tunable system prompt; not a show-character copy).

---

## H. UI Architecture

- Single main window, grid layout: **camera view (center-left)**, **status sidebar (right)**,
  **event log (bottom)**, collapsible **chat panel**, **settings dialog**, optional boot splash (§60).
- Rendering strategy: camera frames drawn via `QPainter` on one dedicated widget; HUD overlays
  (boxes, scanline, crosshair, coordinates, timestamps) painted in the same pass — one blit per
  frame, no stacked translucent widgets (the classic Qt perf trap).
- Boxes: exponential smoothing (Kalman-style interpolation), alpha fade in/out over ~200 ms,
  corner-bracket aesthetic + name/confidence label. Animations timer-driven at paint rate.
- Theme: dark, mono technical font, accent color TBD (green vs amber vs cyan — your call, §J-11).
- States (`OFFLINE/STARTING/READY/LISTENING/THINKING/SPEAKING/ERROR`) drive header badge + subtle
  border/glow changes. Debug info hidden behind Developer Mode toggle.

---

## I. Performance Strategy

- Decoupled rates: **capture 30 FPS → detection every N frames (≈10 FPS) → tracking interpolates
  between detections → UI paints at display refresh**. Latest-result channel (queue size 1, never grows).
- Worker threads with drop-oldest queues; results published as immutable dataclasses.
- ONNX Runtime: `CPUExecutionProvider` default, `CUDAExecutionProvider` auto-detected; graceful fallback.
- Budget targets (CPU-only): face det ≤ 40 ms, obj det ≤ 90 ms per processed frame; total vision
  < 45% of one core at 10 FPS.
- Models lazy-loaded; disabled features cost zero (OCR off = never imported/instantiated).
- Profiling counters exposed in Debug Mode (inference ms, queue size, latencies) — measured, not guessed.

---

## J. Information Needed From You (answers gate the start of Phase 1)

**Hardware**
1. CPU model? 2. GPU (NVIDIA? VRAM?) 3. RAM? 4. Webcam present/model? 5. Mic?
6. Windows or Linux (primary)? 7. Python installed (version)?

**AI**
8. Cloud AI acceptable? (OpenAI-compatible API key available?) 9. Want local LLM (Ollama)?
10. Always-on internet?

**UI**
11. Accent theme: green / amber / blue-cyan? 12. Windowed or fullscreen? 13. Monitor count/resolution?
14. Touchscreen?

**Voice**
15. Primary language TR or EN? 16. Wake word wanted in first releases? 17. Push-to-talk instead?

**Privacy**
18. Frame/event retention: keep defaults (nothing saved)? 19. Audio saved?
20. If cloud AI: only structured text context is sent — OK?

**Scope decisions**
21. Confirm Phase 1 scope exactly as §81 (camera + animated face box + event log + futuristic UI, no LLM yet).
22. OK with ONNX-based detectors (one-time model download to `data/models/`) vs pure Haar/OpenCV
    (zero download, lower quality)? **Recommendation: ONNX YuNet/SCRFD + YOLO-nano; Haar only as demo fallback.**

Once answered, I will scaffold the repo (Phase 0 skeleton) and begin Phase 1.
