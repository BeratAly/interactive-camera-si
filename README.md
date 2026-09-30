# THE MACHINE

A local-first desktop AI system inspired visually by *Person of Interest*:
camera + microphone perception, computer vision (faces/objects/tracking/OCR),
an optional pluggable LLM brain, memory, and a futuristic dark UI.

**Current status: PHASE 0 — Architecture approved? Not yet. See `docs/ARCHITECTURE.md`.**

## Read first

- `docs/ARCHITECTURE.md` — full design: architecture, stack, hardware tiers, roadmap
  (Phase 0 → 14), file structure, security/privacy model, AI design, UI design,
  performance strategy, and the open questions that gate Phase 1.

## Principles (non-negotiable)

1. **Modular** — nothing lives in `main.py` except bootstrap.
2. **Local-first** — vision runs offline; frames/audio are never saved or uploaded by default.
3. **Thread discipline** — GUI thread only paints; capture/inference/LLM run in workers.
4. **Privacy visible** — camera/mic active state always shown in UI.
5. **No arbitrary execution** — system commands go through a whitelist + permissions + confirmations.
6. **Face recognition ethics** — only matches user-created local profiles; strangers stay `UNKNOWN`.
7. **Graceful degradation** — any module can be OFFLINE without crashing the app.

## Planned quick start (after Phase 1 is implemented)

```text
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # optional, only if using cloud AI
python main.py                # add --demo video.mp4 to run without a camera
```
