"""First-run interactive setup (§69, §102).

Asks the user in the terminal for AI provider preferences and writes them to
the local `.env` file (which is gitignored — keys never enter the repository).
Everything is optional: pressing Enter keeps the privacy-first default
(fully offline, no cloud).

Non-interactive contexts (CI, pipes, --no-ask) skip this entirely.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from the_machine.config.settings import PROJECT_ROOT, load_dotenv

ENV_PATH = PROJECT_ROOT / ".env"

# Curated free-tier presets (NVIDIA NIM & OpenRouter need only a free key).
PRESETS: dict[str, dict[str, str]] = {
    "nvidia": {
        "AI_PROVIDER": "nvidia",
        "AI_BASE_URL": "https://integrate.api.nvidia.com/v1",
        "AI_MODEL": "nvidia/llama-3.1-nemotron-nano-8b-v1",
    },
    "openrouter": {
        "AI_PROVIDER": "openai_compat",
        "AI_BASE_URL": "https://openrouter.ai/api/v1",
        "AI_MODEL": "meta-llama/llama-3.1-8b-instruct:free",
    },
    "ollama": {
        "AI_PROVIDER": "ollama",
        "AI_BASE_URL": "",
        "AI_MODEL": "llama3.2",
    },
}


def _clean(value: str) -> str:
    """Strip quotes/inline comments from a .env value."""
    v = value.strip().strip('"').strip("'").split(" #")[0].strip()
    return v


def _input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _read_env_lines(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, v = s.partition("=")
            out[k.strip()] = v.strip()
    return out


def _write_env(updates: dict[str, str], path: Path | None = None) -> None:
    """Update KEY=VALUE lines in .env in place; preserve comments."""
    path = path or ENV_PATH
    existing_template = Path(__file__).resolve().parents[2] / ".env.example"
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    elif existing_template.exists():
        lines = existing_template.read_text(encoding="utf-8").splitlines()

    updates = {k: _clean(v) for k, v in updates.items()}
    if not updates.get("AI_API_KEY", ""):
        updates.pop("AI_API_KEY", None)
    written: set[str] = set()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.partition("=")[0].strip()
            if k in updates:
                out.append(f"{k}={updates[k]}")
                written.add(k)
                continue
        out.append(line)
    for k, v in updates.items():
        if k not in written:
            out.append(f"{k}={v}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _configured_provider() -> str:
    """Effective AI_PROVIDER from real env first, then .env file values."""
    val = os.environ.get("AI_PROVIDER", "")
    if not val and ENV_PATH.exists():
        val = _read_env_lines(ENV_PATH).get("AI_PROVIDER", "")
    return val.partition("#")[0].strip().strip('"').strip("'").lower()


def run_first_time_setup(interactive: bool | None = None) -> None:
    """Prompt once, only when no provider is configured yet."""
    load_dotenv()  # populate env so we can detect existing config
    provider = _configured_provider()
    if provider not in ("", "none"):
        return  # already configured — never nag again

    if interactive is None:
        interactive = sys.stdin.isatty() and "--no-ask" not in sys.argv
    if not interactive:
        print("[SETUP] Non-interactive session: staying OFFLINE (local mode). "
              "Configure AI later by editing .env or deleting it and re-running.")
        return

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  THE MACHINE — INITIAL SETUP                     ║")
    print("║  All steps optional. Press ENTER to skip any     ║")
    print("║  question and keep the offline/private default.  ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("How should the AI brain connect?")
    print("  [1] OFFLINE / LOCAL MODE   — no install, no key, vision commands work")
    print("  [2] NVIDIA NIM             — FREE cloud models (nvapi-... key)")
    print("  [3] OpenRouter             — FREE cloud models (sk-or-... key)")
    print("  [4] Ollama                 — local LLM on your own PC")
    print("  [5] Other OpenAI-compatible API")
    choice = _input("Select [1-5] (default 1): ") or "1"

    updates: dict[str, str] = {}
    if choice == "2":
        preset = PRESETS["nvidia"]
        key = _input("NVIDIA API key (from https://build.nvidia.com , starts with nvapi-, Enter=skip): ")
        if key.startswith("nvapi-") or key:
            updates = {**preset, "AI_API_KEY": key}
            model = _input(f"Model [Enter = {preset['AI_MODEL']}]: ")
            if model:
                updates["AI_MODEL"] = model
        else:
            print("No key entered → staying offline.")
            updates = {"AI_PROVIDER": "none"}
    elif choice == "3":
        preset = PRESETS["openrouter"]
        key = _input("OpenRouter API key (from https://openrouter.ai , starts with sk-or-, Enter=skip): ")
        if key:
            updates = {**preset, "AI_API_KEY": key}
            model = _input(f"Free model [Enter = {preset['AI_MODEL']}]: ")
            if model:
                updates["AI_MODEL"] = model
        else:
            print("No key entered → staying offline.")
            updates = {"AI_PROVIDER": "none"}
    elif choice == "4":
        preset = PRESETS["ollama"]
        host = _input("Ollama host [Enter = http://localhost:11434]: ")
        model = _input("Model name [Enter = llama3.2]: ")
        updates = {**preset,
                   "AI_BASE_URL": host.rstrip("/") if host else "",
                   "AI_MODEL": model or "llama3.2"}
    elif choice == "5":
        base = _input("Base URL (e.g. https://api.openai.com/v1): ")
        model = _input("Model name (e.g. gpt-4o-mini): ")
        key = _input("API key (Enter=skip): ")
        updates = {"AI_PROVIDER": "openai_compat",
                   "AI_BASE_URL": base, "AI_MODEL": model, "AI_API_KEY": key}
    else:
        updates = {"AI_PROVIDER": "none"}

    if updates.get("AI_PROVIDER") == "none":
        updates.setdefault("AI_API_KEY", "")
        updates.setdefault("AI_BASE_URL", "")
        updates.setdefault("AI_MODEL", "")
    if updates:
        _write_env(updates)
        # make sure current process sees the new values too
        import os as _os
        for k, v in updates.items():
            _os.environ[k] = v
        shown_key = "***configured***" if updates.get("AI_API_KEY") else "no key"
        print(f"\n[OK] Saved to .env  (provider={updates.get('AI_PROVIDER', 'none')}, "
              f"model={updates.get('AI_MODEL', '-')}, api_key={shown_key})")
        print("     The .env file is gitignored — your key never leaves this PC.")
    print()
