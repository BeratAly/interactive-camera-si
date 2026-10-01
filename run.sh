#!/usr/bin/env bash
# THE MACHINE — one-command start for Linux/macOS (§70)
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv .venv
    . .venv/bin/activate
    echo "[SETUP] Installing dependencies (first run only)..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    . .venv/bin/activate
fi
python -m the_machine.main "$@"
