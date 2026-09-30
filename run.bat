@echo off
rem THE MACHINE — one-command start for Windows (§70)
cd /d "%~dp0"
if not exist .venv (
    echo [SETUP] Creating virtual environment...
    py -3 -m venv .venv || python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [SETUP] Installing dependencies (first run only)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
python -m the_machine.main %*
pause
