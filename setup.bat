@echo off
rem ================================================================
rem  setup.bat  –  Mirel-Tuner bootstrap (Windows)
rem ================================================================
setlocal enabledelayedexpansion

set "VENV_DIR=.venv"
set "REQ_ROOT=requirements.txt"
set "OS_REQ=requirements\windows.txt"
set "LOCK_FILE=requirements\lock.txt"

:: ── pick python (same logic as before, truncated for brevity) ─────────
set "PY_CMD=python"
for /f "tokens=2" %%v in ('"%PY_CMD%" --version 2^>^&1') do set "PYV=%%v"
echo Using %PY_CMD% (%PYV%)

:: ── create venv if missing ───────────────────────────────────────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    "%PY_CMD%" -m venv "%VENV_DIR%" || (echo Venv creation failed & exit /b 1)
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

:: ── upgrade pip & install pip-tools for lock generation ──────────────
pip install --upgrade pip >nul
pip install --upgrade pip-tools >nul

:: ── install core + OS requirements ──────────────────────────────────
pip install -r "%REQ_ROOT%" -r "%OS_REQ%" || exit /b 1

:: ── pull plugin + user extras (ensure_deps does idempotent merge) ───
python scripts\ensure_deps.py || exit /b 1

:: ── generate / refresh lock file ─────────────────────────────────────
pip-compile ^
    --quiet ^
    --output-file="%LOCK_FILE%" ^
    "%REQ_ROOT%" "%OS_REQ%" || exit /b 1
echo Lock written to %LOCK_FILE%

echo.
echo ✓ Environment ready and activated.
echo   Run:  python main.py --config configs\quickstart.json --dry-run
echo   Deactivate with:  deactivate
echo.
endlocal
