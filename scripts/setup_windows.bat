@echo off
REM ============================================================================
REM Windows one-time setup (PRD ^12.3 venv requirement).
REM Creates .venv and installs pinned requirements.txt. Run from project root.
REM ============================================================================
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0\.."

echo [setup] Working directory: %cd%

where python >nul 2>nul
if errorlevel 1 (
  echo [error] Python is not on PATH. Install Python 3.10+ from
  echo         https://www.python.org/downloads/windows/ - tick Add Python to PATH.
  pause
  exit /b 1
)
python --version

if not exist ".venv" (
  echo [setup] Creating .venv ...
  python -m venv .venv || (echo [error] venv creation failed. & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
echo [setup] Installing pinned dependencies (a few minutes)...
pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo [error] pip install failed. Check internet / proxy.
  pause
  exit /b 1
)

echo [setup] Done. Next:
echo   scripts\train.bat
echo   scripts\run_api.bat
echo   scripts\run_dashboard.bat
pause
endlocal
