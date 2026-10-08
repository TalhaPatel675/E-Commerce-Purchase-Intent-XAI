@echo off
REM Run pytest suite (PRD ^5).
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0\.."
if not exist ".venv\Scripts\activate.bat" (
  echo [error] Run scripts\setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
set PYTHONPATH=.
python -m pytest -v
endlocal
