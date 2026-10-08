@echo off
REM ============================================================================
REM Streamlit dashboard (PRD ^4.8). http://127.0.0.1:8501
REM ============================================================================
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0\.."
if not exist ".venv\Scripts\activate.bat" (
  echo [error] Run scripts\setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
set PYTHONPATH=.
set DASHBOARD_PORT=8501
echo [dashboard] http://127.0.0.1:%DASHBOARD_PORT%
python -m streamlit run dashboard\app.py --server.port %DASHBOARD_PORT% --server.headless false --browser.gatherUsageStats false
endlocal
