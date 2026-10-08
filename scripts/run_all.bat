@echo off
REM Launch FastAPI and Streamlit in two windows.
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0\.."
echo [all] Starting API in a new window...
start "PurchaseIntentAPI" cmd /k "scripts\run_api.bat"
echo [all] Starting dashboard in a new window...
start "PurchaseIntentDashboard" cmd /k "scripts\run_dashboard.bat"
echo [all] Both windows are opening. Ctrl+C in each to stop.
pause
endlocal
