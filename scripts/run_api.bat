@echo off
REM ============================================================================
REM FastAPI service (PRD ^4.7). Swagger UI at http://127.0.0.1:8000/docs.
REM ============================================================================
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0\.."

if not exist ".venv\Scripts\activate.bat" (
  echo [error] Run scripts\setup_windows.bat first.
  pause
  exit /b 1
)
if not exist "models\best_model.joblib" (
  echo [error] No trained model. Run scripts\train.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
set PYTHONPATH=.
set API_HOST=127.0.0.1
set API_PORT=8000

echo [api] FastAPI on http://%API_HOST%:%API_PORT%  (Swagger at /docs)
python -m uvicorn src.api.main:app --host %API_HOST% --port %API_PORT% --workers 1
endlocal
