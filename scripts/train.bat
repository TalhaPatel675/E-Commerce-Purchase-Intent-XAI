@echo off
REM ============================================================================
REM Train all six models, run PageValues ablation, log to MLflow,
REM write SHAP + permutation + LIME + model card.
REM ============================================================================
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0\.."

if not exist ".venv\Scripts\activate.bat" (
  echo [error] Virtual env missing. Run scripts\setup_windows.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
set PYTHONPATH=.

echo [train] Training + explainability + MLflow logging ...
python -m src.models.train
if errorlevel 1 (
  echo [error] Training failed - see logs above.
  pause
  exit /b 1
)
echo [train] Artefacts written to models\ + reports\ + mlruns\
pause
endlocal
