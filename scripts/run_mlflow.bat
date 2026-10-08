@echo off
REM MLflow UI on http://127.0.0.1:5000
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONPATH=.
echo [mlflow] http://127.0.0.1:5000  (run scripts\train.bat first)
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
endlocal
