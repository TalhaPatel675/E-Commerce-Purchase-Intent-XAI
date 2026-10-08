@echo off
REM Black + ruff pass (PRD ^5).
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONPATH=.
echo [lint] black --check ...
python -m black --check src dashboard tests
echo [lint] ruff ...
python -m ruff check src dashboard tests
endlocal
