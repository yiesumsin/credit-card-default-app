@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo The project's Python environment is missing.
    echo Create .venv and install requirements.txt before running the app.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py"
if errorlevel 1 pause
