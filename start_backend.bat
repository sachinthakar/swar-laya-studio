@echo off
REM ============================================================
REM  Swar-Laya detection backend launcher
REM  Creates an isolated virtual environment the first time,
REM  installs dependencies once, then starts the FastAPI server.
REM  Isolated so it never conflicts with your global torch/torchvision.
REM ============================================================
setlocal
cd /d "%~dp0"

set VENV=.venv_backend

if not exist "%VENV%\Scripts\python.exe" (
    echo Creating virtual environment "%VENV%" ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo.
        echo Could not create the venv with "python". Trying the "py" launcher...
        py -3 -m venv "%VENV%"
    )
)

set PY=%VENV%\Scripts\python.exe

if not exist "%VENV%\.installed" (
    echo.
    echo Installing dependencies ^(first run only, this downloads a lot^) ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements_backend.txt
    if errorlevel 1 (
        echo.
        echo Dependency install failed. See the messages above.
        pause
        exit /b 1
    )
    echo done > "%VENV%\.installed"
)

echo.
echo ============================================================
echo  Backend starting on http://127.0.0.1:8000
echo  Health check:      http://127.0.0.1:8000/health
echo  Leave this window open while you transcribe. Ctrl+C to stop.
echo ============================================================
echo.
"%PY%" -m uvicorn server:app --host 127.0.0.1 --port 8000

pause
