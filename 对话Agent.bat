@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv-mcp\Scripts\python.exe"
set PYTHONIOENCODING=utf-8

REM Default backend for this launcher. The CLI reads AIRNAV_DEFAULT_BACKEND
REM and uses it only when --backend is absent, so an explicit --backend below
REM still wins.
set "AIRNAV_DEFAULT_BACKEND=gpu41"

if not exist "%PY%" (
    echo [ERROR] Interpreter not found:
    echo         %PY%
    echo.
    echo Rebuild it with:
    echo    D:\miniforge\envs\dicom\python.exe -m venv --system-site-packages .venv-mcp
    pause
    exit /b 1
)

echo ======================================================================
echo  AirNav-Agent
echo  interpreter: %PY%
echo ======================================================================
echo.

if "%~1"=="" (
    echo Entering interactive mode. Type your question and press Enter.
    echo Commands: tools / cases / exit
    echo For another backend: "%PY%" -m agent.cli repl --backend ollama
    echo Press Ctrl+C to exit.
    echo.
    "%PY%" -m agent.cli repl
) else (
    "%PY%" -m agent.cli ask %* --verbose
)

echo.
echo ----------------------------------------------------------------------
echo  Done. Viewers (if any) are under: outputs\viewers
echo ----------------------------------------------------------------------
pause
