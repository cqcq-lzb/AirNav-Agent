@echo off
chcp 65001 >nul
title AirNav-Agent 网页版
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv-mcp\Scripts\python.exe"
set PYTHONIOENCODING=utf-8

REM Default backend shown in the page. The UI has a selector, so this only
REM decides what is preselected. AIRNAV_WEB_PORT changes the port.
set "AIRNAV_DEFAULT_BACKEND=gpu41"
if not defined AIRNAV_WEB_PORT set "AIRNAV_WEB_PORT=8777"

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
echo  AirNav-Agent  Web UI
echo  interpreter: %PY%
echo  opening     http://127.0.0.1:%AIRNAV_WEB_PORT%
echo  stop with   Ctrl+C
echo ======================================================================
echo.

REM Give the browser a moment to open before the server prints its banner.
start "" "http://127.0.0.1:%AIRNAV_WEB_PORT%"

"%PY%" -m agent.web.server

echo.
echo ----------------------------------------------------------------------
echo  Server stopped. Viewers are under: outputs\viewers
echo ----------------------------------------------------------------------
pause
