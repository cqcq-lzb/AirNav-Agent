@echo off
chcp 65001 >nul
title AirNav-Agent Web UI
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv-mcp\Scripts\python.exe"
set PYTHONIOENCODING=utf-8

set "AIRNAV_DEFAULT_BACKEND=gpu41"
if not defined AIRNAV_WEB_PORT set "AIRNAV_WEB_PORT=8777"

if not exist "%PY%" (
    echo [ERROR] Interpreter not found:
    echo         %PY%
    echo Rebuild it with:
    echo    D:\miniforge\envs\dicom\python.exe -m venv --system-site-packages .venv-mcp
    pause
    exit /b 1
)

echo ======================================================================
echo  AirNav-Agent  Web UI
echo  interpreter: %PY%
echo  url         http://127.0.0.1:%AIRNAV_WEB_PORT%
echo  stop with   Ctrl+C
echo ======================================================================
echo.

rem ---------------------------------------------------------------------
rem Open the browser only AFTER the port answers. The old version opened it
rem first, so the first page load always failed and it looked broken.
rem The waiter runs inside this window (start /b) and the server stays in
rem the foreground, so Ctrl+C still stops it.
rem AIRNAV_WEB_OPEN=0 makes the waiter only probe (never open a browser).
rem Keep this file ASCII-only and CRLF - see the header of 跑评测.bat.
rem ---------------------------------------------------------------------
start "" /b "%PY%" -m agent.scripts.open_web_ui

"%PY%" -m agent.web.server
set "RC=%ERRORLEVEL%"

echo.
echo ----------------------------------------------------------------------
echo  exit code: %RC%
echo  viewers  : outputs\viewers
echo ----------------------------------------------------------------------
pause
exit /b %RC%
