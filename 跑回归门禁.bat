@echo off
chcp 65001 >nul
title AirNav-Agent Regression Gate
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv-mcp\Scripts\python.exe"
set PYTHONIOENCODING=utf-8

if not exist "%PY%" (
    echo [ERROR] Interpreter not found:
    echo         %PY%
    echo.
    echo Rebuild it with:
    echo    D:\miniforge\envs\dicom\python.exe -m venv --system-site-packages .venv-mcp
    pause
    exit /b 1
)

rem ---------------------------------------------------------------------
rem Keep this file ASCII-only and CRLF. cmd.exe re-reads a .bat by byte
rem offset; LF-only or multi-byte text makes it execute misplaced line
rem fragments (it once re-ran the "rebuild the venv" hint above and
rem silently flipped include-system-site-packages to false).
rem ---------------------------------------------------------------------
rem No arguments = full local profile. Arguments are passed through:
rem   this bat --profile ci
rem   this bat --fast
rem   this bat --only geometry,planner
"%PY%" -m agent.scripts.gate %*
set "RC=%ERRORLEVEL%"

echo.
echo ----------------------------------------------------------------------
echo  exit code: %RC%
echo  Report : outputs\gate\gate_report.md
echo ----------------------------------------------------------------------
pause
exit /b %RC%
