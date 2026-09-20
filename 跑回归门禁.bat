@echo off
chcp 65001 >nul
title AirNav-Agent 回归门禁
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

REM 不带参数 = 全量（local 档）；也可以从命令行传，例如：
REM   跑回归门禁.bat --profile ci
REM   跑回归门禁.bat --fast
REM   跑回归门禁.bat --only geometry,planner
"%PY%" -m agent.scripts.gate %*

echo.
echo ----------------------------------------------------------------------
echo  Report: outputs\gate\gate_report.md
echo ----------------------------------------------------------------------
pause
