@echo off
chcp 65001 >nul
title AirNav-Agent Eval
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv-mcp\Scripts\python.exe"
set PYTHONIOENCODING=utf-8

if not exist "%PY%" (
    echo [ERROR] Interpreter not found:
    echo         %PY%
    echo Rebuild it with:
    echo    D:\miniforge\envs\dicom\python.exe -m venv --system-site-packages .venv-mcp
    pause
    exit /b 1
)

rem ---------------------------------------------------------------------
rem Keep this file ASCII-only and CRLF. cmd.exe re-reads a .bat by byte
rem offset; LF-only or multi-byte text makes it execute misplaced line
rem fragments - it can even re-run the "rebuild the venv" hint above and
rem silently flip include-system-site-packages to false.
rem ---------------------------------------------------------------------
rem No arguments = real model (gpu41), all 30 cases, about 7 minutes.
rem A single run is ONE sample - do not quote it as a result.
rem For a quotable number use --repeat 3: it reports per-case pass
rem (N of N runs passed, which is what gates the baseline), attempt-level
rem pass rate, and a flaky-case list.
rem Examples (run from this folder):
rem   this bat --repeat 3
rem   this bat --repeat 3 --include ^<category^>
rem   this bat --cases original
rem   this bat --backend heuristic
rem ---------------------------------------------------------------------

set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=--backend gpu41 --cases both"

echo Running: run_eval %ARGS%
echo.
"%PY%" -m agent.scripts.run_eval %ARGS%
set "RC=%ERRORLEVEL%"

echo.
echo ---------------------------------------------------------------------
echo  exit code: %RC%
echo  report  : outputs\eval\   .json / .md   --repeat N adds _xN suffix
echo  single run = one sample. Use --repeat 3 for a quotable number.
echo ---------------------------------------------------------------------
pause
exit /b %RC%
