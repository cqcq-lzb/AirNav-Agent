@echo off
title Airway Navigation System v3.2
setlocal
cd /d "%~dp0"

set "PYTHON=D:\miniforge\envs\dicom\python.exe"
set "SCRIPT=%~dp0airway_navigation_system_v3.py"

if not exist "%PYTHON%" (
    echo ERROR: Python was not found:
    echo %PYTHON%
    echo.
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo ERROR: Frontend script was not found:
    echo %SCRIPT%
    echo.
    pause
    exit /b 1
)

"%PYTHON%" "%SCRIPT%"

if errorlevel 1 (
    echo.
    echo ERROR: Frontend exited with code %errorlevel%.
    pause
)
