@echo off
setlocal
cd /d "%~dp0"

set "IMAGE=airway-navigation-api:1.0"
set "ARCHIVE=airway-navigation-api-1.0.tar"

docker image inspect %IMAGE% >nul 2>nul
if errorlevel 1 (
    echo ERROR: Image %IMAGE% was not found. Build the backend image first.
    pause
    exit /b 1
)

echo Exporting %IMAGE% ...
docker save -o "%ARCHIVE%" %IMAGE%
if errorlevel 1 (
    echo ERROR: Image export failed.
    pause
    exit /b 1
)

echo.
echo Image archive created: %~dp0%ARCHIVE%
echo Copy this archive with docker-compose.api.yml and .env.api.example.
echo Do not copy .env.api, SSH private keys, or CT data.
pause
