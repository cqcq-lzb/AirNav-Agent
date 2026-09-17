@echo off
setlocal
cd /d "%~dp0"

set "ARCHIVE=airway-navigation-api-1.0.tar"

if not exist "%ARCHIVE%" (
    echo ERROR: %ARCHIVE% was not found in this folder.
    pause
    exit /b 1
)

if not exist ".env.api" (
    echo ERROR: .env.api is missing.
    echo Copy .env.api.example to .env.api and configure SSH_DIR and API_TOKEN.
    pause
    exit /b 1
)

echo Importing backend image...
docker load -i "%ARCHIVE%"
if errorlevel 1 (
    echo ERROR: Image import failed.
    pause
    exit /b 1
)

echo Starting backend...
docker compose --env-file .env.api -f docker-compose.api.yml up -d --no-build
if errorlevel 1 (
    echo ERROR: Backend start failed.
    pause
    exit /b 1
)

echo.
echo Deployment complete: http://127.0.0.1:8000/docs
pause
