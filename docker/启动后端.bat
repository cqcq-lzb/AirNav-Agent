@echo off
setlocal
cd /d "%~dp0"

if not exist ".env.api" (
    echo ERROR: .env.api is missing.
    echo Copy .env.api.example to .env.api and configure SSH_DIR and API_TOKEN.
    pause
    exit /b 1
)

docker compose --env-file .env.api -f docker-compose.api.yml up -d
if errorlevel 1 (
    echo ERROR: API backend start failed.
    pause
    exit /b 1
)

echo API backend is running: http://127.0.0.1:8000/health
echo The service is bound to this computer only.
pause
