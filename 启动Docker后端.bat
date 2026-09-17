@echo off
title Airway Navigation GPU Backend
setlocal
cd /d "%~dp0docker\airway_gpu_backend_bundle"

if not exist ".env" (
    echo ERROR: .env 不存在。
    echo 请先复制 .env.example 为 .env，并设置 API_TOKEN。
    pause
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo ERROR: Docker 引擎未就绪。请先打开 Docker Desktop，等状态变成 Engine running。
    pause
    exit /b 1
)

echo Starting complete GPU backend container...
docker compose --env-file .env -f docker-compose.gpu.yml up -d
if errorlevel 1 (
    echo ERROR: GPU 后端启动失败。
    pause
    exit /b 1
)

echo.
echo Backend is running.
echo   API docs : http://127.0.0.1:8000/docs
echo   Health   : http://127.0.0.1:8000/health
echo.
echo 本地 UI 填写：
echo   Docker 后端地址 = http://127.0.0.1:8000
echo   API 令牌         = docker\airway_gpu_backend_bundle\.env 里的 API_TOKEN
echo   GPU              = 0
echo.
echo 这是 HTTP 后端，不是 VNC。界面请继续用 启动自动导航系统_v3.bat
pause
