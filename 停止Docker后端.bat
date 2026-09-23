@echo off
title Stop Airway Navigation GPU Backend
setlocal
cd /d "%~dp0docker\airway_gpu_backend_bundle"

docker compose --env-file .env -f docker-compose.gpu.yml down
if errorlevel 1 (
    echo ERROR: 停止 GPU 后端失败。
    pause
    exit /b 1
)

echo GPU backend stopped.
echo 病例结果仍保留在 runtime_data / runtime_cases。
pause
