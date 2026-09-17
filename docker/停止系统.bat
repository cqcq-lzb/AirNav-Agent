@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   停止 经支气管肺结节自动导航系统 v3.2 (Docker)
echo ============================================================
echo.

docker info >nul 2>nul
if errorlevel 1 (
    echo Docker 引擎未运行，无需停止。
    echo.
    pause
    exit /b 0
)

echo 正在停止并移除容器...
docker compose down
if errorlevel 1 (
    echo [警告] 停止过程中出现错误，请检查上方输出。
    pause
    exit /b 1
)

echo.
echo 已停止。
echo.
echo 病例数据完好保留在：
echo   D:\airway_navigation_v1\airway_navigation_system_v3_2_complete\cases
echo 待处理 CT 目录：
echo   D:\airway_navigation_v1\docker_shared
echo.
echo 如需彻底退出 Docker Desktop，请右键任务栏托盘图标选择 Quit。
pause
