@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   启动 经支气管肺结节自动导航系统 v3.2 (Docker)
echo ============================================================
echo.

where docker >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 docker 命令，请先安装 Docker Desktop。
    pause
    exit /b 1
)

docker info >nul 2>nul
if not errorlevel 1 goto engine_ready

echo Docker 引擎未运行，正在尝试启动 Docker Desktop...
if exist "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe" (
    start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
) else if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
) else (
    echo [错误] 未找到 Docker Desktop.exe，请手动启动 Docker Desktop。
    pause
    exit /b 1
)

echo 等待引擎就绪（最多 3 分钟，首次启动较慢）...
for /l %%i in (1,1,36) do (
    timeout /t 5 >nul
    docker info >nul 2>nul
    if not errorlevel 1 goto engine_ready
)

echo [错误] Docker 引擎启动超时，请手动打开 Docker Desktop 等状态变为 Engine running 后重试。
pause
exit /b 1

:engine_ready
echo 引擎已就绪。
echo.
echo 启动容器...
docker compose up -d
if errorlevel 1 (
    echo [错误] 容器启动失败。
    pause
    exit /b 1
)

echo.
echo 启动完成，约 10 秒后自动打开界面。
echo 界面地址：http://localhost:6080/vnc.html
timeout /t 10 >nul
start "" "http://localhost:6080/vnc.html?autoconnect=1&resize=scale"

echo.
echo 提示：用完请双击 停止系统.bat 关闭。
pause
