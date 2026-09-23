@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   经支气管肺结节自动导航系统 v3.2 - Docker 构建与启动
echo ============================================================
echo.

where docker >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 docker 命令，请先安装 Docker Desktop。
    pause
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo [错误] Docker 引擎未运行。
    echo        请启动 Docker Desktop，等待状态显示 Engine running 后重试。
    pause
    exit /b 1
)

if not exist "..\..\docker_shared" (
    mkdir "..\..\docker_shared"
    echo [提示] 已创建 CT 交换目录：D:\airway_navigation_v1\docker_shared
)

echo [1/2] 构建镜像（首次构建需要下载依赖，耗时较长）...
docker compose build
if errorlevel 1 (
    echo [错误] 镜像构建失败，请查看上方输出。
    pause
    exit /b 1
)

echo.
echo [2/2] 启动容器...
docker compose up -d
if errorlevel 1 (
    echo [错误] 容器启动失败。
    pause
    exit /b 1
)

echo.
echo 启动完成。
echo   网页界面：http://localhost:6080/vnc.html
echo   查看日志：docker compose logs -f
echo   停止服务：docker compose down
echo.

timeout /t 3 >nul
start "" "http://localhost:6080/vnc.html?autoconnect=1&resize=scale"

pause
