#!/usr/bin/env bash
# ============================================================
# 经支气管肺结节自动导航系统 v3.2 —— 容器入口脚本
#
# 职责：
#   1. 准备虚拟显示（Xvfb），无显示器也能跑 Qt/VTK；
#   2. 启动 x11vnc + noVNC，用浏览器访问图形界面；
#   3. 启动主程序 airway_navigation_system_v3.py。
#
# 环境变量：
#   START_XVFB=0            改用外部 X 服务器（配合 DISPLAY）
#   START_VNC=0             不启动 VNC/noVNC
#   XVFB_SCREEN=1920x1080x24
#   VNC_PASSWORD=xxx        设置后启用 VNC 密码（默认为空密码）
#   DISPLAY=:99             X 显示号
# ============================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/airway_navigation_v1/airway_navigation_system_v3_2_complete}"

export DISPLAY="${DISPLAY:-:99}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"

log() {
    printf '[entrypoint] %s\n' "$*"
}

# ------------------------------------------------------------
# 1) 虚拟显示
# ------------------------------------------------------------
if [ "${START_XVFB:-1}" = "1" ]; then
    SCREEN="${XVFB_SCREEN:-1920x1080x24}"
    log "启动 Xvfb：DISPLAY=${DISPLAY} SCREEN=${SCREEN}"

    rm -f "/tmp/.X${DISPLAY#:}-lock" 2>/dev/null || true

    Xvfb "$DISPLAY" -screen 0 "$SCREEN" -ac +extension GLX +render -noreset \
        >/tmp/xvfb.log 2>&1 &

    for _ in $(seq 1 40); do
        if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
            break
        fi
        sleep 0.25
    done

    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        log "Xvfb 就绪"
    else
        log "警告：Xvfb 未在预期时间内就绪，仍继续启动（详见 /tmp/xvfb.log）"
    fi
else
    log "已跳过 Xvfb，使用外部 X 服务器：DISPLAY=${DISPLAY}"
fi

# ------------------------------------------------------------
# 1.5) 窗口管理器（让主界面自动最大化铺满整屏）
# ------------------------------------------------------------
if [ "${START_WM:-1}" = "1" ]; then
    log "启动 openbox（窗口自动最大化铺满整屏）"
    openbox --config-file /root/.config/openbox/rc.xml \
        >/tmp/openbox.log 2>&1 &
    sleep 1
fi

# ------------------------------------------------------------
# 2) VNC / noVNC
# ------------------------------------------------------------
if [ "${START_VNC:-1}" = "1" ]; then
    VNC_ARGS=(
        -display "$DISPLAY"
        -forever
        -shared
        -rfbport 5900
        -o /tmp/x11vnc.log
        -bg
    )

    if [ -n "${VNC_PASSWORD:-}" ]; then
        mkdir -p /root/.vnc
        x11vnc -storepasswd "$VNC_PASSWORD" /root/.vnc/passwd >/dev/null 2>&1
        VNC_ARGS+=(-rfbauth /root/.vnc/passwd)
        log "x11vnc 已启用密码认证"
    else
        VNC_ARGS+=(-nopw)
        log "x11vnc 未设置密码（内网演示用，请勿暴露到公网）"
    fi

    log "启动 x11vnc :5900"
    x11vnc "${VNC_ARGS[@]}"

    log "启动 noVNC :6080  ->  http://localhost:6080/vnc.html"
    websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/websockify.log 2>&1 &
fi

# ------------------------------------------------------------
# 3) 启动主程序
# ------------------------------------------------------------
log "工作目录：${APP_DIR}"
cd "$APP_DIR"

if [ "$#" -eq 0 ]; then
    set -- airway_navigation_system_v3.py
fi

log "启动命令：python $*"
exec python "$@"
