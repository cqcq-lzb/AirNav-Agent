# ============================================================
# 经支气管肺结节自动导航系统 v3.2 —— 容器镜像
#
# 重要：构建上下文必须是仓库根目录 D:\airway_navigation_v1
#       因为主程序会读取父目录的同级工程 segmentation_optimization。
#
# 构建（在 docker/ 目录内执行）：
#     docker compose build
#
# 运行：
#     docker compose up -d
#     浏览器打开 http://localhost:6080/vnc.html
# ============================================================

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_DIR=/opt/airway_navigation_v1/airway_navigation_system_v3_2_complete \
    DISPLAY=:99 \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    START_XVFB=1 \
    START_VNC=1 \
    XVFB_SCREEN=1920x1080x24

# ------------------------------------------------------------
# 1) 系统依赖
#    - xvfb / x11vnc / novnc / websockify：无显示器环境下提供 GUI
#    - libgl1 / libglu1-mesa / libegl1：VTK 三维渲染（软件 OpenGL）
#    - libxcb-* / libxkbcommon-x11：PySide6 (Qt6) 的 xcb 平台插件
#    - fonts-wqy-zenhei / fonts-noto-cjk：界面中文显示
# ------------------------------------------------------------
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        xvfb \
        x11vnc \
        x11-utils \
        xauth \
        openbox \
        novnc \
        websockify \
        libgl1 \
        libglu1-mesa \
        libegl1 \
        libgles2 \
        libopengl0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        libxt6 \
        libxmu6 \
        libxpm4 \
        libxcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-util1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libxi6 \
        libxcomposite1 \
        libxcursor1 \
        libxinerama1 \
        libfontconfig1 \
        libfreetype6 \
        libdbus-1-3 \
        fonts-wqy-zenhei \
        fonts-noto-cjk \
        fontconfig \
        ca-certificates \
        procps \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 1.5) 窗口管理器：让主界面自动最大化铺满整屏
#      没有窗口管理器时，Qt 窗口只有 1060x760，
#      在 1920x1080 虚拟屏上只占左上角一块。
# ------------------------------------------------------------
RUN mkdir -p /root/.config/openbox \
    && cp /etc/xdg/openbox/rc.xml /root/.config/openbox/rc.xml \
    && sed -i '0,/<applications>/s|<applications>|<applications>\n    <application class="*"><maximized>yes</maximized></application>|' /root/.config/openbox/rc.xml \
    && grep -q '<maximized>yes</maximized>' /root/.config/openbox/rc.xml

# ------------------------------------------------------------
# 2) Python 依赖（与本地 dicom 环境保持同一大版本）
#    若某个版本号在 PyPI 上不存在导致拉取失败，
#    可把对应行改成不带版本约束（例如 "vtk"）后重建。
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "numpy>=2.2,<3" \
        "scipy>=1.15,<2" \
        "scikit-image>=0.25,<1" \
        "SimpleITK>=2.5,<3" \
        "vtk>=9.6,<10" \
        "PySide6>=6.11,<7" \
        "paramiko>=5,<6"

# ------------------------------------------------------------
# 3) 代码
#    保持 airway_navigation_system_v3_2_complete 与
#    segmentation_optimization 为同级目录（主程序按此布局查找）。
# ------------------------------------------------------------
WORKDIR /opt/airway_navigation_v1

COPY airway_navigation_system_v3_2_complete/ \
     /opt/airway_navigation_v1/airway_navigation_system_v3_2_complete/

COPY segmentation_optimization/run_optimized_nnunet.py \
     segmentation_optimization/optimize_segmentations.py \
     segmentation_optimization/optimizer_config.json \
     /opt/airway_navigation_v1/segmentation_optimization/

COPY airway_navigation_system_v3_2_complete/docker/entrypoint.sh \
     /usr/local/bin/entrypoint.sh

# Windows 检出的换行可能是 CRLF，统一转成 LF 后赋予执行权限
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p "${APP_DIR}/cases" /data/ct_input

WORKDIR ${APP_DIR}

EXPOSE 5900 6080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["airway_navigation_system_v3.py"]
