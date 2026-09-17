# 经支气管肺结节自动导航系统 v3.2 —— Docker 封装说明

## 一、封装方案

本系统是 **PySide6 + VTK 桌面 GUI 程序**，容器里没有物理显示器，因此镜像内提供：

```text
Xvfb（虚拟显示） → x11vnc → noVNC → 浏览器访问
```

用浏览器打开 `http://localhost:6080/vnc.html` 即可看到完整的三方位视图、三维气道地图和虚拟支气管镜界面。

容器内**没有 GPU 直通**，VTK 使用 Mesa 软件渲染（`LIBGL_ALWAYS_SOFTWARE=1`）。
功能完整，但三维视图帧率低于本机原生运行。

## 二、前置条件

### 1. Docker 引擎（本机已解决）

本机原先不满足，原因是缺少 WSL2，表现为 Docker Desktop 进程在跑但后端虚拟机起不来，所有命令报：

```text
request returned 500 Internal Server Error for API route ...
```

原状态：

```text
Windows 11 专业版 10.0.22000
Hyper-V   未启用（HypervisorPresent=False）
WSL2      未安装（wsl.exe 为旧版 inbox 版本）
```

解决办法（**管理员 PowerShell**，执行后重启电脑）：

```powershell
wsl --install -d Ubuntu
```

若 `wsl --install` 不可用，可改用分步启用：

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
# 重启电脑
wsl --set-default-version 2
```

重启后确认状态：

```text
WSL 版本   : 2.7.13.0
内核版本   : 6.18.33.2-2
默认版本   : 2
WSL 发行版 : docker-desktop
```

> Ubuntu 发行版并非 Docker 必需，Docker Desktop 会创建并使用自己的 `docker-desktop` 发行版。

### 2. 国内网络（必须配置）

未配置加速时，拉取基础镜像会直接失败：

```text
failed to fetch anonymous token: ... wsarecv: An existing connection was forcibly closed by the remote host
```

已在 `C:\Users\<用户名>\.docker\daemon.json` 中加入可用镜像源：

```json
{
  "registry-mirrors": [
    "https://docker.1panel.live",
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run"
  ]
}
```

改完重启引擎并确认生效：

```powershell
docker desktop restart
docker info --format "{{.RegistryConfig.Mirrors}}"
```

容器内的 apt 与 pip 也已切到清华源（见 Dockerfile），避免构建后半程再卡住。

## 三、构建与运行

进入 `docker` 目录：

```bat
cd D:\airway_navigation_v1\airway_navigation_system_v3_2_complete\docker
```

一键构建并启动：

```bat
构建并运行.bat
```

或手动执行：

```bat
docker compose build
docker compose up -d
docker compose logs -f
```

浏览器打开：

```text
http://localhost:6080/vnc.html
```

### 日常开关（最常用）

Docker Desktop 的 `AutoStart` 是关闭的，**每次开机后需要先手动启动它**，等状态变成 `Engine running`。

| 操作 | 最简单方式 | 命令行方式 |
| --- | --- | --- |
| 开机后使用 | 双击 `启动系统.bat` | `docker compose up -d` |
| 用完关闭 | 双击 `停止系统.bat` | `docker compose down` |
| 临时暂停 | Docker Desktop 里点 Stop | `docker compose stop` |
| 恢复暂停 | Docker Desktop 里点 Start | `docker compose start` |
| 看运行状态 | Docker Desktop → Containers | `docker compose ps` |
| 看实时日志 | Docker Desktop → 容器 → Logs | `docker compose logs -f` |
| 改了代码后重建 | 双击 `构建并运行.bat` | `docker compose build && docker compose up -d` |

界面地址（启动后）：

```text
http://localhost:6080/vnc.html?autoconnect=1&resize=scale
```

**数据不会丢**：病例包、CT 文件都在本机的挂载目录里，停止或删除容器都不影响。
只有删除镜像（`docker rmi airway-navigation-v3:3.2`）才需要重新构建，约十几分钟。

## 四、目录映射

| 容器内路径 | 本机路径 | 用途 |
| --- | --- | --- |
| `/opt/airway_navigation_v1/airway_navigation_system_v3_2_complete/cases` | `D:\airway_navigation_v1\airway_navigation_system_v3_2_complete\cases` | 病例包下载目录，与原生运行共用 |
| `/data/ct_input` | `D:\airway_navigation_v1\docker_shared` | 待上传的 CT 文件，从界面上传 |

镜像内代码布局（**必须保持这种兄弟关系**，主程序按此查找）：

```text
/opt/airway_navigation_v1/
├── airway_navigation_system_v3_2_complete/
│   ├── airway_navigation_system_v3.py
│   ├── interactive_navigation_stage4_v3.py
│   ├── run_stage4_server_pipeline.py
│   ├── server_config.json
│   ├── check_system_environment_v3.py
│   ├── visual_localization_bridge.py
│   └── cases/
└── segmentation_optimization/
    ├── run_optimized_nnunet.py
    ├── optimize_segmentations.py
    └── optimizer_config.json
```

## 五、操作流程

1. 把待处理的 CT（`*_0000.nii.gz`）放到 `D:\airway_navigation_v1\docker_shared\`；
2. 浏览器进入 noVNC 界面；
3. 点击「选择 CT」，路径填 `/data/ct_input/xxx_0000.nii.gz`；
4. 输入服务器 `wcq` 用户密码；
5. 点击「开始服务器分割并打开导航」；
6. 处理完成后自动打开交互导航界面。

容器需要能访问 `192.168.8.41:22`，默认 bridge 网络即可。

## 六、常用命令

```bat
docker compose build            重新构建
docker compose up -d            后台启动
docker compose logs -f          查看日志
docker compose restart          重启
docker compose down             停止并删除容器
docker compose exec airway-navigation bash    进入容器排查
```

环境自检（在容器内）：

```bat
docker compose exec airway-navigation python check_system_environment_v3.py
```

## 七、常见问题

**界面中文显示为方块**
镜像已内置 `fonts-wqy-zenhei` 与 `fonts-noto-cjk`。若仍异常，进入容器执行 `fc-list :lang=zh` 确认字体已注册。

**界面没有铺满整屏**
主程序窗口默认只有 1060x760，在 1920x1080 虚拟屏上只占左上角一块。
镜像内已安装 openbox 窗口管理器并配置「所有窗口自动最大化」，界面会自动铺满整屏。
如需关闭该行为：把 `docker-compose.yml` 里的 `START_WM` 设为 `"0"`。
如需调整整屏分辨率：改 `XVFB_SCREEN`（例如 `1600x900x24`），改完 `docker compose up -d` 重建容器生效。

**三维视图黑屏或崩溃**
软件渲染对共享内存敏感。确认 `shm_size` 为 `1gb`；仍失败时把 `XVFB_SCREEN` 降为 `1280x800x24`。

**构建时卡在拉取基础镜像**
Docker Hub 直连被重置，确认 `daemon.json` 已配置 `registry-mirrors` 并重启过引擎。
另外 Dockerfile 首行原本的 `# syntax=docker/dockerfile:1` 会额外拉取 frontend 镜像，同样容易被重置，**已移除**。

**某个 Python 包版本拉不到**
把 Dockerfile 中带版本约束的行改为不带版本，例如 `"vtk"`。

**VTK 渲染很卡**
这是软件渲染的正常表现。如需接近原生的体验，建议继续在本机直接运行（`启动自动导航系统_v3.bat`），容器版本用于交付与演示。

**x11vnc 默认无密码**
默认 `-nopw` 便于内网演示。若要暴露到非可信网络，请在 `docker-compose.yml` 中设置 `VNC_PASSWORD`。

## 八、局限性

- 容器版本**不改变**系统本身的能力边界：虚拟支气管镜仍是术前 CT 模拟视图；
- 没有真实支气管镜视频、器械跟踪或术中配准；
- 本系统为工程研究演示，**不用于临床诊断或治疗**；
- 软件渲染仅影响显示流畅度，不影响分割与路径规划的计算结果。

## 九、本次封装实测记录（2026-09-10）

环境：Windows 11 专业版 10.0.22000 + Docker Desktop 4.90.0 + WSL2 2.7.13.0

| 检查项 | 结果 |
| --- | --- |
| 镜像构建 | 成功，`airway-navigation-v3:3.2` |
| 容器状态 | `Up`，映射 5900 / 6080 |
| entrypoint 流程 | Xvfb 就绪 → x11vnc :5900 → noVNC :6080 → 主程序启动 |
| 容器内环境自检 | 7 项全部通过 |
| GUI 窗口 | `"经支气管肺结节自动导航系统 v3.2"` 自动最大化至 1920x1063，铺满整屏 |
| noVNC 页面 | HTTP 200 |
| 到分割服务器 192.168.8.41:22 | 连通，返回 SSH-2.0-OpenSSH_8.9p1 |

容器内实际依赖版本：

```text
Python 3.10.21 / numpy 2.2.6 / scipy 1.15.3 / scikit-image 0.25.2
SimpleITK 2.5.6 / VTK 9.7.0 / PySide6 6.11.2 / paramiko 5.0.0
```

**尚未验证**：在界面中真实点击操作、CT 上传到服务器、下载病例包并在导航界面中打开的完整链路。
