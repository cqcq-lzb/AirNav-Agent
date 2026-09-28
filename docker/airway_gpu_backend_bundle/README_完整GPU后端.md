# 气道导航完整 GPU 后端 Docker 使用说明

这是一套纯 API 后端容器，不是 VNC，也不是远程桌面。镜像中包含 nnU-Net 气道分割、MANet 肺结节分割、后处理、质控、路径规划、模型、两套 Python/CUDA 环境和 FastAPI。

## 已完成并实测的部署

- 服务器：`gpu-node`（已脱敏，请替换成你自己的主机）
- API 文档：`http://gpu-node:18000/docs`
- 健康检查：`http://gpu-node:18000/health`
- API Key：当前按要求设置为 `123456`（仅适合受信任局域网测试）
- 容器：`airway-navigation`
- 镜像：`airway-navigation:latest`
- GPU：物理 GPU 3；在容器内编号为 GPU 0
- 并发：默认一次处理 1 个任务
- 共享内存：8 GiB

真实的 `LIDC_0029` CT 已完成上传、nnU-Net、MANet、质控、路径规划、打包和 API 下载，任务最终状态为 `succeeded`。

## 您现在怎么使用

1. 保持 H100 服务器和 Docker 服务运行。
2. 在同一局域网电脑的浏览器打开 `http://gpu-node:18000/docs`。
3. 点击右上角 **Authorize**，在 `X-API-Key` 输入 `123456`。
4. 展开 `POST /api/v1/jobs`，点击 **Try it out**。
5. `ct` 选择 `.nii` 或 `.nii.gz` CT；`case_id` 填不含空格的病例编号；`gpu` 填 `0`。
6. 点击 **Execute**，复制返回的 `job_id`。
7. 在 `GET /api/v1/jobs/{jid}` 中填入 `job_id` 查询状态。
8. 状态为 `succeeded` 后，在 `GET /api/v1/jobs/{jid}/package` 下载结果 ZIP。

前端连接时，把 API Base URL 设置为 `http://gpu-node:18000`，请求头发送 `X-API-Key: 123456`。前端不需要安装 Docker；只有运行后端的服务器需要 Docker 和 NVIDIA Container Toolkit。

## 服务器管理员命令

```bash
cd /data2/home/wcq/airway_gpu_backend_bundle

# 查看
docker compose -f docker-compose.gpu.yml ps

# 查看日志
docker logs -f airway-navigation

# 启动或应用配置
docker compose -f docker-compose.gpu.yml up -d --no-build

# 停止本项目（不会删除镜像和结果）
docker compose -f docker-compose.gpu.yml stop
```

上传文件、任务记录和结果位于部署目录下的 `runtime_data`；病例中间结果位于 `runtime_cases`。重建或重启容器不会删除这些挂载目录。

## 复制到另一台 GPU 服务器

已实测的可迁移镜像包位于：

```text
/data2/home/wcq/airway_gpu_backend_bundle/airway-navigation-h100.tar.gz
/data2/home/wcq/airway_gpu_backend_bundle/airway-navigation-h100.tar.gz.sha256
```

把整个 `airway_gpu_backend_bundle` 文件夹复制到目标 Linux GPU 服务器。目标机需为 x86_64，安装 NVIDIA 驱动、Docker Engine、Docker Compose 插件和 NVIDIA Container Toolkit。

在目标机执行：

```bash
cd /path/to/airway_gpu_backend_bundle
bash import_and_start.sh
```

若脚本首次生成 `.env` 后停止，请编辑 `.env`：

```dotenv
API_TOKEN=换成长随机密钥
API_BIND_ADDRESS=0.0.0.0
API_PORT=8000
DEFAULT_GPU=0
NVIDIA_VISIBLE_DEVICES=目标机物理GPU编号
MAX_CONCURRENT_JOBS=1
MAX_UPLOAD_BYTES=4294967296
```

然后再次运行 `bash import_and_start.sh`。注意：`NVIDIA_VISIBLE_DEVICES` 是宿主机物理 GPU 编号；容器只分配这一张卡，所以 API 的 `gpu` 参数仍填 `0`。

## 当前服务器为什么用 18000 端口

H100 服务器的 `8000`、`8001`、`8002` 已被其他服务占用，因此本项目使用 `18000`。目标服务器若 `8000` 空闲，可在 `.env` 中使用 `API_PORT=8000`。

## 安全提示

- `123456` 很弱，只应在受信任局域网短期测试；正式共享前请改成长随机 API Key。
- 只对受信任网段开放 API 端口，不要直接暴露到公网。
- 本系统仅用于科研验证，不可直接作为临床医疗器械服务。
