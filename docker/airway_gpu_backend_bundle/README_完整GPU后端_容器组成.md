# 气道导航完整 GPU 后端容器

这不是原先 65MB 的 SSH 转发网关。此部署包把分割流水线、两套 Python/CUDA 运行环境、两套上线模型、后处理与 API 放进同一个可迁移镜像。前端仍调用相同的 `/api/v1/jobs` 接口。

## 一、组成

- MANet 肺结节检测与分割：PyTorch 2.4.1 / CUDA 12.1
- nnU-Net 气道分割：PyTorch 2.12.1+cu130 / nnUNetv2 2.8.1
- 气道质控、修复、导航路线和 UI 数据包生成
- FastAPI：上传 CT、查询状态、查看日志、下载结果包
- 仅收集上线需要的约 321MB 模型，不带训练数据、旧权重和历史输出

## 二、为什么必须在 GPU Linux 机器上构建

Windows 本机的 Docker Desktop 没有服务器上的 H100、Linux 推理代码和模型权重。第一次构建必须在保存这些文件的 GPU 服务器上执行。构建完成后得到一个 `.tar` 镜像文件，才可以复制到另一台装有 Docker、NVIDIA 驱动和 NVIDIA Container Toolkit 的 Linux GPU 机器。

## 三、在当前 GPU 服务器上制作镜像

1. 把整个 `gpu_backend_bundle` 文件夹放到 GPU 服务器，例如 `/data2/home/wcq/airway_gpu_backend_bundle`。

2. 进入目录并收集运行必需代码与模型：

   ```bash
   cd /data2/home/wcq/airway_gpu_backend_bundle
   bash prepare_on_gpu_server.sh
   ```

3. 确认账号能访问 Docker 和 GPU：

   ```bash
   docker version
   docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi
   ```

4. 构建、GPU 冒烟测试并导出：

   ```bash
   bash build_and_export.sh
   ```

5. 成功后会得到：

   ```text
   airway-navigation-gpu-backend-2.0.tar
   airway-navigation-gpu-backend-2.0.tar.sha256
   ```

## 四、复制到另一台 GPU 机器部署

目标机器要求：Linux x86_64、NVIDIA 驱动、Docker Engine、NVIDIA Container Toolkit；显卡驱动需支持镜像所需 CUDA。

1. 复制整个文件夹以及导出的两个文件到目标机器。
2. 运行：

   ```bash
   cd /path/to/airway_gpu_backend_bundle
   bash import_and_start.sh
   ```

3. 第一次脚本会生成 `.env` 并停止。编辑 `.env`，至少把 `API_TOKEN` 改成长随机密码，再次运行：

   ```bash
   bash import_and_start.sh
   ```

4. 验证：

   ```bash
   curl http://127.0.0.1:8000/health
   ```

5. 浏览器访问 `http://目标机器IP:8000/docs`。

## 五、前端如何连接

前端 API 地址设置为 `http://GPU机器IP:8000`，API Key 设置为 `.env` 中的 `API_TOKEN`。上传、状态查询、日志和结果下载接口与当前前端已经使用的接口保持一致。

## 六、重要限制

- 默认一次只运行一个任务，避免同事同时提交导致显存竞争。
- 端口 8000 只应向受信任局域网开放。
- `123456` 不适合作为 API 密钥；请使用长随机字符串。
- 本系统仅用于科研验证，不可直接作为临床医疗器械服务。
