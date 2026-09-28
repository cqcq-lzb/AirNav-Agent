# gpu41 上部署 Ollama（AirNav-Agent 的 LLM 后端）

本地跑 7B 会撞 Windows 提交内存上限（详见 `使用指南.md` 第 5 节）。
把推理放到 gpu41 上，本地只做规划计算，两边都用长项。

---

## 1. 现状

| 项 | 值 |
|---|---|
| 服务器 | `gpu-node`（这里本来是真实内网地址，已脱敏；请替换成你自己的 GPU 主机），用户 `your-user` |
| GPU | 8× H100 PCIe 80GB |
| **占用哪张卡** | **只有 GPU 0**（其余 1/2/4~7 常被训练任务占着） |
| Ollama | 0.34.1，装在 `/data2/home/wcq/ollama` |
| 模型目录 | `/data2/home/wcq/.ollama/models` |
| 监听 | `0.0.0.0:11434`（仅内网） |
| 加速后端 | CUDA 13（`cuda_v13`），**Vulkan 已显式关闭** |

---

## 2. 起停

```bash
ssh gpu41
cd ~/ollama
./start_ollama_gpu41.sh     # 启动（已在跑则直接返回）
./stop_ollama_gpu41.sh      # 停止
tail -f serve.log           # 看日志
```

### `deploy/gpu41/` 里的三个脚本

| 脚本 | 作用 | 服务器上的位置 |
|---|---|---|
| `install_ollama.sh` | 装 ollama（免 sudo，自带镜像测速 + 完整性校验） | `~/ollama/` |
| `start_ollama_gpu41.sh` | 起服务，钉死 GPU 0 + 关 Vulkan | `~/ollama/` |
| `stop_ollama_gpu41.sh` | 停服务 | `~/ollama/` |

重装或换台服务器时，把这三个（或前两个）scp 过去，`chmod +x` 即可：

```bash
scp deploy/gpu41/*.sh your-user@gpu-node:/data2/home/your-user/ollama/
```

启动脚本里那几个环境变量**不要删**，每个都对应一个踩过的坑：

### `CUDA_VISIBLE_DEVICES=0` — 只占 GPU 0

服务器上 1/2/4~7 号卡常年被训练任务占着。ollama 默认会去抢显存最空的卡，
必须钉死。

### `OLLAMA_VULKAN=false` — ⚠️ 这条最容易漏

**`CUDA_VISIBLE_DEVICES` 管不住 Vulkan 后端。** 实测不关掉 Vulkan 时：

```
WARN  user overrode visible devices CUDA_VISIBLE_DEVICES=0
INFO  inference compute library=Vulkan ... id=1 ... H100 PCIe  avail=5.6 GiB
INFO  inference compute library=Vulkan ... id=2 ... H100 PCIe  avail=3.2 GiB
...
INFO  vram-based default context  total_vram="636.7 GiB"   ← 8 张卡全被算进来了
```

八张卡全被认出来，`total_vram` 变成 636 GiB。这意味着 ollama 有可能把模型
铺到**别人的训练卡上**。关掉之后只剩一条 `CUDA0`，才安全。

### `OLLAMA_HOST=0.0.0.0:11434` — 否则本机连不上

默认只监听 `127.0.0.1`。**只在可信内网开放，不要把 11434 映射到公网** ——
这个服务没有任何鉴权。

### `OLLAMA_KEEP_ALIVE=30m` — 演示体验

默认 5 分钟就卸载模型，下次提问要重新加载（等十几秒）。演示时很尴尬。

### `OLLAMA_CONTEXT_LENGTH=32768`

不设的话 ollama 会按显存总量推一个很大的默认值（实测算出 262144），
白白吃显存。

---

## 3. 安装（`install_ollama.sh` 已经封装了这些坑）

> 直接跑 `bash install_ollama.sh` 即可 —— 它会自动测速选源、校验压缩包、
> 并检查 GPU 后端库在不在。下面记录的是它为什么这么写。

### 3.1 安装包换了格式：`.tgz` → `.tar.zst`

`https://ollama.com/download/ollama-linux-amd64.tgz` 会 307 跳到 GitHub，
但 **v0.34.1 已经没有 `.tgz` 了，最后一跳是 404**：

```
HTTP/2 307 → github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz
HTTP/2 302 → github.com/ollama/ollama/releases/download/v0.34.1/ollama-linux-amd64.tgz
HTTP/2 404     ← 文件不存在
```

现在的名字是 **`.tar.zst`**：

```bash
# 正确
URL=https://github.com/ollama/ollama/releases/download/v0.34.1/ollama-linux-amd64.tar.zst
```

### 3.2 直连 GitHub 太慢，用镜像

实测同一文件：

| 源 | 速度 |
|---|---|
| GitHub 直连 | 30 KB/s（1 小时才下 143 MB） |
| ghfast.top | 3 KB/s |
| ghproxy.net | 53 KB/s |
| **gh-proxy.com** | **5.4 MB/s** |

差了 **1700 倍**。内网服务器上装东西，先测速再决定，别硬等。

```bash
curl -L -o ollama-linux-amd64.tar.zst \
  https://gh-proxy.com/https://github.com/ollama/ollama/releases/download/v0.34.1/ollama-linux-amd64.tar.zst
```

1.43 GB → 解压后 2.25 GB，65 个条目。校验：

```bash
zstd -t ollama-linux-amd64.tar.zst     # 应输出解压后字节数，无报错
tar --zstd -tf ollama-linux-amd64.tar.zst | wc -l
```

### 3.3 免 sudo 安装

`sudo` 需要密码，所以走用户目录：

```bash
mkdir -p ~/ollama && cd ~/ollama
tar --zstd -xf ~/ollama-linux-amd64.tar.zst   # 得到 bin/ollama + lib/ollama/
./bin/ollama --version
```

> 服务器这份 tarball 里 `cuda_v12` 和 `cuda_v13` 都齐全，
> 不像本机 Windows 那份只有 CPU 后端（那正是本地 ollama 走不了 GPU 的原因）。

---

## 4. 本地侧怎么用

后端预设已经加好了，名字叫 `gpu41`：

```bat
set PY=D:\AirNav-Agent\.venv-mcp\Scripts\python.exe
%PY% -m agent.cli doctor --backend gpu41          REM 确认可达 + 模型在不在
%PY% -m agent.cli ask "LIDC_0089 哪个结节最容易到达" --backend gpu41 --verbose
%PY% -m agent.scripts.run_eval --backend gpu41 --stem eval_report
```

不想每次都写 `--backend`，也可以只改地址沿用本地预设：

```bat
set AIRNAV_GPU41_URL=http://gpu-node:11434/v1
set AIRNAV_GPU41_MODEL=qwen2.5:14b
```

预设定义在 `agent/llm/client.py` 的 `PRESETS`。地址与模型名都可以用上面两个
环境变量覆盖，**不需要改代码**。

> `192.168.x.x` 会被 `_is_local_url()` 判为内网地址，自动设
> `trust_env=False` 绕过系统代理 —— 否则请求会被打到代理端口上得到 502。

---

## 5. 排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 本机连不上 11434 | 没设 `OLLAMA_HOST`，只监听 127.0.0.1 | 用启动脚本重启 |
| `doctor` 报模型不在列表 | 还没 pull | `~/ollama/bin/ollama pull <模型>` |
| 日志里出现 `library=Vulkan` | `OLLAMA_VULKAN=false` 没生效 | 检查环境变量，重启服务 |
| ollama 跑到别的卡上了 | `CUDA_VISIBLE_DEVICES` 没设 | 用启动脚本重启；`nvidia-smi` 确认 |
| `401` / 认证失败 | ollama 不需要 key，但客户端发了错误的 | 预设里 `api_key` 固定为 `ollama` 占位即可 |
| 下载模型极慢 | 走的是 `registry.ollama.ai` | 属正常；或改用 ModelScope / hf-mirror 拉 GGUF 再导入 |
