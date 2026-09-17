#!/usr/bin/env bash
# 在 gpu41 上启动 AirNav 用的 Ollama 服务。
#
# 设计要点（每条都是踩过的坑，别删）：
#
#   CUDA_VISIBLE_DEVICES=0
#       只占 GPU 0。1/2/4~7 号卡常被你自己的训练任务占着，
#       绝对不能碰。
#
#   OLLAMA_VULKAN=false
#       ⚠️ 必须显式关掉。CUDA_VISIBLE_DEVICES 管不住 Vulkan 后端 ——
#       实测不关的话 ollama 会把 8 张卡全部认出来（total_vram=636 GiB），
#       有把模型铺到别人卡上的风险。
#
#   OLLAMA_HOST=0.0.0.0:11434
#       默认只监听 127.0.0.1，本机连不上。仅在内网开放，
#       不要把 11434 映射到公网。
#
#   OLLAMA_KEEP_ALIVE=30m
#       默认 5 分钟就把模型卸载。演示时停一下再问要重新加载，
#       等 10 秒很尴尬，所以拉长到 30 分钟。
#
# 用法：
#   ./start_ollama_gpu41.sh          # 启动（已在跑则直接返回）
#   ./stop_ollama_gpu41.sh           # 停止
#   tail -f serve.log                # 看日志

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_VULKAN=false
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS=/data2/home/wcq/.ollama/models
export OLLAMA_CONTEXT_LENGTH=32768
export OLLAMA_KEEP_ALIVE=30m

if pgrep -x ollama >/dev/null 2>&1; then
    echo "ollama 已在运行 (pid $(pgrep -x ollama | head -1))，不重复启动。"
    echo "想重启请先执行 ./stop_ollama_gpu41.sh"
    exit 0
fi

setsid nohup ./bin/ollama serve > serve.log 2>&1 < /dev/null &
sleep 8

if ! pgrep -x ollama >/dev/null 2>&1; then
    echo "启动失败，serve.log 末尾："
    tail -20 serve.log
    exit 1
fi

echo "已启动，pid $(pgrep -x ollama | head -1)"
echo
echo "--- 识别到的算力设备（应当只有一条 CUDA0）---"
grep -o 'inference compute.*' serve.log 2>/dev/null | sed 's/libdirs=.*//' || echo "(日志尚未写出)"
echo
echo "--- 监听地址 ---"
grep -o 'Listening on.*' serve.log 2>/dev/null | tail -1 || true
