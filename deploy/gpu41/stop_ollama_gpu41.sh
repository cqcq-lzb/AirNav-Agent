#!/usr/bin/env bash
# 停止 gpu41 上的 AirNav Ollama 服务。
#
# 用 pkill -x ollama（精确匹配进程名），不要用 pkill -f 'ollama serve'：
# 后者会匹配到执行它的那个 shell 自身的命令行，把自己一起杀掉
# （症状是 SSH 返回 255 且无任何输出，我踩过一次）。

set -uo pipefail

if ! pgrep -x ollama >/dev/null 2>&1; then
    echo "ollama 未在运行。"
    exit 0
fi

pids=$(pgrep -x ollama | tr '\n' ' ')
echo "停止 ollama: $pids"
pkill -x ollama || true

for _ in $(seq 1 15); do
    pgrep -x ollama >/dev/null 2>&1 || { echo "已停止。"; exit 0; }
    sleep 1
done

echo "15 秒后仍在运行，强制结束。"
pkill -9 -x ollama || true
sleep 2
pgrep -x ollama >/dev/null 2>&1 && echo "仍未能停止，请手动检查。" || echo "已停止。"
