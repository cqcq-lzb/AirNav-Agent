#!/usr/bin/env bash
# 在 gpu41 上把 ollama 装到**用户目录**（不需要 sudo）。
#
# 为什么不用官方 install.sh：它要 sudo 写 /usr/local 并创建 ollama 系统用户。
# 这台机器 sudo 需要密码，且是多人共享的 GPU 服务器，装用户目录更克制、也更好撤。
#
# 用法（在服务器上）：
#   bash install_ollama.sh
#
# 幂等：已装好就跳过下载。
set -euo pipefail

OLLAMA_HOME="${OLLAMA_HOME:-$HOME/ollama}"
BIN="$OLLAMA_HOME/bin/ollama"
PKG="ollama-linux-amd64"
TARBALL="$HOME/${PKG}.tar.zst"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[ERROR] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 前置检查
log "前置检查"
for t in curl zstd tar; do
    command -v "$t" >/dev/null || die "缺少 $t，请先安装"
done
echo "curl : $(command -v curl)"
echo "zstd : $(command -v zstd)"
echo "HOME : $HOME"
df -h "$HOME" | tail -1

# ---------------------------------------------------------------- 已装检测
if [ -x "$BIN" ]; then
    log "已存在 $BIN，跳过下载"
    "$BIN" --version || true
    exit 0
fi

# ---------------------------------------------------------------- 下载
mkdir -p "$OLLAMA_HOME"
log "下载 $PKG.tar.zst（约 1.4 GB 压缩 / 2.25 GB 解压）"
echo "目标文件：$TARBALL"

# ⚠️ 不要直连 ollama.com —— 它会 307 跳到 GitHub，实测只有 30 KB/s
#（1 小时才下 140 MB，本机踩过）。这里先给几个镜像测速，取最快的。
MIRRORS=(
    "https://gh-proxy.com"
    "https://ghproxy.net"
    "https://ghfast.top"
    ""
)
GH_URL="https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION:-0.34.1}/${PKG}.tar.zst"

pick_source() {
    local best_speed=0 best_prefix="" speed url label
    for prefix in "${MIRRORS[@]}"; do
        if [ -z "$prefix" ]; then url="$GH_URL"; label="GitHub 直连"; else url="$prefix/$GH_URL"; label="$prefix"; fi
        speed=$(curl -sL -o /dev/null -w '%{speed_download}' --max-time 12 "$url" 2>/dev/null || echo 0)
        speed=${speed%.*}
        printf '  %-46s %8s B/s\n' "$label" "${speed:-0}"
        if [ "${speed:-0}" -gt "$best_speed" ]; then
            best_speed=$speed
            best_prefix=$prefix
        fi
    done
    echo "  选中：${best_prefix:-GitHub 直连}（${best_speed} B/s）"
    if [ "$best_speed" -lt 200000 ]; then
        echo "  ⚠️ 最快也只有 $((best_speed/1024)) KB/s，会是慢速下载；" >&2
        echo "     也可以用 ModelScope / hf-mirror 拉 GGUF 再导入。" >&2
    fi
    if [ -z "$best_prefix" ]; then echo "$GH_URL"; else echo "$best_prefix/$GH_URL"; fi
}

if [ -s "$TARBALL" ]; then
    echo "已存在同名压缩包，复用（如需重下请先删除）"
else
    echo "测速中……"
    SRC=$(pick_source)
    echo "下载源：$SRC"
    curl -fL --progress-bar "$SRC" -o "$TARBALL" \
        || die "下载失败。可先手动 curl 到 \$HOME/${PKG}.tar.zst 再重跑本脚本（会自动复用）"
fi

echo "压缩包大小：$(du -h "$TARBALL" | cut -f1)"
echo "完整性校验："
zstd -t "$TARBALL" || die "压缩包损坏，删除后重下：rm $TARBALL"

# ---------------------------------------------------------------- 解压
log "解压到 $OLLAMA_HOME"
zstd -d -c "$TARBALL" | tar -xf - -C "$OLLAMA_HOME"

# 官方包结构是 bin/ + lib/
[ -x "$BIN" ] || die "解压后找不到 $BIN，请检查包结构"

# ---------------------------------------------------------------- 验证
log "验证"
"$BIN" --version
echo
echo "可执行文件：$BIN"
echo "自带后端库："
ls -1 "$OLLAMA_HOME/lib/ollama" 2>/dev/null | head -12 || true
echo
echo "GPU 后端库（关键，缺了就只能 CPU 跑）："
find "$OLLAMA_HOME/lib" -name '*cuda*' -o -name '*vulkan*' 2>/dev/null | head -10 || echo "  （未找到，可能是按需下载）"

log "安装完成"
echo "⚠️ 启动时务必设 OLLAMA_VULKAN=false —— 否则 ollama 会把 8 张卡全认出来"
echo "   （CUDA_VISIBLE_DEVICES 管不住 Vulkan 后端），有占别人训练卡的风险。"
echo "   已封装好，直接用：bash start_ollama_gpu41.sh"
