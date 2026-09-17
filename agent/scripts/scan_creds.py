"""提交前自检：凭据泄漏 + DLP 密文混入。

两个检查，都只输出「键名 + 判定」，**绝不打印值本身**：

  1. 凭据扫描 —— 遍历可扫文件，找出疑似真实凭据的字面量
  2. 密文扫描 —— 本机 DLP 会把某些文件改写成密文（文件头 %TSD-Header-###%），
     git.exe 不在解密白名单里，照常提交就会把密文写进仓库

用法::

    python -m agent.scripts.scan_creds              # 扫工作区（默认，跳过 .gitignore 命中项）
    python -m agent.scripts.scan_creds --staged     # 只扫已暂存内容（提交前用）
    python -m agent.scripts.scan_creds --all        # 连被忽略的文件一起扫

退出码：发现高风险项返回 1，否则返回 0。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]

# 体积大户 / 虚拟环境 / 缓存，一律不进扫描
EXCLUDE_DIRS = {
    ".venv-mcp", "cases", ".cache", "__pycache__", ".git", "node_modules",
    "build_context", "runtime_data", "runtime_cases", "runtime_status",
}
# 二进制扩展名
EXCLUDE_EXT = {
    ".dcm", ".nii", ".gz", ".npz", ".npy", ".pkl", ".pt", ".pth", ".onnx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".mp4", ".zip",
    ".tar", ".exe", ".dll", ".so", ".pyd", ".vtk", ".vti", ".stp", ".stl",
    ".obj", ".pdf", ".ttf", ".otf", ".woff", ".woff2", ".icns", ".ico",
}
# 只扫这些扩展名（空集表示不限）
SCAN_EXT = {
    ".env", ".example", ".server", ".api", ".json", ".yml", ".yaml",
    ".bat", ".cmd", ".ps1", ".sh", ".py", ".ini", ".cfg", ".toml", ".txt",
    ".md", ".dockerignore", ".gitignore", ".gitattributes",
}

# 键名里出现这些词就当成潜在凭据槽位
KEY_PATTERN = re.compile(
    r"(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|auth[_-]?key|ssh[_-]?pass)",
    re.IGNORECASE,
)

# 占位符标志。⚠️ "replace" 必须在表里：
# 项目里用的是 "replace-with-a-long-random-token"（32 字符），
# 漏掉它会被长哈希规则误判成真凭据。
PLACEHOLDER_HINTS = (
    "your", "replace", "xxx", "changeme", "placeholder", "todo", "example",
    "sample", "dummy", "填入", "填写", "替换", "示例", "请改",
    "<", ">", "${", "...", "***",
)
# 弱口令字面量（本身就是风险，但也确实是"配置值"）
WEAK_LITERALS = {"123456", "password", "admin", "root", "test", "abc123", "1234", "000000"}

DLP_MAGIC = b"%TSD-Header-###%"


def iter_trackable_files() -> list[Path] | None:
    """列出「git 会管的文件」= 已跟踪 + 未跟踪但未被忽略。

    这才是「即将提交的内容」，比盲目遍历整个工作区准确 ——
    被 .gitignore 排除的文件（如 docker/.env.api）本来就不会进仓库。
    返回 None 表示不是 git 仓库。
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True,
    )
    if out.returncode != 0:
        return None
    return [ROOT / n for n in out.stdout.decode("utf-8", "replace").split("\0") if n]


def iter_workspace_files():
    """用 os.walk 而不是 Path.rglob —— rglob 不支持剪枝，
    会把 1.9G 的 cases/ 全走一遍。"""
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def scannable(path: Path) -> bool:
    name = path.name
    if name.startswith(".env"):
        return True
    suffix = path.suffix.lower()
    if suffix in EXCLUDE_EXT:
        return False
    return suffix in SCAN_EXT or name in (".gitignore", ".gitattributes", ".dockerignore")


def classify(value: str) -> tuple[str, bool]:
    """返回 (判定文本, 是否高风险)。不打印 value 本身，只打印长度与字符类。

    这里刻意做了**两类降噪**，否则误报率会高到没人看结论：
      - 值里有括号/运算符 → 是表达式（os.getenv(...)、math.log(...)），不是字面量
      - 值是裸标识符（self.password = password）→ 变量赋值，不是字面量
    真正的凭据字面量要么是带引号的串，要么是长而高熵的裸串。
    """
    v = value.strip().strip('"').strip("'").strip().rstrip(",")
    if not v:
        return "空", False
    low = v.lower()

    # 布尔 / 空值
    if low in ("true", "false", "none", "null"):
        return "布尔/空值", False
    # 弱口令字面量（纯数字或常见弱口令）
    if low in WEAK_LITERALS:
        return f"弱口令字面量（长度 {len(v)}）", True
    # 占位符
    if any(h in low or h in value for h in PLACEHOLDER_HINTS):
        return "占位符", False
    # 环境变量引用
    if low.startswith("$") or low.startswith("%"):
        return "变量引用", False
    # 函数调用 / 表达式：含括号、方括号、运算符、引号内嵌
    if any(ch in v for ch in "()[]{}+|&"):
        return "表达式/函数调用", False
    # 裸标识符（无引号、无空格、纯 [A-Za-z0-9_.]）→ 变量赋值
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", v):
        return "变量/标识符赋值", False
    if len(v) < 8:
        return f"过短（长度 {len(v)}）", False

    kinds = sum([
        any(c.islower() for c in v),
        any(c.isupper() for c in v),
        any(c.isdigit() for c in v),
    ])
    if len(v) >= 16 and kinds >= 2:
        return f"*疑似真凭据*（长度 {len(v)}，字符类 {kinds}）", True
    if len(v) >= 32:
        return f"*疑似真凭据*（长值，长度 {len(v)}）", True
    return f"短值（长度 {len(v)}）", False


def scan_text(rel: str, text: str, findings: list[tuple[str, str, str]]) -> None:
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "*", "<!--")):
            continue
        # KEY=VALUE 或 "key": "value"
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_.\-\[\]"\']*)\s*[=:]\s*(.+)$', stripped)
        if m and KEY_PATTERN.search(m.group(1)):
            key = m.group(1).strip()
            val = m.group(2)
            # 点号键 + 裸标识符 = 属性赋值（self.password = password），不是字面量
            if "." in key and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", val.strip().rstrip(",")):
                continue
            verdict, risky = classify(val)
            if risky:
                findings.append(("凭据", f"{rel}:{lineno}", f"{key} -> {verdict}"))
            continue
        # 源码里的 name = "literal"
        m2 = re.search(r'([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(["\'])([^"\']{4,})\2', stripped)
        if m2 and KEY_PATTERN.search(m2.group(1)):
            verdict, risky = classify(m2.group(3))
            if risky:
                findings.append(("凭据", f"{rel}:{lineno}", f"{m2.group(1)} -> {verdict}"))


def check_workspace(all_files: bool) -> tuple[int, list[tuple[str, str, str]]]:
    findings: list[tuple[str, str, str]] = []
    scanned = 0
    if all_files:
        candidates: Iterable[Path] = iter_workspace_files()
    else:
        listed = iter_trackable_files()
        candidates = listed if listed is not None else iter_workspace_files()
    for path in candidates:
        if not all_files and not scannable(path):
            continue
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        scanned += 1
        # ⚠️ 注意：Python 在本机能看到 DLP 解密后的明文，所以这里主要
        #    兜住「本来就没加密、但内容异常」的情况；真正的密文检测
        #    必须走 --staged（用 git 自己读，git 不在解密白名单里）。
        if raw.startswith(DLP_MAGIC):
            findings.append(("密文", rel, f"DLP 整体加密（长度 {len(raw)}），git 读到的就是密文"))
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            if b"\x00" not in raw:
                try:
                    raw.decode("gbk")
                    findings.append(("编码", rel, "非 UTF-8（可按 GBK 解码）→ 中文可能已损坏"))
                except UnicodeDecodeError:
                    pass
            continue
        scan_text(rel, text, findings)
    return scanned, findings


def check_staged() -> tuple[int, list[tuple[str, str, str]]]:
    """只检查已暂存内容。用 git 自己读，能直接暴露「git 读到密文」的情况。"""
    findings: list[tuple[str, str, str]] = []
    names = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=ROOT, capture_output=True,
    )
    if names.returncode != 0:
        print("  ⚠️ 不是 git 仓库或 git 不可用，回退到工作区扫描")
        return check_workspace(all_files=False)
    staged = [n for n in names.stdout.decode("utf-8", "replace").split("\0") if n]
    for rel in staged:
        blob = subprocess.run(
            ["git", "cat-file", "blob", f":{rel}"],
            cwd=ROOT, capture_output=True,
        )
        if blob.returncode != 0:
            continue
        raw = blob.stdout
        if raw.startswith(DLP_MAGIC):
            findings.append(("密文", rel, f"暂存的是 DLP 密文（长度 {len(raw)}）→ 千万不要提交"))
            continue
        if b"\x00" in raw:
            findings.append(("填充", rel, "暂存内容含 NUL 字节（DLP 块对齐填充）"))
            continue
        text = raw.decode("utf-8", "replace")
        scan_text(rel, text, findings)
    return len(staged), findings


def main() -> int:
    parser = argparse.ArgumentParser(description="提交前自检：凭据泄漏 + DLP 密文")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="只扫已暂存内容（提交前用）")
    group.add_argument("--all", action="store_true", help="连被忽略的文件一起扫")
    args = parser.parse_args()

    if args.staged:
        scanned, findings = check_staged()
        scope = "已暂存内容"
    else:
        scanned, findings = check_workspace(all_files=args.all)
        scope = "工作区"

    print(f"扫描范围：{scope}，{scanned} 个文件")
    print()

    creds = [f for f in findings if f[0] == "凭据"]
    dlp = [f for f in findings if f[0] in ("密文", "填充")]
    encoding = [f for f in findings if f[0] == "编码"]

    if dlp:
        print(f"🔴 DLP 改写（{len(dlp)} 处）—— 这些内容不能被 git 正确版本化：")
        for _, where, desc in dlp:
            print(f"    {where}")
            print(f"        {desc}")
        print()
    if creds:
        print(f"🔴 疑似凭据（{len(creds)} 处）—— 逐条人工确认，误报很常见：")
        for _, where, desc in creds:
            print(f"    {where}")
            print(f"        {desc}")
        print()
    if encoding:
        print(f"⚠️ 编码异常（{len(encoding)} 处）：")
        for _, where, desc in encoding:
            print(f"    {where}  {desc}")
        print()

    if not findings:
        print("✅ 未发现凭据泄漏或 DLP 密文")
        return 0

    print("提示：凭据扫描的误报率不低（header 名、变量名、占位串都会命中），")
    print("      务必逐个打开对应行确认，不要直接按结论改代码。")
    return 1 if (creds or dlp) else 0


if __name__ == "__main__":
    raise SystemExit(main())
