"""提交前自检：凭据泄漏 + DLP 密文混入。

三个检查，都只输出「键名 + 判定」，**绝不打印值本身**：

  1. 凭据扫描 —— 遍历可扫文件，找出疑似真实凭据的字面量
  2. 密文扫描 —— 本机 DLP 会把某些文件改写成密文（文件头 %TSD-Header-###%），
     git.exe 不在解密白名单里，照常提交就会把密文写进仓库
  3. 内网地址扫描 —— 硬编码的私有网段 IP 会把内网拓扑随仓库发出去。
     ⚠️ 这一项曾经**只写在文档里**（「入库文件 grep 不出内网 IP」），
     从来没被任何检查覆盖 —— 结果是文档里出现了两个互相矛盾的计数
     （「11 文件 / 21 处」「38 文件 / 81 处」），实测是 13 文件 / 22 处。
     **不测就等于没有**：现在它是门禁的一部分。

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
    # ⚠️ "selfcheck" 是自检夹具的标志：`check_web.py` 用
    # "selfcheck-token-2f8a1c9d" 当临时 token 测鉴权层，那是**长值 +
    # 字符类 2**，会被长哈希规则判成真凭据 → 每次跑门禁都多一条红。
    # 判据：值里自带「这是自检」的语义标记，就不是「疑似真凭据」。
    "selfcheck", "fixture",
)
# 弱口令字面量（本身就是风险，但也确实是"配置值"）
WEAK_LITERALS = {"123456", "password", "admin", "root", "test", "abc123", "1234", "000000"}

DLP_MAGIC = b"%TSD-Header-###%"

# ---------------------------------------------------------------- 内网地址
# 私有网段：10/8、172.16/12、192.168/16，以及常见容器网段。
# 只匹配 IPv4 字面量，四段都锚定，避免命中版本号（如 192.168.1.1 之外的 1.2.3）。
PRIVATE_IP_RE = re.compile(
    r"(?<![\d.])("
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r")(?![\d.])"
)
# 占位符主机名 —— 换成这些就等于已经脱敏，不算泄漏。
# ⚠️ 判据钉「地址本身是不是私有网段字面量」，不钉「这行有没有出现 gpu-node」：
#    文档里允许既写占位符、又在同一行解释「这里本来是真实地址」。
PLACEHOLDER_HOSTS = ("gpu-node", "your-host", "192.168.0.1", "127.0.0.1", "localhost")

# 显式豁免标记：确要保留真实地址当「本机兜底默认值」时，在**同一行**写上它。
# 为什么不做成「文件级豁免」：文件级豁免会连注释/文档一起放过，
# 而我们要保护的恰恰是「这个地址被意外复制到别处」。标记放在行上，
# 复制出去的行不带标记 → 立刻报出来。
# ⚠️ 这个标记本身就是**清单**：想查「到底留了几处真地址」，grep 它就够。
IP_ALLOW_MARKER = "airnav-allow-real-ip"

# ⚠️ 为什么不能直接用「全文含 NUL」当 DLP 填充的判据 —— 两种形态必须分开：
#   · DLP 的块对齐填充是**尾部**补 \x00，正文是干净的文本
#   · 二进制文件（.nii.gz / .png / 权重）的 NUL **散布全文**
#   混在一起，每一份二进制都会被判成「DLP 改写」。
#   实测（首次把合成夹具入库时）：4 个 .nii.gz 全部误报，而它们其实
#   gzip magic=1f8b、可正常解压、**暂存区与磁盘逐字节相同**，
#   大小 16786 / 4105 / 1694 / 1814 也都不是 4096 的整数倍。
#
# ⚠️ 也**不能**只靠「首块 8000 字节有没有 NUL」—— 那会漏报**短文件**：
#   一个 20 字节的 .py 被填充到块对齐，NUL 就落在首块里了。


def nul_only_at_tail(raw: bytes) -> bool:
    """NUL 是否**只出现在尾部**（＝文本文件被块对齐填充的特征）。

    去掉尾部连续 NUL 之后如果还有 NUL，说明 NUL 散布在正文里 —— 那是二进制。

    ⚠️ 边界：一个**整份都是 NUL** 的文件会判成「只在尾部」→ 报填充。
    偏保守，但这类文件本身就不该出现在仓库里（宁误报，别放过）。
    """
    return b"\x00" not in raw.rstrip(b"\x00")


def looks_binary(path: Path, raw: bytes) -> bool:
    """这份内容是不是二进制文件。返回 True ＝ **不**做 NUL 填充检查。

    判据：
    1. 扩展名在 `EXCLUDE_EXT` 里 —— 项目既有机制（`.nii.gz` 的 suffix 是 `.gz`）
    2. 否则看 NUL 的**分布**：完全没有 NUL → 文本；
       NUL 只在尾部 → 是被填充的文本（**不是**二进制，要继续查）；
       NUL 散布全文 → 二进制

    ⚠️ `DLP_MAGIC` 密文检查对二进制**仍然生效** —— 那才是二进制也会中的招。
    """
    if path.suffix.lower() in EXCLUDE_EXT:
        return True
    if b"\x00" not in raw:
        return False
    return not nul_only_at_tail(raw)


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


def scan_internal_ip(rel: str, text: str, findings: list[tuple[str, str, str]]) -> None:
    """找出硬编码的私有网段 IP。

    白名单（不算命中）：
      - 该行含有占位符主机名 / 回环地址（已脱敏的形态）
      - 该行带显式豁免标记 `IP_ALLOW_MARKER`（＝有意保留的兜底默认值）
    """
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        # 纯注释行不算「硬编码」—— 注释里提到地址是在**说明**它，
        # 不是在用它。跑不过这条的话，这份检查自身就成了漏报源。
        if not stripped or stripped.startswith(("#", "//", "*", "<!--")):
            continue
        if IP_ALLOW_MARKER in line:
            continue
        for m in PRIVATE_IP_RE.finditer(line):
            ip = m.group(1)
            if ip in PLACEHOLDER_HOSTS:
                continue
            # 同一行出现占位符主机名 → 已经脱敏，地址只是被引用来说明
            if any(h in line for h in PLACEHOLDER_HOSTS if not h[0].isdigit()):
                continue
            findings.append((
                "内网地址", f"{rel}:{lineno}",
                f"硬编码私有网段 IP（前缀 {ip.rsplit('.', 2)[0]}.*）"
                f" → 请改环境变量或 gpu-node 占位符",
            ))


def scan_text(rel: str, text: str, findings: list[tuple[str, str, str]]) -> None:
    scan_internal_ip(rel, text, findings)
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
        if looks_binary(Path(rel), raw):
            # 二进制文件：里面的 NUL 是内容本身，不是 DLP 填充。
            # 文本类检查（凭据正则、编码）对二进制也没有意义，直接跳过。
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
    internal_ip = [f for f in findings if f[0] == "内网地址"]

    if dlp:
        print(f"🔴 DLP 改写（{len(dlp)} 处）—— 这些内容不能被 git 正确版本化：")
        for _, where, desc in dlp:
            print(f"    {where}")
            print(f"        {desc}")
        print()
    if internal_ip:
        print(f"🔴 硬编码内网地址（{len(internal_ip)} 处）—— 会把内网拓扑随仓库发出去：")
        for _, where, desc in internal_ip:
            print(f"    {where}")
            print(f"        {desc}")
        print("    修法：代码走环境变量兜底，文档/示例换 `gpu-node` 占位符；")
        print("          确要保留原地址时，同一行写上 `gpu-node` 说明已脱敏即可放行。")
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
        print("✅ 未发现凭据泄漏、DLP 密文或硬编码内网地址")
        return 0

    print("提示：凭据扫描的误报率不低（header 名、变量名、占位串都会命中），")
    print("      务必逐个打开对应行确认，不要直接按结论改代码。")
    return 1 if (creds or dlp or internal_ip) else 0


if __name__ == "__main__":
    raise SystemExit(main())
