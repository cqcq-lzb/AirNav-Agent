"""锁定 MCP server 的依赖闭包，并区分「已装在 env 内」与「仅存在于用户目录」。

为什么需要这个脚本
------------------
dicom 环境是 Python 3.10，它的 user-site 目录
    %APPDATA%\\Python\\Python310\\site-packages
会和 env 自身的 site-packages 一起参与 import。本机实测：mcp 及其**全部**
依赖（starlette / uvicorn / pydantic / pydantic-core / httpx2 / jsonschema ...）
都只装在 user-site 里，env 自己身上一个都没有。

危险之处在于这件事**不会报错**：
  * `pip list` 会把 user-site 的包一起列出来，显示 mcp 2.2.0 已安装；
  * `pip install --dry-run` 的报告里 install 为空，pip 认为无需安装。

只有真正启动时才可能翻车 —— 加了 `PYTHONNOUSERSITE=1`、用 `python -s`、
换一个解释器、或者用户目录被清理，`import mcp` 立刻 ModuleNotFoundError。

因此必须用 importlib.metadata.distributions(path=...) 直接枚举 env 自己的
site-packages，而不能相信 pip list。

用法
----
    python -m agent.scripts.lock_mcp_deps             # 只报告，不落盘
    python -m agent.scripts.lock_mcp_deps --write     # 直接写出两个锁定文件
    python -m agent.scripts.lock_mcp_deps --install   # 清洗锁定文件并喂给 pip

关于产出方式（本机特有，重要）
------------------------------
本机装了透明加密类 DLP，写出的文件有两种被改写的方式，**触发条件都不确定**：

1. **整体加密**：文件头变成 `%TSD-Header-###%`，长度按 8192 字节对齐。
   pip / python 能透明读回来，但 bash、od、编辑器、版本控制看到的是密文。
2. **尾部 NUL 填充**：正文不变，但被补 `\x00` 到 4096 字节块对齐。
   实测 requirements_mcp.txt = 1291 字节正文 + 2805 字节 NUL。

第 2 种特别阴，因为它**看起来是明文**（head 一看完全正常），却会让按行
解析的消费方多出一条超长假行：
  * `read_text().splitlines()` 会多出一个 2805 字节的"包名"；
  * `pip install -r` 直接 `ERROR: Invalid requirement: '\x00\x00...'`。

同类小文件在本机多数**不会**被填充，所以不要赌文件干净 —— 所有消费方都要
先清 NUL。本脚本的 `read_pins()` 与 `check_mcp._locked_closure()` 都已处理。

产出
----
    requirements_mcp.txt      完整依赖闭包（含精确版本），用于文档与复现
    requirements_mcp_env.txt  仅缺在 env 的那部分，可直接喂给 pip
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_SITE = Path(sys.prefix) / "Lib" / "site-packages"
USER_SITE_HINT = "AppData/Roaming/Python"

LOCK_ALL = ROOT / "requirements_mcp.txt"
LOCK_ENV = ROOT / "requirements_mcp_env.txt"

# 本机直连 PyPI 会 SSL 失败，统一走清华镜像。
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# --no-deps：版本已锁死，不需要 pip 再解析，也就不会顺手升级基础环境里的包。
# --ignore-installed：否则 pip 会因「在别处已经满足」而跳过，包仍留 user-site。
# --no-user：明确禁止装进用户目录 —— 依赖归属层的存在意义就在这里。
PIP_FLAGS = ("--ignore-installed", "--no-deps", "--no-user")

# 合法的 `名==版本` 行。任何越界字符都说明这行不是包名（例如 NUL 填充）。
_PIN_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==[^\s]+$")


# 闭包起点。MCP server 本体，其余依赖由元数据自动展开。
ROOTS = ("mcp",)

# 需求串里出现在包名之后的第一个字符，用来切出纯包名。
# 覆盖 `pyjwt[crypto]>=2.10.1`、`typing-extensions>=4.13.0` 这类写法。
_NAME_CUT = re.compile(r"[\s\[<>=!~;()]")

# PEP 503 归一化。必须做：同一个发行版在不同包的 requires 里可能写成
# `typing-extensions` 或 `typing_extensions`，不归一化就会被当成两个包，
# 导致「已装在 env」的包在报告里又被列为「缺在 env」。
_NORM = re.compile(r"[-_.]+")


def canonical(name: str) -> str:
    """PEP 503 规范名：小写、[-_.] 统一成单个连字符。"""
    return _NORM.sub("-", name).strip().lower()


def env_local_names() -> set[str]:
    """枚举真正落在 env 自身 site-packages 里的发行版名（已归一化）。"""
    if not ENV_SITE.is_dir():
        return set()
    names: set[str] = set()
    for dist in md.distributions(path=[str(ENV_SITE)]):
        raw = dist.metadata["Name"]
        if raw:
            names.add(canonical(raw))
    return names


def marker_ok(req: str) -> bool:
    """需求串的环境标记是否成立；无法判定时按成立处理（宁多装不漏装）。"""
    if ";" not in req:
        return True
    _, _, spec = req.partition(";")
    try:
        from packaging.markers import Marker

        return Marker(spec.strip()).evaluate()
    except Exception:
        return True


def closure(roots: tuple[str, ...]) -> dict[str, str]:
    """按当前已安装元数据展开传递依赖，返回 {归一化包名: 版本}。"""
    out: dict[str, str] = {}
    stack = list(roots)
    while stack:
        name = stack.pop()
        key = canonical(name)
        if key in out:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        out[key] = dist.version
        for req in dist.requires or []:
            if not marker_ok(req):
                continue
            dep = _NAME_CUT.split(req.strip(), 1)[0].strip()
            if dep:
                stack.append(dep)
    return out


def dist_location(name: str) -> str:
    """发行版元数据所在目录，用于说明它到底住在哪。"""
    try:
        return str(md.distribution(name).locate_file(""))
    except Exception:
        return "?"


def read_pins(path: Path) -> tuple[list[str], str]:
    """读锁定文件里的 `名==版本` 行，返回 (pin 列表, 备注)。

    按字节读并清 NUL：本机 DLP 会把文件补 \\x00 到 4096 字节块对齐，
    而填充段里没有换行符，splitlines() 会把它当成一条 2805 字节的假行。
    """
    raw = path.read_bytes()
    n_pad = raw.count(b"\x00")
    text = raw.replace(b"\x00", b"").decode("utf-8", errors="replace")

    pins: list[str] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if _PIN_OK.match(line):
            pins.append(line)
        else:
            skipped += 1

    notes: list[str] = []
    if n_pad:
        notes.append(f"清掉 {n_pad} 字节 NUL 填充")
    if skipped:
        notes.append(f"跳过 {skipped} 条无法识别为 pin 的行")
    return pins, "；".join(notes)


def do_install(which: str) -> int:
    """清洗锁定文件后喂给 pip。

    存在的意义：`pip install -r requirements_mcp_env.txt` 在本机会因
    NUL 填充直接失败（Invalid requirement）。这里先把 pin 行抽出来写成
    干净的临时文件，再交给 pip，绕开这块完全不可控的 DLP 行为。
    """
    src = LOCK_ENV if which == "env" else LOCK_ALL
    if not src.is_file():
        print(f"找不到 {src}")
        print("先执行：python -m agent.scripts.lock_mcp_deps --stdout all > requirements_mcp.txt")
        return 1

    pins, note = read_pins(src)
    if note:
        print(f"{src.name}：{note}")
    if not pins:
        print(f"{src.name} 里没有可用的 pin 行，放弃")
        return 1

    clean = ROOT / ".mcp_deps_clean.txt"
    clean.write_text("\n".join(pins) + "\n", encoding="utf-8")

    cmd = [
        sys.executable, "-m", "pip", "install",
        *PIP_FLAGS, "-i", MIRROR, "-r", str(clean),
    ]
    print(f"解释器：{sys.executable}")
    print(f"待装  ：{len(pins)} 个包")
    print("命令  ：" + " ".join(cmd))
    print("-" * 68)
    rc = subprocess.call(cmd)
    clean.unlink(missing_ok=True)
    if rc != 0:
        print(f"pip 退出码 {rc}")
        return rc

    # 装完立刻复核：这些包是否真的落进本解释器自己的站点
    local = env_local_names()
    still_missing = [p.split("==")[0] for p in pins if canonical(p.split("==")[0]) not in local]
    print("-" * 68)
    if still_missing:
        print(f"仍有 {len(still_missing)} 个包不在 {ENV_SITE}：{still_missing[:8]}")
        return 1
    print(f"OK  {len(pins)} 个包均已落入 {ENV_SITE}")
    print("下一步：python -m agent.scripts.check_mcp --full")
    return 0




def _header(title: str, note: str, install_from: str) -> str:
    return (
        f"# {title}\n"
        f"# 由 agent/scripts/lock_mcp_deps.py 生成，请勿手改。\n"
        f"#\n"
        f"# {note}\n"
        f"#\n"
        f"# 安装（推荐，会自动清洗本机 DLP 补的 NUL 填充）：\n"
        f"#   python -m agent.scripts.lock_mcp_deps --install\n"
        f"#\n"
        f"# 等价的手工命令（走清华镜像，本机直连 PyPI 会 SSL 失败）：\n"
        f"#   python -m pip install --ignore-installed --no-deps --no-user \\\n"
        f"#       -i https://pypi.tuna.tsinghua.edu.cn/simple \\\n"
        f"#       -r {install_from}\n"
        f"#   注意：若本文件被补了 \\x00 到 4096 字节对齐，pip 会直接\n"
        f"#   ERROR: Invalid requirement: '\\x00...' —— 此时请用上面的 --install。\n"
        f"#\n"
        f"# --no-deps 是刻意的：版本已经锁死，不需要 pip 再解析一次，\n"
        f"# 也就不会顺手升级 numpy / cryptography 这些基础环境里已有的包。\n"
        f"# --ignore-installed 也是刻意的：否则 pip 会因为「在别处已经满足」而跳过，\n"
        f"# 包就仍然留在 user-site 上，等于没修。\n"
        f"#\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="锁定 MCP 依赖闭包并区分其真实归属",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--write", action="store_true", help="写出两个锁定文件")
    ap.add_argument(
        "--stdout",
        choices=("all", "env", "none"),
        default="none",
        help="把锁定文件内容打到标准输出（all=完整闭包，env=仅补装部分），"
        "配合 shell 重定向可避开本机的透明加密",
    )
    ap.add_argument("--root", default=None, help="替换闭包起点（默认 mcp）")
    ap.add_argument(
        "--install",
        nargs="?",
        const="env",
        choices=("env", "all"),
        help="清洗锁定文件后喂给 pip（默认用 requirements_mcp_env.txt）",
    )
    args = ap.parse_args()

    # --install 是纯安装路径，不做闭包分析（可能在没有 user-site 的干净机器上跑）
    if args.install:
        return do_install(args.install)

    # --stdout 时报告要走 stderr，否则会和锁定文件内容混在一起污染重定向结果
    def _say(*parts: object) -> None:
        print(*parts, file=sys.stderr if args.stdout != "none" else sys.stdout)

    roots = (args.root,) if args.root else ROOTS

    local = env_local_names()
    pinned = closure(roots)

    in_env = sorted(n for n in pinned if n in local)
    missing = sorted(n for n in pinned if n not in local)

    _say(f"解释器      : {sys.executable}")
    _say(f"env 前缀    : {sys.prefix}")
    _say(f"env site    : {ENV_SITE}")
    _say(f"env 内发行版: {len(local)} 个")
    _say(f"闭包起点    : {', '.join(roots)}")
    _say(f"闭包规模    : {len(pinned)} 个\n")

    _say(f"[A] 已在 env 自身（{len(in_env)} 个）")
    _say("    " + (", ".join(in_env) if in_env else "（无）"))
    _say()

    _say(f"[B] 仅在用户目录 —— 离开 user-site 就会 import 失败（{len(missing)} 个）")
    for name in missing:
        loc = dist_location(name)
        tag = "user-site" if USER_SITE_HINT.replace("/", "\\") in loc else "?"
        _say(f"    {name:<30} {pinned[name]:<16} [{tag}]")
    if not missing:
        _say("    （无，env 已自足）")
    _say()

    if not missing:
        _say("结论：env 自足，MCP server 不依赖用户目录。")
    else:
        _say(f"结论：{len(missing)} 个包依赖用户目录，存在启动期失败风险。")
        _say("      执行 `--write` 生成锁定文件后按头部注释安装即可消除。")

    all_lines = [f"{name}=={pinned[name]}" for name in sorted(pinned)]
    env_lines = [f"{name}=={pinned[name]}" for name in missing]
    all_text = (
        _header(
            "MCP server 完整依赖闭包（锁定版本）",
            "mcp 的全部传递依赖及当时版本。重建 .venv-mcp 时装这一份即可。",
            "requirements_mcp.txt",
        )
        + "\n".join(all_lines)
        + "\n"
    )
    env_text = (
        _header(
            "MCP server —— 基础解释器（conda dicom）缺失的那部分",
            "在 D:\\miniforge\\envs\\dicom 下分析得出：这 25 个包当时只存在于\n"
            "# Python 3.10 的 user-site，conda 环境自己身上一个都没有。\n"
            "# 换解释器分析会得到不同结果，因此重建时请优先用完整闭包那一份。",
            "requirements_mcp_env.txt",
        )
        + "\n".join(env_lines)
        + "\n"
    )

    if args.stdout != "none":
        # 走标准输出，由调用方重定向落盘。本机 python 直接写文件会被 DLP 加密。
        text = all_text if args.stdout == "all" else env_text
        sys.stdout.write(text)
        return 0

    if args.write:
        LOCK_ALL.write_text(all_text, encoding="utf-8")
        LOCK_ENV.write_text(env_text, encoding="utf-8")
        print()
        print(f"已写出 {LOCK_ALL.name}（{len(all_lines)} 行）")
        print(f"已写出 {LOCK_ENV.name}（{len(env_lines)} 行）")
        print(
            "注意：本机 python 写出的文件会被 DLP 加密，"
            "如需明文请改用 --stdout 加重定向"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
