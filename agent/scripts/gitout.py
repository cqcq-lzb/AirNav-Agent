"""在 stdout 被 DLP 切断的环境下读取 git 输出。

## 为什么要这个脚本

本机 git.exe 的 stdout 写入会被拦掉，直接跑：

    git status
    git log --oneline -5

会得到

    fatal: write failure on 'stdout': Bad file descriptor

`git log` 直接失败（exit 128），`git status` 更阴 —— 它 **rc=0 但输出为空**，
看上去像「工作区干净」，据此判断就会得出完全相反的结论。

（`--no-pager`、`GIT_PAGER=cat`、重定向到文件、PowerShell 的 `Out-File`
都试过，全部无效 —— 不是 pager 的问题。）

## ⚠️ 写操作会「报错但成功」，别被吓到

`git commit` / `git add` 同样会打印这行 fatal，因为 git 是**先干活、再写输出**：
提交对象已经落库，只是最后那句 `[main abc1234] ...` 写不出去。实测
commit 返回 128，但 `git log` 里提交确实在。

所以：**看到这个报错不要重试**，直接用本脚本查 `git log` 确认即可。
重复 commit 只会报 nothing to commit（或者更糟，把改动并进第二个提交）。

（`git add` 通常不打印任何东西，因此不受影响。）

## 原理

git 的 stdout 是坏的，但 **它作为子进程被管道接住时是好的**：
Python 用 `subprocess.run(..., capture_output=True)` 拿到的输出完好无损。

## 用法

    PY=D:/AirNav-Agent/.venv-mcp/Scripts/python.exe

    # 不传参 → 一次看 status / log / diff --stat
    "$PY" agent/scripts/gitout.py

    # 传参 → 原样转发给 git（第一个参数是子命令）
    "$PY" agent/scripts/gitout.py log --oneline -8
    "$PY" agent/scripts/gitout.py diff --stat
    "$PY" agent/scripts/gitout.py show --stat HEAD

注意 `git add` / `git commit` 这类**写操作不需要本脚本**（它们不读 stdout）。
只有需要「看到 git 说了什么」时才用它。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# 仓库根：本文件在 agent/scripts/ 下，上溯两层
REPO = Path(__file__).resolve().parents[2]

# 无参时的默认体检项
DEFAULT_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("status", "--short"),
    ("log", "--oneline", "-8"),
    ("diff", "--stat"),
)

SEPARATOR = "-" * 60


def run(args: tuple[str, ...] | list[str]) -> str:
    """跑一条 git 命令，返回「命令 + rc + 输出」的可读文本。"""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
        )
    except FileNotFoundError:
        return f"$ git {' '.join(args)}\n[git 不在 PATH 上]\n"

    body = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
    if not body.strip():
        body = "（无输出）\n"
    return f"$ git {' '.join(args)}\n[rc={proc.returncode}]\n{body}"


def main(argv: list[str]) -> int:
    commands = (tuple(argv),) if argv else DEFAULT_COMMANDS
    for args in commands:
        sys.stdout.write(run(args))
        if len(commands) > 1:
            sys.stdout.write(SEPARATOR + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
