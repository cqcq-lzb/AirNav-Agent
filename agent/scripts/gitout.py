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

## ⚠️ 提交信息不要内联进 shell 字符串

**踩过。** 把带正文的 `git commit -m "..."` 内联在 `bash -c "..."` 的双引号里，
正文中的**反引号**会被 bash 当成命令替换先执行掉：

```
原文：检查 `!line.includes('三维视图')`
落库：「 与 alt 文本都不残留」-> 检查 ，
```

反斜杠也会被吃掉（`D:\AirNav-Agent\...` → `D://AirNav-Agent//...`）。
本项目已第三次踩这类坑（另一次是 MSYS 把 `--noproxy *` 展开成文件列表）。

**正确姿势**：写进脚本文件，用 `subprocess` 以 **argv 列表**传参 —— 字符一个都不会被 shell 碰。
长篇正文更推荐走 stdin，连中间文件都不用落：

```python
subprocess.run(["git", "commit", "-m", subject, "-m", body], cwd=REPO)
subprocess.run(["git", "commit", "--amend", "-F", "-"],   # 从 stdin 读信息
               cwd=REPO, input=message.encode("utf-8"))
```

为什么连 `-F <文件>` 都要绕开：本机 DLP 会在文件落盘后补 NUL 填充，
而且**填充可能落在「你检查完」与「git 去读」之间的那一瞬** ——
实测 `read_bytes()` 刚确认干净（1196 字节、无 NUL），git 读同一路径却报
`error: a NUL byte in commit log message not allowed`。

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
