"""等服务真的起来了，再打开浏览器。

为什么需要它：`启动网页Agent.bat` 原来先 `start "" <url>` 再启动服务，
而服务要几秒才监听（要 import VTK、加载应用）。**首屏必然是「无法访问此网站」**，
用户以为坏了 —— 实测就是这么被问到的。

用法（由 bat 在后台调起，前台留给服务本身）：

    python -m agent.scripts.open_web_ui

环境变量：

- `AIRNAV_WEB_PORT`  端口，默认 8777（与 `agent.web.server` 一致）
- `AIRNAV_WEB_HOST`  主机，默认 127.0.0.1
- `AIRNAV_WEB_OPEN`  设成 0 / false / no 时**只探测不打开浏览器**（自检、CI 用）
- `AIRNAV_WEB_WAIT`  等多久算超时（秒），默认 60

退出码：0 = 已就绪；2 = 超时（服务没起来，此时**不**打开浏览器，并打印一行原因）。
"""

from __future__ import annotations

import os
import socket
import sys
import time
import webbrowser

DEFAULT_WAIT = 60.0


def _flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def wait_ready(host: str, port: int, timeout: float, interval: float = 0.2) -> float | None:
    """返回「第几秒可连上」；超时返回 None。只做 TCP 连通性探测。"""
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            return None
        probe = socket.socket()
        probe.settimeout(0.5)
        try:
            if probe.connect_ex((host, port)) == 0:
                return elapsed
        finally:
            probe.close()
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    host = os.environ.get("AIRNAV_WEB_HOST") or "127.0.0.1"
    port = int(os.environ.get("AIRNAV_WEB_PORT") or 8777)
    timeout = float(os.environ.get("AIRNAV_WEB_WAIT") or DEFAULT_WAIT)
    should_open = _flag("AIRNAV_WEB_OPEN", True)

    ready = wait_ready(host, port, timeout)
    url = f"http://{host}:{port}/"

    if ready is None:
        print(f"[open_web_ui] {timeout:.0f}s 内没等到 {url} —— 服务没起来，不开浏览器。")
        print("[open_web_ui] 先看服务那个窗口里的报错（常见：页面文件太小 / 端口被占）。")
        return 2

    print(f"[open_web_ui] {url} 已就绪（{ready:.1f}s）。")
    if not should_open:
        print("[open_web_ui] AIRNAV_WEB_OPEN=0，跳过打开浏览器。")
        return 0

    if webbrowser.open(url):
        print("[open_web_ui] 已用默认浏览器打开。")
    else:
        print(f"[open_web_ui] 没能自动打开，请手动访问 {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
