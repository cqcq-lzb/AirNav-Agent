"""AirNav-Agent 的网页版。

把 `NavAgent` 包成一个本地 Web 服务，浏览器里就能对话、看工具调用轨迹、
直接打开三维导航视图。

- 服务端：`agent/web/server.py`
- 界面：`agent/web/static/index.html`（单文件，无构建步骤）

启动：`python -m agent.web.server`
"""
