"""AirNav-Agent 网页版服务端。

把 `NavAgent` 包成一个本地 HTTP 服务，浏览器里就能用：

    python -m agent.web.server
    # 然后打开 http://127.0.0.1:8777

## 为什么用 starlette 而不是 fastapi

本机 `.venv-mcp` 里**没有 fastapi**（它是 `backend/api_server.py` 的依赖，
跑在 Docker 容器里，不是本地依赖）。而 `starlette` 和 `uvicorn` 恰好已经随
`mcp` 的 CLI 依赖装进来了 —— 所以这个网页版**不需要新增任何依赖**。
路由、SSE、静态文件 starlette 都能干，用不着 fastapi 的那层封装。

## 为什么 Agent 跑在后台线程里

`NavAgent.run()` 是同步阻塞的（内部用 requests 调 LLM，一次规划 2~10 秒）。
如果在 async 路由里直接调，整个事件循环会被卡住 —— 这期间连静态资源都发不出去，
页面上就是「点了发送之后一片死寂」。

所以：**Agent 放进后台线程跑，事件经 `asyncio.Queue` 回传到事件循环**，
再由 `StreamingResponse` 转成 SSE 推给浏览器。`NavAgent` 的 `on_event` 回调
是同步的，所以转发时必须走 `loop.call_soon_threadsafe` ——
从别的线程直接往 asyncio.Queue 里 put 是不安全的。

## 端点

    GET  /                 界面
    GET  /api/meta         病例 / 工具 / 可用后端
    GET  /api/chat?q=...   SSE 事件流（每一步工具调用都会推）
    GET  /artifacts/<name> 打开生成的 viewer（三维导航视图）
    GET  /health           存活探针
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

from ..agent_loop import NavAgent
from ..llm.client import PRESETS, OpenAICompatClient
from ..tools import NavSession, build_registry

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
VIEWERS_DIR = ROOT / "outputs" / "viewers"

DEFAULT_HOST = os.environ.get("AIRNAV_WEB_HOST") or "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("AIRNAV_WEB_PORT") or 8777)
# 网页版默认走远端 GPU：界面里能随时改，所以默认就该挑最快的那个。
DEFAULT_BACKEND = os.environ.get("AIRNAV_DEFAULT_BACKEND") or "gpu41"

# 只允许通过 /artifacts 暴露这两种后缀，且必须在 outputs/viewers 里
ALLOWED_ARTIFACT_SUFFIXES = (".html", ".json")

# 不属于 PRESETS、但网页版要支持的额外后端。
# `heuristic` 是评测用的规则基线：确定性、不联网、不用模型。
# 放在页面上有两个用处：① 服务器不可达时仍然能演示；
# ② 自检（`agent/scripts/check_web.py`）跑它才能做到又快又稳定。
EXTRA_BACKENDS = {
    "heuristic": "规则基线（不需要模型，离线可用）",
}
KNOWN_BACKENDS = tuple(PRESETS) + tuple(EXTRA_BACKENDS)

# 网页版专用的环境说明，附加在系统提示末尾。
#
# ⚠️ 先说清它**做不到**什么（2026-09-17 实测）：
# 光靠这段提示**挡不住**模型在回答里贴本地文件路径。实测（gpu41 / qwen2.5:14b）
# 即使把「不要粘贴本地路径」写得很直白，模型照样输出
#   [D:\AirNav-Agent\outputs\viewers\viewer_LIDC_0089_c3.html](D:\...)
# 原因很实在：`render_viewer` 的返回值里**本来就带 path 字段**
# （`**output.to_dict()`），模型手上有这个数据，提示词不构成障碍 ——
# 和 E01「提示词压不住模型手里已有的信息」是同一条教训。
# 所以真正管用的修法在前端：`static/index.html` 会把回答里这类路径
# 就地替换成可点的「打开三维视图」按钮（见那边的 `md()` 与 `artByName`）。
#
# 那这段提示还留着干什么：它确实能改变**措辞层面**的行为 ——
# 不说「请在命令行执行 …」，而是直接调工具。这个是有用的，
# 只是别指望它管住路径。
WEB_ENVIRONMENT_NOTE = (
    "你正运行在一个**网页对话界面**里，用户看到的是浏览器页面，不是命令行。\n"
    "因此：\n"
    "- 需要出图时照常调用 render_viewer，界面会自己把三维视图做成卡片展示出来。\n"
    "- 不要提示用户去命令行执行命令；需要做什么就直接调工具。\n"
    "- 回答里不必交代文件位置，界面已经把入口放在手边了。"
)


# ------------------------------------------------------------------ 辅助


def _make_client(backend: str, model: str | None = None):
    """按后端名造 LLM 客户端。PRESETS 之外的走 EXTRA_BACKENDS 分支。"""
    if backend == "heuristic":
        from ..eval.heuristic import HeuristicClient

        return HeuristicClient()
    if backend == "scripted":
        from ..llm.client import ScriptedClient

        return ScriptedClient([ScriptedClient.say("（scripted 后端没有脚本）")])
    return OpenAICompatClient.from_preset(backend, model=model)


def _collect_artifacts(session: NavSession) -> list[dict[str, Any]]:
    """把本次会话产生的产物整理成前端能直接渲染的列表。

    `ArtifactStore.summary()` 已经把 path/case_id/candidate_id 放在 meta 里，
    这里只做一件事：给 viewer 补一个可直接访问的 URL。
    """
    items: list[dict[str, Any]] = []
    for item in session.artifacts.summary():
        record = dict(item)
        path = record.get("path")
        if record.get("kind") == "viewer" and path:
            name = Path(path).name
            record["url"] = f"/artifacts/{name}"
            record["filename"] = name
            record["exists"] = (VIEWERS_DIR / name).is_file()
        items.append(record)
    return items


def _sse(event: dict[str, Any]) -> str:
    """一条 SSE 帧。ensure_ascii=False —— 中文直接进报文，前端不用二次解码。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------ 路由


async def page_index(request) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")


async def api_health(request) -> PlainTextResponse:
    return PlainTextResponse("ok")


async def api_meta(request) -> JSONResponse:
    session = NavSession()
    registry = build_registry()
    cases = session.available_cases()
    tools = []
    for spec in registry.specs():
        schema = spec.json_schema()
        tools.append(
            {
                "name": spec.name,
                "description": spec.description,
                "required": list(schema.get("required", [])),
                "properties": sorted(schema.get("properties", {})),
            }
        )
    labels = {k: v.get("label", k) for k, v in PRESETS.items()}
    labels.update(EXTRA_BACKENDS)
    return JSONResponse(
        {
            "cases": cases,
            "tools": tools,
            "backends": list(PRESETS) + list(EXTRA_BACKENDS),
            "default_backend": DEFAULT_BACKEND,
            "backend_labels": labels,
            "viewer_dir": str(VIEWERS_DIR),
        }
    )


async def api_chat(request) -> Any:
    """SSE：把一次 Agent 运行的每一步实时推给浏览器。"""
    params = request.query_params
    question = (params.get("q") or "").strip()
    if not question:
        return JSONResponse({"error": "缺少 q 参数"}, status_code=400)

    backend = (params.get("backend") or DEFAULT_BACKEND).strip()
    if backend not in KNOWN_BACKENDS:
        return JSONResponse(
            {"error": f"未知后端 {backend}，可选：{list(KNOWN_BACKENDS)}"},
            status_code=400,
        )
    model = (params.get("model") or "").strip() or None

    def _num(name: str, fallback: float) -> float:
        raw = (params.get(name) or "").strip()
        if not raw:
            return fallback
        try:
            return float(raw)
        except ValueError:
            return fallback

    device_diameter = _num("device_diameter", 0.0)
    device_margin = _num("device_margin", 0.2)
    max_steps = int(_num("max_steps", 8))
    use_cache = (params.get("cache") or "1") not in ("0", "false", "no")
    cases_root = (params.get("cases_root") or "").strip() or None

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def push(event: dict[str, Any]) -> None:
        # 从后台线程回事件循环 —— 直接 put 到 asyncio.Queue 是不安全的
        loop.call_soon_threadsafe(queue.put_nowait, event)

    # NavAgent 是在 `run()` 收尾时发 final 的，那时服务端还没算产物列表。
    # 但**产物必须排在 final 之前**：前端要在渲染回答的那一刻就知道
    # 有哪些产物、URL 是什么，才能把回答里贴的本地路径换成可点的按钮。
    # 所以先把 final 扣住，等产物收集完再补发。
    held: list[dict[str, Any]] = []

    def on_event(event: dict[str, Any]) -> None:
        if event.get("type") == "final":
            held.append(event)
            return
        push(event)

    def worker() -> None:
        try:
            session = NavSession(
                root=cases_root,
                device_diameter_mm=device_diameter,
                device_margin_mm=device_margin,
                use_cache=use_cache,
            )
            agent = NavAgent(
                client=_make_client(backend, model),
                registry=build_registry(),
                session=session,
                max_steps=max_steps,
                environment_note=WEB_ENVIRONMENT_NOTE,
                on_event=on_event,
            )
            run = agent.run(question)

            push({"type": "artifacts", "items": _collect_artifacts(session)})
            for event in held:
                push(event)
            # 完整留痕单独发一份，供「下载本次运行记录」这类用途
            push({"type": "run", "data": run.to_dict()})
        except Exception as error:  # noqa: BLE001 - 任何异常都要变成事件，不能只留 traceback
            for event in held:
                push(event)
            push(
                {"type": "error", "message": f"{type(error).__name__}: {error}"}
            )
        finally:
            push({"type": "done"})

    threading.Thread(target=worker, name="airnav-agent", daemon=True).start()

    async def stream() -> Iterator[str]:
        while True:
            event = await queue.get()
            yield _sse(event)
            if event.get("type") == "done":
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 有反向代理时禁用缓冲，否则事件会被攒着一起发，实时性就没了
            "X-Accel-Buffering": "no",
        },
    )


async def artifact(request) -> Any:
    """提供 outputs/viewers 下的产物。

    ⚠️ 这里必须防目录穿越：只看 basename，再校验解析后的真实路径确实落在
    VIEWERS_DIR 里。只允许 .html / .json 两种后缀。
    """
    name = Path(request.path_params["name"]).name
    if not name or not name.endswith(ALLOWED_ARTIFACT_SUFFIXES):
        return JSONResponse({"error": "不支持的文件类型"}, status_code=400)

    target = (VIEWERS_DIR / name).resolve()
    if VIEWERS_DIR.resolve() not in target.parents or not target.is_file():
        return JSONResponse({"error": "文件不存在"}, status_code=404)

    media = "text/html; charset=utf-8" if name.endswith(".html") else "application/json"
    # 三维视图要内联渲染，别让浏览器当附件下载
    return FileResponse(target, media_type=media)


routes = [
    Route("/", page_index),
    Route("/health", api_health),
    Route("/api/meta", api_meta),
    Route("/api/chat", api_chat),
    Route("/artifacts/{name}", artifact),
]

app = Starlette(routes=routes)


def main() -> int:
    import uvicorn

    print("=" * 66)
    print("AirNav-Agent 网页版")
    print(f"  地址   http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    print(f"  后端   {DEFAULT_BACKEND}（页面右上角可改）")
    print(f"  viewer {VIEWERS_DIR}")
    print("  Ctrl+C 停止")
    print("=" * 66)
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
