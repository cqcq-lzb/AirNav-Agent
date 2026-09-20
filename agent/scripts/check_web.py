"""网页版自检（不需要模型、不需要网络、秒级）。

    python -m agent.scripts.check_web

## 为什么要在进程内起服务

本机**没有 httpx**，所以 starlette 的 `TestClient` 用不了。
改成在**后台线程里跑一个真的 uvicorn**，用标准库 `urllib` 打它 ——
好处是测的是真实的 ASGI 栈（含真实的事件循环、真实的 SSE 分块），
而不是一个被简化过的测试替身。

## 为什么能秒级

用 `heuristic` 后端（评测用的规则基线）：确定性、不调模型、不联网。
所以这个自检可以在每次改完网页代码后随手跑，不依赖 gpu41 通不通。

## 五层

    第 0 层 装配       路由齐全、模块可导入
    第 1 层 静态与元信息  /health、/、/api/meta
    第 2 层 SSE 事件流   顺序、字段、start/done 配对
    第 3 层 产物与安全   viewer 能取到；目录穿越/未知后端/缺参数被拒
    第 4 层 鉴权        401 白名单/携带方式/Cookie 流程/失败也写审计
    第 5 层 界面脚本     内联 JS 语法；回答里的本地路径被换成按钮

第 5 层需要 node。**找不到 node 时明确打 SKIP 并跳过，不算失败** ——
界面逻辑不该因为一台机器没装 node 就让整个自检变红。
真正干活的断言在 `agent/scripts/check_ui_paths.js` 里（那边能直接跑 DOM 处理函数）。

## 第 4 层为什么不另起一个带 token 的服务

`auth.token()` / `enabled()` 是**每次请求现读环境变量**的，所以在本进程里
`os.environ["AIRNAV_API_TOKEN"] = ...` 之后，同一个正在跑的服务立刻开始要求凭据 ——
不用重启、不用第二个端口。测的仍然是真实的中间件与真实的路由。

**默认不设 token 时，前四层的行为和以前一字不差**（本地开发零摩擦），
这是刻意的：鉴权不该给日常开发添堵。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = Path(__file__).resolve().parents[2]

HOST = "127.0.0.1"
PORT = 8791
BASE = f"http://{HOST}:{PORT}"
READY_TIMEOUT_S = 30

# 本机常配 http_proxy / https_proxy，urllib 默认会把 127.0.0.1 也丢给代理，
# 于是得到 502（和 requests 连本地 Ollama 时踩的是同一个坑）。
# 显式清空代理，让请求真的打到自己的服务上。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """不跟随重定向。

    `POST /login` 成功回的是 **303 + Set-Cookie**。默认的 opener 会一路跟到首页，
    而重定向链里**中间那个响应的头会被丢掉** —— 于是 `Set-Cookie` 根本拿不到，
    登录 Cookie 流程就测不了（2026-09-20 自检踩过）。
    不跟随之后 3xx 会以 HTTPError 的形式抛上来，头就完整了。
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_OPENER_NO_REDIRECT = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _NoRedirect
)

# `<script ...>` 开标签属性 + 正文，用于抽内联脚本（见 _extract_inline_script）
_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.S | re.I)

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((ok, label, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {label}" + (f"（{detail}）" if detail else ""))
    return ok


def request(path: str, timeout: int = 60) -> tuple[int, str, str]:
    """返回 (状态码, 正文, content-type)。HTTP 错误码不抛异常，按值返回。"""
    url = BASE + path
    try:
        with _OPENER.open(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            return response.status, body, response.headers.get("content-type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace"), ""
    except Exception as error:  # noqa: BLE001 - 连接失败也是一种结果，交给断言去判
        return 0, f"{type(error).__name__}: {error}", ""


def request_full(
    path: str,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    follow: bool = True,
) -> tuple[int, str, dict[str, str], str | None]:
    """同上，但能带请求头/请求体，并把**响应头与最终 URL** 也带回来。

    鉴权用例必须看到 `Set-Cookie`（判断 HttpOnly/SameSite）与重定向后的落地地址，
    这两样 `request()` 都不给，所以单开一个。

    `follow=False` 时不跟随重定向 —— 3xx 会以 HTTPError 形式回来，
    这样才拿得到 303 那个响应自身的 `Set-Cookie`。
    """
    req = urllib.request.Request(BASE + path, data=data, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    opener = _OPENER if follow else _OPENER_NO_REDIRECT
    try:
        with opener.open(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            return response.status, body, dict(response.headers), response.geturl()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        return error.code, body, dict(error.headers or {}), None
    except Exception as error:  # noqa: BLE001
        return 0, f"{type(error).__name__}: {error}", {}, None


def read_sse(path: str, timeout: int = 180, headers: dict[str, str] | None = None) -> list[dict]:
    """读一条 SSE 流，直到收到 done 事件。

    用 urlopen 直接按行读 —— 只有这样才验证了「服务端是边算边推」，
    而不是等全部算完一次性返回。
    """
    events: list[dict] = []
    req = urllib.request.Request(BASE + path)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with _OPENER.open(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[5:].strip())
            events.append(payload)
            if payload.get("type") == "done":
                break
    return events


# ------------------------------------------------------------------ 各层


def layer_0_assembly() -> int:
    print("\n[0] 装配")
    failures = 0

    try:
        from agent.web.server import ALLOWED_ARTIFACT_SUFFIXES, app, routes
    except Exception as error:  # noqa: BLE001
        check(False, "agent.web.server 可导入", f"{type(error).__name__}: {error}")
        return 1
    check(True, "agent.web.server 可导入")

    paths = sorted({getattr(r, "path", "") for r in routes})
    expected = ["/", "/api/chat", "/api/meta", "/artifacts/{name}", "/health", "/login"]
    if not check(paths == expected, "路由集合正确", f"实际 {paths}"):
        failures += 1

    # 中间件必须挂上：能 import 到不代表挂上了（漏挂的表现是「鉴权静默失效」，
    # 而所有功能用例照样全绿 —— 这是最危险的一种失败）
    try:
        from starlette.middleware import Middleware  # noqa: F401

        from agent.web.auth import TokenAuthMiddleware
        from agent.web.server import app as _app

        stack = [m.cls for m in getattr(_app, "user_middleware", [])]
        if not check(TokenAuthMiddleware in stack, "鉴权中间件已挂到应用上",
                     f"{[c.__name__ for c in stack]}"):
            failures += 1
    except Exception as error:  # noqa: BLE001
        check(False, "鉴权中间件已挂到应用上", f"{type(error).__name__}: {error}")
        failures += 1

    ui = Path(__file__).resolve().parents[1] / "web" / "static" / "index.html"
    if not check(ui.is_file() and ui.stat().st_size > 4000, "界面文件存在且非空",
                 f"{ui.name} {ui.stat().st_size if ui.is_file() else 0} 字节"):
        failures += 1

    if not check(ALLOWED_ARTIFACT_SUFFIXES == (".html", ".json"),
                 "产物后缀白名单只有 html/json"):
        failures += 1
    return failures


def layer_1_static_and_meta() -> int:
    print("\n[1] 静态与元信息")
    failures = 0

    status, body, _ = request("/health", timeout=15)
    if not check(status == 200 and body.strip() == "ok", "/health", f"{status} {body[:40]}"):
        failures += 1

    status, body, ctype = request("/", timeout=15)
    if not check(
        status == 200 and "text/html" in ctype and "AirNav-Agent" in body and "/api/chat" in body,
        "首页返回界面 HTML",
        f"{status} {len(body)} 字节",
    ):
        failures += 1

    status, body, _ = request("/api/meta", timeout=30)
    try:
        meta = json.loads(body)
    except Exception:  # noqa: BLE001
        check(False, "/api/meta 返回 JSON", body[:80])
        return failures + 1

    if not check(len(meta.get("tools", [])) == 11, "元信息含 11 个工具",
                 f"实际 {len(meta.get('tools', []))}"):
        failures += 1
    if not check(len(meta.get("cases", [])) > 0, "元信息含病例",
                 f"{len(meta.get('cases', []))} 个"):
        failures += 1
    backends = meta.get("backends", [])
    if not check("heuristic" in backends and "gpu41" in backends,
                 "后端列表含 heuristic 与 gpu41", f"{backends}"):
        failures += 1
    if not check(meta.get("default_backend") in backends, "默认后端在列表内",
                 str(meta.get("default_backend"))):
        failures += 1
    labels = meta.get("backend_labels", {})
    if not check(all(b in labels for b in backends), "每个后端都有标签",
                 f"缺 {[b for b in backends if b not in labels]}"):
        failures += 1
    return failures


def layer_2_sse() -> int:
    print("\n[2] SSE 事件流（heuristic 后端）")
    failures = 0

    question = "给 LIDC_0089 的 3 号候选出一张三维视图"
    quoted = urllib.parse.quote(question)
    started = time.time()
    try:
        events = read_sse(f"/api/chat?q={quoted}&backend=heuristic&device_diameter=2.0")
    except Exception as error:  # noqa: BLE001
        check(False, "SSE 流可读", f"{type(error).__name__}: {error}")
        return 1
    elapsed = time.time() - started

    kinds = [e.get("type") for e in events]
    check(True, "SSE 流可读", f"{len(events)} 个事件 / {elapsed:.1f}s")

    if not check(kinds and kinds[0] == "start", "首个事件是 start", f"{kinds[:3]}"):
        failures += 1
    if not check(kinds and kinds[-1] == "done", "最后一个事件是 done", f"{kinds[-3:]}"):
        failures += 1

    starts = [e for e in events if e.get("type") == "tool_start"]
    dones = [e for e in events if e.get("type") == "tool_done"]
    if not check(len(starts) == len(dones) and len(starts) > 0,
                 "tool_start 与 tool_done 一一配对",
                 f"{len(starts)}/{len(dones)}"):
        failures += 1
    if not check(
        [e["name"] for e in starts] == [e["name"] for e in dones],
        "两个序列的工具名顺序一致",
        f"{[e['name'] for e in starts]}",
    ):
        failures += 1
    if not check(
        all("elapsed_ms" in e and "ok" in e for e in dones),
        "tool_done 带 ok 与 elapsed_ms",
    ):
        failures += 1

    steps = {e["step"] for e in starts}
    if not check(len(steps) >= 1, "事件带 step 编号", f"出现过的 step {sorted(steps)}"):
        failures += 1

    finals = [e for e in events if e.get("type") == "final"]
    if not check(len(finals) == 1, "final 事件恰好一个", f"{len(finals)} 个"):
        failures += 1
    elif not check(
        all(k in finals[0] for k in ("answer", "stop_reason", "steps", "tool_calls", "elapsed_s")),
        "final 含统计字段",
        f"stop={finals[0].get('stop_reason')} steps={finals[0].get('steps')}",
    ):
        failures += 1

    # 事件必须严格早于 done，否则前端渲染顺序就错了
    if not check(
        kinds.index("final") < kinds.index("done"),
        "final 排在 done 之前",
        f"final@{kinds.index('final')} done@{kinds.index('done')}",
    ):
        failures += 1
    return failures


def layer_3_artifacts_and_safety() -> int:
    print("\n[3] 产物与安全")
    failures = 0

    question = "给 LIDC_0089 的 3 号候选出一张三维视图"
    quoted = urllib.parse.quote(question)
    events = read_sse(f"/api/chat?q={quoted}&backend=heuristic&device_diameter=2.0")
    artifacts = [e for e in events if e.get("type") == "artifacts"]

    if not check(len(artifacts) == 1, "有 artifacts 事件", f"{len(artifacts)} 个"):
        return failures + 1
    viewers = [i for i in artifacts[0]["items"] if i.get("kind") == "viewer"]
    if not check(len(viewers) == 1, "含 1 个 viewer 产物", f"{len(viewers)} 个"):
        return failures + 1
    viewer = viewers[0]
    if not check(bool(viewer.get("url")) and bool(viewer.get("exists")),
                 "viewer 有可访问 url 且文件确实落盘",
                 f"{viewer.get('url')} exists={viewer.get('exists')}"):
        failures += 1

    status, body, ctype = request(viewer["url"], timeout=30)
    if not check(status == 200 and "text/html" in ctype and "three" in body.lower(),
                 "产物 URL 能取到三维 HTML",
                 f"{status} {len(body)} 字节"):
        failures += 1

    # ---- 安全与参数校验 ----
    status, _, _ = request("/artifacts/..%2Fserver_config.json", timeout=15)
    if not check(status in (400, 404), "目录穿越被拒", f"状态 {status}"):
        failures += 1

    status, _, _ = request("/artifacts/..%5Cserver_config.json", timeout=15)
    if not check(status in (400, 404), "反斜杠穿越被拒", f"状态 {status}"):
        failures += 1

    status, _, _ = request("/artifacts/nope.html", timeout=15)
    if not check(status == 404, "不存在的产物返回 404", f"状态 {status}"):
        failures += 1

    status, _, _ = request("/artifacts/requirements_mcp.txt", timeout=15)
    if not check(status == 400, "非白名单后缀被拒", f"状态 {status}"):
        failures += 1

    status, body, _ = request("/api/chat?q=hello&backend=does-not-exist", timeout=15)
    if not check(status == 400 and "未知后端" in body, "未知后端返回 400", f"状态 {status}"):
        failures += 1

    status, _, _ = request("/api/chat", timeout=15)
    if not check(status == 400, "缺 q 参数返回 400", f"状态 {status}"):
        failures += 1
    return failures


# ------------------------------------------------------------------ 第 4 层：鉴权


AUTH_TOKEN = "selfcheck-token-2f8a1c9d"
AUTH_COOKIE = "airnav_token"


def layer_4_auth() -> int:
    """临时打开 token，验证「白名单 / 携带方式 / Cookie 流程 / 失败也留痕」。

    ⚠️ 顺序有意为之：**白名单先测**。把 `/health` 一起挡掉是鉴权最常见的自伤 ——
    监控和容器编排全部打不进来，表现为「服务活着但总是被重启」。
    """
    print("\n[4] 鉴权（临时打开 AIRNAV_API_TOKEN）")
    failures = 0

    from agent import audit

    os.environ["AIRNAV_API_TOKEN"] = AUTH_TOKEN
    os.environ["AIRNAV_WEB_ACTOR"] = "selfcheck"
    try:
        # ---- 白名单：没凭据也该放行 ----
        status, body, _ = request("/health", timeout=15)
        if not check(status == 200 and body.strip() == "ok",
                     "未带凭据 /health → 200（探活不受影响）", f"{status}"):
            failures += 1

        status, body, ctype = request("/", timeout=15)
        if not check(status == 200 and "text/html" in ctype,
                     "未带凭据 / → 200（空壳页面公开，不含病例数据）", f"{status}"):
            failures += 1

        status, body, _ = request("/login", timeout=15)
        if not check(status == 200 and 'name="token"' in body and "password" in body,
                     "/login GET 给出登录表单", f"{status} {len(body)} 字节"):
            failures += 1

        # ---- 受保护面：一律 401，且不泄漏任何病例信息 ----
        status, body, headers, _ = request_full("/api/meta", timeout=15)
        if not check(status == 401, "未带凭据 /api/meta → 401", f"{status}"):
            failures += 1
        if not check("LIDC" not in body and "cases" not in body,
                     "401 响应体不含任何病例信息", f"{body[:50]!r}"):
            failures += 1
        if not check("www-authenticate" in {k.lower() for k in headers},
                     "401 带 www-authenticate（脚本能区分「没登录」和别的错）"):
            failures += 1

        status, _, _ = request("/api/chat?q=hello&backend=heuristic", timeout=15)
        if not check(status == 401, "未带凭据 /api/chat → 401（SSE 同样受保护）",
                     f"{status}"):
            failures += 1

        status, _, _ = request("/artifacts/viewer_LIDC_0089_c3_d1.5.html", timeout=15)
        if not check(status == 401,
                     "未带凭据 /artifacts/* → 401（三维视图是敏感数据）", f"{status}"):
            failures += 1

        # ---- 携带方式一：请求头（脚本 / 探针 / 监控）----
        status, body, _, _ = request_full("/api/meta", headers={"X-API-Key": AUTH_TOKEN},
                                          timeout=30)
        if not check(status == 200 and '"cases"' in body,
                     "带 X-API-Key → 200", f"{status}"):
            failures += 1

        status, _, _, _ = request_full("/api/meta",
                                       headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                                       timeout=30)
        if not check(status == 200, "带 Authorization: Bearer → 200", f"{status}"):
            failures += 1

        status, _, _, _ = request_full("/api/meta", headers={"X-API-Key": "wrong-token"},
                                       timeout=15)
        if not check(status == 401, "错误凭据 → 401", f"{status}"):
            failures += 1

        # ---- 携带方式二：Cookie（浏览器；EventSource 不能自定义请求头）----
        form = urllib.parse.urlencode({"token": AUTH_TOKEN}).encode()
        form_headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # 不跟随重定向：要看的是 303 那个响应本身（Set-Cookie 在这里）
        status, _, headers, _ = request_full(
            "/login", method="POST", data=form, headers=form_headers,
            timeout=15, follow=False,
        )
        location = headers.get("Location") or headers.get("location") or ""
        if not check(status == 303 and location == "/",
                     "POST /login 正确令牌 → 303 回首页（PRG）",
                     f"{status} Location={location}"):
            failures += 1

        cookie = headers.get("Set-Cookie") or headers.get("set-cookie") or ""
        if not check(AUTH_COOKIE in cookie, "种下 airnav_token Cookie", f"{cookie[:60]}"):
            failures += 1
        if not check("HttpOnly" in cookie, "Cookie 是 HttpOnly（脚本读不到，防 XSS 窃取）"):
            failures += 1
        if not check("SameSite=Lax" in cookie, "Cookie 带 SameSite=Lax（挡跨站请求）"):
            failures += 1

        # PRG 的另一半：客户端跟随之后必须落在首页，而不是又渲染一次登录页
        status, _, _, final_url = request_full(
            "/login", method="POST", data=form, headers=form_headers, timeout=15
        )
        if not check(status == 200 and final_url == BASE + "/",
                     "跟随 303 之后落在首页", f"{status} → {final_url}"):
            failures += 1

        jar = cookie.split(";")[0]
        status, body, _, _ = request_full("/api/meta", headers={"Cookie": jar}, timeout=30)
        if not check(status == 200 and '"cases"' in body, "带 Cookie → 200", f"{status}"):
            failures += 1

        # 畸形 Cookie 不该 500（中间件里那次 try 就是为它写的）
        status, _, _, _ = request_full("/api/meta", headers={"Cookie": "airnav_token=%%%"},
                                       timeout=15)
        if not check(status in (200, 401), "畸形 Cookie 不会打出 500", f"{status}"):
            failures += 1

        status, _, _, _ = request_full(
            "/login",
            method="POST",
            data=urllib.parse.urlencode({"token": "nope"}).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if not check(status == 401, "POST /login 错误令牌 → 401", f"{status}"):
            failures += 1

        # ---- SSE 在鉴权下要能正常流 ----
        try:
            events = read_sse(
                "/api/chat?q=" + urllib.parse.quote("给 LIDC_0089 的 3 号候选出一张三维视图")
                + "&backend=heuristic&device_diameter=2.0",
                timeout=180,
                headers={"X-API-Key": AUTH_TOKEN},
            )
            kinds = [e.get("type") for e in events]
            if not check(kinds[:1] == ["start"] and kinds[-1:] == ["done"],
                         "带凭据的 SSE 完整跑完（start … done）", f"{kinds[:2]}…{kinds[-1:]}"):
                failures += 1
        except Exception as error:  # noqa: BLE001
            check(False, "带凭据的 SSE 完整跑完（start … done）",
                  f"{type(error).__name__}: {error}")
            failures += 1

        # ---- 审计：被拒与放行都要留痕 ----
        records = [r for r in audit.iter_records() if not r.get("_broken")]
        denied = [r for r in records if r.get("event") == "auth" and r.get("allowed") is False]
        allowed = [r for r in records if r.get("event") == "auth" and r.get("allowed") is True]
        if not check(any(r.get("path") == "/api/meta" for r in denied),
                     "未授权访问被写进审计（谁在什么时候试图进来）",
                     f"被拒 {len(denied)} 条"):
            failures += 1
        if not check(any(r.get("path") == "/login" for r in allowed),
                     "成功登录被写进审计", f"放行 {len(allowed)} 条"):
            failures += 1
        if not check(all(r.get("client_ip") for r in denied),
                     "审计里记下了来源 IP", f"{[r.get('client_ip') for r in denied[:3]]}"):
            failures += 1

        runs = [r for r in records if r.get("event") == "run"]
        if not check(len(runs) >= 1, "Agent 运行也写了审计记录", f"{len(runs)} 条 run"):
            failures += 1
        elif not check(
            all(k in runs[-1] for k in ("run_id", "model", "params", "case_ids",
                                        "answer_sha256", "verdict")),
            "run 记录含可追溯字段（run_id / 模型 / 参数 / 病例 / 答案哈希）",
            f"run_id={str(runs[-1].get('run_id'))[:8]}… verdict={runs[-1].get('verdict')}",
        ):
            failures += 1

        ok, problems = audit.verify_chain()
        if not check(ok, "自检期间写入的审计链完整", " | ".join(problems[:2])):
            failures += 1
    finally:
        os.environ.pop("AIRNAV_API_TOKEN", None)
        os.environ.pop("AIRNAV_WEB_ACTOR", None)

    # 关掉凭据后必须回到「零摩擦」：没设 token 时行为与以前一字不差
    status, _, _ = request("/api/meta", timeout=30)
    if not check(status == 200, "移除 token 后恢复免鉴权（本地开发零摩擦）", f"{status}"):
        failures += 1
    return failures


def _check_element_ids(html: str, script: str) -> tuple[bool, str]:
    """内联 JS 引用的元素 id，必须在 HTML 里真的存在。

    ## 为什么值得单开一条

    `$("lgerr")` 写成 `$("lgErr")` 时，**现有所有测试都会全绿**：
    JS 语法没问题、路径替换没问题、HTTP 层更看不出 —— 直到用户在浏览器里
    点开登录浮层，代码在那行抛 `Cannot read properties of null`。

    这类「拼错的 id」是纯静态可判定的，所以不该留给运行时。之前几轮改动里
    新增了整块登录浮层的 DOM（`lg` / `lgform` / `lgtoken` / `lgerr`），
    正是最容易错的地方。

    只做**子集校验**（引用的 ⊆ 定义的），不反过来要求每个 id 都被用到 ——
    给 CSS 留锚点是正常的。
    """
    defined = set(re.findall(r"""\bid\s*=\s*["']([^"']+)["']""", html))
    referenced = set(re.findall(r"""\$\(\s*["']([^"']+)["']\s*\)""", script))
    referenced |= set(re.findall(r"""getElementById\(\s*["']([^"']+)["']\s*\)""", script))
    missing = sorted(referenced - defined)
    if missing:
        return False, f"引用了不存在的 id：{missing}（HTML 里定义的有 {len(defined)} 个）"
    return True, f"引用 {len(referenced)} 个 id，全部存在"


# ------------------------------------------------------------------ 服务生命周期


def find_node() -> str | None:
    """找一个能用的 node。找不到就返回 None（第 4 层会明确 SKIP）。"""
    env = os.environ.get("AIRNAV_NODE")
    if env and Path(env).is_file():
        return env
    return shutil.which("node") or shutil.which("node.exe")


def layer_5_ui_script(node: str) -> int:
    """界面脚本层：内联 JS 语法 + 回答里本地路径的替换逻辑。

    语法检查用 `node --check`，把 index.html 里的内联 `<script>` 抽到临时文件
    （HTML 里的 JS 不能直接喂给 --check，标签本身会让它语法报错）。

    断言检查直接委托给 `check_ui_paths.js` —— 那边从 index.html 原样抽出函数来跑，
    测的是真源码而不是副本。这里只负责「跑起来没有、输出里有几个 PASS」。
    """
    print("\n[5] 界面脚本")
    failures = 0

    index = REPO / "agent" / "web" / "static" / "index.html"
    html = index.read_text(encoding="utf-8")
    script = _extract_inline_script(html)
    if not check(script is not None, "能在 index.html 里定位内联 <script> 块"):
        return 1

    # 元素 id 交叉校验：拼错 id 是现有测试全都看不见的失效模式（见函数注释）
    ok_ids, detail = _check_element_ids(html, script)
    if not check(ok_ids, "内联 JS 引用的元素 id 在 HTML 里都存在", detail):
        failures += 1

    with tempfile.TemporaryDirectory() as tmp:
        js_file = Path(tmp) / "inline.js"
        js_file.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [node, "--check", str(js_file)], capture_output=True, text=True, timeout=60
        )
        failures += not check(
            proc.returncode == 0,
            "内联 JS 语法通过 node --check",
            (proc.stderr.strip().splitlines() or [""])[0][:120] if proc.returncode else "",
        )

    ui_check = REPO / "agent" / "scripts" / "check_ui_paths.js"
    proc = subprocess.run(
        [node, str(ui_check)], capture_output=True, text=True, timeout=120, cwd=str(REPO)
    )
    out = proc.stdout + proc.stderr
    passed = out.count("  PASS  ")
    failed = out.count("  FAIL  ")
    failures += not check(
        proc.returncode == 0 and failed == 0 and passed > 0,
        "回答里的本地路径被换成可点按钮",
        f"{passed} 过 / {failed} 败",
    )
    if proc.returncode != 0:
        for line in out.splitlines():
            if "FAIL" in line:
                print("       " + line.strip())
    return failures


def _extract_inline_script(html: str) -> str | None:
    """取内联 `<script>` 的正文（本项目只有一个；多个时取最长的那个）。

    ⚠️ 别用「往前看 N 个字符有没有 `src=`」这种土办法判断是不是外链 ——
    界面里 `<iframe src="about:blank">` 离 `<script>` 很近，
    会被误判成外链脚本，于是整层静默跳过。老老实实解析开标签的属性。
    """
    inline = [
        body
        for attrs, body in _SCRIPT_RE.findall(html)
        if not re.search(r"\bsrc\s*=", attrs, re.I)
    ]
    return max(inline, key=len) if inline else None


def wait_ready() -> bool:
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        status, _, _ = request("/health", timeout=3)
        if status == 200:
            return True
        time.sleep(0.3)
    return False


def main() -> int:
    import uvicorn

    from agent.web.server import app

    # 审计落临时目录：自检产生的记录是证据，不是交付物，不该混进 outputs/audit/。
    # 门禁会自己设 AIRNAV_AUDIT_DIR（指到它的 scratch），那时以门禁的为准。
    owned_scratch = None
    if not os.environ.get("AIRNAV_AUDIT_DIR"):
        owned_scratch = tempfile.TemporaryDirectory()
        os.environ["AIRNAV_AUDIT_DIR"] = owned_scratch.name

    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="check-web", daemon=True)
    thread.start()

    print("=" * 68)
    print("AirNav-Agent 网页版自检（进程内起服务，heuristic 后端，不需要模型）")
    print("=" * 68)

    if not wait_ready():
        print(f"\n服务未能在 {READY_TIMEOUT_S}s 内就绪，后续检查跳过。")
        server.should_exit = True
        return 1

    failures = 0
    try:
        failures += layer_0_assembly()
        failures += layer_1_static_and_meta()
        failures += layer_2_sse()
        failures += layer_3_artifacts_and_safety()
        failures += layer_4_auth()
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    # 第 5 层放在服务停掉之后：它只读文件、跑 node，跟这边开着的服务无关。
    node = find_node()
    if node is None:
        print("\n[5] 界面脚本")
        print("  SKIP  没找到 node（设 AIRNAV_NODE 或把 node 放进 PATH 可启用本层）")
    else:
        failures += layer_5_ui_script(node)

    if owned_scratch is not None:
        owned_scratch.cleanup()

    total = len(RESULTS)
    passed = sum(1 for ok, _, _ in RESULTS if ok)
    print("\n" + "=" * 68)
    if failures:
        print(f"未通过：{passed}/{total} 项通过，{failures} 项失败")
        for ok, label, detail in RESULTS:
            if not ok:
                print(f"  FAIL  {label}（{detail}）")
        return 1
    print(f"自检通过：{passed}/{total} 项全部符合预期")
    print(
        "覆盖：路由装配 · 静态与元信息 · SSE 事件顺序与配对 · 产物可访问 · "
        "穿越与参数校验 · 鉴权与审计 · 界面脚本"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
