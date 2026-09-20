"""网页版鉴权：单凭据 + 探活白名单 + 401 也写审计。

## 为什么用纯 ASGI 中间件，而不是 `BaseHTTPMiddleware`

网页版的核心体验是 **SSE 实时流**（每一步工具调用都要推出来）。`BaseHTTPMiddleware`
会把响应再包一层、经由内存队列转发，对流式响应是多余的开销与缓冲风险。
所以这里写的是**纯 ASGI 中间件**：未命中拒绝分支时直接 `await self.app(scope, receive, send)`
原样透传，对流的介入为零。

## 凭据模型（P0 只做单凭据）

```bash
AIRNAV_API_TOKEN=<32 字节随机 hex>   # 空 = 关闭鉴权（本地开发默认不变）
AIRNAV_WEB_ACTOR=qc                  # 可选，写进审计的操作者标识
AIRNAV_WEB_COOKIE_SECURE=1           # 挂在 HTTPS 反代后面时打开
```

向后兼容：新名优先，回落后端的旧名 `API_TOKEN` —— 两个面共用一套凭据，
这也是「统一凭据治理」的第一步。仓库内绝不落盘明文（沿用项目铁律）。

携带方式两条：`X-API-Key` 头（脚本 / 探针 / 监控）与 `airnav_token` Cookie（浏览器）。
比较一律走 `hmac.compare_digest`，失败返回 401 —— **并且写一条审计**：
「谁在什么时候试图进来」本身就是审计要回答的问题。

## 白名单为什么必须有 `/health`

探活不该要凭据，否则监控和容器编排都打不进来 —— 为了安全把可用性搞死。
`/` 也放行：那是纯壳页面（不含任何病例数据），真正的数据入口
（`/api/*` 与 `/artifacts/*`）全部鉴权。三维视图含真实解剖，**必须**在鉴权范围内。
"""
from __future__ import annotations

import hmac
import json
import os
import urllib.parse
from http.cookies import SimpleCookie
from pathlib import Path

from .. import audit

COOKIE_NAME = "airnav_token"
PUBLIC_PATHS = {"/", "/health", "/login", "/favicon.ico"}

LOGIN_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>AirNav-Agent 登录</title>
  <style>
    body {{ font-family: "PingFang SC", "微软雅黑", sans-serif; background: #F7F7F5;
           display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
    form {{ background: #fff; border: 1px solid #E5E7EB; border-radius: 12px; padding: 28px 32px;
            width: 320px; }}
    h1 {{ font-size: 16px; margin: 0 0 4px; color: #111827; }}
    p  {{ font-size: 12px; color: #6B7280; margin: 0 0 18px; }}
    input {{ width: 100%; box-sizing: border-box; padding: 9px 11px; font-size: 13px;
             border: 1px solid #E5E7EB; border-radius: 8px; }}
    button {{ margin-top: 14px; width: 100%; padding: 9px; font-size: 13px; cursor: pointer;
              border: 0; border-radius: 8px; background: #2563EB; color: #fff; }}
    .err {{ color: #A32D2D; font-size: 12px; margin-top: 10px; }}
  </style>
</head>
<body>
  <form method="post" action="/login">
    <h1>AirNav-Agent</h1>
    <p>需要访问令牌才能使用（问运维要）</p>
    <input type="password" name="token" placeholder="访问令牌" autofocus autocomplete="off">
    <button type="submit">进入</button>
    {error}
  </form>
</body>
</html>
"""


# ------------------------------------------------------------------ 配置


def token() -> str:
    """生效的凭据。新的 `AIRNAV_API_TOKEN` 优先，回落后端旧名 `API_TOKEN`。"""
    return (os.environ.get("AIRNAV_API_TOKEN") or os.environ.get("API_TOKEN") or "").strip()


def enabled() -> bool:
    return bool(token())


def actor() -> str:
    """P0 是单凭据，操作者由环境变量声明；P1 上用户表后从会话里取。"""
    return (os.environ.get("AIRNAV_WEB_ACTOR") or "anonymous").strip() or "anonymous"


def public_paths() -> set[str]:
    extra = os.environ.get("AIRNAV_WEB_PUBLIC_PATHS") or ""
    return PUBLIC_PATHS | {item.strip() for item in extra.split(",") if item.strip()}


# ------------------------------------------------------------------ 取值


def client_ip(scope) -> str | None:
    """优先 `X-Forwarded-For` 首段（反代后面才拿得到真实来源）。

    ⚠️ 这个头可以被伪造，只在前置反代可信时才有意义 —— 信任边界写在文档里。
    """
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = scope.get("client")
    return client[0] if client else None


def user_agent(scope) -> str | None:
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
    return headers.get("user-agent")


def _headers(scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


def presented(scope) -> str | None:
    """从 Header 或 Cookie 里取客户端带来的凭据。"""
    headers = _headers(scope)
    given = (headers.get("x-api-key") or "").strip()
    if not given:
        authz = (headers.get("authorization") or "").strip()
        if authz.lower().startswith("bearer "):
            given = authz[7:].strip()
    if not given:
        raw = headers.get("cookie")
        if raw:
            jar = SimpleCookie()
            try:
                jar.load(raw)
            except Exception:  # noqa: BLE001 —— 畸形 Cookie 不该 500
                jar = SimpleCookie()
            morsel = jar.get(COOKIE_NAME)
            given = (morsel.value if morsel else "") or ""
    return given or None


def matches(given: str | None) -> bool:
    expected = token()
    if not expected:
        return True
    if not given:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def should_log_allowed(path: str) -> bool:
    """成功访问只记敏感面。

    全记会把日志淹掉（`/api/meta` 每次开页面都打）—— 审计要的是信号。
    但**三维产物是敏感数据**（真实解剖），谁在什么时候看了哪一份必须留痕，
    而「哪一份」由 run 记录里的 sha256 精确对上。
    """
    return path.startswith("/artifacts/")


# ------------------------------------------------------------------ 中间件


class TokenAuthMiddleware:
    """拒绝则短路；放行则零介入透传。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not enabled():
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in public_paths():
            await self.app(scope, receive, send)
            return

        given = presented(scope)
        ok = matches(given)
        if ok:
            if should_log_allowed(path):
                audit.record_auth(
                    allowed=True,
                    path=path,
                    client_ip=client_ip(scope),
                    user_agent=user_agent(scope),
                    actor=actor(),
                )
            await self.app(scope, receive, send)
            return

        audit.record_auth(
            allowed=False,
            path=path,
            client_ip=client_ip(scope),
            user_agent=user_agent(scope),
            actor="(rejected)",
        )
        payload = json.dumps(
            {"error": "需要访问令牌：请携带 X-API-Key 头，或先在 /login 登录"},
            ensure_ascii=False,
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(payload)).encode()),
                    # 便于脚本与前端区分「没登录」与「没权限/其他错误」
                    (b"www-authenticate", b'Bearer realm="airnav"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


# ------------------------------------------------------------------ 登录路由


def _cookie_flags() -> str:
    flags = f"Path=/; HttpOnly; SameSite=Lax; Max-Age={12 * 3600}"
    if os.environ.get("AIRNAV_WEB_COOKIE_SECURE") == "1":
        flags += "; Secure"
    return flags


def _html(body: str, status: int, cookie: str | None = None) -> tuple[int, dict[str, str], str]:
    """构造登录页响应。

    ⚠️ 响应头必须是 **str 键**。这里原先返回的是字节键的 list，外层再 `dict()` 一下
    看起来「也能用」—— 实际 Starlette 的 `MutableHeaders` 只吃 str，
    于是登录页直接 500。自检抓到的第一个真 bug 就是这个（2026-09-20）。
    """
    headers = {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
    }
    if cookie:
        headers["set-cookie"] = cookie
    return status, headers, body


async def login_page(request):
    """GET：登录表单。POST：校验并种 Cookie。

    换成 Cookie 是为了让浏览器**原生**带上凭据 —— `EventSource` 不能自定义请求头，
    用 Cookie 才能在 SSE 场景下免改协议地鉴权。
    """
    from starlette.responses import Response

    if request.method == "GET":
        status, headers, body = _html(LOGIN_PAGE.format(error=""), 200)
        return Response(body, status_code=status, headers=headers, media_type=None)

    try:
        form = await request.form()
    except Exception:  # noqa: BLE001
        form = {}
    given = str(form.get("token") or "").strip()
    path = request.url.path

    if matches(given):
        audit.record_auth(
            allowed=True,
            path="/login",
            client_ip=client_ip(request.scope),
            user_agent=user_agent(request.scope),
            actor=actor(),
        )
        # 303 让浏览器改用 GET 回首页（PRG 模式，避免刷新重复提交）
        return Response(
            b"",
            status_code=303,
            headers={
                "location": "/",
                "set-cookie": f"{COOKIE_NAME}={given}; {_cookie_flags()}",
            },
        )

    audit.record_auth(
        allowed=False,
        path="/login",
        client_ip=client_ip(request.scope),
        user_agent=user_agent(request.scope),
        actor="(rejected)",
    )
    status, headers, body = _html(
        LOGIN_PAGE.format(error='<p class="err">令牌不对</p>'), 401
    )
    return Response(body, status_code=status, headers=headers)


def status_line() -> str:
    """给启动横幅用的一句话。"""
    if not enabled():
        return "关（未设置 AIRNAV_API_TOKEN / API_TOKEN）—— 只适合本机自用"
    return f"开（actor={actor()}，探活 /health 免鉴权）"
