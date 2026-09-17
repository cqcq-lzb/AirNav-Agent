"""MCP server 自检。

分四层验证，一层比一层更接近真实使用：

0. **依赖归属层**：确认 mcp 及其依赖真的装在**当前解释器自己的** site-packages
   里，而不是靠 Python 3.10 的 user-site 兜底。这一层必须放最前面：如果依赖
   是 user-site 借来的，后面三层在开发机上会全部通过，但换到
   `PYTHONNOUSERSITE=1`、`python -s` 或别的解释器就立刻 import 失败 ——
   典型的「平时能跑、一部署就炸」。本机实测踩过一次，详见
   agent/scripts/lock_mcp_deps.py 的文件头说明。
1. **静态层**：直接 `list_tools()`，检查 11 个工具都在、描述非空、
   参数说明与取值约束有没有被正确透传（这是最容易踩的坑：只写裸类型注解
   会让 schema 丢掉全部参数说明）。
2. **握手层**：真的把 server 作为子进程拉起来，走一遍 stdio 传输的
   initialize 握手，确认能作为 MCP 服务被客户端连接。
3. **调用层**：通过标准 MCP 协议真实调用几个工具，确认参数校验、
   会话共享与知识检索在协议层都正常。

用法：
    python -m agent.scripts.check_mcp
    python -m agent.scripts.check_mcp --full     # 额外跑一次真实规划
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.mcp_server import SERVER_NAME, build_server  # noqa: E402

EXPECTED_TOOLS = 11

# 依赖归属层：以锁定的完整闭包为准（由 lock_mcp_deps.py 生成）。
LOCK_FILE = ROOT / "requirements_mcp.txt"

# 锁定文件缺失时的兜底清单，覆盖会让 import 直接失败的关键包。
FALLBACK_REQUIRED = (
    "mcp",
    "mcp-types",
    "pydantic",
    "pydantic-core",
    "starlette",
    "uvicorn",
    "sse-starlette",
    "anyio",
    "jsonschema",
)

# 必须带参数说明的工具（抽查，覆盖三类参数形态）
DESCRIPTION_SPOT_CHECKS = {
    "plan_route": ("case_id", "candidate_id", "profile", "device_diameter_mm"),
    "render_viewer": ("mesh_step",),
    "search_knowledge": ("query",),
    "explain_route_choice": ("include_knowledge",),
}


def _fail(message: str) -> None:
    print(f"  FAIL  {message}")


def _pass(message: str) -> None:
    print(f"  PASS  {message}")


# 合法发行版名：只允许 ASCII 字母数字与 . _ -，首字符必须字母数字。
# PEP 503 之后官方就是这么归一化的，任何越界字符都说明这行不是包名。
_LOCK_NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _locked_closure() -> tuple[list[str], str]:
    """读取锁定文件里的包名清单，返回 (包名, 备注)。

    这里必须按**字节**读再清 NUL，不能直接 read_text().splitlines()：
    本机 DLP 会把写出的文件补 NUL 到 4096 字节块对齐（实测
    requirements_mcp.txt 1291 字节正文 + 2805 字节 \\x00），整段填充里
    没有换行符，于是会被 splitlines() 当成**一个** 2805 字节长的包名，
    进而误报「位置不可识别」。踩过一次，见 MCP接入说明.md 第 3.5 节。
    """
    if not LOCK_FILE.is_file():
        return list(FALLBACK_REQUIRED), "锁定文件缺失，使用兜底清单"

    raw = LOCK_FILE.read_bytes()
    n_pad = raw.count(b"\x00")
    text = raw.replace(b"\x00", b"").decode("utf-8", errors="replace")

    names: list[str] = []
    illegal: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==")[0].strip()
        if _LOCK_NAME_OK.match(name):
            names.append(name)
        else:
            illegal.append(name[:40])

    notes: list[str] = []
    if n_pad:
        notes.append(f"锁定文件含 {n_pad} 字节 NUL 填充（DLP 块对齐），已忽略")
    if illegal:
        notes.append(f"跳过 {len(illegal)} 行无法识别为包名的内容：{illegal[:3]}")
    if not names:
        notes.append("无可识别包名，使用兜底清单")
        return list(FALLBACK_REQUIRED), "；".join(notes)
    return names, "；".join(notes)


def _norm(path: object) -> str:
    """路径归一成可比较的小写反斜杠形式。

    不能直接用 Path 比较：Windows 上同一个目录会出现 Lib / lib 两种大小写
    （sysconfig 给 Lib，模块 __file__ 给 lib），比不中就会误报。
    """
    return str(path).lower().replace("/", "\\").rstrip("\\")


def _under(child: object, parents: set[str]) -> bool:
    """child 是否落在 parents 里任一条路径之下。"""
    text = _norm(child)
    return any(text == base or text.startswith(base + "\\") for base in parents)


def _site_locations() -> tuple[Path, set[str], set[str]]:
    """返回 (本解释器站点, 可接受站点集合, user-site 集合)。

    「可接受」= 本解释器站点 + 系统/基础环境站点（venv 的
    include-system-site-packages 会让 conda 站点合法可见）。
    「不可接受」= user-site —— 它和别的 Python 3.10 共享、可以被
    PYTHONNOUSERSITE / python -s 关掉，是唯一真正危险的来源。
    """
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    accepted = {_norm(purelib)}
    try:
        for extra in site.getsitepackages():
            accepted.add(_norm(extra))
    except Exception:
        pass

    user: set[str] = set()
    try:
        raw = site.getusersitepackages()
        for item in [raw] if isinstance(raw, str) else list(raw):
            if item:
                user.add(_norm(item))
    except Exception:
        pass
    return purelib, accepted, user


def check_dependency_origin() -> bool:
    """第 0 层：依赖不能靠 user-site 兜底。"""
    print("\n[0] 依赖归属层：mcp 依赖是否依赖 user-site")
    import importlib.metadata as md

    purelib, accepted, user = _site_locations()
    print(f"         本解释器站点：{purelib}")
    on_path = [p for p in user if any(_norm(p) == _norm(item) for item in sys.path)]
    print(
        f"         user-site：{'在 sys.path 上（更危险）' if on_path else '未参与 import'}"
    )
    ok = True

    # 0.1 mcp 本体位置 —— 最直接的红旗
    spec = importlib.util.find_spec("mcp")
    if spec is None or not spec.origin:
        _fail("mcp 无法导入")
        return False
    origin = Path(spec.origin).resolve()
    if _under(origin, user):
        _fail(f"mcp 来自 user-site：{origin.parent}")
        print("         设 PYTHONNOUSERSITE=1 或换解释器即 ModuleNotFoundError")
        ok = False
    elif _under(origin, accepted):
        _pass(f"mcp 位于可接受站点（{origin.parent}）")
    else:
        _fail(f"mcp 位置异常：{origin.parent}")
        ok = False

    # 0.2 逐个核对闭包成员：只拦 user-site，基础环境站点合法
    names, lock_note = _locked_closure()
    if lock_note:
        print(f"         锁定文件：{lock_note}")
    stray: list[str] = []
    from_user: list[str] = []
    for name in names:
        try:
            loc = Path(str(md.distribution(name).locate_file(""))).resolve()
        except md.PackageNotFoundError:
            stray.append(f"{name}(缺失)")
            continue
        if _under(loc, user):
            from_user.append(name)
        elif not _under(loc, accepted):
            stray.append(f"{name}->{loc.parent}")
    if from_user:
        _fail(f"{len(from_user)}/{len(names)} 个包来自 user-site：{from_user[:6]}")
        print("         修复：python -m agent.scripts.lock_mcp_deps --write")
        print("               再按锁定文件头部注释安装（requirements_mcp_env.txt）")
        ok = False
    if stray:
        _fail(f"{len(stray)}/{len(names)} 个包位置不可识别：{[s[:60] for s in stray[:6]]}")
        ok = False
    if not from_user and not stray:
        _pass(f"闭包 {len(names)} 个包均非 user-site 来源")

    # 0.3 实测：屏蔽 user-site 后能否导入到协议层入口
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from mcp.server.mcpserver import MCPServer; "
                "import agent.mcp_server; print('IMPORT_OK')",
            ],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as error:
        _fail(f"屏蔽 user-site 的子进程启动失败：{type(error).__name__}: {error}")
        return False
    if "IMPORT_OK" in (proc.stdout or ""):
        _pass("PYTHONNOUSERSITE=1 下仍可导入 server（不依赖 user-site）")
    else:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        _fail("PYTHONNOUSERSITE=1 下导入失败：")
        for line in detail[-3:]:
            print(f"         {line}")
        ok = False

    return ok


async def check_static() -> tuple[bool, list[str]]:
    print("\n[1] 静态层：工具清单与 schema")
    server = build_server()
    tools = await server.list_tools()
    names = [tool.name for tool in tools]
    ok = True

    if len(tools) == EXPECTED_TOOLS:
        _pass(f"工具数 {len(tools)}")
    else:
        _fail(f"工具数应为 {EXPECTED_TOOLS}，实际 {len(tools)}")
        ok = False

    empty_desc = [tool.name for tool in tools if not (tool.description or "").strip()]
    if empty_desc:
        _fail(f"这些工具没有描述：{empty_desc}")
        ok = False
    else:
        _pass("全部工具有非空描述")

    missing_props: list[str] = []
    missing_constraints: list[str] = []
    for name, fields in DESCRIPTION_SPOT_CHECKS.items():
        tool = next((item for item in tools if item.name == name), None)
        if tool is None:
            missing_props.append(f"{name}(工具缺失)")
            continue
        schema = tool.input_schema or {}
        properties = schema.get("properties") or {}
        for field in fields:
            prop = properties.get(field)
            if prop is None:
                missing_props.append(f"{name}.{field}")
            elif not prop.get("description"):
                missing_props.append(f"{name}.{field}(无说明)")

    if missing_props:
        _fail(f"参数说明缺失：{missing_props}")
        ok = False
    else:
        _pass("抽查的参数说明均已透传")

    # 取值约束：candidate_id 的 ge=1 必须体现为 minimum
    plan_tool = next((item for item in tools if item.name == "plan_route"), None)
    if plan_tool:
        prop = (plan_tool.input_schema or {}).get("properties", {}).get("candidate_id", {})
        if prop.get("minimum") == 1:
            _pass("数值约束已透传（candidate_id minimum=1）")
        else:
            missing_constraints.append(f"candidate_id minimum={prop.get('minimum')}")
    # 枚举：profile 的三个取值
    if plan_tool:
        prop = (plan_tool.input_schema or {}).get("properties", {}).get("profile", {})
        text = json.dumps(prop, ensure_ascii=False)
        if "wide_airway" in text and "gentle_turn" in text:
            _pass("枚举已透传（profile 三个取值）")
        else:
            missing_constraints.append("profile enum 不完整")
    if missing_constraints:
        _fail(f"取值约束缺失：{missing_constraints}")
        ok = False

    print(f"         工具清单：{'、'.join(names)}")
    return ok, names


async def check_handshake(names: list[str]) -> bool:
    print("\n[2] 握手层：真实 stdio 传输")
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent.mcp_server"],
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    try:
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                init = await session.initialize()
                info = getattr(init, "serverInfo", None) or getattr(
                    init, "server_info", None
                )
                print(f"         服务端标识：{getattr(info, 'name', '?')} "
                      f"{getattr(info, 'version', '?')}")
                listed = await session.list_tools()
                remote = [tool.name for tool in listed.tools]
                if sorted(remote) == sorted(names):
                    _pass(f"远端工具清单一致（{len(remote)} 个）")
                    return True
                _fail(f"远端工具清单不一致：{remote}")
                return False
    except Exception as error:
        _fail(f"握手失败：{type(error).__name__}: {error}")
        return False


async def check_calls(full: bool) -> bool:
    print("\n[3] 调用层：通过 MCP 协议真实调用工具")
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent.mcp_server"],
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    ok = True
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 3.1 最轻的调用：列病例
            result = await session.call_tool("list_cases", {})
            text = _text_of(result)
            if '"count"' in text:
                payload = json.loads(text)
                _pass(f"list_cases 返回 {payload.get('count')} 个病例")
            else:
                _fail(f"list_cases 返回异常：{text[:200]}")
                ok = False

            # 3.2 知识检索（验证 RAG 在协议层可用）
            result = await session.call_tool(
                "search_knowledge", {"query": "分叉项为什么没有死区", "top_k": 2}
            )
            text = _text_of(result)
            if "KB-01#5" in text:
                _pass("search_knowledge 命中 KB-01#5")
            else:
                _fail(f"search_knowledge 未命中预期片段：{text[:200]}")
                ok = False

            # 3.3 参数校验（越界候选必须报错而不是崩）
            result = await session.call_tool(
                "plan_route", {"case_id": "LIDC_0089", "candidate_id": 99}
            )
            text = _text_of(result)
            if '"ok": false' in text or '"ok":false' in text:
                _pass("越界参数被工具层拦下并返回 ok=false")
            else:
                _fail(f"越界参数没有被拦下：{text[:200]}")
                ok = False

            if full:
                result = await session.call_tool(
                    "explain_route_choice",
                    {
                        "case_id": "LIDC_0089",
                        "candidate_id": 3,
                        "device_diameter_mm": 2.0,
                    },
                )
                text = _text_of(result)
                if "share_pct" in text and "verified_against_planner" in text:
                    payload = json.loads(text)
                    winner = payload.get("winner") or {}
                    _pass(
                        "explain_route_choice 返回分项占比 "
                        f"{winner.get('share_pct')}，"
                        f"对账 {winner.get('verified_against_planner')}"
                    )
                else:
                    _fail(f"explain_route_choice 返回异常：{text[:200]}")
                    ok = False

    return ok


def _text_of(result) -> str:
    """从 CallToolResult 里取出文本内容。"""
    contents = getattr(result, "content", None) or []
    for item in contents:
        text = getattr(item, "text", None)
        if text:
            return text
    return ""


async def amain(full: bool) -> int:
    print("=" * 72)
    print(f"MCP server 自检（{SERVER_NAME}）")
    print("=" * 72)

    dep_ok = check_dependency_origin()
    static_ok, names = await check_static()
    handshake_ok = await check_handshake(names)
    calls_ok = await check_calls(full)

    print("\n" + "=" * 72)
    layers = (
        ("依赖归属层", dep_ok),
        ("静态层", static_ok),
        ("握手层", handshake_ok),
        ("调用层", calls_ok),
    )
    for label, passed in layers:
        print(f"  {label}：{'通过' if passed else '未通过'}")
    if all(passed for _, passed in layers):
        print("MCP server 自检通过")
        return 0
    print("MCP server 自检未通过")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP server 自检")
    parser.add_argument(
        "--full", action="store_true", help="额外跑一次完整规划归因（较慢）"
    )
    args = parser.parse_args()
    return asyncio.run(amain(args.full))


if __name__ == "__main__":
    raise SystemExit(main())
