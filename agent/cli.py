"""AirNav-Agent 命令行入口。

用法示例：
    # 环境自检（后端探活 + 病例扫描 + 缓存状态）
    python -m agent.cli doctor

    # 查看已注册的工具及其参数
    python -m agent.cli tools

    # 单次提问（默认用本地 Ollama）
    python -m agent.cli ask "LIDC_0089 哪个结节最容易到达"

    # 指定后端与器械约束
    python -m agent.cli ask "给 LIDC_0089 的 2 号结节规划路径" \
        --backend deepseek --device-diameter 2.0

    # 交互模式
    python -m agent.cli repl

    # 缓存管理
    python -m agent.cli cache --clear
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent_loop import NavAgent
from agent.core import cache as cache_mod
from agent.llm.client import LLMError, OpenAICompatClient, PRESETS
from agent.tools import NavSession, build_registry


def add_common_options(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    """公共参数。

    suppress=True 时把默认值设成 SUPPRESS，这样写在子命令后面的同名参数
    不会覆盖写在子命令前面的值 —— 这是 argparse 子解析器的经典坑：
    两边都定义时，子解析器的默认值会把主解析器已经解析出的值冲掉。
    """
    d = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "--backend",
        # 默认后端允许用环境变量给，这样启动脚本能把默认值统一改掉，
        # 而不必改代码：对话Agent.bat 里设 AIRNAV_DEFAULT_BACKEND=gpu41。
        # 显式写 --backend xxx 优先级仍然最高 —— argparse 的 default 只在缺省时生效。
        default=d if suppress else (os.environ.get("AIRNAV_DEFAULT_BACKEND") or "ollama"),
        choices=list(PRESETS) + ["scripted"],
        help="LLM 后端，默认本地 ollama（可用 AIRNAV_DEFAULT_BACKEND 覆盖）",
    )
    parser.add_argument("--model", default=d, help="覆盖默认模型名")
    parser.add_argument("--base-url", default=d, help="覆盖默认 base_url")
    parser.add_argument("--api-key", default=d, help="覆盖 API Key")
    parser.add_argument("--cases-root", default=d, help="病例根目录")
    parser.add_argument(
        "--device-diameter",
        type=float,
        default=d if suppress else 0.0,
        help="默认器械外径 mm",
    )
    parser.add_argument(
        "--device-margin",
        type=float,
        default=d if suppress else 0.2,
        help="默认安全余量 mm",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=d if suppress else 8,
        help="工具调用步数上限",
    )
    parser.add_argument("--json", action="store_true", default=d, help="以 JSON 输出完整留痕")
    parser.add_argument("--verbose", action="store_true", default=d, help="打印每一步的工具调用")
    parser.add_argument("--no-cache", action="store_true", default=d, help="禁用病例加载缓存")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.cli",
        description="AirNav-Agent：经支气管肺结节导航的规划 Agent",
    )
    add_common_options(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="单次提问")
    add_common_options(ask, suppress=True)
    ask.add_argument("question", help="自然语言问题")

    repl = sub.add_parser("repl", help="交互式对话")
    add_common_options(repl, suppress=True)

    tools_cmd = sub.add_parser("tools", help="列出工具")
    add_common_options(tools_cmd, suppress=True)

    doctor = sub.add_parser("doctor", help="环境自检")
    add_common_options(doctor, suppress=True)

    cache_cmd = sub.add_parser("cache", help="缓存管理")
    add_common_options(cache_cmd, suppress=True)
    cache_cmd.add_argument("--clear", action="store_true", help="清空缓存")

    return parser


def make_session(args: argparse.Namespace) -> NavSession:
    return NavSession(
        root=args.cases_root,
        device_diameter_mm=args.device_diameter,
        device_margin_mm=args.device_margin,
        use_cache=not args.no_cache,
    )


def make_client(args: argparse.Namespace):
    if args.backend == "scripted":
        from agent.llm.client import ScriptedClient

        return ScriptedClient([ScriptedClient.say("（scripted 后端未提供脚本）")])
    return OpenAICompatClient.from_preset(
        args.backend, model=args.model, base_url=args.base_url, api_key=args.api_key
    )


# ------------------------------------------------------------------ 命令


def _port_of(base_url: str) -> int:
    parsed = urlparse(base_url)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def _console_text(args: list[str], timeout: int) -> str | None:
    """跑一条 Windows 控制台命令并拿回文本，失败返回 None。

    ⚠️ 不能用 `subprocess.run(..., text=True)`：它按 **UTF-8** 解码，而
    `tasklist` 这类原生工具输出的是 **OEM 代码页（中文 Windows 上是 GBK）**。
    解码会在 reader 线程里抛 `UnicodeDecodeError`，`stdout` 直接变成 `None`，
    外层拿到的是「命令没输出」而不是「解码坏了」—— 极难定位。

    所以先抓 **bytes**，再按 utf-8 → OEM/ANSI → latin-1 逐个试。
    只需要进程名和 PID（都是 ASCII），退到 latin-1 也不影响判断。
    """
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout)
    except Exception:  # noqa: BLE001 - 探测失败就不表态，不能让 doctor 挂掉
        return None
    raw = proc.stdout or b""
    for encoding in ("utf-8", "mbcs", "gbk"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1")


def _probe_local_port_owner(port: int) -> tuple[bool | None, str]:
    """看 `localhost:<port>` 上监听的是不是 ollama 自己。

    返回 (是否 ollama, 说明)：True 是 / False 不是（并给出占用者进程名）/ None 查不出来。

    ## 为什么要查这个

    本机 `localhost:11434` **会被 Cursor / VS Code 的 Remote-SSH 转发到 gpu41**，
    因为转发在，`ollama list` / `ollama ps` 全都正常返回、还显示 `100% GPU`，
    看起来完全像「本机 ollama 跑起来了」—— 但它其实在服务器上。

    这条假象已经造成两次误判：把「本地后端 404」归因成本机模型没装，
    以及把「端口转发」当成「本机 ollama 修好了」。所以做成常驻检查，
    而不是写在文档里等下次再被骗一遍。

    只在 Windows 上查（本项目的本机形态）；其他平台返回 None，不表态。
    """
    if os.name != "nt":
        return None, "非 Windows，跳过"

    listing = _console_text(["netstat", "-ano"], timeout=15)
    if not listing:
        return None, "netstat 不可用"

    pids: set[str] = set()
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[-2].upper() == "LISTENING" and parts[1].endswith(
            f":{port}"
        ):
            pids.add(parts[-1])
    if not pids:
        return None, f"端口 {port} 上没有监听进程"

    table = _console_text(["tasklist", "/FO", "CSV", "/NH"], timeout=25)
    if not table:
        return None, f"端口由 PID {'、'.join(sorted(pids))} 占用（tasklist 不可用）"

    names: dict[str, str] = {}
    for row in csv.reader(io.StringIO(table)):
        if len(row) >= 2:
            names[row[1].strip()] = row[0].strip()
    owners = [names.get(pid, f"PID {pid}") for pid in sorted(pids)]
    is_ollama = all("ollama" in name.lower() for name in owners)
    return is_ollama, "、".join(owners)


def cmd_doctor(args: argparse.Namespace) -> int:
    print("=" * 66)
    print("AirNav-Agent 环境自检")
    print("=" * 66)

    print(f"\n[解释器] {sys.executable}")
    print(f"Python {sys.version.split()[0]}")

    print("\n[病例根目录]")
    session = make_session(args)
    cases = session.available_cases()
    print(f"  {session.root}")
    print(f"  找到 {len(cases)} 个病例")
    for item in cases[:12]:
        flag = "已缓存" if item.get("cached") else "未缓存"
        print(f"    - {item['case_id']:<14} {flag}")
    if len(cases) > 12:
        print(f"    ... 其余 {len(cases) - 12} 个")

    print("\n[加载缓存]")
    info = cache_mod.info()
    print(f"  目录 {info['dir']}")
    print(f"  {info['entries']} 个条目，合计 {info['mb']} MB")

    print("\n[工具集]")
    registry = build_registry()
    print(f"  已注册 {len(registry.names())} 个：{', '.join(registry.names())}")

    print("\n[LLM 后端]")
    try:
        client = make_client(args)
    except LLMError as error:
        print(f"  配置失败：{error}")
        return 1
    print(f"  {client.label}")
    if client.is_local:
        port = _port_of(client.base_url)
        is_ollama, detail = _probe_local_port_owner(port)
        if is_ollama is False:
            print(f"  ⚠️  端口 {port} 的监听进程是 {detail}，不是 ollama。")
            print("      如果这是 Cursor / VS Code 的 Remote-SSH 转发，")
            print("      你连的其实是**服务器**，不是本机 —— 模型列表也跟着变。")
            print("      核实：`ollama list` 里若出现本机没装的模型，即为转发。")
        elif is_ollama is True:
            print(f"  端口 {port} 由 {detail} 监听 —— 确实连到本机。")
    health = client.health()
    if health.get("ok"):
        models = health.get("models") or []
        print(f"  可达。可用模型 {len(models)} 个")
        if not health.get("has_target", True):
            print(f"  注意：模型 {client.model} 不在列表中，可能需要先 ollama pull")
    else:
        print(f"  不可达：{health.get('detail')}")
        if args.backend == "ollama":
            print("  提示：先启动 Ollama（运行 ollama serve），或换 --backend deepseek")
        return 1

    print("\n自检通过。")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    registry = build_registry()
    for spec in registry.specs():
        schema = spec.json_schema()
        required = set(schema.get("required", []))
        print(f"\n{spec.name}")
        print(f"  {spec.description}")
        for key, prop in schema.get("properties", {}).items():
            mark = "*" if key in required else " "
            kind = prop.get("type", "any")
            desc = prop.get("description", "")
            print(f"   {mark} {key}: {kind}  {desc}")
    print("\n（* 表示必填）")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    if args.clear:
        removed = cache_mod.clear()
        print(f"已清空 {removed} 个缓存文件")
        return 0
    print(json.dumps(cache_mod.info(), ensure_ascii=False, indent=2))
    return 0


def probe_backend(client) -> dict[str, Any]:
    """探活 LLM 后端，返回**结构化**结论（而不是只打印一行字）。

    为什么要结构化：模型看不见 stdout。如果降级信息只打在控制台，
    它既不会改变 Agent 的行为，也不会出现在 `--json` 里 ——
    下游（评测、报告、网页端）就无从知道「这次回答是在后端不可达时产生的」。
    所以这里同时给出 `degraded` 字段与一句人话，两边都能消费。

    返回：
      {"ok": bool, "degraded": bool, "detail": str, "models": [...]}
      · ok=True            后端可达且目标模型在列表里
      · degraded=True      后端**不可达**（或目标模型缺失），但流程仍可继续
    """
    if not hasattr(client, "health"):
        # scripted / 回放后端没有网络，天然不算降级
        return {"ok": True, "degraded": False, "detail": f"{client.label}（本地回放）"}

    health = client.health()
    if health.get("ok") and health.get("has_target", True):
        models = health.get("models") or []
        return {
            "ok": True, "degraded": False, "models": models,
            "detail": f"可达，模型 {client.model} 在列（共 {len(models)} 个）",
        }
    if health.get("ok"):
        models = health.get("models") or []
        return {
            "ok": False, "degraded": True, "models": models,
            "detail": f"后端可达，但没有模型 {client.model}"
                      f"（可用 {len(models)} 个）→ 需要先 ollama pull",
        }
    return {
        "ok": False, "degraded": True, "models": [],
        "detail": f"后端不可达：{health.get('detail')}",
    }


def print_degradation(probe: dict[str, Any], backend: str) -> None:
    """把降级状态打成一眼能看懂的横幅。"""
    print("=" * 66)
    print("⚠️  降级运行：推理后端不可用")
    print(f"    {probe['detail']}")
    print("    · 依赖模型判断的步骤会失败；**纯工具查询仍可用**")
    print("      （病例结构、候选列表、几何量、合规报告都不需要模型）")
    if backend == "gpu41":
        print("    核实后端是否起来：ssh gpu41 后跑 ~/ollama/start_ollama_gpu41.sh")
    elif backend == "ollama":
        print("    本机先启动 Ollama（运行 ollama serve），或换 --backend gpu41")
    print("=" * 66)


def run_question(args: argparse.Namespace, question: str) -> int:
    session = make_session(args)
    registry = build_registry()
    client = make_client(args)
    agent = NavAgent(
        client=client,
        registry=registry,
        session=session,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    # 探活一次：不可达时**照常继续**，只是把降级状态显式标出来。
    # ⚠️ 有意的取舍：不在这里 `return 1`。因为大量问题只走工具（不需要模型），
    #    提前退出会把「能用」也一起挡掉 —— 判据是「能不能跑」，不是「后端在不在」。
    probe = probe_backend(client)
    degraded = probe["degraded"]

    if not args.json:
        print(f"[后端] {client.label}")
        if degraded:
            print_degradation(probe, args.backend)
        print(f"[问题] {question}")
        print("-" * 66)

    run = agent.run(question)

    if args.json:
        payload = run.to_dict()
        # 降级状态进 JSON —— 让下游能区分「答错了」与「后端没起来」
        payload["backend_probe"] = probe
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(run.answer)
        print("-" * 66)
        print(
            f"[统计] {len(run.steps)} 步 / {run.tool_call_count} 次工具调用 / "
            f"{run.elapsed_s:.1f}s / 结束原因 {run.stop_reason}"
        )
        if degraded:
            print(f"[降级] ⚠️ 后端不可达（{probe['detail']}）—— 以上结果可能不完整")
        if run.usage:
            print(f"[用量] {json.dumps(run.usage, ensure_ascii=False)}")
        stats = registry.stats()
        if stats["failed_calls"]:
            print(f"[失败调用] {stats['failed_calls']} 次")
            for call in registry.trace():
                if not call["ok"]:
                    print(f"    {call['tool']}: {call['error']}")
    return 0 if run.answer else 1


def cmd_repl(args: argparse.Namespace) -> int:
    session = make_session(args)
    registry = build_registry()
    try:
        client = make_client(args)
    except LLMError as error:
        print(f"后端配置失败：{error}")
        return 1

    print(f"AirNav-Agent 交互模式 | 后端：{client.label}")
    # 启动时探活一次：不可达**不退出**，只打降级横幅。
    # ⚠️ 与 doctor 的分工：doctor 是「环境自检」，后端不可达 = 自检不过 = rc 1；
    #    repl 是「拿来用」，工具类问题不该被后端拖累。
    probe = probe_backend(client)
    if probe["degraded"]:
        print_degradation(probe, args.backend)
    print("输入问题回车执行；exit 退出；tools 查看工具；cases 列出病例。")
    agent = NavAgent(
        client=client,
        registry=registry,
        session=session,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue
        if question in {"exit", "quit", "q"}:
            return 0
        if question == "tools":
            cmd_tools(args)
            continue
        if question == "cases":
            for item in session.available_cases():
                print(f"  {item['case_id']}")
            continue

        run = agent.run(question)
        print("-" * 66)
        print(run.answer)
        print("-" * 66)
        print(
            f"[{len(run.steps)} 步 / {run.tool_call_count} 次工具调用 / "
            f"{run.elapsed_s:.1f}s / {run.stop_reason}]"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "tools":
        return cmd_tools(args)
    if args.command == "cache":
        return cmd_cache(args)
    if args.command == "ask":
        return run_question(args, args.question)
    if args.command == "repl":
        return cmd_repl(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
