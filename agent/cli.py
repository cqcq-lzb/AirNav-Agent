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
import json
import sys
from pathlib import Path

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
        default=d if suppress else "ollama",
        choices=list(PRESETS) + ["scripted"],
        help="LLM 后端，默认本地 ollama",
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

    if not args.json:
        print(f"[后端] {client.label}")
        print(f"[问题] {question}")
        print("-" * 66)

    run = agent.run(question)

    if args.json:
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(run.answer)
        print("-" * 66)
        print(
            f"[统计] {len(run.steps)} 步 / {run.tool_call_count} 次工具调用 / "
            f"{run.elapsed_s:.1f}s / 结束原因 {run.stop_reason}"
        )
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
