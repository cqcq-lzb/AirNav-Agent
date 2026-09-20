"""评测集运行入口。

用法
----
    # 无 LLM，跑规则基线（验证评测链路本身）
    python -m agent.scripts.run_eval --backend heuristic

    # 只跑某几类
    python -m agent.scripts.run_eval --backend heuristic --include 边界拒答 --include 分项归因

    # 真实模型（本地 Ollama；默认模型见 PRESETS，与本机已装的对齐）
    python -m agent.scripts.run_eval --backend ollama --timeout 180

    # 云端
    python -m agent.scripts.run_eval --backend deepseek --api-key $DEEPSEEK_API_KEY

关于后端命名
------------
`heuristic` 是**规则基线**，不是模型：它的回答由模板拼装，只用来证明
用例/打分器/运行器这条链路是通的，同时提供一个下界。报告里会明确标注，
不要把它和模型通过率混起来看。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.eval.cases import CASES  # noqa: E402
from agent.eval.harness import format_progress, run_suite  # noqa: E402
from agent.eval.heuristic import HeuristicClient, SloppyClient  # noqa: E402
from agent.eval.paraphrase import PARAPHRASE_CASES, audit as audit_paraphrases  # noqa: E402
from agent.eval.report import (  # noqa: E402
    merge_reports,
    print_console_summary,
    write_report,
)
from agent.llm.client import PRESETS, OpenAICompatClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "eval"

# 从 PRESETS 推导，不要手写列表 —— 之前手写过一份，里面写着 "dashscope"，
# 而 PRESETS 里的键实际叫 "qwen"，传 --backend dashscope 会直接 KeyError。
# 让两个来源合一，以后加后端就不会漏。
BACKENDS = ("heuristic", "sloppy") + tuple(PRESETS)


def _merge_backend_label(paths: list[str]) -> str:
    labels = []
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            label = payload.get("backend")
        except Exception:
            label = None
        if label and label not in labels:
            labels.append(label)
    return " + ".join(labels) if labels else "merged"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="跑 Agent 评测集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        default="heuristic",
        choices=BACKENDS,
        help="heuristic=规则基线（无 LLM）；其余为真实 LLM 后端",
    )
    parser.add_argument("--model", default=None, help="模型名，缺省用后端预设")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="只跑这些用例 id 或类别，可重复",
    )
    parser.add_argument(
        "--exclude", action="append", default=None, help="排除这些用例 id 或类别"
    )
    parser.add_argument(
        "--cases",
        default="original",
        choices=("original", "paraphrase", "both"),
        help=(
            "original=原有 15 条；paraphrase=同义改写的同一批意图（判定标准逐字段一致）；"
            "both=两套一起跑。改写那套用来把「关键词命中」和「真的理解」分开 —— "
            "规则基线在它上面应当显著掉分"
        ),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--stem", default=None)
    parser.add_argument(
        "--merge",
        nargs="+",
        default=None,
        metavar="JSON",
        help=(
            "只做合并：把这些分片报告 JSON 合成一份完整报告。"
            "整套用例耗时较长时按类别分片跑，再合并"
        ),
    )
    parser.add_argument("--no-json-trace", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _make_client_factory(args):
    if args.backend == "heuristic":

        def factory(_case):
            return HeuristicClient()

        return factory, "规则基线（无 LLM）"

    if args.backend == "sloppy":

        def factory(_case):
            return SloppyClient()

        return factory, "劣化策略（对照，验证评测区分度）"

    def factory(_case):
        return OpenAICompatClient.from_preset(
            args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            temperature=0.0,
        )

    # 探针客户端只用来读出真正生效的模型名：--model 没给时它是预设里的默认值。
    # 不这么做的话报告标题会写成「(预设模型) @ gpu41」，
    # 过几天就没人知道那份报告是 14b 还是 32b 跑的了 —— 报告必须自证身份。
    probe = OpenAICompatClient.from_preset(
        args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    label = f"{probe.model} @ {args.backend}" + (
        f" ({args.base_url})" if args.base_url else ""
    )
    return factory, label


def main() -> int:
    args = build_parser().parse_args()

    # 改写集自检必须在跑之前过。它验的是「题面真的绕开了关键词表」与
    # 「判定字段与基准逐字段一致」—— 这两条破一条，两组分数就不可比，
    # 跑了也白跑。宁可在这里 exit 2，也不要产出一份看着像结论的东西。
    if args.cases in {"paraphrase", "both"}:
        problems = audit_paraphrases()
        if problems:
            print("同义改写集自检未通过，评测不开始：")
            for problem in problems:
                print(f"  - {problem}")
            return 2
        print(
            f"同义改写集自检通过：{len(PARAPHRASE_CASES)} 条，"
            "判定字段与基准逐字段一致"
        )

    if args.cases == "paraphrase":
        selected_cases = list(PARAPHRASE_CASES)
    elif args.cases == "both":
        selected_cases = list(CASES) + list(PARAPHRASE_CASES)
    else:
        selected_cases = list(CASES)

    # 评测用例里有「必须出图」那一类（E05），它会真的渲染 HTML。渲染默认落在
    # outputs/viewers —— 那是**入库的演示产物**目录（2.0 mm / step 1 那一份），
    # 被评测用的参数覆盖掉之后，跑一次评测就在工作区留下无关改动。改到临时目录。
    from agent.render.viewer import use_scratch_output

    scratch = use_scratch_output("run_eval")
    print(f"渲染输出改到临时目录：{scratch}")

    # ---- 合并模式：不跑用例，只把分片报告合成完整报告 ----
    if args.merge:
        report = merge_reports(args.merge)
        report.backend = _merge_backend_label(args.merge)
        print_console_summary(report)
        stem = args.stem or "eval_report"
        paths = write_report(
            report,
            args.out_dir,
            stem=stem,
            title=f"Agent 评测报告（{report.backend}）",
            include_run=False,
        )
        print("\n合并报告已写入：")
        for kind, path in paths.items():
            print(f"  {kind:<9s} {path}")
        return 0 if report.passed == report.total else 1

    # 真实模型先探活，避免跑一半才发现连不上
    if args.backend not in {"heuristic", "sloppy"}:
        client = OpenAICompatClient.from_preset(
            args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        print(f"探测后端 {client.label} ...")
        health = client.health()
        ok = bool(health.get("ok"))
        detail = health.get("detail") or (
            f"模型列表 {len(health.get('models') or [])} 个"
            f"，目标模型{'在列' if health.get('has_target') else '不在列'}"
        )
        print(f"  {'可用' if ok else '不可用'}：{detail}")
        if not ok:
            print(
                "\n后端不可用，评测无法开始。\n"
                "本地内存不足时可先改用 --backend heuristic 验证评测链路。"
            )
            return 2

    factory, label = _make_client_factory(args)

    print(f"开始评测：{label}")
    print(f"用例集：{args.cases}（{len(selected_cases)} 条）")
    print(f"用例筛选：include={args.include} exclude={args.exclude}")

    def on_result(result) -> None:
        print(format_progress(result))

    report = run_suite(
        cases=selected_cases,
        client_factory=factory,
        backend=label,
        on_result=on_result,
        include=args.include,
        exclude=args.exclude,
    )

    print_console_summary(report)

    stem = args.stem or f"eval_{args.backend}" + (
        "" if args.cases == "original" else f"_{args.cases}"
    )
    paths = write_report(
        report,
        args.out_dir,
        stem=stem,
        title=f"Agent 评测报告（{label}）",
        include_run=not args.no_json_trace,
    )
    print("\n报告已写入：")
    for kind, path in paths.items():
        print(f"  {kind:<9s} {path}")

    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
