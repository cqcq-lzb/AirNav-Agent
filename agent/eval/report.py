"""评测报告：把 EvalReport 渲染成可读的 markdown 与结构化 JSON。

报告刻意把「失败了什么」放在最前面 —— 通过率是个瞬时数字，
失败明细才是能拿去做改进的东西。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cases import case_by_id
from .graders import GradeResult
from .harness import CaseResult, EvalReport

GRADER_LABELS = {
    "tool_selection": "工具选型",
    "tool_arguments": "参数正确性",
    "grounding": "数字可溯源",
    "citations": "引用有效性",
    "refusal": "边界拒答",
    "robustness": "稳健性",
    "runner": "运行",
}


def load_report(path: str | Path) -> EvalReport:
    """从 JSON 报告还原 EvalReport（用于分片跑完再合并）。

    注意：还原出来的 CaseResult 不带 AgentRun（原始轨迹在 json 里是可选字段），
    所以合并后的报告只能做汇总与失败明细，不能再看逐步轨迹。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    report = EvalReport(
        backend=payload.get("backend", "unknown"),
        started_at=payload.get("started_at", ""),
    )
    for row in payload.get("cases", []):
        try:
            case = case_by_id(row["case_id"])
        except KeyError:
            continue
        grades = [
            GradeResult(
                grader=item.get("grader", "?"),
                passed=bool(item.get("passed")),
                detail=item.get("detail", ""),
                severity=item.get("severity", "error"),
                extra=item.get("extra") or {},
            )
            for item in row.get("grades", [])
        ]
        report.results.append(
            CaseResult(
                case=case,
                passed=bool(row.get("passed")),
                grades=grades,
                error=row.get("error"),
                elapsed_s=float(row.get("elapsed_s") or 0.0),
                tools_called=list(row.get("tools_called") or []),
                answer=row.get("answer") or "",
                stop_reason=row.get("stop_reason") or "",
            )
        )
    return report


def merge_reports(paths: list[str | Path]) -> EvalReport:
    reports = [load_report(path) for path in paths]
    if not reports:
        raise ValueError("没有可合并的报告")
    merged = reports[0]
    for other in reports[1:]:
        merged.merge(other)
    return merged


def to_markdown(report: EvalReport, title: str = "Agent 评测报告") -> str:
    summary = report.to_dict()
    lines: list[str] = [
        f"# {title}",
        "",
        f"- 后端：`{report.backend}`",
        f"- 时间：{report.started_at}",
        f"- 用例：**{summary['passed']} / {summary['total']}** "
        f"通过（{summary['pass_rate'] * 100:.1f}%）",
        "",
    ]

    failures = [item for item in report.results if not item.passed]
    if failures:
        lines += ["## 失败明细", ""]
        for item in failures:
            lines.append(f"### {item.case.id}（{item.case.category}）")
            lines.append("")
            lines.append(f"> 判定口径：{item.case.rubric}")
            lines.append("")
            if item.error:
                lines.append(f"- **运行异常**：`{item.error}`")
            for grade in item.failures:
                label = GRADER_LABELS.get(grade.grader, grade.grader)
                lines.append(f"- **{label}**：{grade.detail}")
            lines.append(
                f"- 实际调用链：`{' -> '.join(item.tool_chain) or '（未调用任何工具）'}`"
            )
            if item.answer_text:
                first = item.answer_text.strip().splitlines()[0][:120]
                lines.append(f"- 回答开头：{first}")
            lines.append("")
    else:
        lines += ["## 失败明细", "", "无。全部用例通过。", ""]

    lines += ["## 分类通过率", "", "| 类别 | 通过 / 总数 | 通过率 |", "|---|---|---|"]
    for category, bucket in report.by_category_stats().items():
        lines.append(
            f"| {category} | {bucket['passed']} / {bucket['total']} | "
            f"{bucket['pass_rate'] * 100:.0f}% |"
        )
    lines.append("")

    lines += ["## 各打分器命中情况", "", "| 打分器 | 通过 | 失败 |", "|---|---|---|"]
    for grader, bucket in report.grader_stats().items():
        label = GRADER_LABELS.get(grader, grader)
        lines.append(f"| {label} | {bucket['passed']} | {bucket['failed']} |")
    lines.append("")

    lines += ["## 全部用例", "", "| 用例 | 类别 | 结果 | 工具调用 |", "|---|---|---|---|"]
    for item in report.results:
        mark = "通过" if item.passed else "**失败**"
        tool_text = " → ".join(item.tool_chain) or "—"
        lines.append(f"| {item.case.id} | {item.case.category} | {mark} | {tool_text} |")
    lines.append("")

    return "\n".join(lines)


def write_report(
    report: EvalReport,
    out_dir: str | Path,
    *,
    stem: str = "eval_report",
    title: str = "Agent 评测报告",
    include_run: bool = True,
) -> dict[str, Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    markdown_path = directory / f"{stem}.md"
    json_path = directory / f"{stem}.json"

    markdown_path.write_text(
        to_markdown(report, title=title), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(
            report.to_dict(include_run=include_run), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    return {"markdown": markdown_path, "json": json_path}


def print_console_summary(report: EvalReport) -> dict[str, Any]:
    summary = report.to_dict()
    print("\n" + "=" * 72)
    print("评测汇总")
    print("=" * 72)
    print(f"后端：{report.backend}")
    print(
        f"通过：{summary['passed']} / {summary['total']}"
        f"（{summary['pass_rate'] * 100:.1f}%）"
    )
    print("\n分类：")
    for category, bucket in report.by_category_stats().items():
        print(
            f"  {category:<10s} {bucket['passed']}/{bucket['total']}"
            f"  {bucket['pass_rate'] * 100:5.0f}%"
        )
    print("\n打分器：")
    for grader, bucket in report.grader_stats().items():
        label = GRADER_LABELS.get(grader, grader)
        print(f"  {label:<12s} 通过 {bucket['passed']:>3d}  失败 {bucket['failed']:>3d}")

    failures = [item for item in report.results if not item.passed]
    if failures:
        print(f"\n失败用例（{len(failures)} 条）：")
        for item in failures:
            print(f"  - {item.case.id}")
            if item.error:
                print(f"      运行异常：{item.error}")
            for grade in item.failures:
                label = GRADER_LABELS.get(grade.grader, grade.grader)
                print(f"      {label}：{grade.detail[:150]}")
    return summary


__all__ = [
    "GRADER_LABELS",
    "load_report",
    "merge_reports",
    "print_console_summary",
    "to_markdown",
    "write_report",
]
