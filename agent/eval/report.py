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
from .harness import AttemptRecord, CaseResult, EvalReport

GRADER_LABELS = {
    "tool_selection": "工具选型",
    "tool_arguments": "参数正确性",
    "grounding": "数字可溯源",
    # ⚠️ 加打分器时这张表也要跟着加。漏了不会报错，只会让失败明细里
    # 混进一个裸 key（`id_binding`）—— 加第 7 类时漏过一次，第九条纪律
    # 「把数量/清单写死在断言里的地方，加东西时要全局搜一遍」说的就是这类表。
    "id_binding": "编号口径",
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
        attempts_detail = [
            AttemptRecord(
                index=int(item.get("index") or 0),
                passed=bool(item.get("passed")),
                grades=[
                    GradeResult(
                        grader=g.get("grader", "?"),
                        passed=bool(g.get("passed")),
                        detail=g.get("detail", ""),
                        severity=g.get("severity", "error"),
                        extra=g.get("extra") or {},
                    )
                    for g in item.get("failures", [])
                ],
                error=item.get("error"),
                elapsed_s=float(item.get("elapsed_s") or 0.0),
                tools_called=list(item.get("tools") or []),
                stop_reason=item.get("stop_reason") or "",
            )
            for item in row.get("attempts_detail", [])
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
                attempts_detail=attempts_detail,
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


def _attempt_sequence(item: CaseResult) -> str:
    """`1:过 2:挂 3:过` —— 把「哪次挂」写进报告，而不是只给一个布尔。"""
    return " ".join(
        f"{a.index}:{'过' if a.passed else '挂'}" for a in item.attempts_detail
    )


def to_markdown(report: EvalReport, title: str = "Agent 评测报告") -> str:
    summary = report.to_dict()
    lines: list[str] = [
        f"# {title}",
        "",
        f"- 后端：`{report.backend}`",
        f"- 时间：{report.started_at}",
        f"- 用例：**{summary['passed']} / {summary['total']}** "
        f"通过（{summary['pass_rate'] * 100:.1f}%）",
    ]
    if report.repeat > 1:
        # 报告必须自证身份：不写这一行，过几天没人分得清这 28/30 是
        # 「单次恰好」还是「3 次里稳定过的」—— 两者的分量完全不同。
        lines += [
            f"- **每条重复 {report.repeat} 次**。用例级判定 = **N 次全过**"
            f"（保守读数，用来卡基线）；"
            f"尝试级通过率 **{summary['attempts_passed']} / "
            f"{summary['attempts_total']}**"
            f"（{summary['run_pass_rate'] * 100:.1f}%）",
        ]
    lines.append("")

    flaky = report.flaky_cases()
    if flaky:
        lines += [
            "## 偶发用例（既过过也挂过）",
            "",
            "> 这类用例的处置与「稳定失败」相反：稳定失败要改代码，"
            "偶发先看采样 —— 所以必须单独列出来，不能被平均进通过率里。",
            "",
            "| 用例 | 重复结果 | 尝试级 |",
            "|---|---|---|",
        ]
        for item in flaky:
            lines.append(
                f"| {item.case.id} | {_attempt_sequence(item)} | "
                f"{item.pass_count}/{item.attempts} |"
            )
        lines.append("")

    failures = [item for item in report.results if not item.passed]
    if failures:
        lines += ["## 失败明细", ""]
        for item in failures:
            lines.append(f"### {item.case.id}（{item.case.category}）")
            lines.append("")
            lines.append(f"> 判定口径：{item.case.rubric}")
            lines.append("")
            if item.attempts > 1:
                verdict = "**偶发**，不是稳定失败" if item.flaky else "每次都失败"
                lines.append(
                    f"- 重复 {item.attempts} 次：`{_attempt_sequence(item)}`"
                    f"（{verdict}）"
                )
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

    header = ["用例", "类别", "结果", "工具调用"]
    if report.repeat > 1:
        header.insert(3, "重复")
    lines += ["## 全部用例", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for item in report.results:
        if item.flaky:
            mark = "**偶发**"
        elif item.passed:
            mark = "通过"
        else:
            mark = "**失败**"
        tool_text = " → ".join(item.tool_chain) or "—"
        row = [item.case.id, item.case.category, mark]
        if report.repeat > 1:
            row.append(f"{item.pass_count}/{item.attempts}")
        row.append(tool_text)
        lines.append("| " + " | ".join(row) + " |")
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
    if report.repeat > 1:
        print(f"重复：每条 {report.repeat} 次（用例级 = N 次全过）")
        print(
            f"尝试级通过率：{summary['attempts_passed']} / "
            f"{summary['attempts_total']}"
            f"（{summary['run_pass_rate'] * 100:.1f}%）"
        )
        flaky = report.flaky_cases()
        if flaky:
            print(f"\n偶发用例（{len(flaky)} 条，先看采样、别急着改代码）：")
            for item in flaky:
                print(
                    f"  - {item.case.id}  {_attempt_sequence(item)}"
                    f"  {item.pass_count}/{item.attempts}"
                )
        else:
            print("偶发用例：无（这份读数没有歧义）")

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
