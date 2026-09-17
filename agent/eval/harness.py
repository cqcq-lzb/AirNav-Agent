"""评测运行器。

两条原则
--------
1. **用例之间必须隔离**：每个用例都新建一套 `ToolRegistry` + `NavSession`。
   复用会话会让上一个用例的病例/规划缓存和 trace 漏进下一个用例，
   分数就不再可信 —— 这是评测框架最常见的自欺方式。
2. **打分器出错必须暴露**：某条用例在运行期抛异常时，记为失败并保留异常，
   而不是跳过。被静默跳过的用例会让通过率虚高。
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence

from ..agent_loop import AgentRun, NavAgent
from ..llm.client import ScriptedClient
from ..tools import NavSession, build_registry
from .cases import CASES, EvalCase, by_category
from .graders import GradeResult, grade_case


@dataclass
class CaseResult:
    case: EvalCase
    passed: bool
    grades: list[GradeResult] = field(default_factory=list)
    run: AgentRun | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    # 从 JSON 报告还原时没有 AgentRun，但摘要信息要保留下来，
    # 否则「合并报告」会丢掉调用链，看板上会整列变成空
    tools_called: list[str] = field(default_factory=list)
    answer: str = ""
    stop_reason: str = ""

    @property
    def failures(self) -> list[GradeResult]:
        return [g for g in self.grades if not g.passed and g.severity == "error"]

    @property
    def tool_chain(self) -> list[str]:
        if self.run is not None:
            return [
                call["name"] for step in self.run.steps for call in step.tool_calls
            ]
        return list(self.tools_called)

    @property
    def answer_text(self) -> str:
        return self.run.answer if self.run is not None else self.answer

    def to_dict(self, include_run: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case.id,
            "category": self.case.category,
            "question": self.case.question,
            "rubric": self.case.rubric,
            "passed": self.passed,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 2),
            "grades": [g.to_dict() for g in self.grades],
            "tools_called": self.tool_chain,
            "answer": self.answer_text,
            "stop_reason": (
                self.run.stop_reason if self.run is not None else self.stop_reason
            ),
        }
        if include_run and self.run is not None:
            payload["trace"] = self.run.to_dict()
        return payload


@dataclass
class EvalReport:
    backend: str
    started_at: str
    results: list[CaseResult] = field(default_factory=list)

    # ---- 汇总 ----

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for item in self.results if item.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def by_category_stats(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for item in self.results:
            bucket = out.setdefault(
                item.case.category, {"total": 0, "passed": 0, "case_ids": []}
            )
            bucket["total"] += 1
            bucket["passed"] += 1 if item.passed else 0
            bucket["case_ids"].append(item.case.id)
        for bucket in out.values():
            bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 3)
        return out

    def grader_stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for item in self.results:
            for grade in item.grades:
                bucket = out.setdefault(grade.grader, {"passed": 0, "failed": 0})
                if grade.passed:
                    bucket["passed"] += 1
                else:
                    bucket["failed"] += 1
        return out

    def all_graders_pass(self) -> bool:
        return all(
            grade.passed for item in self.results for grade in item.grades
        ) and bool(self.results)

    def to_dict(self, include_run: bool = False) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "started_at": self.started_at,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "all_graders_pass": self.all_graders_pass(),
            "by_category": self.by_category_stats(),
            "by_grader": self.grader_stats(),
            "cases": [item.to_dict(include_run=include_run) for item in self.results],
        }

    # ---- 分片合并 ----

    def merge(self, other: "EvalReport") -> "EvalReport":
        """把另一个报告合并进来（同一用例 id 以 other 为准覆盖）。

        整套用例跑一次可能要十几分钟，超过单次命令的上限；
        按类别分片跑再合并，是 CI 里更常见的做法。
        """
        merged = {item.case.id: item for item in self.results}
        for item in other.results:
            merged[item.case.id] = item
        order = {case.id: index for index, case in enumerate(CASES)}
        self.results = sorted(
            merged.values(),
            key=lambda item: (order.get(item.case.id, 10_000), item.case.id),
        )
        self.backend = self.backend if self.backend == other.backend else (
            f"{self.backend} + {other.backend}"
        )
        self.started_at = min(self.started_at, other.started_at)
        return self


def make_agent(
    client,
    *,
    max_steps: int = 10,
    verbose: bool = False,
) -> NavAgent:
    """为单个用例创建一套全新的 agent（注册表与会话都是新的）。"""
    return NavAgent(
        client=client,
        registry=build_registry(),
        session=NavSession(),
        max_steps=max_steps,
        verbose=verbose,
    )


def evaluate_run(case: EvalCase, run: AgentRun, elapsed_s: float = 0.0) -> CaseResult:
    grades = grade_case(case, run)
    failed = [g for g in grades if not g.passed and g.severity == "error"]
    return CaseResult(
        case=case,
        passed=not failed,
        grades=grades,
        run=run,
        elapsed_s=elapsed_s,
    )


def run_suite(
    cases: Sequence[EvalCase] | None = None,
    *,
    client_factory: Callable[[EvalCase], Any],
    backend: str = "unknown",
    on_result: Callable[[CaseResult], None] | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> EvalReport:
    """跑一整套用例。

    client_factory 接收用例、返回该用例要用的 LLM 客户端。
    真实模型可以按用例构造（例如注入 few-shot 或调整温度）；
    脚本回放可以按用例返回预设的调用序列。
    """
    selected = _select(cases or CASES, include, exclude)
    report = EvalReport(
        backend=backend,
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    for case in selected:
        started = datetime.now()
        try:
            client = client_factory(case)
            agent = make_agent(client)
            run = agent.run(case.question)
            elapsed = (datetime.now() - started).total_seconds()
            result = evaluate_run(case, run, elapsed)
        except Exception as error:  # 运行期异常 -> 记为失败，不跳过
            result = CaseResult(
                case=case,
                passed=False,
                error=f"{type(error).__name__}: {error}",
                grades=[
                    GradeResult("runner", False, traceback.format_exc(limit=4))
                ],
            )
        report.results.append(result)
        if on_result is not None:
            on_result(result)

    return report


def _select(
    cases: Sequence[EvalCase],
    include: Iterable[str] | None,
    exclude: Iterable[str] | None,
) -> list[EvalCase]:
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    out: list[EvalCase] = []
    for case in cases:
        if include_set and not (
            case.id in include_set or case.category in include_set
        ):
            continue
        if case.id in exclude_set or case.category in exclude_set:
            continue
        out.append(case)
    return out


def format_progress(result: CaseResult) -> str:
    mark = "PASS" if result.passed else "FAIL"
    line = f"[{mark}] {result.case.id}  ({result.case.category})"
    if result.error:
        return f"{line}\n        运行异常：{result.error}"
    details = [
        f"        - {grade.grader}: {grade.detail}"
        for grade in result.failures
    ]
    if not details:
        details = [
            f"        调用链：{' -> '.join(result.tool_chain) or '（未调用工具）'}"
        ]
    return "\n".join([line, *details])


__all__ = [
    "CaseResult",
    "EvalReport",
    "by_category",
    "evaluate_run",
    "format_progress",
    "make_agent",
    "run_suite",
]
