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


def _chain_of(run: AgentRun | None, fallback: Sequence[str] = ()) -> list[str]:
    if run is not None:
        return [call["name"] for step in run.steps for call in step.tool_calls]
    return list(fallback)


@dataclass
class AttemptRecord:
    """**一次尝试**的结果。

    为什么要把「一次尝试」独立出来（`--repeat N`）：
    单次评测分数不能当结论 —— 实测同一个用例同一份代码，09-17 过、09-20 挂；
    E07 是**稳定失败 0/5**（协议问题），E15 是**偶发**（5/5 过）。
    只跑一次的话，「偶发挂」和「稳定挂」在报告里长得一模一样，
    而这两种东西的处置完全相反（一个要改代码，一个先看采样）。
    """

    index: int
    passed: bool
    grades: list[GradeResult] = field(default_factory=list)
    run: AgentRun | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    tools_called: list[str] = field(default_factory=list)
    answer: str = ""
    stop_reason: str = ""

    @property
    def failures(self) -> list[GradeResult]:
        return [g for g in self.grades if not g.passed and g.severity == "error"]

    @property
    def tool_chain(self) -> list[str]:
        return _chain_of(self.run, self.tools_called)

    @property
    def answer_text(self) -> str:
        return self.run.answer if self.run is not None else self.answer

    def to_dict(self) -> dict[str, Any]:
        """紧凑形态：重复跑时每个用例都有 N 份，不能把 trace 也塞进来。"""
        return {
            "index": self.index,
            "passed": self.passed,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "stop_reason": self.run.stop_reason if self.run is not None else self.stop_reason,
            "tools": self.tool_chain,
            "failures": [
                {"grader": g.grader, "detail": g.detail} for g in self.failures
            ],
        }


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
    # `--repeat N` 时的 N 次尝试；空列表 = 只跑过一次（缺省语义，与旧报告兼容）
    attempts_detail: list[AttemptRecord] = field(default_factory=list)

    # ---- 重复运行（`--repeat N`）派生量 ----
    #
    # ⚠️ `passed` 在 repeat>1 时表示「**N 次全过**」，不是「过过一次」。
    # 这是刻意的：偶然过一次不算通过，否则偶发失败会被采样掩盖，
    # 而 `baseline.json` 的「通过数不得下降」也会被同一批偶发噪声来回触发。

    @property
    def attempts(self) -> int:
        return len(self.attempts_detail) or 1

    @property
    def pass_count(self) -> int:
        if self.attempts_detail:
            return sum(1 for item in self.attempts_detail if item.passed)
        return 1 if self.passed else 0

    @property
    def flaky(self) -> bool:
        """**偶发**：既过过也挂过。这才是「单次分数不能当结论」的那个量。"""
        return 0 < self.pass_count < self.attempts

    @property
    def failures(self) -> list[GradeResult]:
        return [g for g in self.grades if not g.passed and g.severity == "error"]

    @property
    def tool_chain(self) -> list[str]:
        return _chain_of(self.run, self.tools_called)

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
            "attempts": self.attempts,
            "pass_count": self.pass_count,
            "flaky": self.flaky,
        }
        if self.attempts_detail:
            payload["attempts_detail"] = [a.to_dict() for a in self.attempts_detail]
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

    # ---- 尝试级（`--repeat N`）----
    #
    # 「用例级通过数」与「尝试级通过率」是两个不同的量，必须都报：
    #   · 用例级 `passed`（= N 次全过）是**保守**读数，用来卡基线，防偶发抬分
    #   · 尝试级通过率是**能力**读数，用来比较「改前 / 改后」
    # 只报前者会把偶发失败放大成能力退化；只报后者会把偶发失败洗成达标。

    @property
    def repeat(self) -> int:
        """用例最多的尝试次数（`--repeat N` 的 N；单次运行为 1）。"""
        return max((item.attempts for item in self.results), default=1)

    @property
    def attempts_total(self) -> int:
        return sum(item.attempts for item in self.results)

    @property
    def attempts_passed(self) -> int:
        return sum(item.pass_count for item in self.results)

    @property
    def run_pass_rate(self) -> float:
        return (
            self.attempts_passed / self.attempts_total if self.attempts_total else 0.0
        )

    def flaky_cases(self) -> list[CaseResult]:
        """偶发用例：既过过也挂过。空列表 = 这份读数没有歧义。"""
        return [item for item in self.results if item.flaky]

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
            # repeat=1 时下面四项与上面三项取值一致，多报一遍是为了让
            # 「这份报告跑了几次」永远能自证 —— 报告必须自证身份，
            # 否则过几天没人分得清 28/30 是单次还是 3 次里稳定过的。
            "repeat": self.repeat,
            "attempts_total": self.attempts_total,
            "attempts_passed": self.attempts_passed,
            "run_pass_rate": round(self.run_pass_rate, 4),
            "flaky": [
                {
                    "case_id": item.case.id,
                    "pass_count": item.pass_count,
                    "attempts": item.attempts,
                }
                for item in self.flaky_cases()
            ],
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


def run_once(
    case: EvalCase,
    client_factory: Callable[[EvalCase], Any],
    *,
    index: int = 0,
) -> AttemptRecord:
    """跑**一次**用例，永不抛异常 —— 运行期异常记为失败并保留异常。

    被静默跳过的用例会让通过率虚高，所以这里宁可返回一个 passed=False 的记录。
    """
    started = datetime.now()
    try:
        client = client_factory(case)
        agent = make_agent(client)
        run = agent.run(case.question)
        grades = grade_case(case, run)
        failed = [g for g in grades if not g.passed and g.severity == "error"]
        return AttemptRecord(
            index=index,
            passed=not failed,
            grades=grades,
            run=run,
            elapsed_s=(datetime.now() - started).total_seconds(),
        )
    except Exception as error:  # noqa: BLE001 - 运行期异常 -> 失败，不跳过
        return AttemptRecord(
            index=index,
            passed=False,
            grades=[GradeResult("runner", False, traceback.format_exc(limit=4))],
            error=f"{type(error).__name__}: {error}",
            elapsed_s=(datetime.now() - started).total_seconds(),
        )


def aggregate_attempts(
    case: EvalCase, attempts: Sequence[AttemptRecord]
) -> CaseResult:
    """把同一个用例的 N 次尝试合成一条结论。

    代表结果取「**第一次失败**」而不是最后一次：
    失败明细要展示真实失败的那一次 —— 否则「3 次里挂 2 次、最后一次侥幸过了」
    会在报告里显示成一个通过的用例，把两次失败**抹掉**。
    全过时才取第一次。
    """
    if not attempts:
        raise ValueError("aggregate_attempts 需要至少一次尝试")

    representative = next(
        (item for item in attempts if not item.passed), attempts[0]
    )
    return CaseResult(
        case=case,
        passed=all(item.passed for item in attempts),
        grades=representative.grades,
        run=representative.run,
        error=representative.error,
        # 累计耗时：repeat 之后「这条用例花了多久」应是 N 次之和，
        # 不是最后一次的耗时（否则耗时统计会随 N 变小，看着像变快了）
        elapsed_s=sum(item.elapsed_s for item in attempts),
        tools_called=representative.tool_chain,
        answer=representative.answer_text,
        stop_reason=(
            representative.run.stop_reason
            if representative.run is not None
            else representative.stop_reason
        ),
        # ⚠️ 只跑过一次时**不挂** attempts_detail：单次报告的 JSON 里原本没有
        # 这个键，挂上一个只含一项的列表会让旧报告与新报告不再是同一种形状
        # （而且那一项与顶层字段完全重复）。repeat=1 必须保持旧语义。
        attempts_detail=list(attempts) if len(attempts) > 1 else [],
    )


def run_suite(
    cases: Sequence[EvalCase] | None = None,
    *,
    client_factory: Callable[[EvalCase], Any],
    backend: str = "unknown",
    on_result: Callable[[CaseResult], None] | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    repeat: int = 1,
    on_attempt: Callable[[EvalCase, AttemptRecord], None] | None = None,
) -> EvalReport:
    """跑一整套用例。

    client_factory 接收用例、返回该用例要用的 LLM 客户端。
    真实模型可以按用例构造（例如注入 few-shot 或调整温度）；
    脚本回放可以按用例返回预设的调用序列。

    `repeat > 1` 时每个用例跑 N 次（每次都是**全新的** agent 与会话），
    报告给出「N 次全过」的保守读数 + 尝试级通过率 + 偶发用例清单。
    ⚠️ 缺省 `repeat=1` 的行为与加这个参数之前**逐字段一致** ——
    基线、看板、CI 都建立在旧语义上，不能悄悄改。
    """
    if repeat < 1:
        raise ValueError(f"repeat 必须 >= 1，收到 {repeat}")

    selected = _select(cases or CASES, include, exclude)
    report = EvalReport(
        backend=backend,
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    for case in selected:
        attempts: list[AttemptRecord] = []
        for index in range(1, repeat + 1):
            record = run_once(case, client_factory, index=index)
            attempts.append(record)
            if on_attempt is not None:
                on_attempt(case, record)
            # repeat=1 时别多跑一轮：逐个试跑会在单次路径上引入
            # 「第 1 次失败就不必再试」这类新逻辑，而单次路径要保持原样。
        result = aggregate_attempts(case, attempts)
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
    if result.passed:
        mark = "PASS"
    elif result.flaky:
        # 偶发既不是「通过」也不是「稳定失败」—— 单看 PASS/FAIL 会把它两边的
        # 信息都丢掉：改成 PASS 就掩盖了它挂过，改成 FAIL 就看不出它过过。
        mark = "FLAKY"
    else:
        mark = "FAIL"
    line = f"[{mark}] {result.case.id}  ({result.case.category})"
    if result.attempts > 1:
        line += f"  {result.pass_count}/{result.attempts} 次通过"
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


def format_attempt(case: EvalCase, record: AttemptRecord) -> str:
    """重复跑时逐次打印 —— 90 次的长任务必须能实时看见进度与抖动。"""
    mark = "PASS" if record.passed else "FAIL"
    detail = ""
    if not record.passed:
        names = "、".join(g.grader for g in record.failures) or "运行异常"
        detail = f"  ← {names}"
    return f"    [{mark}] {case.id} 第 {record.index} 次{detail}"


__all__ = [
    "AttemptRecord",
    "CaseResult",
    "EvalReport",
    "aggregate_attempts",
    "by_category",
    "evaluate_run",
    "format_attempt",
    "format_progress",
    "make_agent",
    "run_once",
    "run_suite",
]
