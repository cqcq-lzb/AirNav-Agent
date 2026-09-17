"""Agent 评测层：用例、可判定打分器、运行器与自检。

对齐关系：
  cases.py     —— 问题与期望（硬指标）
  graders.py   —— 六类打分器（工具选型/参数/可溯源/引用/拒答/稳健性）
  harness.py   —— 逐用例隔离运行 + 汇总
  selftest.py  —— 用合成轨迹验证打分器本身有效（评测的评测）
  report.py    —— 生成可读报告
"""
from __future__ import annotations

from .cases import CASES, EvalCase, by_category, case_by_id
from .graders import GradeResult, grade_case
from .harness import CaseResult, EvalReport, evaluate_run, make_agent, run_suite

__all__ = [
    "CASES",
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "GradeResult",
    "by_category",
    "case_by_id",
    "evaluate_run",
    "grade_case",
    "make_agent",
    "run_suite",
]
