"""AirNav-Agent 核心层：病例加载与路径规划（无 GUI）。"""
from .case_loader import (
    CaseContext,
    NoduleCandidateInfo,
    list_local_cases,
    load_case,
    resolve_case_files,
)
from .planner import (
    RoutePlan,
    choose_target_nodes,
    classify_reachability,
    compare_with_baseline,
    plan_all_candidates,
    plan_candidate,
    resolve_profiles,
)

__all__ = [
    "CaseContext",
    "NoduleCandidateInfo",
    "RoutePlan",
    "list_local_cases",
    "load_case",
    "resolve_case_files",
    "choose_target_nodes",
    "classify_reachability",
    "compare_with_baseline",
    "plan_all_candidates",
    "plan_candidate",
    "resolve_profiles",
]
