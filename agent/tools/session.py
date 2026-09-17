"""会话状态：病例缓存、规划结果缓存、Artifact 存储。

三个职责：
1. 病例按需加载并常驻（配合磁盘缓存，重复加载 0.6s）
2. 规划结果按 (病例, 候选, 配置, 器械) 缓存 —— 一次 3 档对比要跑 18 次 A*，
   不缓存的话 Agent 多问一句就要重算 20 秒
3. Artifact 存储：路径折线、网格这些大对象不进上下文，只传 id
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..core.case_loader import (
    CaseContext,
    cases_root,
    list_local_cases,
    load_case,
)
from ..core.planner import RoutePlan, plan_candidate
from .registry import ArtifactStore

PROFILE_ALIASES = {
    "balanced": "balanced",
    "平衡型": "balanced",
    "平衡": "balanced",
    "wide_airway": "wide_airway",
    "宽气道优先": "wide_airway",
    "宽气道": "wide_airway",
    "gentle_turn": "gentle_turn",
    "平缓转弯优先": "gentle_turn",
    "平缓转弯": "gentle_turn",
    "避弯": "gentle_turn",
}


def normalize_profile(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip()
    if key in PROFILE_ALIASES:
        return PROFILE_ALIASES[key]
    lowered = key.lower()
    return PROFILE_ALIASES.get(lowered, lowered)


class NavSession:
    """一次 Agent 会话的全部状态。"""

    def __init__(
        self,
        root: str | Path | None = None,
        device_diameter_mm: float = 0.0,
        device_margin_mm: float = 0.2,
        use_cache: bool = True,
    ) -> None:
        self.root = Path(root) if root else cases_root()
        self.device_diameter_mm = float(device_diameter_mm)
        self.device_margin_mm = float(device_margin_mm)
        self.use_cache = use_cache
        self.artifacts = ArtifactStore()
        self._cases: dict[str, CaseContext] = {}
        self._plans: dict[tuple, RoutePlan] = {}
        self._all_proposals: dict[tuple, list[RoutePlan]] = {}

    # ------------------------------------------------------------ 病例

    def available_cases(self) -> list[dict[str, Any]]:
        return list_local_cases(self.root)

    def case_ids(self) -> list[str]:
        return [item["case_id"] for item in self.available_cases()]

    def resolve_case_id(self, value: str) -> str:
        """把用户写的病例名解析成真实目录名（容忍大小写与部分匹配）。"""
        if value in self._cases:
            return value
        ids = self.case_ids()
        if value in ids:
            return value

        lowered = value.strip().lower()
        exact = [i for i in ids if i.lower() == lowered]
        if exact:
            return exact[0]
        partial = [i for i in ids if lowered in i.lower()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            raise KeyError(f"病例名 {value} 有多个匹配：{partial}")
        raise KeyError(f"找不到病例 {value}，本地可选：{ids}")

    def load(self, case_id: str) -> CaseContext:
        resolved = self.resolve_case_id(case_id)
        if resolved in self._cases:
            return self._cases[resolved]

        matches = [c for c in self.available_cases() if c["case_id"] == resolved]
        if not matches:
            raise KeyError(f"找不到病例 {resolved}")
        package = Path(matches[0]["package_dir"])

        case = load_case(package, case_id=resolved, use_cache=self.use_cache)
        self._cases[resolved] = case
        return case

    def loaded_case_ids(self) -> list[str]:
        return list(self._cases)

    # ------------------------------------------------------------ 规划

    def _key(
        self,
        case_id: str,
        candidate_id: int,
        profile: str | None,
        device_diameter_mm: float | None,
        device_margin_mm: float | None,
    ) -> tuple:
        return (
            case_id,
            int(candidate_id),
            profile or "auto",
            round(float(device_diameter_mm if device_diameter_mm is not None else self.device_diameter_mm), 4),
            round(float(device_margin_mm if device_margin_mm is not None else self.device_margin_mm), 4),
        )

    def plan(
        self,
        case_id: str,
        candidate_id: int,
        profile: str | None = None,
        device_diameter_mm: float | None = None,
        device_margin_mm: float | None = None,
    ) -> RoutePlan:
        case = self.load(case_id)
        normalized = normalize_profile(profile)
        diameter = (
            self.device_diameter_mm if device_diameter_mm is None else device_diameter_mm
        )
        margin = self.device_margin_mm if device_margin_mm is None else device_margin_mm
        key = self._key(case_id, candidate_id, normalized, diameter, margin)

        if key in self._plans:
            return self._plans[key]

        best, proposals = plan_candidate(
            case,
            int(candidate_id),
            profile_name=normalized,
            device_diameter_mm=float(diameter),
            device_margin_mm=float(margin),
        )
        self._plans[key] = best
        self._all_proposals[key] = proposals
        return best

    def proposals(
        self,
        case_id: str,
        candidate_id: int,
        profile: str | None = None,
        device_diameter_mm: float | None = None,
        device_margin_mm: float | None = None,
    ) -> list[RoutePlan]:
        """拿到全部候选方案（靶点 × 代价配置），按 score 升序。"""
        self.plan(case_id, candidate_id, profile, device_diameter_mm, device_margin_mm)
        key = self._key(
            case_id,
            candidate_id,
            normalize_profile(profile),
            device_diameter_mm,
            device_margin_mm,
        )
        return self._all_proposals.get(key, [])

    def best_per_profile(
        self,
        case_id: str,
        candidate_id: int,
        device_diameter_mm: float | None = None,
        device_margin_mm: float | None = None,
    ) -> dict[str, RoutePlan]:
        """三档代价配置各自的最优方案。"""
        proposals = self.proposals(
            case_id, candidate_id, None, device_diameter_mm, device_margin_mm
        )
        out: dict[str, RoutePlan] = {}
        for plan in proposals:
            current = out.get(plan.profile_name)
            if current is None or (
                plan.score,
                plan.metrics["route_length_mm"],
            ) < (
                current.score,
                current.metrics["route_length_mm"],
            ):
                out[plan.profile_name] = plan
        return out

    def is_reachable(
        self,
        case_id: str,
        candidate_id: int,
        device_diameter_mm: float,
        device_margin_mm: float,
    ) -> bool:
        try:
            self.plan(
                case_id,
                candidate_id,
                None,
                device_diameter_mm,
                device_margin_mm,
            )
            return True
        except RuntimeError:
            return False

    def clear_plans(self) -> None:
        self._plans.clear()
        self._all_proposals.clear()

    # ------------------------------------------------------------ 元信息

    def status(self) -> dict[str, Any]:
        return {
            "cases_root": str(self.root),
            "available_cases": len(self.available_cases()),
            "loaded_cases": self.loaded_case_ids(),
            "cached_plans": len(self._plans),
            "artifacts": len(self.artifacts.keys()),
            "default_device": {
                "diameter_mm": self.device_diameter_mm,
                "margin_mm": self.device_margin_mm,
            },
        }
