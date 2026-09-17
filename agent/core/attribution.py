"""路径代价的分项归因。

「为什么选这条」如果只回答「因为它的总分最低」，等于没回答。总分是四项加权的
结果，同一个总分可能有完全不同的构成。这个模块把一条路径的代价**拆成四项**，
让人看到取舍真正发生在哪一项上。

实现方式
--------
不改 V1，也不重新规划，而是**沿已选出的路径重放代价函数**：
按 V1 `edge_cost` 的公式逐边累加 length / radius / curvature / branch 四项，
再乘上对应 profile 的权重。

为了保证拆出来的四项确实是 V1 算的那个数，`decompose` 会做逐边对账：
对每条边同时调用 V1 的 `nav.edge_cost`，比较「四项加权之和」与原函数返回值。
对账不通过就标 `verified=False` 并把最大偏差报出来 —— 宁可暴露不一致，
也不给一个看起来专业但和规划器对不上的解释。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .case_loader import CaseContext
from .navbridge import nav
from .planner import RoutePlan, resolve_profiles

# 与 V1 edge_cost 内部一致的常量
FREE_ANGLE_RAD = math.radians(15.0)
MIN_LOCAL_RADIUS_MM = 0.25

TERM_NAMES = ("length", "radius", "curvature", "branch")

TERM_LABELS = {
    "length": "长度项",
    "radius": "半径项（气道宽窄）",
    "curvature": "转弯项（非分叉处）",
    "branch": "分叉项（分叉处转弯）",
}

TERM_WEIGHT_ATTRS = {
    "length": "weight_length",
    "radius": "weight_radius",
    "curvature": "weight_curvature",
    "branch": "weight_branch",
}


@dataclass
class CostBreakdown:
    """一条路径的代价构成。"""

    profile: str
    profile_title: str
    weights: dict[str, float]
    raw: dict[str, float]  # 未加权的四项原始和
    weighted: dict[str, float]  # 加权后的四项
    edges: int
    junction_edges: int
    verified: bool = False
    max_deviation: float = 0.0
    total_weighted: float = 0.0

    @property
    def shares(self) -> dict[str, float]:
        """各项占总代价的比例（0~1）。"""
        if self.total_weighted <= 0:
            return {name: 0.0 for name in TERM_NAMES}
        return {name: self.weighted[name] / self.total_weighted for name in TERM_NAMES}

    @property
    def dominant(self) -> str:
        return max(TERM_NAMES, key=lambda name: self.weighted[name])

    def to_dict(self) -> dict[str, Any]:
        shares = self.shares
        return {
            "profile": self.profile,
            "profile_title": self.profile_title,
            "weights": {name: round(v, 3) for name, v in self.weights.items()},
            "weighted_cost": {name: round(self.weighted[name], 3) for name in TERM_NAMES},
            "share_pct": {name: round(shares[name] * 100, 1) for name in TERM_NAMES},
            "dominant_term": self.dominant,
            "dominant_label": TERM_LABELS[self.dominant],
            "edges": self.edges,
            "junction_edges": self.junction_edges,
            "verified_against_planner": self.verified,
            "max_deviation": round(self.max_deviation, 9),
            "total_weighted_cost": round(self.total_weighted, 3),
        }


def _edge_lookup(case: CaseContext) -> dict[tuple[int, int], float]:
    """(a, b) -> 边长 mm，取自邻接表，避免重算。"""
    table: dict[tuple[int, int], float] = {}
    adjacency = case.adjacency
    for node, neighbors in enumerate(adjacency):
        for neighbor, step_mm in neighbors:
            table[(int(node), int(neighbor))] = float(step_mm)
    return table


def decompose(
    case: CaseContext,
    plan: RoutePlan,
    *,
    verify: bool = True,
) -> CostBreakdown:
    """把一条已规划路径的代价拆成四项。"""
    profile = resolve_profiles(plan.profile_name)[0]
    weights = {
        name: float(getattr(profile, attr)) for name, attr in TERM_WEIGHT_ATTRS.items()
    }

    nodes = [int(node) for node in plan.nodes]
    coords = case.centerline_coords
    radii = case.centerline_radii
    degrees = case.degrees
    spacing = case.spacing_zyx
    edges = _edge_lookup(case)

    raw = {name: 0.0 for name in TERM_NAMES}
    deviation = 0.0
    junction_edges = 0

    for index in range(len(nodes) - 1):
        previous_node = nodes[index - 1] if index > 0 else -1
        current_node = nodes[index]
        next_node = nodes[index + 1]

        step_mm = edges.get((current_node, next_node))
        if step_mm is None:
            # 邻接表里没有这条边 —— 数据不一致，明确报错而不是猜一个值
            raise KeyError(
                f"路径里有邻接表中不存在的边 ({current_node} -> {next_node})，"
                "无法归因"
            )

        length_term = step_mm
        local_radius = float(min(radii[current_node], radii[next_node]))
        radius_term = step_mm * (2.0 / max(local_radius, MIN_LOCAL_RADIUS_MM)) ** 2.0

        angle = nav.transition_angle_rad(
            None if previous_node < 0 else coords[previous_node],
            coords[current_node],
            coords[next_node],
            spacing,
        )
        excess = max(0.0, angle - FREE_ANGLE_RAD)
        curvature_term = step_mm * (excess / math.pi) ** 2.0

        branch_term = 0.0
        if int(degrees[current_node]) >= 3:
            branch_term = step_mm * (angle / math.pi) ** 2.0
            junction_edges += 1

        raw["length"] += length_term
        raw["radius"] += radius_term
        raw["curvature"] += curvature_term
        raw["branch"] += branch_term

        if verify:
            reference = nav.edge_cost(
                previous_node,
                current_node,
                next_node,
                step_mm,
                coords,
                radii,
                degrees,
                spacing,
                profile,
                float(plan.device_diameter_mm),
                float(plan.device_margin_mm),
            )
            mine = (
                weights["length"] * length_term
                + weights["radius"] * radius_term
                + weights["curvature"] * curvature_term
                + weights["branch"] * branch_term
            )
            if reference is not None:
                deviation = max(deviation, abs(float(reference) - float(mine)))

    weighted = {name: weights[name] * raw[name] for name in TERM_NAMES}
    total = sum(weighted.values())

    return CostBreakdown(
        profile=plan.profile_name,
        profile_title=plan.profile_title,
        weights=weights,
        raw=raw,
        weighted=weighted,
        edges=max(len(nodes) - 1, 0),
        junction_edges=junction_edges,
        verified=verify and deviation < 1e-6,
        max_deviation=deviation,
        total_weighted=total,
    )


def compare_plans(
    case: CaseContext,
    plans: dict[str, RoutePlan],
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """把多条（通常三档）路径的代价构成放在一起比较。"""
    breakdowns = {
        name: decompose(case, plan, verify=verify) for name, plan in plans.items()
    }
    return {
        name: {
            **breakdown.to_dict(),
            "metrics": _metrics_subset(plan),
        }
        for name, (breakdown, plan) in zip(breakdowns, plans.items())
    }


def _metrics_subset(plan: RoutePlan) -> dict[str, Any]:
    metrics = plan.metrics
    return {
        "route_length_mm": round(float(metrics["route_length_mm"]), 3),
        "minimum_diameter_mm": round(float(metrics["minimum_diameter_mm"]), 3),
        "minimum_clearance_mm": (
            None
            if metrics.get("minimum_clearance_mm") is None
            else round(float(metrics["minimum_clearance_mm"]), 3)
        ),
        "maximum_turn_angle_deg": round(float(metrics["maximum_turn_angle_deg"]), 2),
        "target_distance_mm": round(float(metrics["target_distance_mm"]), 3),
        "waypoint_count": int(metrics.get("waypoint_count", plan.waypoint_count)),
        "score": round(float(plan.score), 3),
        "topology_signature": plan.topology_signature,
    }


def describe_tradeoff(
    winner: CostBreakdown,
    challenger: CostBreakdown,
) -> str:
    """用一句话说清两条路径的取舍点在哪里。"""
    deltas = {
        name: winner.weighted[name] - challenger.weighted[name] for name in TERM_NAMES
    }
    key = max(deltas, key=lambda name: abs(deltas[name]))
    share = winner.shares[key] * 100
    return (
        f"相对 {challenger.profile_title}，{winner.profile_title} 的主要差异在"
        f"{TERM_LABELS[key]}（加权代价相差 {deltas[key]:+.3f}），"
        f"该项占胜出方案总代价的 {share:.1f}%"
    )


__all__ = [
    "CostBreakdown",
    "TERM_LABELS",
    "TERM_NAMES",
    "compare_plans",
    "decompose",
    "describe_tradeoff",
]
