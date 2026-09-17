"""无 GUI 的多代价 A* 路径规划。

严格复刻 interactive_navigation_stage4_v3.InteractiveNavigationWindow
.select_nodule_candidate 的规划段与 compare_with_baseline_route，
剥离 Qt 依赖，输出可序列化的 RoutePlan。

规划语义与 V1 完全一致：
  - 靶点候选：在中心线上取距结节掩膜最近的若干节点（去重间距 4mm）
  - 代价配置：balanced / wide_airway / gentle_turn 三选一（或全跑取最优）
  - 排序：score = A* 代价 + 2.0 * 靶点距离，相同则取路径更短者
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from . import geometry
from .case_loader import (
    DEFAULT_MAX_TARGET_NODES,
    DEFAULT_TARGET_DISTANCE_SLACK_MM,
    CaseContext,
)
from .navbridge import choose_entry_node_no_edt, nav

TARGET_DEDUP_MM = 4.0
TARGET_DISTANCE_CAP_MM = 60.0

# 退化路径守卫：A* 偶尔会返回「起点即终点」的单节点路径，
# 其 route_length_mm 为 0，但对上层 Agent 来说是一个虚假的成功。
# V1 GUI 只在界面上展示，不会察觉；Agent 必须把它拦掉。
MIN_VALID_ROUTE_MM = 1.0

# 可达性分级阈值（靶点到气道中心线的最近距离，单位 mm）
# 注意：这不是临床阈值，只是给 Agent 一个可解释的分级口径，
# 具体判断仍需医生结合 CT 与器械条件确认。
REACHABILITY_BANDS: tuple[tuple[float, str, str], ...] = (
    (15.0, "adjacent", "紧邻气道，支气管镜路径可达性好"),
    (30.0, "reachable", "位于气道邻近区域，可规划路径，需关注远端细小气道连通性"),
    (50.0, "marginal", "明显偏离气道，当前气道树的覆盖可能不足，需人工复核"),
    (float("inf"), "unreachable", "远离气道，当前分割结果无法覆盖，不建议作为支气管镜导航目标"),
)


def classify_reachability(target_distance_mm: float) -> dict[str, Any]:
    """把「靶点到中心线最近距离」翻译成 Agent 可解释的分级。"""
    for upper, grade, label in REACHABILITY_BANDS:
        if target_distance_mm <= upper:
            return {
                "grade": grade,
                "distance_mm": round(float(target_distance_mm), 3),
                "label": label,
            }
    return {"grade": "unknown", "distance_mm": None, "label": "无法判定"}


@dataclass
class RoutePlan:
    """一条规划好的路径（V1 RouteResult 的无 GUI 版本）。"""

    candidate_id: int
    profile_name: str
    profile_title: str
    topology_signature: str
    topology_tokens: tuple[str, ...]
    score: float
    target_node: int
    nodes: list[int]
    coords_zyx: np.ndarray
    points_xyz_mm: np.ndarray
    radii_mm: np.ndarray
    cumulative_mm: np.ndarray
    turn_angles_deg: np.ndarray
    entry_mode: str
    device_diameter_mm: float
    device_margin_mm: float
    metrics: dict[str, Any] = field(default_factory=dict)

    # ---- 序列化 ----

    def to_dict(self, include_geometry: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target_candidate_id": self.candidate_id,
            "profile": self.profile_name,
            "profile_title": self.profile_title,
            "topology_signature": self.topology_signature,
            "metrics": self.metrics,
            "entry_mode": self.entry_mode,
            "device_diameter_mm": self.device_diameter_mm,
            "device_margin_mm": self.device_margin_mm,
            "target_node": self.target_node,
        }
        if include_geometry:
            payload["route_points_xyz_mm"] = [
                [round(float(v), 4) for v in row] for row in self.points_xyz_mm
            ]
        return payload

    @property
    def waypoint_count(self) -> int:
        return int(self.points_xyz_mm.shape[0])

    @property
    def bottleneck_xyz_mm(self) -> list[float]:
        """最窄处坐标 —— 逐点展示时的重点位置。"""
        idx = int(np.argmin(self.radii_mm))
        return [round(float(v), 3) for v in self.points_xyz_mm[idx]]

    @property
    def sharpest_turn(self) -> dict[str, Any]:
        idx = int(np.argmax(self.turn_angles_deg))
        return {
            "angle_deg": round(float(self.turn_angles_deg[idx]), 2),
            "at_xyz_mm": [round(float(v), 3) for v in self.points_xyz_mm[idx]],
        }

    def branch_sequence(self) -> list[str]:
        return list(self.topology_tokens)


# --------------------------------------------------------------- 靶点选择


def _distance_to_mask(case: CaseContext, target_mask: np.ndarray) -> np.ndarray:
    """中心线各节点到结节掩膜的最近距离（mm）。

    等价于 V1 的 `ndi.distance_transform_edt(~target_mask, sampling=spacing)`，
    但不会分配整卷 882MB 的特征变换数组。
    为什么能这样替换、以及为什么结果逐位相同，见 `agent/core/geometry.py`。
    """
    return geometry.distance_to_mask_at(
        target_mask, case.centerline_coords, case.spacing_zyx
    )


def choose_target_nodes(
    case: CaseContext,
    target_mask: np.ndarray,
    max_target_nodes: int = DEFAULT_MAX_TARGET_NODES,
    slack_mm: float = DEFAULT_TARGET_DISTANCE_SLACK_MM,
) -> tuple[list[int], np.ndarray, float]:
    """与 V1 的 choose_target_nodes 逐行对应（距离计算方式见 _distance_to_mask）。"""
    values = _distance_to_mask(case, target_mask)

    nearest = float(values.min())
    eligible = np.flatnonzero(values <= min(nearest + slack_mm, TARGET_DISTANCE_CAP_MM))
    if eligible.size == 0:
        eligible = np.array([int(np.argmin(values))], dtype=np.int64)

    ranking = values[eligible] - 0.20 * case.centerline_radii[eligible]
    ordered = eligible[np.argsort(ranking)]

    selected: list[int] = []
    for raw_node in ordered:
        node = int(raw_node)
        if not selected:
            selected.append(node)
        else:
            separation = np.linalg.norm(
                case.centerline_points[selected] - case.centerline_points[node], axis=1
            )
            if float(separation.min()) >= TARGET_DEDUP_MM:
                selected.append(node)
        if len(selected) >= max_target_nodes:
            break

    if not selected:
        selected = [int(np.argmin(values))]
    return selected, values, nearest


def resolve_profiles(profile_name: str | Iterable[str] | None):
    """把自然语言侧选出的 profile 名字解析为 CostProfile 列表。"""
    all_profiles = {p.name: p for p in nav.profiles()}

    if profile_name is None:
        return list(all_profiles.values())

    names = [profile_name] if isinstance(profile_name, str) else list(profile_name)
    unknown = [n for n in names if n not in all_profiles]
    if unknown:
        raise KeyError(
            f"未知的代价配置 {unknown}，可选：{list(all_profiles)}"
            f"（中文别名：平衡型 / 宽气道优先 / 平缓转弯优先）"
        )
    return [all_profiles[n] for n in names]


# --------------------------------------------------------------- 主规划


def plan_candidate(
    case: CaseContext,
    candidate_id: int,
    profile_name: str | Iterable[str] | None = None,
    device_diameter_mm: float = 0.0,
    device_margin_mm: float = 0.2,
    max_target_nodes: int = DEFAULT_MAX_TARGET_NODES,
    slack_mm: float = DEFAULT_TARGET_DISTANCE_SLACK_MM,
    with_baseline: bool = True,
    min_route_length_mm: float = MIN_VALID_ROUTE_MM,
) -> tuple[RoutePlan, list[RoutePlan]]:
    """为指定结节候选规划路径。

    返回 (最优方案, 全部候选方案) —— 后者按 score 升序，可用于对比解释。

    min_route_length_mm 用于拦截退化路径（起点=终点、长度为 0），
    这类解在 V1 里会被当成成功返回，但对 Agent 是不可用的。
    """
    target_mask = case.candidate_mask(candidate_id)
    target_nodes, target_distances, _nearest = choose_target_nodes(
        case, target_mask, max_target_nodes=max_target_nodes, slack_mm=slack_mm
    )

    device_diameter = float(device_diameter_mm or 0.0)
    device_margin = float(device_margin_mm or 0.0)
    required_radius = (
        device_diameter / 2.0 + device_margin if device_diameter > 0 else 0.0
    )

    profiles = resolve_profiles(profile_name)

    proposals: list[RoutePlan] = []
    degenerate = 0
    for target_node in target_nodes:
        for profile in profiles:
            try:
                nodes, search_cost = nav.stateful_astar(
                    start_node=case.entry_node,
                    target_node=target_node,
                    adjacency=case.adjacency,
                    coords_zyx=case.centerline_coords,
                    points_xyz=case.centerline_points,
                    radii_mm=case.centerline_radii,
                    degrees=case.degrees,
                    spacing_zyx=case.spacing_zyx,
                    profile=profile,
                    device_diameter_mm=device_diameter,
                    device_margin_mm=device_margin,
                )
            except RuntimeError:
                continue

            node_array = np.asarray(nodes, dtype=np.int64)
            route_coords = case.centerline_coords[node_array]
            route_points = case.centerline_points[node_array]
            route_radii = case.centerline_radii[node_array]
            route_diameters = 2.0 * route_radii
            cumulative = nav.cumulative_distance(route_points)

            # 退化路径：没有实际位移，不能作为导航路径交付
            if (
                node_array.size < 2
                or float(cumulative[-1]) < float(min_route_length_mm)
            ):
                degenerate += 1
                continue

            turns = nav.smoothed_turn_angles(route_points)
            topology_tokens, topology_signature = nav.route_topology_signature(
                nodes, case.junction_labels, case.segment_labels
            )

            target_distance = float(target_distances[target_node])
            minimum_radius = float(route_radii.min())
            score = search_cost + 2.0 * target_distance

            metrics: dict[str, Any] = {
                "route_length_mm": float(cumulative[-1]),
                "minimum_radius_mm": minimum_radius,
                "minimum_diameter_mm": 2.0 * minimum_radius,
                "mean_radius_mm": float(route_radii.mean()),
                "mean_diameter_mm": float(route_diameters.mean()),
                "maximum_turn_angle_deg": float(turns.max()),
                "target_distance_mm": target_distance,
                "required_radius_mm": required_radius,
                "minimum_clearance_mm": (
                    minimum_radius - required_radius if device_diameter > 0 else None
                ),
                "device_passable": (
                    bool(minimum_radius >= required_radius)
                    if device_diameter > 0
                    else True
                ),
                "reachability": classify_reachability(target_distance),
            }

            proposals.append(
                RoutePlan(
                    candidate_id=candidate_id,
                    profile_name=profile.name,
                    profile_title=profile.title,
                    topology_signature=str(topology_signature),
                    topology_tokens=tuple(topology_tokens),
                    score=float(score),
                    target_node=int(target_node),
                    nodes=[int(n) for n in nodes],
                    coords_zyx=route_coords,
                    points_xyz_mm=route_points,
                    radii_mm=route_radii,
                    cumulative_mm=cumulative,
                    turn_angles_deg=turns,
                    entry_mode=case.entry_mode,
                    device_diameter_mm=device_diameter,
                    device_margin_mm=device_margin,
                    metrics=metrics,
                )
            )

    if not proposals:
        detail = (
            f"（器械外径 {device_diameter}mm，安全余量 {device_margin}mm；"
            f"已拦截 {degenerate} 条退化路径）"
        )
        raise RuntimeError(
            f"候选 {candidate_id} 在当前器械约束下没有可行路径{detail}"
        )

    proposals.sort(key=lambda item: (item.score, item.metrics["route_length_mm"]))
    best = proposals[0]

    for plan in proposals:
        plan.metrics["waypoint_count"] = plan.waypoint_count
        plan.metrics["route_valid"] = True
    best.metrics["degenerate_routes_rejected"] = degenerate

    # 修复/桥接体素统计（与 V1 一致）
    route_coords = tuple(best.coords_zyx.T)
    best.metrics["bridge_voxels_on_route"] = (
        int(np.count_nonzero(case.airway_added_mask[route_coords]))
        if case.airway_added_mask is not None
        else 0
    )
    best.metrics["recovered_voxels_on_route"] = (
        int(np.count_nonzero(case.airway_recovered_mask[route_coords]))
        if case.airway_recovered_mask is not None
        else 0
    )

    if with_baseline:
        comparison = compare_with_baseline(case, target_mask, device_diameter, device_margin)
        if comparison is not None:
            best.metrics["baseline_comparison"] = comparison

    return best, proposals


# ----------------------------------------------------------- 基线对照


def compare_with_baseline(
    case: CaseContext,
    target_mask: np.ndarray,
    device_diameter: float,
    device_margin: float,
) -> dict[str, Any] | None:
    """与 V1 的 compare_with_baseline_route 一致。

    基线 = 优化前的原始气道掩膜（airway_raw_baseline.nii.gz）。
    它回答「气道修复到底有没有让路径变好」。
    """
    if case.airway_baseline_mask is None:
        return None

    baseline_mask = case.airway_baseline_mask
    from skimage.morphology import skeletonize

    centerline = skeletonize(baseline_mask, method="lee").astype(bool)
    coords = np.argwhere(centerline).astype(np.int32)
    if coords.shape[0] == 0:
        return None

    points = nav.physical_points(coords, case.ct_image)
    radii = geometry.radius_at_points(baseline_mask, coords, case.spacing_zyx)
    adjacency, degrees = nav.build_graph(coords, baseline_mask.shape, case.spacing_zyx)
    entry_node, _mode = choose_entry_node_no_edt(
        coords, points, radii, case.entry_mask, case.spacing_zyx
    )

    target_distances = geometry.distance_to_mask_at(
        target_mask, coords, case.spacing_zyx
    )
    nearest = float(target_distances.min())
    eligible = np.flatnonzero(
        target_distances <= min(nearest + DEFAULT_TARGET_DISTANCE_SLACK_MM, TARGET_DISTANCE_CAP_MM)
    )
    ranking = target_distances[eligible] - 0.20 * radii[eligible]
    ordered = eligible[np.argsort(ranking)]

    target_nodes: list[int] = []
    for raw_node in ordered:
        node = int(raw_node)
        if not target_nodes:
            target_nodes.append(node)
        else:
            separation = np.linalg.norm(points[target_nodes] - points[node], axis=1)
            if float(separation.min()) >= TARGET_DEDUP_MM:
                target_nodes.append(node)
        if len(target_nodes) >= DEFAULT_MAX_TARGET_NODES:
            break
    if not target_nodes:
        target_nodes = [int(np.argmin(target_distances))]

    proposals: list[dict[str, Any]] = []
    for target_node in target_nodes:
        for profile in nav.profiles():
            try:
                nodes, search_cost = nav.stateful_astar(
                    start_node=entry_node,
                    target_node=target_node,
                    adjacency=adjacency,
                    coords_zyx=coords,
                    points_xyz=points,
                    radii_mm=radii,
                    degrees=degrees,
                    spacing_zyx=case.spacing_zyx,
                    profile=profile,
                    device_diameter_mm=device_diameter,
                    device_margin_mm=device_margin,
                )
            except RuntimeError:
                continue

            node_array = np.asarray(nodes, dtype=np.int64)
            route_points = points[node_array]
            route_radii = radii[node_array]
            cumulative = nav.cumulative_distance(route_points)
            turns = nav.smoothed_turn_angles(route_points)
            target_distance = float(target_distances[target_node])
            proposals.append(
                {
                    "score": float(search_cost + 2.0 * target_distance),
                    "profile_title": profile.title,
                    "route_length_mm": float(cumulative[-1]),
                    "minimum_diameter_mm": float(2.0 * route_radii.min()),
                    "maximum_turn_angle_deg": float(turns.max()),
                    "target_distance_mm": target_distance,
                }
            )

    if not proposals:
        return {
            "available": True,
            "route_found": False,
            "reason": "原始气道在当前器械约束下没有可行路径",
        }

    proposals.sort(key=lambda item: (item["score"], item["route_length_mm"]))
    result = proposals[0]
    result["available"] = True
    result["route_found"] = True
    return result


def plan_all_candidates(
    case: CaseContext,
    profile_name: str | None = None,
    device_diameter_mm: float = 0.0,
    device_margin_mm: float = 0.2,
) -> list[dict[str, Any]]:
    """为所有候选各规划一条最优路径（用于「帮我挑一个最好到的结节」）。"""
    rows: list[dict[str, Any]] = []
    for candidate in case.candidates:
        try:
            best, _ = plan_candidate(
                case,
                candidate.candidate_id,
                profile_name=profile_name,
                device_diameter_mm=device_diameter_mm,
                device_margin_mm=device_margin_mm,
                with_baseline=False,
            )
        except RuntimeError as error:
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "reachable": False,
                    "reason": str(error),
                    "volume_mm3": round(candidate.volume_mm3, 3),
                }
            )
            continue
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "reachable": True,
                "component_label": candidate.component_label,
                "volume_mm3": round(candidate.volume_mm3, 3),
                "profile": best.profile_name,
                "profile_title": best.profile_title,
                "route_length_mm": round(best.metrics["route_length_mm"], 3),
                "minimum_diameter_mm": round(best.metrics["minimum_diameter_mm"], 3),
                "maximum_turn_angle_deg": round(
                    best.metrics["maximum_turn_angle_deg"], 3
                ),
                "target_distance_mm": round(best.metrics["target_distance_mm"], 3),
                "reachability": best.metrics.get("reachability"),
                "bottleneck_xyz_mm": best.bottleneck_xyz_mm,
            }
        )
    rows.sort(key=lambda row: (not row["reachable"], row.get("route_length_mm", 1e9)))
    return rows
