"""影像规划工具集。

这些是 Agent 真正能调用的能力。设计原则：

1. **工具返回摘要，大对象走 artifact** —— 路径折线动辄上千点，
   直接返回会把上下文窗口塞爆，只返回 id。
2. **主动给出 warnings** —— 不只报数字，还要指出「这条路径依赖修复气道」
   「最窄处 1.6mm，实际器械可能过不去」这类需要人注意的点。
3. **编号口径必须显式** —— 客户端编号与服务端 manifest 编号不一致，
   每个涉及候选的返回都同时给出两者，避免 Agent 张冠李戴。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.planner import RoutePlan, classify_reachability
from .registry import ToolRegistry
from .session import NavSession

# 低于这个直径的路径在实际支气管镜操作中基本不可用
CLINICAL_NARROW_WARNING_MM = 2.0

ProfileName = Literal["balanced", "wide_airway", "gentle_turn"]


# ------------------------------------------------------------------ 参数模型


class EmptyArgs(BaseModel):
    """无参数工具。"""


class CaseArgs(BaseModel):
    case_id: str = Field(description="病例编号，例如 LIDC_0089")


class CandidateArgs(CaseArgs):
    candidate_id: int = Field(
        ge=1,
        description="结节候选编号，使用导航系统界面的客户端编号（从 1 开始）",
    )


class PlanArgs(CandidateArgs):
    profile: ProfileName | None = Field(
        default=None,
        description=(
            "路径代价配置：balanced=平衡型（默认）、wide_airway=宽气道优先、"
            "gentle_turn=平缓转弯优先。不填则由系统自动在三种里选最优"
        ),
    )
    device_diameter_mm: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="器械外径（mm）。0 表示不加器械尺寸约束；常用值 1.0~3.0",
    )
    device_margin_mm: float = Field(
        default=0.2,
        ge=0.0,
        le=5.0,
        description="器械与气道壁的安全余量（mm），默认 0.2",
    )


class CompareArgs(CandidateArgs):
    device_diameter_mm: float = Field(default=0.0, ge=0.0, le=20.0)
    device_margin_mm: float = Field(default=0.2, ge=0.0, le=5.0)


class DeviceScanArgs(CandidateArgs):
    max_diameter_mm: float = Field(
        default=6.0,
        ge=0.5,
        le=20.0,
        description="扫描上界（mm）。系统会在 0.1~该值之间二分查找可通过的最大器械外径",
    )
    device_margin_mm: float = Field(default=0.2, ge=0.0, le=5.0)


class RankArgs(CaseArgs):
    device_diameter_mm: float = Field(default=0.0, ge=0.0, le=20.0)
    device_margin_mm: float = Field(default=0.2, ge=0.0, le=5.0)


class ViewerArgs(PlanArgs):
    mesh_step: int = Field(
        default=1,
        ge=1,
        le=4,
        description=(
            "气道表面网格的降采样步长。1=原始分辨率（最清晰，文件约 2MB）；"
            "2=体积和生成时间减半，细支气管仍然保留。一般用默认值即可"
        ),
    )
    open_browser: bool = Field(
        default=False,
        description="是否在系统默认浏览器里直接打开生成的视图",
    )


# ------------------------------------------------------------------ 辅助


def _device_block(plan: RoutePlan) -> dict[str, Any]:
    return {
        "diameter_mm": plan.device_diameter_mm,
        "margin_mm": plan.device_margin_mm,
        "required_radius_mm": round(plan.metrics.get("required_radius_mm", 0.0), 4),
    }


def _route_block(plan: RoutePlan) -> dict[str, Any]:
    metrics = plan.metrics
    return {
        "length_mm": round(metrics["route_length_mm"], 3),
        "waypoints": plan.waypoint_count,
        "minimum_diameter_mm": round(metrics["minimum_diameter_mm"], 3),
        "mean_diameter_mm": round(metrics["mean_diameter_mm"], 3),
        "max_turn_angle_deg": round(metrics["maximum_turn_angle_deg"], 3),
        "target_distance_mm": round(metrics["target_distance_mm"], 3),
        "device_passable": metrics.get("device_passable"),
        "minimum_clearance_mm": (
            round(metrics["minimum_clearance_mm"], 3)
            if metrics.get("minimum_clearance_mm") is not None
            else None
        ),
        "bottleneck_xyz_mm": plan.bottleneck_xyz_mm,
        "sharpest_turn": plan.sharpest_turn,
        "reachability": metrics.get("reachability"),
        "branches_crossed": len(plan.topology_tokens),
        "branch_sequence": plan.branch_sequence(),
        "topology_signature": plan.topology_signature,
        "repaired_voxels_on_route": {
            "bridge_added": metrics.get("bridge_voxels_on_route", 0),
            "gap_recovered": metrics.get("recovered_voxels_on_route", 0),
        },
        "entry_mode": plan.entry_mode,
    }


def _warnings(
    plan: RoutePlan,
    baseline_note: str | None = None,
) -> list[str]:
    """把数字翻译成需要注意的事项 —— Agent 解释路径时要引用这些。"""
    out: list[str] = []
    metrics = plan.metrics

    reach = metrics.get("reachability") or {}
    grade = reach.get("grade")
    if grade == "unreachable":
        out.append(
            f"靶点距气道 {reach.get('distance_mm')}mm，当前气道分割无法覆盖，"
            f"该路径终点并未真正到达结节，不可作为导航方案"
        )
    elif grade == "marginal":
        out.append(
            f"靶点距气道 {reach.get('distance_mm')}mm，偏离明显，需人工复核"
            f"气道分割是否完整、结节是否真的适合经支气管镜路径"
        )

    if metrics.get("device_passable") is False:
        out.append(
            f"器械外径 {plan.device_diameter_mm}mm + 余量 {plan.device_margin_mm}mm "
            f"超过最窄处半径 {metrics['minimum_radius_mm']:.3f}mm，器械无法通过"
        )
    elif plan.device_diameter_mm == 0 and metrics["minimum_diameter_mm"] < CLINICAL_NARROW_WARNING_MM:
        out.append(
            f"路径最窄处仅 {metrics['minimum_diameter_mm']:.3f}mm，"
            f"未施加器械约束；实际选用器械前需确认该处能否通过"
        )

    bridge = metrics.get("bridge_voxels_on_route", 0)
    recovered = metrics.get("recovered_voxels_on_route", 0)
    if bridge:
        out.append(
            f"路径经过 {bridge} 个桥接体素（属于分割后补出的连接段），"
            f"该段在原始 CT 上未必是真实管腔，需重点复核"
        )
    if recovered:
        out.append(
            f"路径经过 {recovered} 个远端修复体素，该段来自细小气道连通性修复"
        )

    if baseline_note:
        out.append(baseline_note)

    return out


def _baseline_block(plan: RoutePlan) -> tuple[dict[str, Any] | None, str | None]:
    """把与「原始气道」的对照整理成 Human/LLM 都能读的形式。"""
    baseline = plan.metrics.get("baseline_comparison")
    if not baseline:
        return None, None

    if not baseline.get("route_found"):
        return (
            baseline,
            "关键结论：优化前的原始气道在该器械约束下完全无可行路径，"
            "当前路径依赖气道修复/桥接步骤才成立",
        )

    metrics = plan.metrics
    note = None
    length_delta = metrics["route_length_mm"] - baseline["route_length_mm"]
    turn_delta = (
        metrics["maximum_turn_angle_deg"] - baseline["maximum_turn_angle_deg"]
    )
    diameter_delta = (
        metrics["minimum_diameter_mm"] - baseline["minimum_diameter_mm"]
    )
    parts: list[str] = []
    if abs(length_delta) > 1.0:
        parts.append(
            f"路径长度{'增加' if length_delta > 0 else '缩短'} {abs(length_delta):.1f}mm"
        )
    if abs(turn_delta) > 0.5:
        parts.append(
            f"最大转角{'增加' if turn_delta > 0 else '降低'} {abs(turn_delta):.1f}°"
        )
    if abs(diameter_delta) > 0.05:
        parts.append(
            f"最窄直径{'加宽' if diameter_delta > 0 else '收窄'} {abs(diameter_delta):.2f}mm"
        )
    if parts:
        note = "相比未修复的原始气道：" + "，".join(parts)

    return baseline, note


# ------------------------------------------------------------------ 工具实现


def register_tools(registry: ToolRegistry) -> None:
    """把所有工具注册进 registry。"""

    @registry.tool(
        "list_cases",
        "列出本机已下载的病例（stage4_package）。规划任何病例前先用它确认病例编号。",
        EmptyArgs,
    )
    def list_cases(ctx: NavSession) -> dict[str, Any]:
        cases = ctx.available_cases()
        return {
            "cases_root": str(ctx.root),
            "count": len(cases),
            "cases": [
                {
                    "case_id": item["case_id"],
                    "cached": item.get("cached", False),
                    "required_files_ok": item.get("required_ok", False),
                }
                for item in cases
            ],
        }

    @registry.tool(
        "inspect_case",
        "查看病例概况：体素尺寸、气道规模、中心线节点数、入口点模式、结节候选数量，"
        "以及候选编号对照表（客户端编号 vs 服务端清单编号）。"
        "如果病例来自服务器清单，务必先看编号对照表再指定 candidate_id。",
        CaseArgs,
    )
    def inspect_case(ctx: NavSession, case_id: str) -> dict[str, Any]:
        case = ctx.load(case_id)
        summary = case.summary()
        return {
            "case": summary,
            "id_mapping": case.id_mapping(),
            "id_mapping_note": (
                "candidate_id 为导航系统客户端编号（连通域标签升序）；"
                "server_candidate_id 为服务端 stage4_manifest.json 编号（按体积降序）。"
                "下游工具一律使用 client_candidate_id"
            ),
        }

    @registry.tool(
        "list_nodule_candidates",
        "列出病例内全部结节候选，含体积、等效直径、三维中心坐标与两种编号。",
        CaseArgs,
    )
    def list_nodule_candidates(ctx: NavSession, case_id: str) -> dict[str, Any]:
        case = ctx.load(case_id)
        return {
            "case_id": case.case_id,
            "count": case.candidate_count,
            "candidates": [item.to_dict() for item in case.candidates],
        }

    @registry.tool(
        "plan_route",
        "为指定结节候选规划一条经支气管路径。返回长度、最窄直径、最大转角、"
        "分叉序列、可达性分级、注意事项等摘要，并把路径折线存入 artifact（只返回 id）。"
        "规划失败会返回 ok=false 与原因，此时应改用更细的器械或换代价配置重试。",
        PlanArgs,
    )
    def plan_route(
        ctx: NavSession,
        case_id: str,
        candidate_id: int,
        profile: ProfileName | None = None,
        device_diameter_mm: float = 0.0,
        device_margin_mm: float = 0.2,
    ) -> dict[str, Any]:
        case = ctx.load(case_id)
        plan = ctx.plan(
            case_id, candidate_id, profile, device_diameter_mm, device_margin_mm
        )
        baseline, baseline_note = _baseline_block(plan)

        artifact_id = ctx.artifacts.put(
            "route",
            {
                "points_xyz_mm": plan.points_xyz_mm,
                "radii_mm": plan.radii_mm,
                "cumulative_mm": plan.cumulative_mm,
                "turn_angles_deg": plan.turn_angles_deg,
            },
            meta={
                "case_id": case.case_id,
                "candidate_id": candidate_id,
                "profile": plan.profile_name,
            },
        )

        candidate = case.candidate(candidate_id)
        return {
            "case_id": case.case_id,
            "candidate_id": candidate_id,
            "server_candidate_id": candidate.server_candidate_id,
            "profile": plan.profile_name,
            "profile_title": plan.profile_title,
            "device": _device_block(plan),
            "route": _route_block(plan),
            "baseline_comparison": baseline,
            "warnings": _warnings(plan, baseline_note),
            "artifact_id": artifact_id,
            "artifact_note": "路径折线数据已存入 artifact，渲染三维视图时用 render_viewer 引用该 id",
        }

    @registry.tool(
        "compare_profiles",
        "把 balanced / wide_airway / gentle_turn 三种代价配置各规划一遍并对比。"
        "当医生提出「尽量走宽气道」或「避开急转弯」这类偏好时，用本工具给出量化取舍依据。",
        CompareArgs,
    )
    def compare_profiles(
        ctx: NavSession,
        case_id: str,
        candidate_id: int,
        device_diameter_mm: float = 0.0,
        device_margin_mm: float = 0.2,
    ) -> dict[str, Any]:
        best = ctx.best_per_profile(
            case_id, candidate_id, device_diameter_mm, device_margin_mm
        )
        rows: list[dict[str, Any]] = []
        for name, plan in best.items():
            metrics = plan.metrics
            rows.append(
                {
                    "profile": name,
                    "profile_title": plan.profile_title,
                    "score": round(plan.score, 3),
                    "length_mm": round(metrics["route_length_mm"], 3),
                    "minimum_diameter_mm": round(metrics["minimum_diameter_mm"], 3),
                    "max_turn_angle_deg": round(
                        metrics["maximum_turn_angle_deg"], 3
                    ),
                    "branches_crossed": len(plan.topology_tokens),
                    "target_distance_mm": round(metrics["target_distance_mm"], 3),
                }
            )
        rows.sort(key=lambda row: row["score"])

        if not rows:
            return {
                "ok": False,
                "error": (
                    f"候选 {candidate_id} 在器械 {device_diameter_mm}mm"
                    f"+余量 {device_margin_mm}mm 约束下三种配置均无可行路径"
                ),
            }

        best_row = rows[0]
        others = rows[1:]
        note = f"综合代价最低的是 {best_row['profile_title']}"
        if others:
            widest = max(rows, key=lambda r: r["minimum_diameter_mm"])
            flat = min(rows, key=lambda r: r["max_turn_angle_deg"])
            note += (
                f"；若优先保证管腔宽度应选 {widest['profile_title']}"
                f"（最窄 {widest['minimum_diameter_mm']}mm）"
                f"；若优先减少转向应选 {flat['profile_title']}"
                f"（最大转角 {flat['max_turn_angle_deg']}°）"
            )
        return {"case_id": case_id, "candidate_id": candidate_id, "rows": rows, "note": note}

    @registry.tool(
        "scan_device_fit",
        "二分查找该结节路径能通过的最大器械外径（mm）。"
        "回答「这个结节最粗能过多粗的镜子」这类问题。"
        "注意：可行域单调，器械越粗越难通过，因此二分有效。",
        DeviceScanArgs,
    )
    def scan_device_fit(
        ctx: NavSession,
        case_id: str,
        candidate_id: int,
        max_diameter_mm: float = 6.0,
        device_margin_mm: float = 0.2,
    ) -> dict[str, Any]:
        case = ctx.load(case_id)
        resolution = 0.1

        if ctx.is_reachable(case_id, candidate_id, max_diameter_mm, device_margin_mm):
            probe = ctx.plan(
                case_id, candidate_id, None, max_diameter_mm, device_margin_mm
            )
            return {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "max_device_diameter_mm": max_diameter_mm,
                "bounded_by_scan_limit": True,
                "minimum_diameter_on_route_mm": round(
                    probe.metrics["minimum_diameter_mm"], 3
                ),
                "note": f"在扫描上界 {max_diameter_mm}mm 处即可通过，可提高上界继续测试",
            }

        low, high = 0.1, max_diameter_mm
        if not ctx.is_reachable(case_id, candidate_id, low, device_margin_mm):
            return {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "max_device_diameter_mm": None,
                "note": "即使 0.1mm 器械也无法规划出路径，问题不在器械尺寸，而在气道本身",
            }

        steps = 0
        while high - low > resolution and steps < 12:
            mid = round((low + high) / 2.0, 3)
            if ctx.is_reachable(case_id, candidate_id, mid, device_margin_mm):
                low = mid
            else:
                high = mid
            steps += 1

        best = ctx.plan(case_id, candidate_id, None, low, device_margin_mm)
        return {
            "case_id": case_id,
            "candidate_id": candidate_id,
            "max_device_diameter_mm": round(low, 2),
            "search_steps": steps,
            "resolution_mm": resolution,
            "minimum_diameter_on_route_mm": round(
                best.metrics["minimum_diameter_mm"], 3
            ),
            "profile_used": best.profile_name,
            "note": (
                f"该结节最粗可通过约 {round(low, 2)}mm 外径器械"
                f"（安全余量 {device_margin_mm}mm）"
            ),
        }

    @registry.tool(
        "explain_route",
        "给出路径的完整可解释信息：分叉序列逐级展开、最窄处与最急转弯的位置、"
        "与未修复原始气道的对照、可达性分级与建议。"
        "当需要向医生解释「为什么选这条路」时调用本工具，不要自己编造理由。",
        PlanArgs,
    )
    def explain_route(
        ctx: NavSession,
        case_id: str,
        candidate_id: int,
        profile: ProfileName | None = None,
        device_diameter_mm: float = 0.0,
        device_margin_mm: float = 0.2,
    ) -> dict[str, Any]:
        case = ctx.load(case_id)
        plan = ctx.plan(
            case_id, candidate_id, profile, device_diameter_mm, device_margin_mm
        )
        baseline, baseline_note = _baseline_block(plan)
        candidate = case.candidate(candidate_id)

        reach = plan.metrics.get("reachability") or classify_reachability(
            plan.metrics["target_distance_mm"]
        )

        # 逐级分叉序列，标出每一跳跨过的段与分叉点
        tokens = plan.branch_sequence()
        hops: list[dict[str, Any]] = []
        for index in range(0, len(tokens) - 1, 2):
            segment = tokens[index] if index < len(tokens) else None
            junction = tokens[index + 1] if index + 1 < len(tokens) else None
            hops.append({"order": index // 2 + 1, "segment": segment, "junction": junction})

        return {
            "case_id": case.case_id,
            "candidate_id": candidate_id,
            "server_candidate_id": candidate.server_candidate_id,
            "nodule": {
                "volume_mm3": round(candidate.volume_mm3, 3),
                "equivalent_diameter_mm": round(
                    candidate.equivalent_diameter_mm, 3
                ),
                "center_xyz_mm": [round(v, 3) for v in candidate.center_xyz_mm],
            },
            "profile": {
                "name": plan.profile_name,
                "title": plan.profile_title,
                "why": _profile_rationale(plan),
            },
            "device": _device_block(plan),
            "route": _route_block(plan),
            "branch_hops": hops,
            "baseline_comparison": baseline,
            "reachability": reach,
            "warnings": _warnings(plan, baseline_note),
            "artifact_id": ctx.artifacts.put(
                "route",
                {
                    "points_xyz_mm": plan.points_xyz_mm,
                    "radii_mm": plan.radii_mm,
                    "cumulative_mm": plan.cumulative_mm,
                    "turn_angles_deg": plan.turn_angles_deg,
                },
                meta={
                    "case_id": case.case_id,
                    "candidate_id": candidate_id,
                    "profile": plan.profile_name,
                },
            ),
        }

    @registry.tool(
        "rank_candidates",
        "对病例内所有结节候选各规划一次，按「好到达程度」排序，"
        "用于回答「哪个结节最容易取到」或在某个目标不可达时给出替代目标。",
        RankArgs,
    )
    def rank_candidates(
        ctx: NavSession,
        case_id: str,
        device_diameter_mm: float = 0.0,
        device_margin_mm: float = 0.2,
    ) -> dict[str, Any]:
        case = ctx.load(case_id)
        rows: list[dict[str, Any]] = []

        for candidate in case.candidates:
            base = {
                "candidate_id": candidate.candidate_id,
                "server_candidate_id": candidate.server_candidate_id,
                "volume_mm3": round(candidate.volume_mm3, 3),
                "equivalent_diameter_mm": round(
                    candidate.equivalent_diameter_mm, 3
                ),
            }
            try:
                plan = ctx.plan(
                    case_id,
                    candidate.candidate_id,
                    None,
                    device_diameter_mm,
                    device_margin_mm,
                )
            except RuntimeError as error:
                rows.append({**base, "reachable": False, "reason": str(error)})
                continue

            metrics = plan.metrics
            reach = metrics.get("reachability") or {}
            rows.append(
                {
                    **base,
                    "reachable": True,
                    "profile": plan.profile_name,
                    "length_mm": round(metrics["route_length_mm"], 3),
                    "minimum_diameter_mm": round(metrics["minimum_diameter_mm"], 3),
                    "max_turn_angle_deg": round(
                        metrics["maximum_turn_angle_deg"], 3
                    ),
                    "target_distance_mm": round(metrics["target_distance_mm"], 3),
                    "reachability_grade": reach.get("grade"),
                    "branches_crossed": len(plan.topology_tokens),
                }
            )

        order = {"adjacent": 0, "reachable": 1, "marginal": 2, "unreachable": 3}
        rows.sort(
            key=lambda row: (
                not row["reachable"],
                order.get(row.get("reachability_grade"), 9),
                row.get("length_mm", 1e9),
            )
        )
        return {
            "case_id": case_id,
            "device": {
                "diameter_mm": device_diameter_mm,
                "margin_mm": device_margin_mm,
            },
            "ranked": rows,
        }

    @registry.tool(
        "render_viewer",
        "把某条规划路径渲染成一个自包含的三维交互视图（单文件 HTML，离线可打开），"
        "内含气道表面、中心线、按器械余量着色的路径双管、最窄处标记、结节候选、"
        "分割修复段，以及腔内视角与沿路径飞行。"
        "回答里凡是需要让人看清路径走向、或者用户索要「图/三维/可视化/演示」时调用它。"
        "返回 viewer_path（绝对路径），直接把它给用户即可。",
        ViewerArgs,
    )
    def render_viewer(
        ctx: NavSession,
        case_id: str,
        candidate_id: int,
        profile: ProfileName | None = None,
        device_diameter_mm: float = 0.0,
        device_margin_mm: float = 0.2,
        mesh_step: int = 1,
        open_browser: bool = False,
    ) -> dict[str, Any]:
        # 局部导入：render 子包依赖 skimage / scipy，不加载时不该被拖进来
        from ..render.viewer import render_case_viewer, warnings_for

        case = ctx.load(case_id)
        plan = ctx.plan(
            case_id,
            candidate_id,
            profile,
            device_diameter_mm,
            device_margin_mm,
        )

        # 视图里的注意事项直接用渲染层的 warnings_for：
        # 它和工具层的 _warnings 口径一致，另外还带一句「工程研究演示」免责说明。
        # 关键是「对话里说的」和「视图里写的」必须是同一份清单。
        output = render_case_viewer(
            case,
            plan,
            device_diameter_mm,
            mesh_step=mesh_step,
        )

        artifact = ctx.artifacts.put(
            kind="viewer",
            payload=str(output.path),
            meta={
                "case_id": case.case_id,
                "candidate_id": plan.candidate_id,
                "path": str(output.path),
                "size_mb": round(output.size_mb, 2),
            },
        )

        if open_browser:
            _open_in_browser(output.path)

        return {
            "case_id": case.case_id,
            "candidate_id": plan.candidate_id,
            "server_candidate_id": next(
                (
                    c.server_candidate_id
                    for c in case.candidates
                    if c.candidate_id == plan.candidate_id
                ),
                None,
            ),
            **output.to_dict(),
            "artifact_id": artifact,
            "mesh_step": mesh_step,
            "contents": [
                "气道表面（半透明，可剖切）",
                "中心线（含分叉点层级）",
                "路径管腔 + 器械双管（器械管按余量逐点着色）",
                "最窄处标记",
                "结节候选（选中项高亮）",
                "分割修复段",
            ],
            "interactions": [
                "拖动旋转 / 右键平移 / 滚轮缩放",
                "视角预设：全局 / 正位 / 路径 / 最窄处 / 结节 / 腔内",
                "「自动演示」一键走完整套镜头，适合直接录屏",
                "腔内视角下可按「沿路径飞行」或拖时间轴逐点查看",
                "横断面剖切滑块、深色 / 浅色主题",
            ],
            "note": (
                "viewer 是单文件 HTML，不依赖网络，可直接打开或转发。"
                "把 viewer_path 原样给用户即可，不需要复述里面的数字。"
            ),
        }


def _open_in_browser(path) -> bool:
    """在系统默认浏览器里打开 viewer。失败不影响工具返回。"""
    import webbrowser

    try:
        return bool(webbrowser.open(path.resolve().as_uri()))
    except Exception:  # noqa: BLE001 - 打不开浏览器只是体验问题，不该让工具失败
        return False


def _profile_rationale(plan: RoutePlan) -> str:
    """解释这套代价配置在权衡什么 —— 权重来自 V1 的 profiles()。"""
    table = {
        "balanced": "平衡型：长度、管腔宽度、曲率、分支惩罚权重均衡，默认首选",
        "wide_airway": "宽气道优先：管腔宽度权重最高（radius 3.30），"
        "倾向于绕行以换取更宽的管腔，适合口径较大的器械",
        "gentle_turn": "平缓转弯优先：曲率与分支惩罚权重最高（curvature 0.84 / branch 1.92），"
        "主动避开分叉角大与转弯急的路径",
    }
    return table.get(plan.profile_name, "未定义的代价配置")


__all__ = ["register_tools"]
