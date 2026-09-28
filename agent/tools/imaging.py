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

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.planner import RoutePlan, classify_reachability
from .registry import ToolRegistry
from .session import NavSession

# 低于这个直径的路径在实际支气管镜操作中基本不可用
CLINICAL_NARROW_WARNING_MM = 2.0

# 器械外径二分的取值口径。抽成常量，是因为 `scan_device_fit` 与
# `plan_route` 的失败增强**都要做这件事** —— 两处各写一套二分迟早会走偏
# （改了一处忘了另一处），而且会给医生两个不一样的上限值。
DEVICE_SEARCH_LOW_MM = 0.1
DEVICE_SEARCH_HIGH_MM = 6.0
DEVICE_SEARCH_RESOLUTION_MM = 0.1
DEVICE_SEARCH_MAX_STEPS = 12

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


# ------------------------------------------------------------------ 规划失败的结构化归因


def _error_text(error: BaseException) -> str:
    """异常的展示文案。与 registry 兜异常时的写法保持一致，报告里不会有两种格式。"""
    return f"{type(error).__name__}: {error}"


def _search_max_device_diameter(
    ctx: NavSession,
    case_id: str,
    candidate_id: int,
    device_margin_mm: float,
    upper_mm: float = DEVICE_SEARCH_HIGH_MM,
) -> dict[str, Any]:
    """二分查找该候选**能通过的最大器械外径**。

    可行性对直径单调（越粗越难过），所以二分有效。

    返回：
      max_device_diameter_mm  最大可通过外径；None 表示「连最细的也不行」
      bounded_by_scan_limit   True 表示**限制来自解剖本身**（路径最窄处已不允许更粗的
                              器械），此时 max_device_diameter_mm 等于最窄处、再调大
                              扫描上界也不会有别的答案。**不是**「你给的上界太小」
      search_steps / resolution_mm
      minimum_diameter_on_route_mm / profile_used

    单独抽出来，是因为 `scan_device_fit` 与 `plan_route` 的失败增强都要用它。
    注意 `min_route_diameter` 是路径最窄处的**直径**，与「最大可通过外径」
    是两个量 —— 调用方报给医生时不要混。
    """
    low = DEVICE_SEARCH_LOW_MM
    if not ctx.is_reachable(case_id, candidate_id, low, device_margin_mm):
        return {
            "max_device_diameter_mm": None,
            "bounded_by_scan_limit": False,
            "search_steps": 0,
            "resolution_mm": DEVICE_SEARCH_RESOLUTION_MM,
            "minimum_diameter_on_route_mm": None,
            "profile_used": None,
        }

    # 二分的不变量：`high` 必须是**已知不可通过**的值。
    if not ctx.is_reachable(case_id, candidate_id, upper_mm, device_margin_mm):
        # 常见路径：请求上界本身就过不去，直接拿它当 high。
        # 与加下面那段之前**逐字段一致**（一次 is_reachable(upper) + 一次 is_reachable(low)）。
        high = upper_mm
    else:
        # 上界处就能过 → 需要一个更高的 high 才能收敛。
        #
        # ⚠️ 原来这里**直接返回**（bounded_by_scan_limit=True + 把 `max_diameter_mm`
        # 原样回显成答案，再附一句「调大再测一次」）。实测模型会自己传
        # `max_diameter_mm=3`（3mm 细镜的临床先验）→ 答案就成了「至少 3.0mm」
        # **而不是真值 5.46mm**，还要医生「调大参数重测」：答案错了题，评测却
        # 照样判通过（它只断言「调过 scan_device_fit」+ 数字可溯源，而 3.0
        # 恰好来自返回体）。现在改成**内部自动抬高上界继续收敛**，一次调用即给真值，
        # 不再依赖模型是否愿意重试（见 docs/评测区分度_同义改写实验.md §8.9 实测表）。
        # 代价：这条路径每次 +8~18s（工具内多 ~7 次规划探测），换来答案从
        # 「回显的参数」变成几何真值。
        #
        # 抬到哪：器械外径的**天然硬上界**是「路径最窄处的直径」——
        # 比最窄处还粗的器械在几何上不可能通过。
        probe = ctx.plan(case_id, candidate_id, None, upper_mm, device_margin_mm)
        ceiling = round(probe.metrics["minimum_diameter_mm"], 3)
        if ceiling <= upper_mm:
            # 请求上界已达到/超过最窄处 → 没有更粗的几何可能
            return {
                "max_device_diameter_mm": upper_mm,
                "bounded_by_scan_limit": True,
                "search_steps": 0,
                "resolution_mm": DEVICE_SEARCH_RESOLUTION_MM,
                "minimum_diameter_on_route_mm": ceiling,
                "profile_used": probe.profile_name,
            }
        if ctx.is_reachable(case_id, candidate_id, ceiling, device_margin_mm):
            # 连最窄处都能通过（余量为 0 之类的边界）→ 真值即最窄处，无须二分
            return {
                "max_device_diameter_mm": ceiling,
                "bounded_by_scan_limit": True,
                "search_steps": 1,
                "resolution_mm": DEVICE_SEARCH_RESOLUTION_MM,
                "minimum_diameter_on_route_mm": ceiling,
                "profile_used": probe.profile_name,
            }
        high = ceiling
    steps = 0
    while (
        high - low > DEVICE_SEARCH_RESOLUTION_MM
        and steps < DEVICE_SEARCH_MAX_STEPS
    ):
        mid = round((low + high) / 2.0, 3)
        if ctx.is_reachable(case_id, candidate_id, mid, device_margin_mm):
            low = mid
        else:
            high = mid
        steps += 1

    best = ctx.plan(case_id, candidate_id, None, low, device_margin_mm)
    return {
        "max_device_diameter_mm": low,
        "bounded_by_scan_limit": False,
        "search_steps": steps,
        "resolution_mm": DEVICE_SEARCH_RESOLUTION_MM,
        "minimum_diameter_on_route_mm": round(best.metrics["minimum_diameter_mm"], 3),
        "profile_used": best.profile_name,
    }


def _plan_failure(
    ctx: NavSession,
    case_id: str,
    candidate_id: int,
    error: BaseException,
    device_diameter_mm: float,
    device_margin_mm: float,
    candidate_count: int,
    candidate_ids: list[int],
) -> dict[str, Any]:
    """把 `plan_route` 的失败整理成**模型能直接照做**的结构化返回体。

    动机（2026-09-22 实测，见 `docs/评测区分度_同义改写实验.md` 第八节）：

    原来的失败体只有两个 key —— `{"ok": false, "error": "RuntimeError: 候选 2 …"}`。
    模型只拿到一句话，而工具描述里那句「失败就改用更细的器械重试」是**无条件**的。
    对「编号越界」这种失败，「换个参数重试」正好等于**静默换目标**：
    E14 的 trace 里，模型 step 2 的推理已经写对了
    「客户端编号 99 超出了有效范围…范围是 1 到 4」，最终回答却交付了
    3 号的完整路径 + 三维图，通篇不提 99 —— **知道，但没说**。

    所以修法不是再补一句提示词（试过，零收益且把 E01P/E02P 压坏，已回滚），
    而是让**返回体自己按原因分支**：`next_step` 随 `reason` 变，模型照抄即可。

    两种异常是可以区分的（`agent/core/case_loader.py` 与 `planner.py` 的约定）：
      越界            -> KeyError     （`candidate_mask` / `candidate`）
      无可行路径      -> RuntimeError （器械太粗，或气道本身不通）
    """
    if isinstance(error, KeyError):
        return {
            "ok": False,
            "error": _error_text(error),
            "failure": {
                "reason": "candidate_out_of_range",
                "candidate_id": candidate_id,
                "valid_candidate_id_range": [1, candidate_count],
                "available_candidate_ids": candidate_ids,
                "next_step": (
                    f"**不要**自己挑一个别的编号去规划。先把「{candidate_id} 号不存在、"
                    f"本病例可用编号是 1..{candidate_count}」告知医生，"
                    f"请医生指定要规划哪一个。"
                ),
                "hint": (
                    f"本病例只有 {candidate_count} 个结节候选"
                    f"（编号 1..{candidate_count}），没有 {candidate_id} 号。"
                ),
            },
        }

    # ---- RuntimeError：没有可行路径。先分清「器械太粗」还是「气道本身不通」，
    #      这两种的诊断结论完全不同，不能让模型自己去猜。
    if device_diameter_mm <= 0:
        # 连器械约束都没施加仍然规划不出来 → 换器械无用
        return {
            "ok": False,
            "error": _error_text(error),
            "failure": {
                "reason": "route_infeasible_by_airway",
                "candidate_id": candidate_id,
                "requested_device_diameter_mm": device_diameter_mm,
                "max_device_diameter_mm": None,
                "next_step": (
                    "该目标在当前气道分割下没有可行路径，**换更细的器械也不会变**。"
                    "请复核靶点位置或气道分割质量，并如实告知医生此路不通。"
                ),
                "hint": "未施加器械尺寸约束仍然没有路径，问题不在器械。",
            },
        }

    # 已知该外径过不去，所以上限一定 ≤ 请求值 —— 二分上界取请求值即可，
    # 比默认的 6.0mm 少搜一段（每次探测都是一次完整规划，不便宜）。
    upper = max(DEVICE_SEARCH_LOW_MM, min(DEVICE_SEARCH_HIGH_MM, device_diameter_mm))
    search = _search_max_device_diameter(
        ctx, case_id, candidate_id, device_margin_mm, upper_mm=upper
    )
    max_mm = search["max_device_diameter_mm"]

    if max_mm is None:
        return {
            "ok": False,
            "error": _error_text(error),
            "failure": {
                "reason": "route_infeasible_by_airway",
                "candidate_id": candidate_id,
                "requested_device_diameter_mm": device_diameter_mm,
                "max_device_diameter_mm": None,
                "next_step": (
                    "即使最细的器械（0.1mm）也无法规划出路径，问题不在器械尺寸，"
                    "而在气道本身。换器械无用，请复核靶点或气道分割。"
                ),
                "hint": "连 0.1mm 器械都无可行路径，属几何上不通。",
            },
        }

    return {
        "ok": False,
        "error": _error_text(error),
        "failure": {
            "reason": "device_too_thick",
            "candidate_id": candidate_id,
            "requested_device_diameter_mm": device_diameter_mm,
            "device_margin_mm": device_margin_mm,
            "max_device_diameter_mm": round(max_mm, 2),
            "minimum_diameter_on_route_mm": search["minimum_diameter_on_route_mm"],
            "search_steps": search["search_steps"],
            "search_resolution_mm": search["resolution_mm"],
            # ⚠️ 这个上限是**二分收敛到的某个「已实测可行」的点**，不是无限精度的真值：
            # 上界取得不同（比如请求 3.0mm vs 1.5mm），收敛点会差在分辨率以内
            # （实测同一候选得到 1.19 / 1.24 / 1.21）。所以别把它当唯一数值去对账，
            # 报告时也只说「约 X mm」。真值区间是 (X, X + resolution]。
            "precision_mm": search["resolution_mm"],
            "next_step": (
                f"把 device_diameter_mm 降到 {round(max_mm, 2)}mm 或更细，"
                f"再对**同一个候选 {candidate_id}** 调一次本工具"
                f"（安全余量 {device_margin_mm}mm 不变）。"
            ),
            "hint": (
                f"器械外径 {device_diameter_mm}mm 过不去；"
                f"该候选的路径最粗只能过约 {round(max_mm, 2)}mm。"
            ),
        },
    }


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
        "为指定结节候选规划一条经支气管路径。"
        "**这是该目标规划结论的唯一权威来源** —— 器械可通过性 device_passable、"
        "安全余量 minimum_clearance_mm、瓶颈位置、可达性分级与注意事项 warnings "
        "全都只在这里给出，rank_candidates 不返回这些。"
        "用户给了器械外径时必须把 device_diameter_mm 传进来（不要留默认值 0）。"
        "返回长度、最窄直径、最大转角、分叉序列等摘要。"
        "**规划成功时三维视图已随本次规划渲染好**，就在 viewer_path 里 ——"
        "把它原样写进回答即可，这次说「三维视图已生成」是真的；"
        "不要复述视图里的数字，也不必为同一目标再调 render_viewer。"
        "（viewer_error 非空说明这次没出成图，那就如实说没出图，别许诺链接。）"
        "规划失败会返回 ok=false，并在 `failure` 字段里给出 `reason` 与 `next_step`："
        "**先读 failure.next_step，按它做**。"
        "`reason=candidate_out_of_range` 表示编号不存在（可用范围在 "
        "`failure.valid_candidate_id_range`）—— **此时不要自己挑别的编号去规划**，"
        "要把越界事实与可用范围告诉医生；"
        "`reason=device_too_thick` 表示器械过粗（上限已在 "
        "`failure.max_device_diameter_mm`，用它以内的外径重试**同一个候选**）；"
        "`reason=route_infeasible_by_airway` 表示几何上不通，换器械无用。"
        "（换代价配置 profile 属于同一目标的重试，任何时候都允许。）",
        PlanArgs,
        # 与 rank_candidates 同批发出时，本调用会被循环自动延后一步 ——
        # 因为「规划哪个候选」必须先看到排序结果才能定，
        # 否则 candidate_id 只能是瞎猜（实测会猜中不可达候选）。
        after=("rank_candidates",),
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
        try:
            plan = ctx.plan(
                case_id, candidate_id, profile, device_diameter_mm, device_margin_mm
            )
        except (KeyError, RuntimeError) as error:
            # 失败也要**结构化** —— 见 `_plan_failure` 里记的现场。
            # 只接这两种异常：越界与「无可行路径」是可预期、可指导下一步的失败；
            # 其它异常继续往上抛，由 registry 兜住。否则「程序坏了」会被
            # 伪装成「路径不通」，模型会照着一条错的方向去改参数。
            if isinstance(error, KeyError) and 1 <= candidate_id <= case.candidate_count:
                # ⚠️ 编号合法却仍抛 KeyError —— 那不是越界，是别的地方出了错。
                # 这种情况**必须重抛**：如果硬说成「编号越界」，模型会去改一个
                # 本来正确的编号，等于我们亲手制造一次静默换目标。
                raise
            return _plan_failure(
                ctx,
                case_id,
                candidate_id,
                error,
                device_diameter_mm,
                device_margin_mm,
                case.candidate_count,
                [item.candidate_id for item in case.candidates],
            )
        baseline, baseline_note = _baseline_block(plan)

        # 折线数据留痕（供 artifacts 列表与 trace 使用）。
        # **不把返回的 id 带进 payload** —— 见下方 note 里的原因。
        ctx.artifacts.put(
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
            # 三维视图随规划一起给 —— 见 `_auto_view` 里记的现场。
            **_auto_view(ctx, case, plan, device_diameter_mm),
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
        # 二分搜索本体抽在 `_search_max_device_diameter` —— `plan_route` 的失败增强
        # 也要用它，两处必须是同一个答案，否则医生会拿到两个不同的上限值。
        found = _search_max_device_diameter(
            ctx, case_id, candidate_id, device_margin_mm, upper_mm=max_diameter_mm
        )
        max_mm = found["max_device_diameter_mm"]

        if found["bounded_by_scan_limit"]:
            # ⚠️ 这一支的含义**收紧了**：不是「你给的上界太小，请调大重试」，
            # 而是「限制来自解剖本身」—— 路径最窄处已经不允许更粗的器械，
            # 再调大 max_diameter_mm 也不会有别的答案。
            # 原来那句「调大再测一次」会把模型推成线性试（§8.9 实测 3 次连续调用），
            # 现在搜索已在工具内收敛，**不要把活推回给模型**。
            #
            # 注意这里返回 `max_mm`（= found 里的值），**不是** `max_diameter_mm`
            # （那是调用方传进来的扫描上界）—— 旧代码回显参数，让「上限」这个字段
            # 名不副实；现在它必须是真值。
            return {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "max_device_diameter_mm": max_mm,
                "bounded_by_scan_limit": True,
                "minimum_diameter_on_route_mm": found["minimum_diameter_on_route_mm"],
                "note": (
                    f"最粗可通过约 {max_mm}mm 外径器械"
                    f"（安全余量 {device_margin_mm}mm）；"
                    f"限制来自**解剖本身** —— 路径最窄处 "
                    f"{found['minimum_diameter_on_route_mm']}mm，"
                    f"再调大 max_diameter_mm 也不会得到更粗的结果，**不必重试**。"
                ),
            }

        if max_mm is None:
            return {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "max_device_diameter_mm": None,
                "note": "即使 0.1mm 器械也无法规划出路径，问题不在器械尺寸，而在气道本身",
            }

        return {
            "case_id": case_id,
            "candidate_id": candidate_id,
            "max_device_diameter_mm": round(max_mm, 2),
            "search_steps": found["search_steps"],
            "resolution_mm": found["resolution_mm"],
            "minimum_diameter_on_route_mm": found["minimum_diameter_on_route_mm"],
            "profile_used": found["profile_used"],
            "note": (
                f"该结节最粗可通过约 {round(max_mm, 2)}mm 外径器械"
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
            # 本工具**不额外存折线** —— 同一份数据 plan_route 已经存过了，
            # 再存一次只会在 artifacts 列表里多出一张重复的「路径结果」卡片。
            #
            # 也**不返回 id**：一个裸的 `route-001` 会被模型当成「可视化已生成」的
            # 句柄写进回答（实测过「三维视图已生成，ID 为 route-001」这句假话），
            # 所以只给一句正确且可执行的说明。
            "note": (
                "本工具只返回数值与拓扑结果，**不产出任何可查看的文件**。"
                "需要三维视图请调用 render_viewer。"
            ),
        }

    @registry.tool(
        "rank_candidates",
        "【筛选工具】对病例内所有结节候选各规划一次，按「好到达程度」排序，"
        "用于回答「哪个结节最容易取到」，或在某个目标不可达时给出替代目标。"
        "⚠️ 本工具**只用于挑目标**：它不返回器械可通过性 device_passable、"
        "安全余量 minimum_clearance_mm、瓶颈位置与注意事项 warnings，"
        "因此**不能替代 plan_route 的结论**。"
        "选定目标后，必须再用该目标调一次 plan_route（带上器械外径）拿完整结论。",
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
            # 在模型最容易「看到数字就下结论」的位置再声明一次工具边界。
            # 实测（2026-09-17，E01）模型拿到本表的 minimum_diameter_mm 后
            # 直接当规划结论用，跳过了 plan_route，还自己算了个余量 1.715（真值 1.758）。
            "note": (
                "本排序**只用于挑目标**：不返回器械可通过性、安全余量、瓶颈位置与注意事项。"
                f"请对选定目标再调一次 plan_route（device_diameter_mm={device_diameter_mm}）"
                "拿完整结论；也**不要**用本表的 minimum_diameter_mm 自行换算安全余量。"
            ),
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

        # viewer 文件已落盘，登记进 artifacts 供网页版的产物列表使用。
        # **返回的 id 不带进 payload** —— 见下方 viewer_file 的注释。
        ctx.artifacts.put(
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
            # 不给裸 id（`viewer-002` 这种），只给真文件。
            # 理由同 plan_route：模型拿到一个 `xxx-NNN` 就会把它写成链接，
            # `![](viewer-002)` 在浏览器里是一张破图。
            # viewer_path 会被命令行场景直接交代给用户；
            # 网页版则在前端把本地路径就地换成可点按钮（见 agent/web/static/index.html）。
            "viewer_file": Path(output.path).name,
            # 同一路径用同一批参数再渲染一次时不会重写文件（内容只差 generatedAt），
            # 这里如实说明，免得模型把「复用了旧文件」讲成「刚刚重新生成了」。
            "viewer_reused": output.reused,
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


def _auto_view(
    ctx: NavSession,
    case: Any,
    plan: Any,
    device_diameter_mm: float,
) -> dict[str, Any]:
    """规划成功时顺带把三维视图渲染出来，让回答里的链接**是真的**。

    ## 现场的两种修法

    2026-09-17 用户实测的回答结尾是：

        三维路径可视化文件已生成，路径折线数据已存入 artifact，ID 为 route-001。
        如需查看路径的三维可视化，请点击下方链接：
        ![](route-001)          ← 用户回了一句「连接呢」

    点不开，因为模型手里根本没有文件 —— `route-001` 只是个内部留痕 id。
    当时能走两条路：**别许诺**，或者**让许诺成真**。

    第一轮先走了「别许诺」（删掉 artifact_id、写一句「不产出任何可查看的文件」），
    结果模型老实了，但用户想要的东西还是没有：问一条路径，拿回一段文字。
    这说明真正的问题是**交付形态**，不是措辞 —— 规划结论和三维视图本来就该一起来。

    所以现在改成让许诺成真：规划成功就把视图渲染好，`viewer_path` 是磁盘上真实的
    单文件 HTML。模型照着念，用户点得开；网页版还会把它就地渲染成按钮
    （`agent/web/static/index.html` 的 markArtifacts）。

    ## 几处刻意的取舍

    - **不让渲染失败拖垮规划**：出图依赖 mesh 构建（skimage / scipy），
      而规划结论本身与它无关。失败就只记 `viewer_error`，并把「没出成图」写清楚，
      免得模型又凭 viewer_path 的存在去许诺。
    - **固定 mesh_step=1**：与 `outputs/viewers/` 那份入库演示产物同精度，
      同一组参数再跑演示会命中内容去重（`viewer_reused`），不产生无谓的 diff。
    - **`viewer_path` 而不是 artifact id**：没有任何工具吃 id 入参，
      给出去只会被模型写成死链。
    """
    try:
        from ..render.viewer import render_case_viewer

        output = render_case_viewer(case, plan, device_diameter_mm, mesh_step=1)
    except Exception as exc:  # noqa: BLE001 - 出图失败不该让规划结论失败
        return {
            "viewer_path": None,
            "viewer_error": f"{type(exc).__name__}: {exc}",
            "viewer_note": (
                "本次三维视图渲染失败，**没有生成任何文件**。"
                "回答里如实说明没出图，不要给出链接。"
            ),
        }

    # 登记进 artifacts，供网页版的产物列表使用（同一份折线已在上面 put 过 route）。
    ctx.artifacts.put(
        "viewer",
        str(output.path),
        meta={
            "case_id": case.case_id,
            "candidate_id": plan.candidate_id,
            "path": str(output.path),
            "size_mb": round(output.size_mb, 2),
        },
    )
    return {
        "viewer_path": str(output.path),
        "viewer_file": Path(output.path).name,
        "viewer_reused": output.reused,
        "viewer_note": (
            "三维视图已随本次规划渲染完成，把 viewer_path 原样交给用户即可 ——"
            "网页版会显示成可点开的按钮，命令行场景它本身就是一份离线可打开的 HTML。"
            "不要复述视图里的数字。"
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
