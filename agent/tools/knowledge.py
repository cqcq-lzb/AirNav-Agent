"""知识检索与路径归因工具。

这两个工具一起支撑「为什么选这条」：

- `search_knowledge`：从领域知识库取回可引用的依据片段，带 `[KB-xx#n]` 引用号
- `explain_route_choice`：把代价拆成四项，并给出跨配置的量化取舍

设计上刻意把「查依据」和「算归因」分开：
归因给的是**数字证据**，检索给的是**文字依据**。分开之后，模型既不能拿
知识库里的通用说法冒充本病例的实测数字，也不能用本病例的数字去解释
「为什么系统这么设计」。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.attribution import decompose, describe_tradeoff
from ..core.planner import RoutePlan
from ..rag import get_retriever
from .registry import ToolRegistry
from .session import NavSession, normalize_profile

ProfileName = Literal["balanced", "wide_airway", "gentle_turn"]


class SearchArgs(BaseModel):
    query: str = Field(
        min_length=1,
        description=(
            "要查的知识点，用自然语言描述即可。例如「分叉项为什么没有死区」"
            "「器械选型的余量怎么定」「候选编号为什么对不上」"
        ),
    )
    top_k: int = Field(default=4, ge=1, le=8, description="返回片段数，默认 4")
    tags: list[str] | None = Field(
        default=None,
        description=(
            "可选，按标签过滤。可用标签例如：代价函数 / 代价配置 / 可达性 / 编号口径 / "
            "气道 / 规划 / 器械 / 边界"
        ),
    )


class ChoiceArgs(BaseModel):
    case_id: str = Field(description="病例编号，例如 LIDC_0089")
    candidate_id: int = Field(
        ge=1,
        description="结节候选编号，使用导航系统界面的客户端编号（从 1 开始）",
    )
    profile: ProfileName | None = Field(
        default=None,
        description=(
            "要归因的代价配置。不填则用系统自动选出的最优配置作为胜出方案，"
            "并把另外两档作为落选方案一起对比"
        ),
    )
    device_diameter_mm: float = Field(default=0.0, ge=0.0, le=20.0)
    device_margin_mm: float = Field(default=0.2, ge=0.0, le=5.0)
    include_knowledge: bool = Field(
        default=True,
        description="是否同时检索知识库，把代价项与分叉/转弯相关的设计依据一并返回",
    )


KNOWLEDGE_QUERIES = {
    "length": "长度项 weight_length 为什么要乘步长",
    "radius": "半径项平方反比 窄气道代价 安全余量",
    "curvature": "转弯项 15 度死区 为什么设死区",
    "branch": "分叉项 为什么分叉处转弯更贵 抖矩传递",
}


def register_knowledge_tools(registry: ToolRegistry) -> None:
    """注册知识检索与路径归因两个工具。"""

    @registry.tool(
        "search_knowledge",
        "检索本项目的领域知识库，返回带引用号（形如 [KB-01#3]）的资料片段。"
        "内容涵盖：路径代价函数与三档配置的设计意图、可达性分级口径、"
        "结节候选双编号口径、气道重建与拓扑记号、器械通过性、以及系统的能力边界。"
        "当需要解释「系统为什么这么设计」「某个数字的含义是什么」时必须调用它，"
        "不要凭常识回答。若返回空列表，说明知识库里没有相关依据，"
        "此时应当直接说明「知识库中没有相关依据」，不要编造。",
        SearchArgs,
    )
    def search_knowledge(
        query: str, top_k: int = 4, tags: list[str] | None = None
    ) -> dict[str, Any]:
        retriever = get_retriever()
        hits = retriever.search(query, top_k=top_k, tags=tags)
        if not hits:
            return {
                "query": query,
                "hits": [],
                "note": (
                    "知识库中没有与该问题相关的依据。"
                    "请直接说明没有依据，不要凭常识编造解释。"
                ),
            }
        return {
            "query": query,
            "count": len(hits),
            "hits": [hit.to_dict() for hit in hits],
            "citation_rule": (
                "引用时请带上引用号，例如「依据 [KB-01#5]」，"
                "并把片段内容用自己的话复述，不要整段照抄。"
            ),
        }

    @registry.tool(
        "explain_route_choice",
        "回答「为什么选这条路径」的定量归因工具。它把胜出路径的总代价拆成长度、"
        "半径、转弯、分叉四项，给出各项的绝对值与占比，并与另外两档代价配置的"
        "方案做对比，指出取舍发生在哪一项上。同时给出关键几何指标（长度、最窄直径、"
        "最窄余量、最大转角）。调用它会顺带检索知识库，返回代价项的设计依据。"
        "凡是要解释选路理由、比较不同偏好下的取舍、或说明某条路「难在哪」，"
        "都应当调用本工具，而不是自己解释分数。",
        ChoiceArgs,
    )
    def explain_route_choice(
        ctx: NavSession,
        case_id: str,
        candidate_id: int,
        profile: ProfileName | None = None,
        device_diameter_mm: float = 0.0,
        device_margin_mm: float = 0.2,
        include_knowledge: bool = True,
    ) -> dict[str, Any]:
        case = ctx.load(case_id)
        normalized = normalize_profile(profile)

        # 三档各自的最优方案 —— 用 best_per_profile 而不是全部 proposals，
        # 否则同一档会有多条候选方案混进来，对比表会变得不可读
        best_by_profile = ctx.best_per_profile(
            case_id, candidate_id, device_diameter_mm, device_margin_mm
        )

        if normalized is not None:
            winner_plan = ctx.plan(
                case_id, candidate_id, normalized, device_diameter_mm, device_margin_mm
            )
        else:
            winner_plan = ctx.plan(
                case_id, candidate_id, None, device_diameter_mm, device_margin_mm
            )

        winner = decompose(case, winner_plan)

        alternatives: list[dict[str, Any]] = []
        for name, plan in sorted(best_by_profile.items()):
            if name == winner_plan.profile_name:
                continue
            other = decompose(case, plan)
            alternatives.append(
                {
                    **other.to_dict(),
                    "metrics": _metrics(plan),
                    "tradeoff_vs_winner": describe_tradeoff(winner, other),
                }
            )

        payload: dict[str, Any] = {
            "case_id": case_id,
            "candidate_id": candidate_id,
            "profile_was_auto_selected": normalized is None,
            "winner": {
                **winner.to_dict(),
                "metrics": _metrics(winner_plan),
                "topology_signature": winner_plan.topology_signature,
                "branch_sequence": list(winner_plan.topology_tokens),
            },
            "alternatives": alternatives,
            "how_to_read": (
                "share_pct 是各项占该方案总加权代价的比例。主导项占比高说明"
                "这条路的困难集中在该项：长度项高=路远，半径项高=被迫走窄气道，"
                "转弯项高=弯多，分叉项高=在分叉处反复拐弯。"
                "把 winner 与 alternatives 的同一个指标对比，就能看出换配置的代价。"
            ),
            "note": (
                "verify_against_planner=true 表示这四项加权之和与规划器原函数"
                "逐边吻合，归因数字可信。"
            ),
        }

        if include_knowledge:
            retriever = get_retriever()
            queries = [KNOWLEDGE_QUERIES[winner.dominant]]
            queries.append("三档代价配置的适用场景与取舍")
            if winner.shares["curvature"] + winner.shares["branch"] > 0.15:
                queries.append("平缓转弯优先 分叉项 急转弯 操作难度")
            seen: set[str] = set()
            hits: list[dict[str, Any]] = []
            for query in queries:
                for hit in retriever.search(query, top_k=2):
                    if hit.chunk.kb_id in seen:
                        continue
                    seen.add(hit.chunk.kb_id)
                    hits.append(hit.to_dict())
            payload["knowledge"] = hits

        return payload


def _metrics(plan: RoutePlan) -> dict[str, Any]:
    metrics = plan.metrics
    clearance = metrics.get("minimum_clearance_mm")
    return {
        "route_length_mm": round(float(metrics["route_length_mm"]), 3),
        "minimum_diameter_mm": round(float(metrics["minimum_diameter_mm"]), 3),
        "minimum_clearance_mm": None if clearance is None else round(float(clearance), 3),
        "maximum_turn_angle_deg": round(float(metrics["maximum_turn_angle_deg"]), 2),
        "target_distance_mm": round(float(metrics["target_distance_mm"]), 3),
        "waypoint_count": int(metrics.get("waypoint_count", plan.waypoint_count)),
        "score": round(float(plan.score), 3),
        "device_passable": bool(metrics.get("device_passable", True)),
        "reachability": metrics.get("reachability"),
    }


__all__ = ["ChoiceArgs", "KNOWLEDGE_QUERIES", "SearchArgs", "register_knowledge_tools"]
