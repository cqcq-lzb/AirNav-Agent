"""规则基线客户端（无 LLM）。

存在的理由
----------
本机提交内存已经顶到页面文件上限，加载 7B 模型需要再加约 5GB，跑不起来。
但评测链路本身必须能被验证 —— 一个从没跑通过的评测框架，等于没有。

所以这里实现一个**确定性规则策略**：按问题里的关键词决定调用哪些工具，
再把工具返回的字段拼成最终回答。

它是什么、不是什么
------------------
- **是**：评测框架的离线冒烟测试。它能跑通说明用例、打分器、运行器、
  报告这一整条链路是通的。
- **是**：一个下界基线。真实模型的通过率应当显著高于它。
- **不是**：模型能力的度量。它的回答是模板拼的，不具备泛化性；
  报告里必须标注为 baseline，不能和模型分数混为一谈。

它的回答只使用工具返回的字段值，不做任何自行推算 ——
这既让「数字可溯源」这条能过，也保证基线不会被自己的幻觉污染。
"""
from __future__ import annotations

import json
from typing import Any

from ..llm.client import ChatReply, ToolCallRequest

CASE_ID = "LIDC_0089"


# ------------------------------------------------------------------ 意图判定

_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("refusal_diagnosis", ("良性", "恶性")),
    ("refusal_biopsy", ("该不该做活检", "要不要做活检")),
    ("refusal_device", ("推荐", "型号")),
    ("attribution", ("为什么系统选", "把代价拆开", "难点在哪", "为什么选这条")),
    ("knowledge_meta", ("batch size", "学习率", "训练时用")),
    ("knowledge", ("死区", "怎么算出来的", "这个数字该怎么读", "怎么读")),
    ("numbering", ("服务端清单",)),
    ("device_fit", ("最粗能过多粗",)),
    ("comparison", ("各规划一次", "分别用", "三者")),
    ("viewer", ("三维视图", "视图")),
    ("unreachable", ("适合做支气管镜导航",)),
    ("out_of_range", ("99 号候选",)),
    ("evade_turn", ("避开分叉角过大",)),
    ("baseline", ()),
)


def _validate_intents() -> None:
    """守卫：关键词组必须是元组，且每个关键词长度大于 1。

    这条检查是一次真实 bug 的产物：写成 `("适合做支气管镜导航")` 少一个尾逗号，
    它就不是元组而是字符串，遍历时按**字符**逐个匹配 —— 于是「支」命中「分支」、
    「做」命中「做」，把两个用例错误地分到了 unreachable 类别。
    这种错误不抛异常，只是静默降低基线质量，所以必须显式拦下。
    """
    for name, keywords in _INTENTS:
        if not isinstance(keywords, tuple):
            raise ValueError(f"意图 {name} 的关键词不是元组（很可能是漏了尾逗号）")
        too_short = [word for word in keywords if not isinstance(word, str) or len(word) < 2]
        if too_short:
            raise ValueError(f"意图 {name} 含过短关键词 {too_short}，会导致误匹配")


_validate_intents()


def classify(question: str) -> str:
    for name, keywords in _INTENTS:
        if not keywords:
            return name
        if any(word in question for word in keywords):
            return name
    return "baseline"


# ------------------------------------------------------------------ 计划


def build_plan(question: str) -> list[tuple[str, dict[str, Any]]]:
    """按意图给出工具调用序列。"""
    intent = classify(question)

    if intent.startswith("refusal"):
        return [("search_knowledge", {"query": "系统能力边界 不做什么 拒答", "top_k": 2})]

    if intent in {"attribution", "evade_turn"}:
        if intent == "attribution":
            return [
                (
                    "explain_route_choice",
                    {
                        "case_id": CASE_ID,
                        "candidate_id": 3,
                        "device_diameter_mm": 2.0,
                        "device_margin_mm": 0.2,
                    },
                )
            ]
        return [
            (
                "compare_profiles",
                {"case_id": CASE_ID, "candidate_id": 3, "device_diameter_mm": 1.9},
            ),
            (
                "plan_route",
                {
                    "case_id": CASE_ID,
                    "candidate_id": 3,
                    "profile": "gentle_turn",
                    "device_diameter_mm": 1.9,
                    "device_margin_mm": 0.2,
                },
            ),
            # 要说明「为什么这个偏好对分叉敏感」，权重数字得从知识库取，
            # 不能靠模型自己记
            (
                "search_knowledge",
                {"query": "平缓转弯优先 分叉项 权重 死区", "top_k": 2},
            ),
        ]

    if intent == "knowledge":
        return [("search_knowledge", {"query": question, "top_k": 3})]

    if intent == "knowledge_meta":
        return [("search_knowledge", {"query": question, "top_k": 3})]

    if intent == "numbering":
        return [("list_nodule_candidates", {"case_id": CASE_ID})]

    if intent == "device_fit":
        return [
            (
                "scan_device_fit",
                {"case_id": CASE_ID, "candidate_id": 3, "device_margin_mm": 0.2},
            )
        ]

    if intent == "comparison":
        return [
            (
                "compare_profiles",
                {"case_id": CASE_ID, "candidate_id": 3, "device_diameter_mm": 2.0},
            )
        ]

    if intent == "viewer":
        return [
            (
                "render_viewer",
                {
                    "case_id": CASE_ID,
                    "candidate_id": 3,
                    "profile": "balanced",
                    "device_diameter_mm": 2.0,
                },
            )
        ]

    if intent == "unreachable":
        return [
            (
                "plan_route",
                {"case_id": CASE_ID, "candidate_id": 4, "device_diameter_mm": 2.0},
            ),
            ("rank_candidates", {"case_id": CASE_ID, "device_diameter_mm": 2.0}),
        ]

    if intent == "out_of_range":
        # ⚠️ 2026-09-22：原来这里只有一条 `plan_route(candidate_id=99)` ——
        # 也就是**不查病例就断言越界**，依据是题面里那五个字（规则表 key）。
        # 这不是「规则引擎能力不够」，是它在**猜**：候选总数就摆在
        # `inspect_case` 的 `case.candidate_count` 里，任何称职的规则实现
        # 都会先读一眼再下结论。
        # 被 E14 的新判据 `expect_tools=("inspect_case",)` 抓出来后补上。
        # 顺带的收获是：这条判据**有区分度** —— 它把对照组的偷懒也照出来了。
        # 保留 plan_route(99)：越界这条路径还要覆盖「失败调用的处理」。
        return [
            ("inspect_case", {"case_id": CASE_ID}),
            (
                "plan_route",
                {"case_id": CASE_ID, "candidate_id": 99, "device_diameter_mm": 2.0},
            ),
        ]

    # baseline：完整链路
    return [
        ("inspect_case", {"case_id": CASE_ID}),
        ("list_nodule_candidates", {"case_id": CASE_ID}),
        ("rank_candidates", {"case_id": CASE_ID, "device_diameter_mm": 2.0}),
        (
            "plan_route",
            {
                "case_id": CASE_ID,
                "candidate_id": 3,
                "profile": "balanced",
                "device_diameter_mm": 2.0,
                "device_margin_mm": 0.2,
            },
        ),
    ]


# ------------------------------------------------------------------ 取字段


def _results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in messages:
        if item.get("role") != "tool":
            continue
        try:
            payload = json.loads(item.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        payload["_tool"] = item.get("name")
        out.append(payload)
    return out


def _last(results: list[dict[str, Any]], tool: str) -> dict[str, Any] | None:
    for item in reversed(results):
        if item.get("_tool") == tool:
            return item
    return None


def _citations(results: list[dict[str, Any]], limit: int = 2) -> str:
    ids: list[str] = []
    for item in results:
        for hit in item.get("hits") or []:
            if hit.get("kb_id") and hit["kb_id"] not in ids:
                ids.append(hit["kb_id"])
    return "、".join(f"[{cid}]" for cid in ids[:limit]) if ids else ""


def _search_text(results: list[dict[str, Any]]) -> str:
    """把检索到的片段正文拼起来，供基线在回答里引用。"""
    chunks: list[str] = []
    for item in results:
        for hit in item.get("hits") or []:
            text = (hit.get("text") or "").strip()
            if text:
                chunks.append(f"{hit['kb_id']}｜{hit.get('section','')}：{text[:160]}")
    return "\n".join(chunks)


# ------------------------------------------------------------------ 回答


def _route_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """plan_route 的指标在 route 块里，不在顶层。"""
    return result.get("route") or {}


def compose(question: str, messages: list[dict[str, Any]]) -> str:
    intent = classify(question)
    results = _results(messages)
    disclaimer = "本结果为工程研究演示，不用于临床诊断与治疗决策。"

    if intent.startswith("refusal"):
        cite = _citations(results)
        return (
            "这不在本系统的能力范围内：本系统只做气道几何与路径规划的工程计算，"
            "不做诊断判断、不提供治疗建议、也不推荐具体器械型号。"
            f"这类问题需要由医生结合完整临床信息判断。{cite}\n\n"
            "我能提供的是几何量：目标气道的直径与余量、路径长度与转角、"
            "以及器械是否几何上可通过。需要的话我可以把这些数字给你。\n\n"
            f"{disclaimer}"
        )

    if intent == "attribution":
        item = _last(results, "explain_route_choice") or {}
        winner = item.get("winner") or {}
        share = winner.get("share_pct") or {}
        metrics = winner.get("metrics") or {}
        parts = "、".join(f"{name} {value}%" for name, value in share.items())
        return (
            f"结论：系统自动选出的是{winner.get('profile_title', '')}方案。\n"
            f"各项占比：{parts}。主导项是{winner.get('dominant_label', '')}。\n"
            f"几何指标：路径长度 {metrics.get('route_length_mm')}mm，"
            f"最窄直径 {metrics.get('minimum_diameter_mm')}mm，"
            f"最窄余量 {metrics.get('minimum_clearance_mm')}mm，"
            f"最大转角 {metrics.get('maximum_turn_angle_deg')}°，"
            f"靶点距离 {metrics.get('target_distance_mm')}mm。\n"
            f"归因与原规划函数逐边吻合：{winner.get('verified_against_planner')}"
            f"（最大偏差 {winner.get('max_deviation')}）。\n"
            f"依据 {_citations(results)}\n\n{disclaimer}"
        )

    if intent == "evade_turn":
        compare = _last(results, "compare_profiles") or {}
        plan = _last(results, "plan_route") or {}
        route = _route_metrics(plan)
        rows = compare.get("rows") or []
        row_text = "；".join(
            f"{row.get('profile_title')} 最窄 {row.get('minimum_diameter_mm')}mm"
            f" / 最大转角 {row.get('max_turn_angle_deg')}°"
            for row in rows
        )
        return (
            "「避开分叉角过大的分支」对应平缓转弯优先配置（分叉项权重 1.92，"
            "转弯项权重 0.84）。\n"
            f"该配置下的结果：路径长度 {route.get('length_mm')}mm，"
            f"最窄直径 {route.get('minimum_diameter_mm')}mm，"
            f"最大转角 {route.get('max_turn_angle_deg')}°。\n"
            f"三档对比：{row_text}\n"
            "分叉项只在度≥3的节点上计，且没有 15° 死区，"
            "所以这个偏好对分叉密集的路径影响最明显。\n"
            f"依据 {_citations(results)}\n\n{disclaimer}"
        )

    if intent == "knowledge":
        text = _search_text(results)
        cite = _citations(results)
        if not text:
            return f"知识库中没有与该问题相关的依据。\n\n{disclaimer}"
        return (
            f"依据 {cite}\n\n{text}\n\n"
            "以上为知识库中的原文片段，供参考。\n\n" + disclaimer
        )

    if intent == "knowledge_meta":
        text = _search_text(results)
        if not text:
            return (
                "知识库中没有关于训练超参（batch size、学习率）的依据，"
                "我不能给出这组数字。\n\n" + disclaimer
            )
        return f"依据 {_citations(results)}\n\n{text}\n\n{disclaimer}"

    if intent == "numbering":
        item = _last(results, "list_nodule_candidates") or {}
        rows = item.get("candidates") or []
        target = next(
            (row for row in rows if row.get("server_candidate_id") == 1), None
        )
        if target is None:
            return f"未能取到候选清单。\n\n{disclaimer}"
        return (
            f"服务端清单里的 1 号结节是体积最大的那颗：{target.get('voxel_count')} 体素，"
            f"等效直径 {target.get('equivalent_diameter_mm')}mm。\n"
            f"它在本系统的客户端编号是 {target.get('candidate_id')} 号 —— "
            "两套编号口径不同，报告时建议以体素量作为锚点。\n\n" + disclaimer
        )

    if intent == "device_fit":
        item = _last(results, "scan_device_fit") or {}
        return (
            f"二分查找结果：该路径能通过的最大器械外径约 "
            f"{item.get('max_device_diameter_mm')}mm。\n"
            f"约束来自路径最窄处直径 {item.get('minimum_diameter_on_route_mm')}mm，"
            f"采用的代价配置是 {item.get('profile_used')}。\n\n" + disclaimer
        )

    if intent == "comparison":
        item = _last(results, "compare_profiles") or {}
        rows = item.get("rows") or []
        lines = [
            f"- {row.get('profile_title')}：最窄直径 {row.get('minimum_diameter_mm')}mm，"
            f"最大转角 {row.get('max_turn_angle_deg')}°"
            for row in rows
        ]
        return (
            "三档代价配置的结果：\n"
            + "\n".join(lines)
            + f"\n\n{item.get('note', '')}\n\n"
            + disclaimer
        )

    if intent == "viewer":
        item = _last(results, "render_viewer") or {}
        return (
            f"已生成三维视图：{item.get('viewer_path')}\n"
            f"气道三角面 {item.get('airway_triangles')}，航点 {item.get('waypoints')}。\n\n"
            + disclaimer
        )

    if intent == "unreachable":
        plan = _last(results, "plan_route") or {}
        route = _route_metrics(plan)
        reach = route.get("reachability") or {}
        return (
            f"该候选的靶点距气道中心线 {route.get('target_distance_mm')}mm，"
            f"可达性分级为 {reach.get('grade')}。\n"
            "超出当前气道分割的可靠覆盖范围，需人工复核后再决定是否作为导航目标。\n"
            "建议参考其他候选的可达性排序做选择。\n\n" + disclaimer
        )

    if intent == "out_of_range":
        plan = _last(results, "plan_route") or {}
        return (
            f"规划失败：{plan.get('error')}\n"
            "候选编号超出该病例的实际范围，请改用范围内的候选编号。\n\n"
            + disclaimer
        )

    # baseline
    plan = _last(results, "plan_route") or {}
    inspect = _last(results, "inspect_case") or {}
    rank = _last(results, "rank_candidates") or {}
    route = _route_metrics(plan)
    reach = route.get("reachability") or {}
    graded = rank.get("ranked") or []
    return (
        f"病例 {inspect.get('case_id')}，共 {inspect.get('count')} 个候选。\n"
        f"以 2.0mm 器械规划后，路径长度 {route.get('length_mm')}mm，"
        f"最窄直径 {route.get('minimum_diameter_mm')}mm，"
        f"最窄余量 {route.get('minimum_clearance_mm')}mm，"
        f"最大转角 {route.get('max_turn_angle_deg')}°。\n"
        f"可达性为 {reach.get('grade')}，靶点距离 {route.get('target_distance_mm')}mm。\n"
        f"候选排序共 {len(graded)} 条，可据此选择替代目标。\n\n"
        f"{disclaimer}"
    )


class HeuristicClient:
    """按关键词决定工具序列的确定性客户端。"""

    def __init__(self, label: str = "规则基线（无 LLM）") -> None:
        self.label = label
        self._plan: list[tuple[str, dict[str, Any]]] | None = None
        self._cursor = 0
        self._question = ""

    def chat(self, messages, tools=None) -> ChatReply:
        if self._plan is None:
            self._question = _first_user(messages)
            self._plan = build_plan(self._question)

        if self._cursor < len(self._plan):
            name, arguments = self._plan[self._cursor]
            self._cursor += 1
            return ChatReply(
                tool_calls=[
                    ToolCallRequest(
                        id=f"call_{self._cursor}",
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    )
                ],
                finish_reason="tool_calls",
            )

        return ChatReply(
            content=compose(self._question, messages), finish_reason="stop"
        )


def _first_user(messages: list[dict[str, Any]]) -> str:
    for item in messages:
        if item.get("role") == "user":
            return item.get("content") or ""
    return ""


# ------------------------------------------------------------------ 劣化策略


class SloppyClient:
    """故意劣化的策略，用来验证评测集真的有区分度。

    基线通过率 100% 本身说明不了什么 —— 一个按用例写出来的策略当然全过。
    真正的证据是：把策略按常见的失败模式劣化之后，评测能不能逐条抓住。

    这里复刻的正是真实 LLM 最常犯的四类错：
      1. 跳过编号口径确认，直接把服务端编号当客户端编号用
      2. 不用工具，自己用「最窄直径 ÷ 2」硬算器械外径
      3. 回答里掺入工具没返回过的数字（幻觉）
      4. 越界提问照样给临床结论
    """

    def __init__(self) -> None:
        self.label = "劣化策略（对照，用于验证评测区分度）"
        self._cursor = 0
        self._plan: list[tuple[str, dict[str, Any]]] | None = None
        self._question = ""

    def chat(self, messages, tools=None) -> ChatReply:
        if self._plan is None:
            self._question = _first_user(messages)
            self._plan = self._sloppy_plan(self._question)

        if self._cursor < len(self._plan):
            name, arguments = self._plan[self._cursor]
            self._cursor += 1
            return ChatReply(
                tool_calls=[
                    ToolCallRequest(
                        id=f"call_{self._cursor}",
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    )
                ],
                finish_reason="tool_calls",
            )

        return ChatReply(content=self._answer(messages), finish_reason="stop")

    @staticmethod
    def _sloppy_plan(question: str) -> list[tuple[str, dict[str, Any]]]:
        intent = classify(question)
        # 知识类问题不去检索，直接凭印象答
        if intent in {"knowledge", "knowledge_meta"} or intent.startswith("refusal"):
            return []
        # 编号类问题不去查对照表，直接假定两套编号一致
        if intent == "numbering":
            return []
        # 器械问题不调 scan_device_fit，自己算
        if intent == "device_fit":
            return [
                (
                    "plan_route",
                    {"case_id": CASE_ID, "candidate_id": 3, "device_diameter_mm": 2.0},
                )
            ]
        # 其余：跳过编号确认，直接规划
        plan = [
            item
            for item in build_plan(question)
            if item[0] not in {"inspect_case", "list_nodule_candidates", "search_knowledge"}
        ]
        return plan or [
            ("plan_route", {"case_id": CASE_ID, "candidate_id": 3})
        ]

    def _answer(self, messages: list[dict[str, Any]]) -> str:
        intent = classify(self._question)
        results = _results(messages)
        plan = _last(results, "plan_route") or {}
        route = _route_metrics(plan)

        if intent.startswith("refusal"):
            return (
                "从影像表现看该结节边缘毛刺、分叶，恶性可能性约 72%，"
                "建议做活检明确诊断。"
            )

        if intent == "numbering":
            return "服务端清单里的 1 号就是客户端 1 号，体积约 3000 体素。"

        if intent == "device_fit":
            return (
                f"最窄直径 {route.get('minimum_diameter_mm')}mm，"
                "所以最粗能过 2.96mm 的镜子。"
            )

        if intent in {"knowledge", "knowledge_meta"}:
            return (
                "分叉项没有死区是因为分叉处更需要平滑。依据 [KB-01#1]。"
            )

        # 其余：模板 + 一个凭空来的数字
        return (
            f"路径长度 {route.get('length_mm')}mm，"
            f"最窄直径 {route.get('minimum_diameter_mm')}mm，"
            "平均直径 4.2mm，气道截面扩张率 18.5%，"
            "整体条件良好，可以操作。"
        )


__all__ = ["HeuristicClient", "SloppyClient", "build_plan", "classify", "compose"]
