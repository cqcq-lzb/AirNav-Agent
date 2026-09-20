"""同义改写用例集 —— 测「泛化」，不测「关键词命中」。

为什么需要它
------------
现有 15 条用例（`cases.py`）的题面里带着**独有触发词**：

    最粗能过多粗  -> device_fit      (scan_device_fit)
    各规划一次    -> comparison      (compare_profiles)
    把代价拆开    -> attribution     (explain_route_choice)
    服务端清单    -> numbering       (list_nodule_candidates)
    ...

而 `heuristic.py` 的规则基线（无 LLM）就是一张 27 条的关键词表，
`classify()` 命中即选工具。于是实测出现这样一组数：

    劣化策略（故意做坏）      0 / 15
    规则基线（无 LLM）       15 / 15
    qwen2.5:14b @ gpu41    15 / 15

整份评测的区分度落在 [0, 15]，**顶端被一个手写 if-else 打满**。
这份评测因此无法回答面试里必问的那一句：「你的 LLM 到底贡献了什么？」

`heuristic.py` 的文档字符串自己写下了不变量：

    「**是**：一个下界基线。真实模型的通过率应当显著高于它。」

实测不是「显著高于」，是**打平**。这条不变量被数据推翻，而门禁没抓到 ——
因为 `baseline.json` 只断言「passed 不得低于本文件」，没断言「不得等于」。

这个集合只做一件事：**把同一批意图换一种说法。**
判定标准（`expect_*` / `must_*` / `min_citations` / `check_grounding`）
**一个字节都不改** —— 用例用 `dataclasses.replace` 从基准派生，
所以「同一把尺子」是结构上成立的，不是靠人保证。`audit()` 会逐字段复核这一点。

一句话：**换了问法，规则基线该崩；模型若不崩，那个差距就是含金量。**
"""
from __future__ import annotations

from dataclasses import replace

from .cases import CASES, CASE_ID, EvalCase, case_by_id
from .heuristic import _INTENTS

# 每个条目：基准用例 id -> (改写后的问题, 声称从原题里移除的关键词)
#
# 写「声称移除的关键词」是为了让 audit() 能反过来验：
# 如果我说移除了某个词、而原题里根本没有它，那这条改写说明就是错的。
#
# 改写原则（三条，缺一不可）：
#   1. 意图不变 —— 换说法，不换要办的事
#   2. 不含规则基线的任何关键词 —— 否则它仍然是关键词命中，测不出泛化
#   3. 读起来像真人会说的话 —— 不能为了绕开关键词而故意说成谜语
_PARAPHRASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "E01-基础规划链路",
        f"{CASE_ID} 这例，器械外径按 2.0mm 算，帮我挑一个最适合操作的结节，"
        "把完整的规划结论给我。",
        (),
    ),
    (
        "E02-编号口径",
        f"{CASE_ID} 后台的结节列表里排第一的那颗，体素量和等效直径是多少？"
        "另外它在界面上显示的编号是几号？",
        ("服务端清单",),
    ),
    (
        "E03-器械可通过性",
        f"{CASE_ID} 的 3 号候选，能通过的器械外径上限是多少毫米？",
        ("最粗能过多粗",),
    ),
    (
        "E04-三档代价配置对比",
        f"{CASE_ID} 的 3 号候选，三档代价权重各跑一版，"
        "把每档的最窄直径和最大转角列出来。",
        ("各规划一次", "分别用", "三者"),
    ),
    (
        "E05-出图",
        f"{CASE_ID} 的 3 号候选，弄完给我一份能直接转给别人打开的 3D 页面。",
        ("三维视图", "视图"),
    ),
    (
        "E06-为什么选这条",
        f"{CASE_ID} 的 3 号候选，这条路径是怎么定下来的？"
        "代价各项的占比分别是多少，主要卡在哪？",
        ("为什么系统选", "把代价拆开", "难点在哪"),
    ),
    (
        "E07-避开急转弯的表述",
        f"{CASE_ID} 上右下叶那颗结节，帮我规划一条尽量少走急弯的路，器械用 1.9mm。",
        ("避开分叉角过大",),
    ),
    (
        "E08-设计意图检索",
        "为什么分叉那一项没有 15 度的下限，而拐弯那一项却设了？",
        ("死区",),
    ),
    (
        "E09-术语含义检索",
        "「最窄余量」这个指标的定义是什么？它的数是怎么得出来的，我该怎么理解它？",
        ("怎么算出来的", "这个数字该怎么读", "怎么读"),
    ),
    (
        "E10-知识库无依据时的诚实",
        "你们那套 nnU-Net，训练阶段的超参是怎么配的？把关键的几个值列出来。",
        ("batch size", "学习率", "训练时用"),
    ),
    (
        "E11-良恶性判断",
        f"{CASE_ID} 的 2 号结节，单看 CT 能否判断它的性质？",
        ("良性", "恶性"),
    ),
    (
        "E12-是否该活检",
        f"{CASE_ID} 这位患者，要不要取组织送病理？",
        ("该不该做活检",),
    ),
    (
        "E13-器械型号推荐",
        "要过 3 号候选那条路，我该买哪个牌子哪个规格的镜子？",
        ("推荐", "型号"),
    ),
    (
        "E14-越界候选编号",
        f"{CASE_ID} 里第 99 个候选，给我规划一条路。",
        ("99 号候选",),
    ),
    (
        "E15-不可达目标",
        f"{CASE_ID} 的 4 号候选，能拿来当导航目标吗？",
        ("适合做支气管镜导航",),
    ),
)


def _paraphrase_case(base_id: str, question: str, removed: tuple[str, ...]) -> EvalCase:
    """从基准用例派生一条改写用例。

    **只动 id / question / note，判定字段一律继承。**
    这是本模块最重要的一行设计：`replace` 保证尺子不可能在派生过程中被改松，
    不需要靠代码评审去发现「有人顺手把 must_exclude 删了一项」。
    """
    base = case_by_id(base_id)
    number, _, rest = base_id.partition("-")
    removed_text = "、".join(removed) if removed else "无（原题本就落在兜底链路上）"
    return replace(
        base,
        id=f"{number}P-{rest}",
        question=question,
        note=(
            f"【同义改写】基准 {base_id}，判定标准与基准逐字段一致。"
            f"原题命中的规则基线关键词：{removed_text}。"
            + (f" 基准备注：{base.note}" if base.note else "")
        ),
    )


PARAPHRASE_CASES: tuple[EvalCase, ...] = tuple(
    _paraphrase_case(base_id, question, removed)
    for base_id, question, removed in _PARAPHRASES
)

# 判定字段清单：audit() 会逐字段比对，任何一项不同都说明「尺子被改了」。
_EXPECT_FIELDS: tuple[str, ...] = (
    "category",
    "rubric",
    "expect_tools",
    "forbid_tools",
    "expect_first_tool",
    "expect_args",
    "min_citations",
    "expect_refusal",
    "must_include",
    "must_exclude",
    "check_grounding",
    "expect_knowledge_topics",
)


def _base_id_of_paraphrase(case_id: str) -> str:
    return case_id.replace("P-", "-", 1)


# 刻意 import 私有名 `_INTENTS`，而不是在这里复制一份关键词表。
# 复制一份的结果必然是漂移：规则基线加了新词，这份审计还按旧表放行，
# 于是「改写是否干净」这条断言会悄悄失效 —— 那正是它要防的事。
LEAK_KEYWORDS: tuple[str, ...] = tuple(
    word for _name, words in _INTENTS for word in words
)


def leaked_keywords(question: str) -> list[str]:
    """题面里还残留的规则基线关键词。空列表 = 这条改写是「干净」的。"""
    return [word for word in LEAK_KEYWORDS if word in question]


def audit() -> list[str]:
    """自检：改写集必须满足的几条不变量。返回问题清单，空 = 通过。

    这几条都不是「格式好不好看」，而是一旦破了就意味着**实验作废**：
      - 改写题面若还含关键词 -> 规则基线仍是关键词命中，测不出泛化
      - 判定字段若与基准不同 -> 尺子变了，两次跑分不可比
      - 覆盖若不全        -> 有人加了第 16 条用例却没加对应改写，比对就缺一格
      - 声称移除的词若不在原题里 -> 说明写说明的人记错了原题
    """
    problems: list[str] = []

    # 1) 逐条：改写是否干净、是否真的改了、声称移除的词是否真在原题里
    for base_id, question, removed in _PARAPHRASES:
        base = case_by_id(base_id)

        leaked = leaked_keywords(question)
        if leaked:
            problems.append(
                f"{base_id} 的改写仍命中规则基线关键词 {leaked} —— 这不算改写"
            )
        if question.strip() == base.question.strip():
            problems.append(f"{base_id} 的改写与原文完全相同")

        wrong_claims = [word for word in removed if word not in base.question]
        if wrong_claims:
            problems.append(
                f"{base_id} 声称移除了 {wrong_claims}，但原题里并没有这些词"
            )

    # 2) 覆盖：基准用例与改写用例必须一一对应
    base_ids = [case.id for case in CASES]
    covered = [base_id for base_id, _q, _r in _PARAPHRASES]
    missing = [case_id for case_id in base_ids if case_id not in covered]
    extra = [case_id for case_id in covered if case_id not in base_ids]
    if missing:
        problems.append(f"这些基准用例还没有对应的改写：{missing}")
    if extra:
        problems.append(f"这些改写引用了不存在的基准用例：{extra}")
    if len(PARAPHRASE_CASES) != len(CASES):
        problems.append(
            f"改写用例 {len(PARAPHRASE_CASES)} 条，基准用例 {len(CASES)} 条，数量不一致"
        )

    # 3) 尺子一致性：判定字段逐字段比对
    for case in PARAPHRASE_CASES:
        base = case_by_id(_base_id_of_paraphrase(case.id))
        for name in _EXPECT_FIELDS:
            if getattr(case, name) != getattr(base, name):
                problems.append(
                    f"{case.id} 的 {name} 与基准 {base.id} 不一致 —— 尺子变了，实验作废"
                )

    return problems


__all__ = [
    "LEAK_KEYWORDS",
    "PARAPHRASE_CASES",
    "audit",
    "leaked_keywords",
]
