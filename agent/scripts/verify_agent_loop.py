"""Agent 循环控制流的针对性验证（不需要 LLM）。

针对的是一类**很隐蔽**的失败：模型输出「接下来，我将调用 plan_route 重新规划路径。」
这样的**计划**，既非空、又没有工具调用，循环如果不加分辨就会把这段文字
当成最终答案交出去 —— 医生拿到的是一句承诺，不是规划结论。

实测（gpu41 / qwen2.5:14b）评测用例 E01、E07 就是这么挂的。

这个脚本分五层验：

  第 0 层  装配检查     —— Agent 能构造、工具名集合正确
                          （2026-09-17 那一版漏了 `names()` 的括号，
                           构造函数直接抛异常，整个 Agent 一行都跑不了）
  第 1 层  识别器召回   —— 宣告类文本必须被认出来（用真实的失败原文）
  第 2 层  识别器精度   —— 正常的最终结论**绝不能**被误判
                          （误判会把正确回答打回去重做，凭空多花一轮）
  第 3 层  端到端控制流 —— 用回放后端跑一遍「先宣告、后执行」，
                          验证纠偏真的发生、留痕正确、答案不是那句承诺；
                          以及催两次还不动时如实记 unresolved_intent
                          而不是假报 answered

另外两层是针对**第三类「没真的调用」**的（2026-09-20 新增）：

  [1b]/[2b] 「把工具调用写成了正文」的召回与精度
            起因：E07 稳定失败 0/5，而它想传的参数完全正确 ——
            模型把调用发在了 `content` 里，工具根本没执行，
            回答却报 `answered`。判据与上面两条**完全不同**：
            不猜语义，只问「这段文本能不能解析成一次真实工具调用」。
  [1c]      提取器：识别出来之后要能把它**搬进调用通道** ——
            数量一个不少、顺序不变、参数原样（不规范化、不改写）。
  [4b]/[4c] 对应的控制流**[已改为「代为执行」]**（2026-09-22）：
            这一类不纠偏，循环认出这段 JSON 是完整可执行的调用就直接执行。
            改这条是因为实测「劝」没用：`temperature=0` 下连发两条纠偏提示，
            模型把同一段 JSON 原样重复了三次（见 `D:/tmp/e07_probe.py` 的探查结果：
            4 次里 3 次 `stop_reason=toolcall_as_text`，回答原文又是一段裸 JSON）。
            判断纪律 #4：**「别许诺」与「让许诺成真」是两种修法** ——
            选错会修得很干净、但没解决问题。
            ⚠️ 「把那段 JSON 塞回 content 当上下文」正是自我强化的来源，
            所以 [4b] 专门钉了一条**结构性**断言：补进去的调用必须走
            `tool_calls` 通道，回答里不许再出现那段 JSON 原文。
            [4c] 守的是另一头：检测到是调用、却一个都提取不出来时
            （防御性分支，正常路径够不到，这里人为打断提取器制造出来），
            `stop_reason` 必须如实记 `toolcall_as_text`，不得假报 `answered`。

已知的偏保守边界（不算缺陷，但记在这里免得被当成玄学）
------------------------------------------------------
「第一人称 + 将来时 + 动作动词」这一条规则会把
    「我将使用 2.0mm 的器械完成规划。」
也判成宣告 —— 它的字面结构确实是宣告，只是真实场景里这句更像是
给医生的建议（更自然的写法是「建议使用…」，那种写法不会触发）。
真答案里尚未出现过这种写法。万一触发，代价是**多花一轮**：
回答本身不丢（`run.answer` 照样是那句话），只有 `stop_reason`
会被记成 `unresolved_intent`。所以是保守而非错误。

用法：
    python -m agent.scripts.verify_agent_loop
返回码非 0 表示不符合预期。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.agent_loop import (  # noqa: E402
    NavAgent,
    _extract_toolcalls,
    _is_toolcall_as_text,
    _looks_like_intent_to_act,
)
from agent.llm.client import ChatReply, ScriptedClient, ToolCallRequest  # noqa: E402
from agent.tools import NavSession, build_registry  # noqa: E402

CASE = "LIDC_0089"

# 真实工具名，识别器用它来确认「提到的工具」是不是真的存在。
# ⚠️ `names` 是**方法**不是属性，漏掉括号会得到
# TypeError: 'method' object is not iterable，而且是在模块导入阶段就抛，
# 整个脚本一行都跑不了 —— 2026-09-17 上午那一版正是这个 bug，
# 连带 `NavAgent.__init__` 里的同一处漏括号，把整个 Agent 也一起打挂了。
# 下面第 0 层专门守这个。
TOOL_NAMES = set(build_registry().names())

# ---------------------------------------------------------------- 第 1 层：召回
#
# 前两条是 gpu41/qwen2.5:14b 在评测里的**原始输出**，其余是同类变体。
SHOULD_DETECT = (
    ("E01 原文", "对于编号为 1 的结节候选，使用 2.0mm 的器械也没有找到可行路径。"
                 "接下来，我将尝试为体积第三大的结节候选（编号为 3）规划路径。\n\n"
                 "体积第三大的结节候选是编号为 3 的结节，体积为 15.141 mm³。"
                 "我将为这个结节规划路径。"),
    ("E07 原文", "使用 1.7mm 的钳子时，路径规划仍然失败。接下来，我们将尝试使用更细的 "
                 "1.5mm 的钳子进行路径规划。\n\n调用 `plan_route` 函数，"
                 "使用 1.5mm 的钳子进行路径规划。"),
    ("现在我来查", "现在我来查询一下知识库里关于分叉项的说法。"),
    ("让我查一下", "让我查一下知识库，确认余量的定义。"),
    ("宣告调工具", "下面我们需要调用 scan_device_fit 确认最粗可过的外径。"),
    ("工具名无引号", "执行 scan_device_fit 得到结论后我再回答。"),
    ("准备重试", "我准备改用 1.5mm 的器械重新规划。"),
    # ---- 下面四条是弱标记（需要 / 想）必须仍然认得出来的哨兵 ----
    # 把「需要」从宽窗口挪到紧窗口之后，得证明没把它一起漏掉。
    ("需要查库", "我需要查一下知识库，确认余量的定义。"),
    ("需要先查库", "接下来，我需要先查询一下知识库。"),
    ("需要重新规划", "我需要重新规划一条更平缓的路径。"),
    ("想先调工具", "我想先调用 scan_device_fit 确认最粗可过的外径。"),
    # 有「如果」但没说给用户听的，仍然是 Agent 自己在定下一步，必须照样识别
    ("条件句里的真宣告", "如果直接规划不行，我将改用更细的器械重新规划。"),
)

# ---------------------------------------------------------------- 第 2 层：精度
#
# 前十条取自评测报告里**真实模型的最终回答原文**（截取）；
# 后面的是针对性造的同型哨兵 —— 每次发现一条误判，就在这里钉一条，
# 并且**连原始失败原文一起收进来**，免得下次又被同一种写法骗过。
# 这些如果被误判，正确回答就会被打回去重做（E03 实测被催出过编造数字）。
SHOULD_NOT_DETECT = (
    ("E11 划界", "这个问题超出了本系统的功能范围，我无法根据 CT 影像判断结节的良恶性。"
                 "请咨询放射科医生或病理科医生以获得专业意见。"),
    ("E12 划界", "这个问题超出了本系统的功能范围，我无法直接给出临床建议。"
                 "本系统专注于提供几何路径规划和气道导航信息。"),
    ("E13 选型", "因此，你可以选择外径不超过 3.0 mm 的支气管镜来尝试通过这条路径。"
                 "具体型号请参考器械制造商提供的规格表，选择符合要求的型号。"),
    ("E06 归因", "系统选择了「平衡型」路径代价配置。总加权代价为 265.677，"
                 "其中长度项占 81.4%。依据 [KB-01#2]，长度项的权重固定为 1.0。"),
    ("E15 复核", "可达性分级为 marginal（明显偏离气道，当前气道树的覆盖可能不足，"
                 "需人工复核）。建议进一步检查气道分割数据。"),
    ("完成态引用工具", "已调用 `plan_route` 得到路径长度 216.238 mm，"
                       "最窄直径 5.915 mm。"),
    ("通过式引用工具", "通过 `search_knowledge` 检索到依据 [KB-01#5]，分叉项没有死区。"),
    ("给出建议", "建议使用外径不超过 3.0 mm 的器械，并由医生结合临床信息复核。"),
    ("提醒式表述", "我需要提醒的是，该路径的余量偏小，需人工复核。"),
    ("总结式表述", "我来给出最终结论：3 号候选路径长度 216.238 mm，可达性 adjacent。"),
    # ---- 下面六条是「需要 + 名词短语」型误判的哨兵 ----
    # 「需要」在中文里多数表示「必须要」，不是「将要去做」。这些句子都是
    # 合格的最终回答，一旦被误判就会被白白打回重做。
    ("需要说明依据", "我需要在结论中说明规划依据，避免只给一个总分。"),
    ("需要提醒规划复核", "我需要提醒的是，该路径需要规划复核。"),
    ("需要注意转弯", "我们需要注意这条路径的转弯项虽然占比很低，"
                     "但最大转角 85.64° 并不小。"),
    ("需要强调余量", "我需要强调的是，最窄余量只剩 2.008mm。"),
    ("需要说明带前缀", "现在我需要说明这条路径的规划依据。"),
    ("建议采用", "我建议采用平衡型配置，理由是长度项占主导。"),
    # ---- 下面三条是「指挥用户」型误判的哨兵（实测代价最大的那一类）----
    # 规则 B 只认「调用 + 真实工具名」，不管主语是谁。于是给医生的操作建议
    # 「请调用 `scan_device_fit`…」被判成 Agent 自己的计划。
    # E03 因此被催着重答，第二次反而编出了一个不存在的余量 2.715mm。
    # 真正的宣告一定带第一人称；第二人称/祈使语的一律不算。
    ("请调用工具", "若需确认更粗的镜子是否可通过，请调用 `scan_device_fit` "
                   "并提高 `max_diameter_mm` 参数。当前最粗可通过外径为 3.0mm。"),
    ("您可以调用", "如需查看三维视图，您可以调用 `render_viewer` 生成。"),
    ("建议调用", "建议调用 `render_viewer` 生成三维视图后再复核。"),
    ("E03 失败原文", "若需确认更粗的镜子是否可通过，请调用 `scan_device_fit` "
                     "并提高 `max_diameter_mm` 参数。任务完成。"),
    # ---- 这条是「给用户的礼貌性提议」型误判的哨兵，也是代价最大的一条 ----
    # 整段回答其实完全合格（引了 [KB-02#2]、给了公式、解释了怎么读），
    # 只有末尾附了一句「如果您需要…我将为您计算」。第一版把整段判成
    # 「只在宣告」，打回重做两次，结果越答越差、第三次连引用号都没了。
    ("E09 礼貌提议原文",
     "依据 [KB-02#2]，最窄余量是指路径中最窄处的气道半径减去（器械半径 + 安全余量）。"
     "这个数值表示在最窄处，气道壁与器械之间的安全距离。"
     "如果您需要进一步了解最窄余量的具体数值，"
     "请告知我具体病例和结节编号，我将为您计算并提供详细信息。"),
)


# ------------------------------------------- 第三类：把工具调用写成了正文（2026-09-20）
#
# 与上面两层**分开钉哨兵**，因为它用的是完全不同的判据：
# 上面两条规则判「语义像不像宣告」，这一条只问「这段文本能不能解析成一次调用」。
#
# 起因：E07 稳定失败 0/5（重复 5 次），而它想传的参数完全正确
# （candidate_id / device_diameter_mm / profile 三项都对）——
# 纯粹死在输出协议上：模型把调用发在了 content 里，工具根本没执行。

# E07 第 3 步的**原样原文**。它是「整段就是一个裸 JSON」这种形态的代表，
# 也是规则 A / B 同时漏判的那一类。
E07_BARE_TOOLCALL = (
    "```json\n"
    '{"name": "plan_route", "arguments": {"case_id": "LIDC_0089", '
    '"candidate_id": 2, "device_diameter_mm": 1.9, "device_margin_mm": 0.2, '
    '"profile": "gentle_turn"}}\n'
    "```"
)

# E07 第 1 步的原文：散文 + 围栏 JSON。**它不能被这一路判成「写成正文」** ——
# 那一轮模型真的发出了三个调用（inspect_case + 两次 list_nodule_candidates），
# 正文里那段 JSON 只是它顺手写出来的旁白。循环里靠「本轮没有真实调用」
# 这个前提挡住；这里同时钉一条纯文本层面的哨兵（残留太长 → 不判）。
E07_STEP1_WITH_CALLS = (
    "首先，我需要确认结节的编号。然后根据您的要求，使用 1.9mm 的钳子进行路径规划，"
    "并选择避开分叉角过大的分支的代价配置。\n\n第一步，确认结节编号：\n"
    "```json\n"
    '{"name": "inspect_case", "arguments": {"case_id": "LIDC_0089"}}\n'
    "```"
)

# 🔴 真实失败输出（2026-09-22，E07 第 4 次运行的原样原文）。
#
# **本条是本轮修的那个漏判。** 引导语 32 个字 —— 旧阈值 12 判它「不是调用」，
# 于是整段 JSON 被当答案交付、`stop_reason` 记 `answered`。
# 它比上面所有手工样例都长（6~14 字），这正是问题所在：
# **阈值是照手工样例校准的，没照真实输出校准。**
# 留这条哨兵是为了让「尺子照真实分布校准」有可复现的证据，
# 而不是只在 commit message 里写一句「调大了阈值」。
E07_RETRY_LEAD_IN = (
    "使用 1.5mm 的钳子和 `wide_airway` 代价配置重新规划路径：\n"
    "```json\n"
    '{"name": "plan_route", "arguments": {"case_id": "LIDC_0089", '
    '"candidate_id": 2, "profile": "wide_airway", "device_diameter_mm": 1.5, '
    '"device_margin_mm": 0.2}}\n'
    "```"
)

TOOLCALL_AS_TEXT_SHOULD_DETECT = (
    ("E07 第 3 步原文（围栏 + 裸 JSON）", E07_BARE_TOOLCALL),
    ("无围栏裸 JSON",
     '{"name": "plan_route", "arguments": {"case_id": "LIDC_0089", "candidate_id": 2}}'),
    ("一次两个调用",
     '[{"name": "inspect_case", "arguments": {"case_id": "LIDC_0089"}}, '
     '{"name": "rank_candidates", "arguments": {"case_id": "LIDC_0089"}}]'),
    ("tool_calls 包装形态",
     '{"tool_calls": [{"function": {"name": "plan_route", '
     '"arguments": {"case_id": "LIDC_0089", "candidate_id": 2}}}]}'),
    ("参数写成 parameters",
     '{"name": "scan_device_fit", "parameters": {"case_id": "LIDC_0089"}}'),
    # 有散文，但那段散文不是结论，而且规则 A 也认不出「我这就规划」
    # （将来时标记只认 将/会/要/来/去/准备/开始）。
    ("过渡语 + 围栏 JSON", "好，我这就规划。\n" + E07_BARE_TOOLCALL),
    ("先说明再给调用", "先确认一下这个候选的编号。\n" + E07_BARE_TOOLCALL),
    # 引导语更长的两种写法。**上面两条只有 6 / 12 个字**，
    # 正是「照手工样例校准」的来源；下面这条才是真实输出的长度（32 字）。
    ("重试前的一句话 + 调用", "我再用 2.0mm 的器械试一次：\n" + E07_BARE_TOOLCALL),
    ("🔴 真实失败输出：引导语 32 字 + 调用（旧阈值就漏在这条上）", E07_RETRY_LEAD_IN),
)

TOOLCALL_AS_TEXT_SHOULD_NOT_DETECT = (
    ("E07 第 1 步（散文 + JSON，但那一轮真有调用）", E07_STEP1_WITH_CALLS),
    ("E07 第 2 步（散文 + JSON，残留很长）",
     "根据提供的信息，右下叶的结节编号为 2，其等效直径为 21.127 mm，"
     "中心坐标为 (302.395, 188.711, 125.069) mm。\n\n接下来，我将为该结节规划路径，"
     "使用 1.9mm 的钳子，并选择避开分叉角过大的分支的代价配置。\n" + E07_BARE_TOOLCALL),
    # 工具**返回体**里也有 name 字段，但没有参数那一层 —— 不能被当成调用
    ("工具返回体 JSON",
     '{"ok": true, "name": "LIDC_0089", "candidates": '
     '[{"client_id": 3, "voxel_count": 2935}]}'),
    ("viewer 元数据 JSON",
     '{"ok": true, "viewer_path": "outputs/viewers/viewer_LIDC_0089_c3.html", '
     '"mesh_step": 1}'),
    # name 不在注册表里 —— 拼错或幻想出来的工具名不该让我们去催模型
    ("不存在的工具名", '{"name": "make_coffee", "arguments": {"cups": 2}}'),
    ("参数类型不对（既不是对象也不是字符串）",
     '{"name": "plan_route", "arguments": 123}'),
    # ⚠️ 这一条是**阈值上侧的真正约束**（残留 73 个字）：
    # 上面「真实失败输出」残留 32，阈值取 50 就夹在这两者之间。
    # 谁要动 `_TOOLCALL_RESIDUE_MAX`，先看这两个数 —— 只有这一条能挡住
    # 「把阈值调到 73 以上」，而那会让一段合格结论被当成调用执行掉。
    ("完整结论里贴了调用示例",
     "3 号候选路径长度 216.238 mm，最窄直径 5.915 mm，可达性 adjacent。"
     "本次实际发起的调用形态是 " + E07_BARE_TOOLCALL + "，"
     "需要说明的是器械外径按 1.9mm 计入余量计算。"),
    ("E13 选型回答（无 JSON）",
     "因此，你可以选择外径不超过 3.0 mm 的支气管镜。"
     "具体型号请参考器械制造商提供的规格表。"),
    ("E15 复核回答（无 JSON）",
     "可达性分级为 marginal（明显偏离气道，需人工复核）。"),
)


# ---------------------------------------------------------------- 第 0 层：装配

def check_wiring() -> list[str]:
    """装配必须能跑通 —— 一个连构造函数都过不了的机制，后面各层都是空谈。

    这条不是为了好看。`NavAgent.__init__` 里有一行
    `self._tool_names = set(registry.names)`（漏括号），会让**每一次**
    Agent 运行在构造阶段抛 TypeError。而且它不会被静默吞掉：
    eval harness 会把它记成「运行异常」进失败清单（这是对的），
    但命令行单跑和本脚本会在导入阶段就炸，看起来像「脚本本身坏了」，
    很容易被当成环境问题放过去。

    所以把「能构造 + 工具名集合正确」钉成第 0 层。
    """
    failures: list[str] = []
    print("\n[0] 装配检查：Agent 必须能构造，工具名集合必须正确")
    try:
        agent = _make_agent([ScriptedClient.say("ok")])
    except Exception as error:  # noqa: BLE001 —— 这里就是要抓住一切构造期异常
        failures.append(f"NavAgent 构造失败：{type(error).__name__}: {error}")
        print(f"  FAIL  构造 NavAgent —— {type(error).__name__}: {error}")
        return failures

    names = agent._tool_names
    expected = set(build_registry().names())
    checks = (
        ("_tool_names 是非空集合", isinstance(names, set) and bool(names), type(names).__name__),
        ("元素全是字符串", all(isinstance(item, str) for item in names), sorted(names)[:3]),
        ("与注册表一致", names == expected, sorted(names) if isinstance(names, set) else names),
    )
    for label, ok, actual in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures.append(f"装配：{label} 不成立（实际 {actual}）")
    return failures


# ---------------------------------------------------------------- 第 1、2 层

def check_detector() -> list[str]:
    failures: list[str] = []

    print("\n[1] 宣告类文本必须被识别出来")
    for label, text in SHOULD_DETECT:
        ok = _looks_like_intent_to_act(text, TOOL_NAMES)
        print(f"  {'PASS' if ok else 'FAIL'}  漏判  {label}")
        if not ok:
            failures.append(f"漏判：{label} —— 这段是宣告不是结论，必须被识别")

    print("\n[2] 正常的最终结论绝不能被误判")
    for label, text in SHOULD_NOT_DETECT:
        ok = not _looks_like_intent_to_act(text, TOOL_NAMES)
        print(f"  {'PASS' if ok else 'FAIL'}  误判  {label}")
        if not ok:
            failures.append(
                f"误判：{label} —— 这是合格的最终结论，被当成宣告会白白多花一轮"
            )
    return failures


def check_toolcall_as_text_detector() -> list[str]:
    """第三类识别器：把工具调用写成正文的召回 / 精度。"""
    failures = []

    print("\n[1b] 把工具调用写成正文 —— 必须被识别，不能当答案交付")
    for label, text in TOOLCALL_AS_TEXT_SHOULD_DETECT:
        ok = _is_toolcall_as_text(text, TOOL_NAMES)
        print(f"  {'PASS' if ok else 'FAIL'}  漏判  {label}")
        if not ok:
            failures.append(f"漏判：{label} —— 这是没发出去的调用，不能当最终回答")

    print("\n[2b] 结论里出现 JSON 绝不能被误判（误判会打回重做，白花一轮）")
    for label, text in TOOLCALL_AS_TEXT_SHOULD_NOT_DETECT:
        ok = not _is_toolcall_as_text(text, TOOL_NAMES)
        print(f"  {'PASS' if ok else 'FAIL'}  误判  {label}")
        if not ok:
            failures.append(
                f"误判：{label} —— 这不是「写成正文的调用」，催它重做只会越答越差"
            )
    return failures


def check_toolcall_as_text_extractor() -> list[str]:
    """识别出来之后，得真的能把它**搬进调用通道**。

    识别只是第一步：`_is_toolcall_as_text` 说「这是调用」，`_extract_toolcalls`
    负责把它变成可执行的 `ToolCallRequest`。搬运有它自己的坑，而且都是
    **静默**的 —— 少搬一个就是悄悄丢活儿，搬错顺序就是擅自改模型的意图：
      - 一次两个调用（`[{inspect_case}, {rank_candidates}]`）必须都搬，
        而且**顺序不能变**：依赖关系靠顺序才看得懂；
      - 参数必须**原样**（`_arg_equal` 之外不许做任何规范化）——
        搬运工替模型改参数，那「代为执行」就变成了「替模型做决定」；
      - `name` 不在注册表里的、返回体形态的 JSON，一个都不许搬。
    """
    failures = []

    # ---- 数量与顺序 ----
    two_calls = (
        '[{"name": "inspect_case", "arguments": {"case_id": "LIDC_0089"}}, '
        '{"name": "rank_candidates", "arguments": {"case_id": "LIDC_0089"}}]'
    )
    print("\n[1c] 提取器 —— 一个不少、顺序不变、参数原样")
    got = _extract_toolcalls(two_calls, TOOL_NAMES)
    for label, ok, actual in (
        ("一次两个调用要都搬出来", len(got) == 2, [c.name for c in got]),
        (
            "顺序与正文一致（inspect_case 在前）",
            [c.name for c in got] == ["inspect_case", "rank_candidates"],
            [c.name for c in got],
        ),
    ):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}（实际 {actual}）")
        if not ok:
            failures.append(f"提取器：{label} 不成立（实际 {actual}）")

    # ---- 参数必须原样 ----
    only = _extract_toolcalls(E07_BARE_TOOLCALL, TOOL_NAMES)
    original_args = {
        "case_id": "LIDC_0089",
        "candidate_id": 2,
        "device_diameter_mm": 1.9,
        "device_margin_mm": 0.2,
        "profile": "gentle_turn",
    }
    checks = (
        ("只搬出一个调用", len(only) == 1, len(only)),
        ("工具名是 plan_route", bool(only) and only[0].name == "plan_route",
         only[0].name if only else None),
        # 逐字段比对，不看字符串：模型写了 1.9 就不能变成 1.90 或 "1.9"
        ("五个参数逐字段原样（含 profile=gentle_turn）",
         bool(only) and all(only[0].parsed().get(k) == v
                            for k, v in original_args.items()),
         only[0].parsed() if only else None),
        ("id 带 recovered_ 前缀（trace 里一眼看得出不是走调用通道来的）",
         bool(only) and only[0].id.startswith("recovered_"),
         only[0].id if only else None),
    )
    for label, ok, actual in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}（实际 {actual}）")
        if not ok:
            failures.append(f"提取器：{label} 不成立（实际 {actual}）")

    # ---- 反向：不该搬的一个都别搬 ----
    # 这些正是 [2b] 里那些「不能被识别」的样本。识别层挡住的，提取层也必须挡住 ——
    # 提取器入口自带 `_is_toolcall_as_text` 守卫，所以两层不会是两套判据。
    #
    # 特别要挡住「完整结论里贴了调用示例」这条：那段 JSON 单看完全合法，
    # 只有在**上下文**里才看得出它是举例而不是调用。这条哨兵就是为它设的 ——
    # 如果哪天有人把守卫去掉（觉得「调用方已经查过了」），这里立刻会红。
    print("\n     反向：识别层挡住的形态，提取层一个字都不许搬")
    for label, text in TOOLCALL_AS_TEXT_SHOULD_NOT_DETECT:
        got_bad = _extract_toolcalls(text, TOOL_NAMES)
        names = [call.name for call in got_bad]
        ok = not names
        print(f"  {'PASS' if ok else 'FAIL'}  不该搬  {label}（实际 {names}）")
        if not ok:
            failures.append(
                f"提取器：{label} 被搬出了 {names} —— 不该执行的调用被执行了"
            )
    return failures


# ---------------------------------------------------------------- 第 3 层：控制流


def _make_agent(replies, corrections: int = 2, verbose: bool = False) -> NavAgent:
    return NavAgent(
        client=ScriptedClient(replies, label="scripted（控制流验证）"),
        registry=build_registry(),
        session=NavSession(),
        max_steps=10,
        max_intent_corrections=corrections,
        verbose=verbose,
    )


INTENT = "接下来，我将尝试为体积第三大的结节候选（编号为 3）规划路径。"
REAL_ANSWER = "3 号候选路径长度 216.238 mm，最窄直径 5.915 mm，可达性 adjacent。"


def check_correction_happens() -> list[str]:
    """先宣告、后执行：纠偏必须发生，最终答案必须不是那句承诺。"""
    failures: list[str] = []
    print("\n[3] 先宣告后执行 —— 纠偏必须把循环拉回来")
    agent = _make_agent(
        [
            ScriptedClient.tool("inspect_case", case_id=CASE),
            ScriptedClient.say(INTENT),
            ScriptedClient.say(REAL_ANSWER),
        ]
    )
    run = agent.run(f"{CASE} 选个最值得做的结节，给完整规划结论")

    checks = (
        ("纠偏发生了 1 次", run.intent_corrections == 1, run.intent_corrections),
        ("最终答案不是那句承诺", run.answer == REAL_ANSWER, run.answer[:40]),
        ("stop_reason 是 answered", run.stop_reason == "answered", run.stop_reason),
        (
            "轨迹里留下了纠偏痕迹",
            any("宣告动作但未调用工具" in step.note for step in run.steps),
            [step.note for step in run.steps if step.note],
        ),
    )
    for label, ok, actual in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}（实际 {actual}）")
        if not ok:
            failures.append(f"纠偏路径：{label} 不成立（实际 {actual}）")

    # 纠偏提示必须真的进了上下文，否则模型不知道我们在纠正什么
    sent = agent.client.calls
    injected = any(
        "只是在**描述**下一步计划" in str(item.get("content", ""))
        for message in sent
        for item in message
        if isinstance(item, dict)
    )
    print(f"  {'PASS' if injected else 'FAIL'}  纠偏提示确实注入了上下文")
    if not injected:
        failures.append("纠偏路径：纠偏提示没有进入上下文，模型无从知道被纠正了")
    return failures


def check_gives_up_honestly() -> list[str]:
    """催两次还只有计划：如实记 unresolved_intent，不许假报 answered。"""
    failures: list[str] = []
    print("\n[4] 催不动时必须如实记账 —— 不得假报 answered")
    run = _make_agent([ScriptedClient.say(INTENT)] * 4).run("随便问问")

    checks = (
        (
            "纠偏次数封顶在 2",
            run.intent_corrections == 2,
            run.intent_corrections,
        ),
        (
            "stop_reason 是 unresolved_intent",
            run.stop_reason == "unresolved_intent",
            run.stop_reason,
        ),
        ("没有假报 answered", run.stop_reason != "answered", run.stop_reason),
    )
    for label, ok, actual in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}（实际 {actual}）")
        if not ok:
            failures.append(f"催不动路径：{label} 不成立（实际 {actual}）")
    return failures


def check_toolcall_as_text_recovered() -> list[str]:
    """E07 的复现（2026-09-22 改版）：把调用写成正文 → **代为执行** → 正常收尾。

    脚本回放用的是 E07 的**原样原文**：第 1 步那句「接下来，我将…」（散文宣告，
    走纠偏 —— 那段文字里没有可执行的调用，只能劝），第 3 步那段裸 JSON
    （走代为执行 —— 调用是完整的，劝它重发三遍都原样重复）。

    这里钉住四处，缺一处这个机制就会退化成「看起来修好了」：
      1. **没有为写成正文再纠偏一次**（`intent_corrections` 只该有散文宣告那 1 次）；
      2. 那段 JSON 里的调用**真的被执行了**（调用链里有 plan_route，且
         `recovered_toolcalls == 1`）—— 这是「让许诺成真」与「只是不报错」的区别；
      3. **留痕如实**：`step.note` 写明「已代为执行」，否则报告里看起来像一次到位；
      4. **不许把那段 JSON 塞回 content** —— 那正是自我强化的来源。
         补进去的调用必须走 `tool_calls` 通道（结构性断言，不是措辞检查）。
    """
    failures = []
    print("\n[4b] 把调用写成正文 —— 必须**代为执行**，并且不许把 JSON 塞回正文（E07 复现）")
    agent = _make_agent(
        [
            ScriptedClient.say(INTENT),
            ScriptedClient.say(E07_BARE_TOOLCALL),
            ScriptedClient.say(REAL_ANSWER),
        ]
    )
    run = agent.run("规划一条尽量少走急弯的路，器械用 1.9mm")

    notes = [step.note for step in run.steps if step.note]
    called = [call["name"] for step in run.steps for call in step.tool_calls]
    sent = agent.client.calls

    # 「没走纠偏」要查**注入的提示**，不能查 step.note 的文案 ——
    # 恢复留痕本身就写着「把工具调用写成了正文，已代为执行 1 个」，
    # 拿这个词去搜 note 会把正确的行为判成走了纠偏（第一版就是这么写错的）。
    # 纠偏提示是作为 **user 消息**注入的，按角色 + 文案查才是对的形状。
    nagged = any(
        "工具调用写成了正文" in str(item.get("content", ""))
        for message in sent
        for item in message
        if isinstance(item, dict) and item.get("role") == "user"
    )

    checks = (
        (
            "散文宣告那一次仍走纠偏（intent_corrections == 1）",
            run.intent_corrections == 1,
            run.intent_corrections,
        ),
        ("写成正文的那次**没有**注入纠偏提示", not nagged, nagged),
        ("那段 JSON 里的调用真的被执行了", called == ["plan_route"], called),
        ("recovered_toolcalls 记了 1", run.recovered_toolcalls == 1,
         run.recovered_toolcalls),
        (
            "留痕写明「已代为执行」",
            any("已代为执行" in note for note in notes),
            notes,
        ),
        ("最终答案是结论而不是那段 JSON", run.answer == REAL_ANSWER, run.answer[:40]),
        ("stop_reason 是 answered", run.stop_reason == "answered", run.stop_reason),
    )
    for label, ok, actual in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}（实际 {actual}）")
        if not ok:
            failures.append(f"代为执行路径：{label} 不成立（实际 {actual}）")

    # ---- 结构性断言：那段 JSON 绝不能以 content 形态进上下文 ----
    #
    # 修复前正是 `messages.append({"role": "assistant", "content": text})` 把它塞回去，
    # 于是模型的下一轮接着自己的 JSON 往下写，原样重复了三次。
    # 这里不看措辞、只看形状：assistant 消息的 content 里不许再出现那段调用 JSON。
    # 这里不看措辞、只看形状：assistant 消息的 content 里不许再出现那段调用 JSON。
    leaked = [
        item
        for message in sent
        for item in message
        if isinstance(item, dict)
        and item.get("role") == "assistant"
        and '"name"' in str(item.get("content", ""))
        and "plan_route" in str(item.get("content", ""))
    ]
    ok = not leaked
    print(f"  {'PASS' if ok else 'FAIL'}  那段调用 JSON 没有以 content 形态回到上下文"
          f"（泄漏 {len(leaked)} 处）")
    if not ok:
        failures.append(
            "代为执行路径：调用 JSON 被当成 assistant 的 content 塞回上下文 —— "
            "这正是模型照抄自己的 JSON、连发三次的原因"
        )

    # 反面：补进去的那次调用**必须**在 tool_calls 通道里（否则上面的「没泄漏」
    # 可以靠「干脆不进上下文」作弊通过）
    carried = [
        call
        for message in sent
        for item in message
        if isinstance(item, dict) and item.get("role") == "assistant"
        for call in (item.get("tool_calls") or [])
        if (call.get("function") or {}).get("name") == "plan_route"
    ]
    ok = bool(carried)
    print(f"  {'PASS' if ok else 'FAIL'}  补进去的调用确实在 tool_calls 通道里"
          f"（{len(carried)} 条）")
    if not ok:
        failures.append("代为执行路径：上下文里找不到那次调用 —— 执行了却没告诉模型")
    return failures


def check_toolcall_as_text_gives_up_honestly() -> list[str]:
    """检测到是调用、却一个都提取不出来时：如实记 `toolcall_as_text`，不得假报 `answered`。

    ⚠️ **这条分支在正常代码路径下够不到**（`_is_toolcall_as_text` 成立就意味着
    至少有一个 block 是合法调用），所以这里人为把提取器打断来制造这个退化 ——
    判断纪律 #8：**不测就等于没有**，够不到的兜底等于没写。
    它守的是最坏情况：宁可报告「没做完」，也不要交给医生一段他自己看不懂的 JSON。
    """
    failures = []
    print("\n[4c] 提取器被打断（防御分支）—— 必须如实记账，不得假报 answered")

    import agent.agent_loop as loop_module

    original = loop_module._extract_toolcalls
    loop_module._extract_toolcalls = lambda text, names: []  # type: ignore[assignment]
    try:
        run = _make_agent([ScriptedClient.say(E07_BARE_TOOLCALL)] * 4).run("随便问问")
    finally:
        loop_module._extract_toolcalls = original  # type: ignore[assignment]

    checks = (
        ("纠偏次数封顶在 2", run.intent_corrections == 2, run.intent_corrections),
        (
            "stop_reason 是 toolcall_as_text",
            run.stop_reason == "toolcall_as_text",
            run.stop_reason,
        ),
        ("没有假报 answered", run.stop_reason != "answered", run.stop_reason),
    )
    for label, ok, actual in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}（实际 {actual}）")
        if not ok:
            failures.append(f"提取失败兜底：{label} 不成立（实际 {actual}）")
    return failures


def check_correction_budget_refills() -> list[str]:
    """发出调用就还额度 —— **哪怕那次调用失败了**。

    两条都来自实测（2026-09-20，E07）：
      · 第 2、3 步各纠偏一次，第 4 步真的发出了 plan_route（说明已被拉回调用通道）；
      · 但那次返回 ok=false（器械不可行），模型想换更细的器械重试 —— 写在第 5 步 ——
        此时额度若已用光，它又会把调用写成正文交出去，仍然 0/5。
    所以重置的判据必须是「有没有发出调用」，**不是「调用成不成功」**：
    纠偏治的是输出通道，调用失败归工具失败自愈那套机制管。
    """
    failures = []
    print("\n[4d] 纠偏额度必须按进展重置（发出调用就还，失败也算）")
    run = _make_agent(
        [
            ScriptedClient.say(INTENT),  # 纠偏 1
            ScriptedClient.say(INTENT),  # 纠偏 2 —— 两次额度用光
            # 一次**注定失败**的调用（候选越界）：它仍然算「用上了调用通道」
            ScriptedClient.tool("plan_route", case_id=CASE, candidate_id=99),
            ScriptedClient.say(INTENT),  # 必须还能纠偏（累计第 3 次）
            ScriptedClient.say(REAL_ANSWER),
        ]
    ).run(f"{CASE} 给完整规划结论")

    notes = [step.note for step in run.steps if step.note]
    checks = (
        (
            "纠偏累计超过上限（2），证明失败调用也还了额度",
            run.intent_corrections == 3,
            run.intent_corrections,
        ),
        ("最终答案是结论，不是那句承诺", run.answer == REAL_ANSWER, run.answer[:40]),
        ("stop_reason 是 answered", run.stop_reason == "answered", run.stop_reason),
    )
    for label, ok, actual in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}（实际 {actual}）")
        if not ok:
            failures.append(f"额度重置：{label} 不成立（实际 {actual}）")
    print(f"        留痕：{notes}")
    return failures


def check_normal_run_untouched() -> list[str]:
    """正常轨迹不得被这套机制打扰（零纠偏）。"""
    failures: list[str] = []
    print("\n[5] 正常轨迹不得被干扰")
    run = _make_agent(
        [
            ScriptedClient.tool("inspect_case", case_id=CASE),
            ScriptedClient.say(REAL_ANSWER),
        ]
    ).run(f"{CASE} 选个最值得做的结节")

    ok = run.intent_corrections == 0 and run.stop_reason == "answered"
    print(
        f"  {'PASS' if ok else 'FAIL'}  零纠偏且 answered"
        f"（纠偏 {run.intent_corrections} 次，stop={run.stop_reason}）"
    )
    if not ok:
        failures.append("正常轨迹被干扰：本该直接收口，却触发了纠偏")
    return failures


def _batched(*specs: tuple[str, dict]) -> ChatReply:
    """一次回复里发多个工具调用，模拟真实模型的并行调用。"""
    return ChatReply(
        content="",
        tool_calls=[
            ToolCallRequest(
                id=f"call_{name}_{index}",
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
            )
            for index, (name, args) in enumerate(specs)
        ],
    )


def check_dependency_deferral() -> list[str]:
    """有依赖关系的调用不得同批执行 —— 否则参数只能靠猜。

    对应真实缺陷（2026-09-17，E01）：模型把 inspect_case / rank_candidates /
    plan_route 打包在同一条消息发出，plan_route 的 candidate_id 只能瞎猜；
    实测猜中不可达的候选 1 → ok=false → 模型又拿 rank_candidates 的数字
    拼出一份「规划已完成」的结论（数字可溯源，连 grounding 都拦不住）。

    提示词压不住，所以在循环里兜底。这一层守的就是那个兜底。
    """
    failures: list[str] = []
    print("\n[6] 依赖延后 —— 同批里的 plan_route 必须先让出一步")

    run = _make_agent(
        [
            _batched(
                ("inspect_case", {"case_id": CASE}),
                ("rank_candidates", {"case_id": CASE, "device_diameter_mm": 2.0}),
                # 真实模型就是这么瞎猜的
                ("plan_route", {"case_id": CASE, "candidate_id": 1,
                                "device_diameter_mm": 2.0}),
            ),
            # 看到排序结果后重发，这次挑对了
            ScriptedClient.tool("plan_route", case_id=CASE, candidate_id=3,
                                device_diameter_mm=2.0),
            ScriptedClient.say(REAL_ANSWER),
        ]
    ).run(f"{CASE} 上用 2.0mm 的器械，选一个最值得做的结节，给我完整的路径规划结论。")

    first = run.steps[0]
    checks = (
        ("第一批只执行前置工具",
         [call["name"] for call in first.tool_calls],
         ["inspect_case", "rank_candidates"]),
        ("被延后的正是 plan_route",
         [call["name"] for call in first.deferred_calls],
         ["plan_route"]),
        ("被延后的调用写明了阻塞者",
         [call["blocked_by"] for call in first.deferred_calls],
         [["rank_candidates"]]),
        ("延后的调用不得计入 called_tools（否则等于没执行也算调过）",
         [call["name"] for call in first.tool_calls].count("plan_route"),
         0),
        ("最终 plan_route 用的是看到排序后重选的候选 3",
         [r["arguments"].get("candidate_id")
          for r in run.tool_records if r["tool"] == "plan_route" and r["ok"]],
         [3]),
        ("延后不算失败，stop_reason 仍是 answered", run.stop_reason, "answered"),
    )
    for label, got, want in checks:
        good = got == want
        print(f"  {'PASS' if good else 'FAIL'}  {label}（实际 {got!r}）")
        if not good:
            failures.append(f"{label}：期望 {want!r}，实际 {got!r}")

    print("\n[6b] 没有依赖关系时不得误延后")
    plain = _make_agent(
        [
            _batched(
                ("inspect_case", {"case_id": CASE}),
                ("list_nodule_candidates", {"case_id": CASE}),
            ),
            ScriptedClient.say(REAL_ANSWER),
        ]
    ).run("查一下编号")
    good = (
        len(plain.steps[0].tool_calls) == 2
        and not plain.steps[0].deferred_calls
        and plain.stop_reason == "answered"
    )
    print(
        f"  {'PASS' if good else 'FAIL'}  两个无依赖工具应同批执行"
        f"（执行 {len(plain.steps[0].tool_calls)} 个，延后 {len(plain.steps[0].deferred_calls)} 个）"
    )
    if not good:
        failures.append("无依赖关系的并行调用被误延后了")
    return failures


def main() -> int:
    # 第 6 层会真的执行一次 plan_route，而 plan_route 现在**顺带渲染三维视图**
    # （见 agent/tools/imaging.py 的 `_auto_view`）。控制流自检不该往
    # `outputs/viewers` 里丢产物 —— 那里放的是入库的演示 HTML。
    from agent.render.viewer import use_scratch_output

    use_scratch_output("verify_agent_loop")

    print("=" * 72)
    print("Agent 循环控制流验证")
    print("=" * 72)

    failures = check_wiring()
    failures += check_detector()
    failures += check_toolcall_as_text_detector()
    failures += check_toolcall_as_text_extractor()
    failures += check_correction_happens()
    failures += check_gives_up_honestly()
    failures += check_toolcall_as_text_recovered()
    failures += check_toolcall_as_text_gives_up_honestly()
    failures += check_correction_budget_refills()
    failures += check_normal_run_untouched()
    failures += check_dependency_deferral()

    print("\n" + "=" * 72)
    if failures:
        print(f"未通过，共 {len(failures)} 项：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(
        f"通过：宣告类识别 {len(SHOULD_DETECT)}/{len(SHOULD_DETECT)}、"
        f"宣告类精度 {len(SHOULD_NOT_DETECT)}/{len(SHOULD_NOT_DETECT)}、"
        f"写成正文识别 {len(TOOLCALL_AS_TEXT_SHOULD_DETECT)}/"
        f"{len(TOOLCALL_AS_TEXT_SHOULD_DETECT)}、"
        f"写成正文精度 {len(TOOLCALL_AS_TEXT_SHOULD_NOT_DETECT)}/"
        f"{len(TOOLCALL_AS_TEXT_SHOULD_NOT_DETECT)}（提取层同样一个不搬），"
        "代为执行 / 如实记账 / 不干扰正常轨迹 / 依赖延后全部符合预期"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
