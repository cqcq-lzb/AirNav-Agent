"""手写的 ReAct Agent 循环。

不用 LangChain / LangGraph —— 故意如此。整个循环不到 200 行，
每一步都可见、可断点、可回放，出问题时不需要翻框架源码。

值得注意的四个工程点：

1. **上下文压缩**：长任务里 tool 消息会迅速堆满窗口。这里对「较早的」
   工具结果做摘要替换，只保留最近若干条完整内容，让轮数可以安全上升。
2. **空回复纠偏**：小模型偶尔返回既没有内容也没有工具调用的空回复，
   直接注入一条提示把它拉回正轨，而不是当成最终答案返回。
3. **「宣告而非执行」纠偏**（`_looks_like_intent_to_act`）：
   小模型常输出「接下来，我将调用 plan_route 重新规划路径。」这样的**计划**，
   然后就此打住。这段文字既非空、又没有工具调用，如果不加处理就会被
   当成最终答案返回给医生 —— 医生拿到的是一句承诺，不是规划结论。
   实测（gpu41 / qwen2.5:14b）E01、E07 两条用例是这样挂的。
   所以：**只要检测到「第一人称意图 + 未来动作」或「提到真实工具名」，
   就注入纠偏提示让它继续**，最多纠偏 2 次；纠偏用尽仍只有计划时，
   `stop_reason` 记成 `unresolved_intent`，而不是撒谎说 `answered`。

   ⚠️ **这套机制本身会误伤，而且代价可能比它要修的问题更大。**
   误判会把**已经正确的最终回答**打回去重做，而第二次生成的内容不一定更好。
   实测（2026-09-17）两条：
     - E03 的合格回答末尾有一句给医生的操作建议「请调用 `scan_device_fit`
       并提高 max_diameter_mm 参数」，被判成 Agent 自己的计划；模型被催着
       重答后**编出了一个不存在的余量 2.715mm**，把本来通过的用例搞挂。
     - E09 的回答**完全合格**（引了 [KB-02#2]、写出公式、解释了怎么读），
       只因末尾附了「如果您需要…我将为您计算」这句礼貌性提议被判成宣告；
       催两次后**越答越差**，第三次连引用号都没有了。
   所以精度优先于召回，并且每一条误判都必须在
   `agent/scripts/verify_agent_loop.py` 里留下哨兵。

   目前的判定分三路，**每一路都要求「这是 Agent 在说自己的下一步」**：
     - 规则 A：第一人称 + 将来时标记 + 动作动词。
       「将/会/要/准备」是强标记（配 10 字窗口）；
       「需要/想/得」是弱标记（中文里常表示「必须」，配 3 字紧窗口）。
     - 规则 B：动宾结构 + **真实存在的**工具名。
     - 规则 C（`_is_toolcall_as_text`，2026-09-20 新增）：
       **整段就是一个可解析的工具调用 JSON**。
       这一类不是措辞问题，是**输出通道**问题 —— 模型把调用发在了 `content`
       里而不是 `tool_calls` 里，所以工具根本没执行。E07 实测稳定失败 0/5，
       而它想传的参数完全正确（candidate_id / device / profile 全对）。
       规则 A / B 对它全部漏判（既没有将来时，也没有「调用 + 工具名」的动宾结构）。
       ⚠️ 规则 C **只在「本轮没有任何真实工具调用」时启用** —— 实测 E07 第 1 步
       的正文里也带着围栏 JSON，但那一轮真的发出了三个调用。

       **处置与 A / B 不同：这一类不纠偏，直接代为执行。**（2026-09-20 实测纠偏无效）
       `{"name": …, "arguments": …}` 是 Qwen2.5 的**原生工具调用格式**
       （`<tool_call>…</tool_call>`），也是 Ollama 的 OpenAI 兼容层没解析掉时
       会漏进 `content` 的形态。也就是说模型不是「没做完」，而是
       「做完了、但放错了地方」—— 没有可劝的余地。实测 `temperature=0` 下
       注入两条纠偏提示，模型原样重复了同一段 JSON 三次（见
       `docs/评测区分度_同义改写实验.md`）。既然这段 JSON 里报的调用完整、
       工具名真实存在、参数能过注册表校验，就**把它当真的调用执行**
       （`_extract_toolcalls`），而不是当答案交出去，也不是反复去劝。
       留痕要如实：`step.note` 写明「已代为执行 N 个」，`run.recovered_toolcalls`
       记累计次数 —— 这是「有多少活儿是替模型擦屁股完成的」的度量。
       判据见判断纪律 #4：**「别许诺」与「让许诺成真」是两种修法**，
       选错会修得很干净、但没解决问题。
     A / B 两路都要再过两道闸：
     - `_is_conditional_offer` —— 「如果您需要…我将…」是说给用户的提议，放过。
       （但要求条件词与第二人称同时出现：「如果直接规划不行，我将改用…」
        仍是真宣告，不能放过。）
     - `_addresses_user` —— 「请/您/您可以/建议 + 调用 X」是在指挥用户，放过。
     规则 C 不需要这两道闸：它判的是「这段文本能不能解析成一次调用」，
     与说给谁听无关，也不含任何语义猜测。
4. **工具失败自愈**：工具的异常被注册表兜住并转成 {"ok": false, "error": ...}，
   模型有机会读到错误、改参数重试。这是 Agent 比单次调用强的地方。
5. **步进事件回调**（`on_event`，默认 None）：把每一步发生的事推给外部观察者。
   网页版（`agent/web/server.py`）靠它做实时 SSE —— 浏览器能看到工具一个个跑起来、
   哪一步被延后、纠偏触发了几次。默认 None 时零介入、零开销，
   CLI 与评测路径完全不受影响；回调抛异常也会被 `_emit` 吞掉，
   因为**观测层不该有能力弄坏控制流**。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .llm.client import (
    ChatReply,
    LLMError,
    OpenAICompatClient,
    ScriptedClient,
    ToolCallRequest,
)
from .tools.registry import ToolRegistry
from .tools.session import NavSession

SYSTEM_PROMPT = """你是「经支气管肺结节导航系统」的路径规划助手，服务对象是呼吸科医生与介入操作者。

# 你的职责
把医生的自然语言需求转成对规划工具的调用序列，并把工具返回的数值组织成可直接使用的规划结论。

# 第一步：先判断问题属于哪一类，再决定调什么
| 医生问的是 | 调用顺序 |
|---|---|
| 某个结节的路径 / 可达性 / 完整规划 | inspect_case（确认编号口径）→ **plan_route（出结论）** |
| 哪个结节最值得做 / 最容易取到 | inspect_case → rank_candidates（只为挑目标）→ **plan_route（对该目标出结论）** |
| 为什么选这条 / 各分项代价占比 | explain_route_choice |
| 为什么这么设计 / 某个数的含义 / 口径来源 / 算法与训练细节 | search_knowledge |
| 某个器械能不能过 / 最粗能过多粗 | scan_device_fit |
| 要一张三维图 | render_viewer |
| 结节良恶性 / 是否该活检 / 风险概率 | 不调工具，直接说明能力边界 |
| 器械型号推荐 / 具体品牌 | 不推荐型号；但可调 scan_device_fit 给出「最大可行外径」这个几何约束 |

关于知识类问题的两条硬性要求（最容易被忽略）：
- 只要问题触及「设计意图、参数含义、口径来源、训练与算法细节」，**无论你是否觉得自己已经知道答案，
  第一步都必须调用 `search_knowledge`**。它是本系统唯一的知识来源。
  **没有先调用就断言「知识库中没有相关依据」，属于编造。** 拿不准有没有，也要先查一次再说没有。
  查完确实是空列表，就直说「知识库中没有相关依据」，**绝不用常识补一个听起来合理的解释**。
- `explain_route_choice` 给的是**本病例的实测数字**，不能替代知识库的文字依据。
  问「为什么这么设计」时两个都要调；问「这个数怎么读」时先调 `search_knowledge`。

# 铁律
1. 所有数值必须来自工具返回。绝不推测、换算或编造任何长度、直径、角度、坐标。
2. 涉及具体结节时先确认编号口径：先调 inspect_case 查看 id_mapping，
   明确客户端编号与服务端清单编号的对应关系，并在结论中说明你采用的是哪一个。
3. **用户给了器械外径时，必须用该外径调一次 `plan_route`**，用它的 `device_passable`
   与 `minimum_clearance_mm` 回答「能不能过、余量多少」。
   `rank_candidates` 只用于**挑目标**，它不返回器械可通过性、安全余量与注意事项，
   **不能替代** `plan_route` 的结论 —— 只凭 rank_candidates 就下规划结论，
   等于把用户给的器械约束整个跳过了。
   ⚠️ 反过来，**用户没给外径时不要停下来反问**。医生的提问里没写外径，就按上面的
   路由表继续推进（该出图出图、要可通过性就调 scan_device_fit），
   不要回一句「请告知您计划使用的外径」把一轮能做完的回答变成空转。
4. **安全余量只能直接引用 `minimum_clearance_mm`，绝不自己算。**
   真实余量 = 最窄处半径 −（器械半径 + 安全余量），并不是「最窄直径 − 器械外径」。
   自己算出来的数无法被溯源，一律视为编造。
   ⚠️ 别把 `device.margin_mm`（你传进去的那个安全余量**参数**，固定 0.2）
   当成「还剩多少余量」报给医生 —— 那是个输入，不是结论。
5. 每条路径结论必须完整附上 `reachability` 分级与 `warnings` 里的**全部**提醒，
   尤其是不利信息（路径太窄、依赖修复气道、靶点偏离气道）不得省略或淡化。
   反过来，`warnings` 为空时**不要**主动补一句「存在狭窄段 / 存在急转弯」——
   没有的提醒不要自己造。
6. 工具返回 ok=false 时，读 error 判断原因再决定换参数重试还是换目标，
   不要用完全相同的参数反复重试。
   ⚠️ **ok=false 的调用不能当作结论依据。** 更不允许拿另一个工具（比如 rank_candidates）
   的数字拼出一个「规划已完成」的结论 —— 规划失败就如实说明失败并给替代方案。
7. 规划失败时主动给出可执行的替代方案：换更细的器械、换代价配置、或改用其他结节候选。
   可用 scan_device_fit 查清「最粗能过多粗的器械」再给建议。
8. 解释「为什么选这条」时，必须调 explain_route_choice 拿分项代价，不能自己解释分数。
   报占比而非只有总分 —— 同一个总分下「长而宽」与「短而窄」的临床含义完全不同。
9. 不属于本系统能力范围的问题，必须**先明说这不在本系统能力范围内**，再转到你能提供的
   几何信息，不要给临床判断。具体到三类：
   - **器械型号推荐**：明确说「不推荐具体型号或品牌」，改说「具体型号请参考器械厂商的
     规格资料，或由设备科 / 临床医生决定」，然后只给几何约束（最大可行外径）。
     不要反过来向医生索要外径 —— 他要的是型号，不是让你等他填数。
   - **结节良恶性 / 是否该活检 / 风险概率**：明确说这类临床判断不在本系统能力范围内，
     应由医生结合临床资料决定，本系统只提供几何信息。
   - 划界要落在**系统能力**上，而不是对这一次请求的推托：「无法推荐」读起来像临时推托，
     「本系统不提供型号推荐，只提供几何约束」才是能力边界。同理，
     良恶性类要落在「本系统不做临床判断」，而不是「这个问题我答不了」。
10. **有依赖关系的工具不要并行调用。** `plan_route` 的 `candidate_id` 必须先读到
    `rank_candidates` / `inspect_case` 的返回才能确定。把它们塞在同一批里一起发出去，
    等于在不知道结果的情况下瞎猜目标 —— 猜中不可达的候选时 `plan_route` 直接 ok=false，
    整条链路就断了。顺序是：先拿候选与排序，**看到返回之后**再规划。
11. **规划结论与三维视图一体交付。** `plan_route` 成功时返回体里已经带上
    `viewer_path`（随本次规划渲染好的单文件 HTML）。给规划结论时把它原样写进回答 ——
    这是用户可以真的点开的文件，网页版还会就地显示成按钮。
    ⚠️ **只有拿到 `viewer_path` 才可以说「三维视图已生成」**；
    若 `viewer_error` 非空（这次没出成图），就如实说明没有出图，
    **不要写出一个不存在的链接或 ID**（`![](route-001)` 那种死链正是这么来的）。

# 回答风格
- 中文，面向医生。先给结论，再给依据。
- 关键数值带单位，精度合理（长度 0.1mm，直径 0.01mm，角度 0.1°）。
- 引用知识库时带引用号，例如「依据 [KB-01#5]」；本病例的数字不要标引用号。
- 不确定的地方明确写「需人工复核」，不要给模糊的乐观表述。
- 篇幅克制，不要罗列全部中间数据。

# 可用工具
{tools}

# 结束条件
当你能给出完整的规划结论时，直接输出最终回答，不要再调用工具。

⚠️ **绝不要在回答里「宣告」你接下来要做什么。** 下面这些写法都是错的：
- 「接下来，我将调用 plan_route 规划编号 3 的路径。」（宣告而非执行）
- 「调用 `plan_route` 函数，使用 1.5mm 的钳子进行路径规划。」（宣告而非执行）
- 「我先查一下知识库。」（宣告而非执行）

要么**真的发起工具调用**，要么**直接给出最终结论**。只描述计划不算完成任务。"""

# ⚠️ 2026-09-22 试过在铁律 7 后面补一段「替代方案是附加不是替换…中间推理里说清楚了不算」，
# 想治 E14 的「静默换目标」。**实测无效且有害，已回滚**：
#   · 无效：E14 / E14P 各 3 次，共 6 次里 4 次仍然静默换目标（换成 4 号 / 2 号交付）；
#   · 有害：E01P 从「先查病例再挑目标」退化成 `rank_candidates → plan_route`，
#           E02P 认错结节对象、体素量报 9（应为 2935）；
#           两条**各 3 次全部失败、三次回答一字不差** —— 不是波动，是稳定退步。
#   原因想通了：铁律 5 / 11 关于「结论必须完整附上 reachability 与 warnings 的**全部**提醒」
#   是**无条件**格式要求，而「越界时要**先**说明」是**条件性**的 ——
#   条件性规范竞争不过无条件格式；而多出来的那串动作序列指令
#   （「先说明…再附上…」）只会把模型往「尽快去规划」推。
#   → 这类需求要靠**结构**：让 `plan_route` 失败时把「被挡下的目标编号 + 可用范围」
#     做成返回体里的**独立字段**（而不是只写在 `error` 文案里），比再写一句提示词靠谱。
#   证据：`docs/评测区分度_同义改写实验.md` 第七节、`D:/tmp/pair_check.py`。


# 「宣告动作但没真的调用」的识别规则。
#
# 精度比召回重要：误判会把本来正确的最终回答打回去重做，白白多花一轮。
# 所以只认两类高置信度信号 ——
#   A. 第一人称 + 将来时态 + 动作动词（「我将…」「接下来，我们将…」）
#   B. 明确提到**真实存在的工具名**且处在「调用/执行/发起」的动宾结构里
# 实测能覆盖 E01（「接下来，我将尝试为…规划路径。」）与
# E07（「调用 `plan_route` 函数，使用 1.5mm 的钳子进行路径规划。」）。
#
# ⚠️ 规则 A 内部按「将来时标记的强度」分两级，别把它们合并：
#
#   强标记（将 / 会 / 要 / 准备）—— 语义就是「将要去做」，配 10 字宽窗口
#   弱标记（需要 / 想 / 得）    —— 语义多半是「必须要」，配 3 字紧窗口
#
# 混在一起会同时漏判和误判。实测（2026-09-17）：
#   「我准备改用 1.5mm 的器械重新规划。」   <- 真宣告，因为宽表里没有「改用」，
#                                              要等 12 字才碰到「重新规划」→ 漏判
#   「我需要在结论中说明规划依据。」        <- 正常回答，因为 8 字外有个「规划」
#                                              落在 10 字窗口内 → 误判
# 两句都只跟窗口宽度和动词表有关，所以按强度拆开、各自配窗口。
_ACTION_VERBS = (
    "调用|尝试|规划|查询|检索|检查|计算|执行|使用|获取|发起|重新|看看|查看|做一次"
    # 补进「改用 / 换成 / 换用 / 复查」：这些是紧跟在「我准备」后面的动作词，
    # 缺了它们就只能靠更远处的「重新规划」兜底，而那个距离超出窗口。
    "|改用|换成|换用|复查"
)
# 紧表：只认紧挨着弱标记的动作动词，不含「使用 / 重新 / 尝试」这类
# 在名词短语里也常见的词（「需要说明规划依据」里的「规划」就是名词用法）。
_TIGHT_ACTION_VERBS = "调用|查询|查|检索|查看|看看|执行|获取|试|计算|规划|改用|换成"

_INTENT_RULES = (
    # 「接下来，我将…」「现在我来…」（注意**刻意不含「需要」**，理由见上）
    re.compile(
        r"(?:接下来|下面|下一步|现在|然后|接着)[^。；\n]{0,12}?"
        r"(?:我|我们)(?:将|会|要|来|去|准备|开始)"
    ),
    # 「我将尝试…」「我们会先调用…」
    re.compile(
        r"我(?:们)?(?:将|会|要|准备)"
        r"(?:立即|马上|现在|先|再|继续)?[^。；\n]{0,10}?(?:" + _ACTION_VERBS + r")"
    ),
    # 「让我查一下」「我先试一遍」—— 同样是宣告，只是换了主语
    re.compile(
        r"(?:让我|我先|我们先|我来)"
        r"[^。；\n]{0,10}?(?:查|检索|调用|检查|查看|看看|规划|计算|获取|试|开始)"
    ),
    # 弱标记单独一条紧规则：窗口只给 3 字，只认紧邻的动作动词。
    # 这样「我需要查一下知识库」（查，距离 0）命中，
    # 而「我需要在结论中说明规划依据」（规划，距离 8）不命中。
    re.compile(
        r"我(?:们)?(?:需要|想|得)"
        r"(?:立即|马上|现在|先|再|继续)?[^。；\n]{0,3}?(?:" + _TIGHT_ACTION_VERBS + r")"
    ),
)

# 纠偏提示：说清楚「为什么不行」而不是笼统地催
_INTENT_CORRECTION = (
    "你上一条消息只是在**描述**下一步计划，并没有真正发起工具调用，所以任务还没完成。"
    "请二选一：\n"
    "1) 如果还需要数据 —— **立刻发起工具调用**。不要写「我将调用 X」，直接调用 X；\n"
    "2) 如果已经掌握足够信息 —— 直接给出完整的最终结论。\n"
    "不要再输出任何计划或意图描述。"
)

# 纠偏提示（第三类的**兜底**分支：把工具调用写成了正文，但提取不出可执行的调用）。
#
# ⚠️ 这条现在只在「检测到是工具调用、却一个都提取不出来」时用得上 ——
# 正常情况下规则 C 走的是**代为执行**（见模块 docstring「规则 C」与
# `_extract_toolcalls`），根本不进纠偏。理论上这两件事不该脱节
# （`_is_toolcall_as_text` 成立就意味着至少有一个 block 是合法调用），
# 所以这条兜底是**防御性**的：宁可留一条能劝就劝的路径，
# 也不要让一段既不是结论、又没执行的东西被当成答案交出去。
# 哨兵见 `verify_agent_loop.py` 的 `[4c]`（人为让提取返回空来制造这个退化）。
_TOOLCALL_TEXT_CORRECTION = (
    "你上一条消息把**工具调用写成了正文**（一整段 ```json {\"name\": …} ``` 代码块），"
    "这不是一次真正的调用 —— 它在 `content` 里，不在调用通道里，**所以工具并没有执行**，"
    "任务也还没完成。\n"
    "请二选一：\n"
    "1) 如果还需要数据 —— **通过调用通道真的发起这次调用**，不要把 JSON 写进回答正文；\n"
    "2) 如果已经掌握足够信息 —— 直接给出完整的最终结论，不要再输出任何 JSON。"
)

# 「我要调用某个工具」的第三种写法：动宾结构 + 真实工具名。
#
# ⚠️ 必须排除过去时。已完成的调用在回答里很常见：
#     「已调用 `plan_route` 得到长度 216.238mm」  <- 这是结论，不是宣告
# 所以对动词加否定后顾，挡住「已/经/过」开头的完成态。
_TOOL_REF_RE = re.compile(
    r"(?<!已)(?<!经)(?<!过)(?:调用|执行|发起|请求|运行)\s*"
    r"[`「“\"']?([A-Za-z_][A-Za-z0-9_]{2,})"
)

# 「请 / 您 + 调用 X」是在**指挥用户**，不是 Agent 自己的下一步计划。
#
# 实测（2026-09-17，E03-器械可通过性）这条误判的代价比想象中大：
# 回答结尾是一句给医生的操作建议 ——
#     「若需确认更粗的镜子是否可通过，请调用 `scan_device_fit`
#       并提高 `max_diameter_mm` 参数。」
# 它本该是合格的最终回答，却因为出现「调用 scan_device_fit」被判成「只在宣告」。
# 后果不只是多花一轮：模型被催着重答，第二次**编出了一个不存在的余量 2.715mm**，
# 把上一轮本来通过的用例直接搞挂。
#
# 所以：真正的宣告一定是第一人称的（「我将调用」「让我调用」），
# 第二人称的一律不算。规则 A（正则那三条）本来就都要求第一人称，
# 只有规则 B（认工具名）缺了这一层，在这里补上。
_ADDRESSES_USER = ("请", "您", "你", "建议", "如需", "可以", "可", "用户", "医生")
_USER_ADDRESS_WINDOW = 6

# 「如果您需要…我将…」是**给用户的礼貌性提议**，不是 Agent 的下一步计划。
#
# 这是实测代价最大的一条误判（2026-09-17，E09-术语含义检索）。
# 模型给出的回答**完全合格** —— 引了 [KB-02#2]、写出了计算公式、解释了怎么读，
# 只是末尾附了一句
#     「如果您需要进一步了解最窄余量的具体数值，请告知我具体病例和结节编号，
#        我将为您计算并提供详细信息。」
# 整段被规则 A 判成「只在宣告」，于是被就地打回重做两次。
# 而重答的结果**一次比一次差**：第三次连引用号都没有了，用例从通过变成失败。
#
# 关键区别：这句话的将来时是有**前提**的（「如果您需要」），
# 它既不是自我计划，也不影响前面那份回答的完整性。
#
# ⚠️ 但不能见到「如果」就放过 —— 下面这句仍然是真宣告：
#     「如果直接规划不行，我将改用更细的器械重新规划。」
# 区别在于**这句话是说给谁的**：「如果您需要…我将为您计算」主语是用户的需求，
# 是提议；「如果直接规划不行，我将…」是 Agent 自己在定下一步。
# 所以要求条件词与第二人称**同时出现**才算提议。
#
# 另外窗口必须按**整句**切（以 。！？；\n 为界，不是以 ，为界）——
# 「如果」离「我将」隔了两个逗号，按小句切窗会把它切掉。
_CONDITIONAL_MARKERS = (
    "如果", "若您", "如需", "如您", "如有需要", "需要的话", "倘若", "假如", "要是",
)
_USER_PRONOUNS = ("您", "你")

# ---------------------------------------------------------------- 第三类「没真的调用」
#
# 实测（2026-09-20，E07）模型输出的**整段**就是
#
#     ```json
#     {"name": "plan_route", "arguments": {"case_id": "LIDC_0089",
#      "candidate_id": 2, "device_diameter_mm": 1.9, "profile": "gentle_turn"}}
#     ```
#
# 没有散文、没有反引号包住的工具名 —— 规则 A（要第一人称将来时）与规则 B
# （要「调用/执行 + 工具名」）**全部漏判**，于是这段 JSON 被当成最终回答
# 交付给医生，`stop_reason` 还记成 `answered`。
#
# 而它想传的参数其实**完全正确**（候选编号、器械外径、代价档位都对），
# 纯粹死在输出协议上。所以「分数掉了」未必等于「模型变笨」。
#
# ⚠️ 为什么不去放宽规则 A / B 的召回：那两个有误判前科（E03 被催后编出
# 不存在的余量 2.715mm、E09 被催后丢掉带引用和公式的合格回答，见上文）。
# 放宽召回就是重蹈覆辙。这一路用的是**完全不同的判据** —— 不猜语义，
# 只问「这段文本能不能解析成一次真实的工具调用」。
# 给医生的最终结论不可能是裸 JSON，所以这条判据不会误伤；
# 它也不要求任何散文线索，所以不会漏判。
#
# 🔴 只在**本轮没有任何真实工具调用**时启用（见循环里 `if not reply.wants_tools`）。
# 实测 E07 第 1 步的正文里同样带着一段围栏 JSON，但那一轮模型**真的**发出了
# 三个调用 —— 若不加这个前提，那些调用会被误判成「没做完」而被打回去重做。
_TOOLCALL_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*\s*([\s\S]*?)```")

# 去掉 JSON 与围栏之后，剩下的「非标点字符」不超过这个数，就认为这条消息
# 本身就是一次（写成正文的）工具调用，而不是一段回答。
#
# ⚠️ 这个数**从 12 改到 50**（2026-09-22），原因值得记下来 ——
# 12 是照着手工写的样例校准的，不是照着**真实输出**校准的：
#
#   必须判成调用（残留字符数）      绝不能判成调用（残留字符数）
#     0  纯裸调用                    73  合格结论里贴了调用示例
#     0  纯围栏调用                 102  E07 第 2 步（散文 + 调用）
#     6  「好，我这就规划。」        36  工具返回体       ← 这两条是被
#    12  「先确认一下这个候选的编号。」 37  不存在的工具名   ← `_is_toolcall_payload`
#    14  「我再用 2.0mm 的器械试一次：」                       挡掉的，与阈值无关
#    32  **真实失败输出**（见下）
#
# 漏判的那次真实输出是：
#     「使用 1.5mm 的钳子和 `wide_airway` 代价配置重新规划路径：」+ 围栏 JSON
# 引导语 32 个字 > 12，于是识别器判「这不是调用」，把一整段 JSON 当答案交付、
# `stop_reason` 还记 `answered`。**手工样例（6~14 字）比真实输出（32 字）短得多，
# 阈值就落在真实分布之外了** —— 尺子要照真实输出校准，这条教训是通用的。
#
# 50 落在 32 与 73 之间：对最大真阳性留 1.56 倍余量，对最小真阴性留 1.46 倍。
# 为什么不是「刚好 33」：再出现一条更长的引导语就又会漏判，而两侧余量对等
# 才能让「下次该往哪边挪」有据可依。
# 重新校准用 `D:/tmp/residue_scale.py`（打印两侧分布），改完两侧哨兵都要过。
# 哨兵见 `agent/scripts/verify_agent_loop.py` 的第 [1b] / [2b] 层。
_TOOLCALL_RESIDUE_MAX = 50


def _is_conditional_offer(text: str, position: int) -> bool:
    """命中点所在的那一句，是不是「如果您需要…我将…」式的、说给用户的提议。"""
    start = 0
    for sep in "。！？；\n":
        cut = text.rfind(sep, 0, position)
        if cut != -1:
            start = max(start, cut + 1)
    sentence = text[start:position]
    has_condition = any(marker in sentence for marker in _CONDITIONAL_MARKERS)
    has_user = any(pronoun in sentence for pronoun in _USER_PRONOUNS)
    return has_condition and has_user


def _addresses_user(text: str, position: int) -> bool:
    """工具名之前的小句里有没有第二人称/祈使语。"""
    window = text[max(0, position - _USER_ADDRESS_WINDOW) : position]
    for sep in "，。；、\n：:（(":
        cut = window.rfind(sep)
        if cut != -1:
            window = window[cut + 1 :]
    return any(marker in window for marker in _ADDRESSES_USER)


def _is_self_plan(text: str, position: int) -> bool:
    """命中点是不是「第一人称 + 真要去做」，而不是提议或指挥用户。"""
    return not _is_conditional_offer(text, position) and not _addresses_user(text, position)


def _looks_like_intent_to_act(text: str, tool_names: set[str]) -> bool:
    """这段文字是不是「只在宣告下一步动作」，而不是最终回答？

    判定只看高置信度信号，宁可漏判也不误判：误判会把正确的最终回答
    打回去重做，凭空多花一轮，还可能把模型带偏 —— 实测两次教训：
      - E03 被误判后重答，第二次**编出了一个不存在的余量 2.715mm**；
      - E09 被误判后重答两次，**把一份带引用和公式的合格回答丢掉了**。
    """
    if not text:
        return False
    for pattern in _INTENT_RULES:
        for match in pattern.finditer(text):
            if _is_self_plan(text, match.start()):
                return True
    for match in _TOOL_REF_RE.finditer(text):
        if match.group(1) not in tool_names:
            continue
        if _is_self_plan(text, match.start()):
            return True
    return False


def _brace_blocks(text: str) -> list[str]:
    """切出文本里所有**顶层**的 `{...}` 片段（按引号与转义正确配平）。

    不用正则：`\\{[\\s\\S]*\\}` 是贪婪的，会把两段 JSON 和夹在中间的散文一起
    吞进去，于是「残留有多少字」就算不准 —— 而残留量正是这条判据的唯一阈值。
    """
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    blocks.append(text[start : index + 1])
                    start = -1
    return blocks


def _is_toolcall_payload(candidate: str, tool_names: set[str]) -> bool:
    """这段文本本身是不是一次工具调用？name 必须是**注册表里真实存在**的工具。

    要求 name 真实存在，是为了不误伤「回答里贴了一段示例 JSON」这种写法；
    要求同时带 arguments/parameters，是为了不误伤工具**返回体**（那些 JSON
    里也有 name 字段，但没有参数那一层）。

    兼容三种常见形态：
      {"name": ..., "arguments": {...}}
      {"name": ..., "parameters": {...}}
      {"tool_calls": [{"function": {"name": ..., "arguments": ...}}]}
    """
    try:
        payload = json.loads(candidate.strip())
    except (TypeError, ValueError):
        return False

    items: Any = payload
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        items = payload["tool_calls"]
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        return False

    for item in items:
        if not isinstance(item, dict):
            return False
        function = item.get("function")
        if isinstance(function, dict):
            item = function
        name = item.get("name")
        if not isinstance(name, str) or name not in tool_names:
            return False
        if not isinstance(item.get("arguments", item.get("parameters")), (dict, str)):
            return False
    return True


def _is_toolcall_as_text(text: str, tool_names: set[str]) -> bool:
    """整条消息就是一次（被写成正文的）工具调用 —— 它不是给医生的最终回答。

    判定分两级，任何一级成立即可：
      1. 整段（剥掉 ``` 围栏后）就是一个可解析的工具调用；
      2. 里面**确实**含一个工具调用 JSON，且去掉所有 JSON 与围栏之后，
         剩下的够不上一段话（非标点字符 ≤ `_TOOLCALL_RESIDUE_MAX`）。

    第 2 级是为了覆盖「好，我这就规划。 + JSON」这种过渡语开头的写法 ——
    它有散文，但那段散文不是结论，而且规则 A 也认不出「我这就规划」。
    """
    if not text or not tool_names:
        return False

    unwrapped = _TOOLCALL_FENCE_RE.sub(lambda match: match.group(1), text)
    if _is_toolcall_payload(unwrapped, tool_names):
        return True

    residue = unwrapped
    found = False
    for block in _brace_blocks(unwrapped):
        if _is_toolcall_payload(block, tool_names):
            found = True
        residue = residue.replace(block, " ")
    if not found:
        return False

    residue = _TOOLCALL_FENCE_RE.sub(" ", residue)
    prose = re.sub(r"[\s，。；、：！？~～—…\-—+*#>`'\"（）()\[\]]+", "", residue)
    return len(prose) <= _TOOLCALL_RESIDUE_MAX


def _extract_toolcalls(text: str, tool_names: set[str]) -> list[ToolCallRequest]:
    """把「被写成正文的」工具调用提取成真的 `ToolCallRequest`，交给执行通道。

    ⚠️ **入口自带 `_is_toolcall_as_text` 守卫，这是刻意的。**
    单独看「这段 JSON 能不能解析成调用」是不够的 ——
    一段合格的最终结论里如果贴了调用示例（
    「本次实际发起的调用形态是 ```{"name": "plan_route", …}```，需要说明的是…」），
    单看那一段 JSON 也完全合法。区别在**上下文**：正文的主体是结论，不是调用。
    那个判断属于 `_is_toolcall_as_text`（它数了「去掉 JSON 之后还剩多少字」）。
    守卫放在这里而不是只放在调用方，是因为**约定挡不住未来的误用**：
    少写一次前置检查，结果就是「把示例当调用执行」，而且静默。
    代价是重复跑一遍识别（毫秒级），换「不可能用错」，值。

    提取的是**全部**顶层 `{…}` 块，不只是第一段：模型把两个调用写在一起
    （`[{inspect_case}, {rank_candidates}]`）时，只执行第一个等于偷偷丢活儿。
    执行顺序**按它们在正文里出现的先后**，与模型写下的顺序一致；
    依赖关系（`plan_route` 依赖 `candidate_id`）仍由循环里原有的
    prerequisites 机制处理，这里不重复判断 —— 两处都管会互相打架。

    `id` 前缀用 `recovered_`：一眼能从 trace 里看出这次调用不是走
    `tool_calls` 通道来的。参数按原样重新序列化，不做规范化 ——
    模型写了什么就执行什么，否则「代为执行」就变成了「替模型改参数」。
    """
    if not _is_toolcall_as_text(text, tool_names):
        return []

    unwrapped = _TOOLCALL_FENCE_RE.sub(lambda match: match.group(1), text)
    calls: list[ToolCallRequest] = []
    for block in _brace_blocks(unwrapped):
        if not _is_toolcall_payload(block, tool_names):
            continue
        try:
            payload = json.loads(block.strip())
        except (TypeError, ValueError):
            continue
        items: Any = payload
        if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
            items = payload["tool_calls"]
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if isinstance(function, dict):
                item = function
            raw = item.get("arguments", item.get("parameters"))
            calls.append(
                ToolCallRequest(
                    id=f"recovered_{len(calls)}_{item['name']}",
                    name=item["name"],
                    arguments=raw if isinstance(raw, str)
                    else json.dumps(raw or {}, ensure_ascii=False),
                )
            )
    return calls


@dataclass
class AgentStep:
    """循环中的一步。"""

    index: int
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    # 因依赖关系被延后到下一步的调用。
    # ⚠️ 刻意不放进 tool_calls —— 评测器的 called_tools() 直接读 tool_calls，
    #    放进去等于「没执行也算调过」，那是在放水。
    deferred_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.content,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "deferred_calls": self.deferred_calls,
            "note": self.note,
        }


@dataclass
class AgentRun:
    """一次完整运行的留痕。"""

    question: str
    answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    stop_reason: str = ""
    elapsed_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    # 完整的工具调用记录（含未截断的 result）。
    # registry.trace() 里的 result 是截断过的预览，做「数字可溯源」不够用，
    # 这里单独留一份全文，评测器直接读它。
    tool_records: list[dict[str, Any]] = field(default_factory=list)
    # 「宣告动作但没真的调用」被纠偏了几次。>0 说明模型出现过这种行为，
    # 留痕是为了能区分「一次就答对」和「催了才答对」。
    intent_corrections: int = 0
    # 「把工具调用写成了正文」被**代为执行**了几个调用。
    #
    # 与 `intent_corrections` 分开记，因为它们是两种东西：前者是「催了几次」，
    # 后者是「替模型把 N 个调用补进了调用通道」。E07 从 0/5 到 5/5 靠的就是这条，
    # 不记的话报告里会显得「一次到位」，把擦屁股的活儿藏掉了。
    recovered_toolcalls: int = 0

    @property
    def tool_call_count(self) -> int:
        return sum(len(step.tool_calls) for step in self.steps)

    def registry_calls(self) -> list[dict[str, Any]]:
        """带完整返回值的工具调用记录，供评测打分器使用。"""
        return self.tool_records

    def to_dict(self, include_messages: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "steps": len(self.steps),
            "tool_calls": self.tool_call_count,
            "intent_corrections": self.intent_corrections,
            "recovered_toolcalls": self.recovered_toolcalls,
            "elapsed_s": round(self.elapsed_s, 2),
            "usage": self.usage,
            "backend": self.backend,
            "trace": [step.to_dict() for step in self.steps],
        }
        if include_messages:
            payload["messages"] = self.messages
        return payload


class NavAgent:
    """把工具注册表、会话状态与 LLM 串起来的 ReAct 循环。"""

    def __init__(
        self,
        client: OpenAICompatClient | ScriptedClient,
        registry: ToolRegistry,
        session: NavSession,
        max_steps: int = 8,
        keep_recent_tools: int = 4,
        max_context_chars: int = 24000,
        max_intent_corrections: int = 2,
        max_deferrals: int = 2,
        verbose: bool = False,
        system_prompt: str | None = None,
        environment_note: str = "",
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.session = session
        self.max_steps = max_steps
        self.keep_recent_tools = keep_recent_tools
        self.max_context_chars = max_context_chars
        # 纠偏次数上限。给 2 次：一次可能是模型没反应过来，
        # 两次还只有计划就说明它真的卡住了，再催也是浪费 token，
        # 这时如实记 stop_reason=unresolved_intent 交给人看。
        self.max_intent_corrections = max_intent_corrections
        # 依赖延后的次数上限。给 2 次是为了防死循环：
        # 万一模型每次都把前置工具和依赖工具打包发出来，延到一定次数就直接放行执行
        # （宁可让它猜一次，也不能把 max_steps 耗光导致整条链路无结论）。
        self.max_deferrals = max_deferrals
        # 真实存在的工具名，用于识别「宣告要调用某工具」这种文本
        # ⚠️ names 是方法不是属性 —— 写成 `set(registry.names)` 会得到
        # TypeError: 'method' object is not iterable，而且是在构造函数里炸，
        # 整个 Agent 一行都跑不了。eval harness 会把它记成「运行异常」而不是
        # 静默跳过，所以能立刻暴露；但命令行单跑更容易看出来。
        self._tool_names = set(registry.names())
        self.intent_corrections = 0
        self.deferrals_used = 0
        self.verbose = verbose
        # 步进事件回调。**默认 None = 完全不介入** ——
        # CLI 与评测路径一行都不受影响，也不会多出任何开销。
        # 网页版（`agent/web/server.py`）用它把每一步实时推给浏览器：
        # Agent 是同步阻塞的，所以服务端把它放进后台线程，
        # 事件经 asyncio.Queue 转发成 SSE。
        #
        # ⚠️ 观测层绝不能拖垮主流程：`_emit` 会吞掉回调抛出的任何异常。
        # 浏览器断开、前端写错字段，都不该让一次已经算对的规划失败。
        self.on_event = on_event
        # 「这台 Agent 正跑在什么界面里」。网页版用它告诉模型：
        # 出图之后系统会自己把三维视图做成卡片放在界面右侧，
        # **不要再往回答里贴本地文件路径** —— 浏览器点不开 `D:\...\x.html`，
        # 那是命令行场景才需要的交代。默认空串，CLI / 评测完全不受影响。
        base_prompt = system_prompt or SYSTEM_PROMPT.format(tools=registry.describe())
        if environment_note:
            base_prompt = f"{base_prompt}\n\n# 运行环境\n{environment_note.strip()}\n"
        self._system_prompt = base_prompt

    def _emit(self, event: dict[str, Any]) -> None:
        """把一步事件推给外部观察者（没有观察者就直接返回）。

        刻意吞异常：事件通道是**观测层**，不是控制流。
        前端断开连接、回调里写错字段，都不应该让 Agent 本身失败。
        """
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:  # noqa: BLE001 —— 见上面注释，这里就是要全吞
            pass

    # ------------------------------------------------------------ 主循环

    def run(self, question: str) -> AgentRun:
        started = time.time()
        self.registry.reset_trace()
        self.intent_corrections = 0
        self.deferrals_used = 0
        run = AgentRun(question=question, backend=self.client.label)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]
        self._emit(
            {
                "type": "start",
                "question": question,
                "backend": self.client.label,
                "max_steps": self.max_steps,
            }
        )

        for step_index in range(1, self.max_steps + 1):
            step = AgentStep(index=step_index)

            try:
                reply = self.client.chat(
                    messages, tools=self.registry.openai_tools()
                )
            except LLMError as error:
                run.stop_reason = f"llm_error: {error}"
                run.answer = f"模型调用失败：{error}"
                self._emit({"type": "error", "message": str(error)})
                break

            _merge_usage(run.usage, reply.usage)
            step.content = reply.content

            if not reply.wants_tools:
                text = reply.content.strip()
                if text:
                    # 先分辨这是「最终回答」还是「根本没做完」。
                    # 两类没做完都不能当答案交出去 —— 医生拿到的会是一句承诺，
                    # 或者一段他自己看不懂的 JSON。
                    announcing = _looks_like_intent_to_act(text, self._tool_names)
                    # 只在「本轮没有任何真实调用」时才查第三类。E07 第 1 步的正文里
                    # 也带着一段围栏 JSON，但那一轮真的发出了三个调用 —— 不加这个
                    # 前提，那些调用会被误判成没做完而被打回去重做。
                    as_text = not announcing and _is_toolcall_as_text(
                        text, self._tool_names
                    )
                    kind = "intent" if announcing else "toolcall_text" if as_text else ""

                    # ---------------------------------------------- 代为执行
                    # 规则 C 的**主路径**（不是纠偏）。这一段 JSON 里报的调用是完整的、
                    # 工具名真实存在、参数能过注册表校验，只是发错了通道 ——
                    # 那就按它说的执行，别去劝，更别把 JSON 当结论交给医生。
                    # 「劝」这条路已经用实测排除：temperature=0 下连发两条纠偏提示，
                    # 模型把同一段 JSON 原样重复了三次（2026-09-20，E07 探查）。
                    recovered = (
                        _extract_toolcalls(text, self._tool_names) if as_text else []
                    )
                    if recovered:
                        run.recovered_toolcalls += len(recovered)
                        step.note = (
                            f"把工具调用写成了正文，已代为执行 {len(recovered)} 个"
                            f"（{'、'.join(call.name for call in recovered)}）"
                        )
                        if self.verbose:
                            print(f"  [step {step_index}] ** {step.note}")
                        self._emit(
                            {
                                "type": "recovered",
                                "step": step_index,
                                "names": [call.name for call in recovered],
                                "note": step.note,
                            }
                        )
                        # 补成一次**合规的**调用轮次：下面执行完会依次追加 role=tool 的
                        # 回应，那些 tool_call_id 必须先有主，否则下一轮请求不合法。
                        # 注意这里走的是 `tool_calls` 通道 —— 与「把那段 JSON 塞回
                        # content 当上下文」是两回事：后者会让模型照着自己那段 JSON
                        # 继续写下去（正是连发三次的原因）。
                        reply = ChatReply(
                            content="",
                            tool_calls=recovered,
                            usage=reply.usage,
                            finish_reason=reply.finish_reason,
                            raw=reply.raw,
                        )
                        # 不 continue —— 落到下面的执行块，与真实调用走同一条路
                    elif kind and self.intent_corrections < self.max_intent_corrections:
                        self.intent_corrections += 1
                        # run 上记的是**累计**次数（报告要的是「一共被催了几次」），
                        # self 上的是**本轮额度**（可被进展重置，见循环末尾）
                        run.intent_corrections += 1
                        reason = (
                            "宣告动作但未调用工具"
                            if kind == "intent"
                            else "把工具调用写成了正文（但提取不出可执行的调用）"
                        )
                        step.note = (
                            f"第 {run.intent_corrections} 次「{reason}」，已注入纠偏提示"
                        )
                        if self.verbose:
                            print(f"  [step {step_index}] !! {step.note}")
                        self._emit(
                            {
                                "type": "correction",
                                "kind": kind,
                                "step": step_index,
                                "count": run.intent_corrections,
                                "note": step.note,
                            }
                        )
                        # 把它自己的话放回上下文，模型才知道我们在纠正什么。
                        # ⚠️ 只有「散文宣告」这一路可以这么做 —— 它那段话里没有
                        # 完整 JSON，照抄不会自我强化。规则 C 走的是上面那条路。
                        messages.append({"role": "assistant", "content": text})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    _INTENT_CORRECTION
                                    if kind == "intent"
                                    else _TOOLCALL_TEXT_CORRECTION
                                ),
                            }
                        )
                        run.steps.append(step)
                        continue
                    else:
                        run.answer = text
                        # 都没救成：如实记下是哪一种，不要假报 answered。
                        # 「把调用写成正文」单列一个值 —— 因为回答了「有没有完成」的
                        # stop_reason 一旦撒谎，所有只看它的上层（网页版、演示脚本）
                        # 都会被骗：E07 就是这样在交付一段裸 JSON 的同时报 answered。
                        run.stop_reason = (
                            "unresolved_intent"
                            if kind == "intent"
                            else "toolcall_as_text"
                            if kind == "toolcall_text"
                            else "answered"
                        )
                        run.steps.append(step)
                        break

                else:
                    # 空回复：既没内容也没工具调用，纠偏一次。
                    #
                    # ⚠️ 这一支必须是 `else`，不能靠「上面 `if text:` 的每条路都以
                    # continue/break 收尾」这个**隐式约定**。2026-09-22 加「代为执行」
                    # 分支时就是踩了这个坑：那条路故意不 continue（要落到执行块），
                    # 结果掉进这里，`step.note` 被覆盖成「空回复」，调用根本没执行，
                    # 但 `recovered_toolcalls` 已经记上了 1 —— 一个静默的假账。
                    # 是 `verify_agent_loop` 的 [4b] 把它抓出来的。
                    step.note = "空回复，已注入纠偏提示"
                    self._emit(
                        {
                            "type": "correction",
                            "kind": "empty",
                            "step": step_index,
                            "note": step.note,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "你刚才没有返回任何内容。请继续："
                                "要么调用工具获取数据，要么直接给出最终结论。"
                            ),
                        }
                    )
                    run.steps.append(step)
                    continue

            # ---- 执行工具 ----
            messages.append(_assistant_message(reply))

            # 依赖关系处理：同一批里若既有前置工具、又有依赖它的工具，后者延后一步。
            #
            # 为什么必须做成结构性保证（2026-09-17，E01）：
            # 模型为了省一轮，把 inspect_case / rank_candidates / plan_route 打包在同一条
            # 消息里一起发出来。于是 plan_route 的 candidate_id 只能是**瞎猜**的 ——
            # 实测猜中了不可达的候选 1，plan_route 直接 ok=false，
            # 而模型手里已经有了 rank_candidates 的全套数字，就拼出一份
            # 「规划已完成」的结论交差（数字还能溯源，因此连 grounding 都拦不住）。
            # 加了三轮提示词（含「有依赖关系的工具不要并行调用」）都压不住，
            # 所以在循环里兜底。
            batch = {call.name for call in reply.tool_calls}
            runnable: list[Any] = []
            deferred: list[tuple[Any, list[str]]] = []
            for call in reply.tool_calls:
                blocked_by = sorted(set(self.registry.prerequisites(call.name)) & batch)
                if blocked_by and self.deferrals_used < self.max_deferrals:
                    deferred.append((call, blocked_by))
                else:
                    runnable.append(call)

            for call in runnable:
                arguments_preview = call.parsed()
                if self.verbose:
                    print(f"  [step {step_index}] -> {call.name}({arguments_preview})")

                # 先发「开始」再发「完成」：网页版靠这一对把卡片渲染成
                # running → ok/fail，否则一次规划里几个工具会一起冒出来。
                self._emit(
                    {
                        "type": "tool_start",
                        "step": step_index,
                        "name": call.name,
                        "arguments": arguments_preview,
                    }
                )
                _tool_started = time.time()
                result = self.registry.execute(
                    call.name, call.arguments, context=self.session, step=step_index
                )
                step.tool_calls.append(
                    {"id": call.id, "name": call.name, "arguments": arguments_preview}
                )
                step.tool_results.append(
                    {"name": call.name, "ok": result.get("ok", False)}
                )
                self._emit(
                    {
                        "type": "tool_done",
                        "step": step_index,
                        "name": call.name,
                        "ok": bool(result.get("ok", False)),
                        "elapsed_ms": int((time.time() - _tool_started) * 1000),
                        "error": result.get("error"),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": _pack(result),
                    }
                )

            if deferred:
                self.deferrals_used += 1
                blocked_names = sorted({name for _, names in deferred for name in names})
                step.deferred_calls = [
                    {"name": call.name, "arguments": call.parsed(), "blocked_by": blocked_by}
                    for call, blocked_by in deferred
                ]
                step.note = (
                    f"延后一步：{[call.name for call, _ in deferred]}"
                    f"（须先看到 {blocked_names} 的结果，否则参数只能靠猜）"
                )
                if self.verbose:
                    print(f"  [step {step_index}] ~~ {step.note}")
                self._emit(
                    {
                        "type": "deferred",
                        "step": step_index,
                        "names": [call.name for call, _ in deferred],
                        "blocked_by": blocked_names,
                        "note": step.note,
                    }
                )
                # 每个 tool_call_id 都必须有回应，否则下一轮请求不合法。
                # 这里面说清「为什么没执行」和「接下来该怎么做」，模型才知道要重发。
                for call, blocked_by in deferred:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": _pack(
                                {
                                    "ok": False,
                                    "deferred": True,
                                    "error": (
                                        f"{call.name} 本次未执行，已延后到你看到 "
                                        f"{blocked_by} 的结果之后。"
                                        "它的参数（例如 candidate_id）必须先有那些结果"
                                        "才能确定，同一条消息里一起发出来只能靠猜。"
                                        "请阅读上面已返回的结果，再重新发起一次 "
                                        f"{call.name}。"
                                    ),
                                }
                            ),
                        }
                    )
                # 延后不是失败，也不要模型基于残缺信息收尾：
                # 什么都不执行时也必须回到循环顶部，让它带着新结果重新决策。

            # 纠偏额度按「进展」重置 —— 而不是整个 run 只给两次。
            #
            # ⚠️ 判据是「**这一轮有没有通过调用通道发出调用**」，**不是调用成功**。
            # 第一版写成「工具执行成功才还额度」，实测（2026-09-20，E07）不够用：
            # 模型在第 2、3 步各触发一次纠偏（先用散文宣告、再把调用写成正文），
            # 第 4 步**真的发出了** plan_route —— 但那次返回 ok=false（器械不可行），
            # 于是额度没还；第 5 步它想换更细的器械重试，写成正文时已无额度可用，
            # 又把 JSON 交了出去，仍然是 0/5。
            #
            # 想清楚这个机制管的是什么就不会写错：**纠偏治的是「输出通道」，
            # 不是「调用结果」**。调用失败归工具失败自愈那套机制管；只要它开始用
            # 调用通道说话，纠偏的活儿就干完了，额度该还。
            # 不会失控 —— 外侧还有 `max_steps` 兜底，且累计次数仍如实记在
            # `run.intent_corrections` 上（报告看的是累计值，不是剩余额度）。
            if self.intent_corrections:
                self.intent_corrections = 0

            run.steps.append(step)
            messages = self._compact(messages)

        else:
            run.stop_reason = "max_steps"
            run.answer = run.answer or self._force_summary(messages)

        if not run.answer and run.stop_reason == "max_steps":
            run.answer = self._force_summary(messages)

        run.elapsed_s = time.time() - started
        run.messages = messages
        run.tool_records = [
            {
                "step": call.step,
                "tool": call.name,
                "arguments": call.arguments,
                "ok": call.ok,
                "result": call.result,
                "error": call.error,
                "elapsed_ms": call.elapsed_ms,
            }
            for call in self.registry.calls
        ]
        # 收尾事件只发一次，把渲染统计栏要用的东西一并带上。
        # （不放循环内部发：answer 有「正常回答」和「max_steps 兜底总结」两条来路，
        #  在两处各发一次迟早会重复或漏掉一条。）
        self._emit(
            {
                "type": "final",
                "answer": run.answer,
                "stop_reason": run.stop_reason,
                "elapsed_s": round(run.elapsed_s, 2),
                "steps": len(run.steps),
                "tool_calls": run.tool_call_count,
                "intent_corrections": run.intent_corrections,
                "recovered_toolcalls": run.recovered_toolcalls,
                "usage": run.usage,
            }
        )
        return run

    # ------------------------------------------------------------ 上下文管理

    def _compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把较早的工具结果压缩成摘要，只保留最近若干条完整内容。

        这是长任务能跑下去的关键：一条 rank_candidates 的结果可能上千字符，
        跑十轮就会把窗口吃光，模型随后开始丢参数、重复调用。
        """
        tool_indexes = [
            index for index, item in enumerate(messages) if item.get("role") == "tool"
        ]
        keep = set(tool_indexes[-self.keep_recent_tools :])
        total = sum(len(str(item.get("content", ""))) for item in messages)

        if total <= self.max_context_chars:
            return messages

        compacted: list[dict[str, Any]] = []
        for index, item in enumerate(messages):
            if (
                item.get("role") == "tool"
                and index not in keep
                and not str(item.get("content", "")).startswith("[已压缩]")
            ):
                compacted.append(
                    {
                        **item,
                        "content": "[已压缩] 早期 "
                        + str(item.get("name", "工具"))
                        + " 的结果："
                        + _summarize(str(item.get("content", ""))),
                    }
                )
            else:
                compacted.append(item)
        return compacted

    def _force_summary(self, messages: list[dict[str, Any]]) -> str:
        """步数用尽时，要求模型基于已有信息直接收口。"""
        try:
            reply = self.client.chat(
                messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "已达到工具调用步数上限。请立即基于现有信息给出"
                            "最终结论，明确说明哪些部分尚未验证、需要人工复核。"
                        ),
                    }
                ],
                tools=None,
            )
            _merge_usage({}, reply.usage)
            return reply.content.strip() or "（模型未能给出结论）"
        except LLMError as error:
            return f"（收口时模型调用失败：{error}）"


# ------------------------------------------------------------------ 工具函数


def _pack(result: dict[str, Any]) -> str:
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"ok": False, "error": "工具结果无法序列化"}, ensure_ascii=False)


def _assistant_message(reply: ChatReply) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": reply.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in reply.tool_calls
        ],
    }


def _summarize(text: str, limit: int = 240) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:limit]
    if isinstance(data, dict):
        keys = list(data)
        return f"字段 {keys}（原始 {len(text)} 字符）"
    return text[:limit]


def _merge_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in (source or {}).items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
        else:
            target[key] = value


__all__ = [
    "AgentRun",
    "AgentStep",
    "NavAgent",
    "SYSTEM_PROMPT",
    "_looks_like_intent_to_act",
]
