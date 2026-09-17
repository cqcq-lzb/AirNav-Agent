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

   目前的判定分两路，**每一路都要求「这是 Agent 在说自己的下一步」**：
     - 规则 A：第一人称 + 将来时标记 + 动作动词。
       「将/会/要/准备」是强标记（配 10 字窗口）；
       「需要/想/得」是弱标记（中文里常表示「必须」，配 3 字紧窗口）。
     - 规则 B：动宾结构 + **真实存在的**工具名。
   两路都要再过两道闸：
     - `_is_conditional_offer` —— 「如果您需要…我将…」是说给用户的提议，放过。
       （但要求条件词与第二人称同时出现：「如果直接规划不行，我将改用…」
        仍是真宣告，不能放过。）
     - `_addresses_user` —— 「请/您/您可以/建议 + 调用 X」是在指挥用户，放过。
4. **工具失败自愈**：工具的异常被注册表兜住并转成 {"ok": false, "error": ...}，
   模型有机会读到错误、改参数重试。这是 Agent 比单次调用强的地方。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .llm.client import ChatReply, LLMError, OpenAICompatClient, ScriptedClient
from .tools.registry import ToolRegistry
from .tools.session import NavSession

SYSTEM_PROMPT = """你是「经支气管肺结节导航系统」的路径规划助手，服务对象是呼吸科医生与介入操作者。

# 你的职责
把医生的自然语言需求转成对规划工具的调用序列，并把工具返回的数值组织成可直接使用的规划结论。

# 第一步：先判断问题属于哪一类，再决定调什么
| 医生问的是 | 第一步必须调用 |
|---|---|
| 某个结节的路径 / 可达性 / 完整规划 | inspect_case（先确认编号口径） |
| 为什么选这条 / 各分项代价占比 | explain_route_choice |
| 为什么这么设计 / 某个数的含义 / 口径来源 / 算法与训练细节 | search_knowledge |
| 某个器械能不能过 / 最粗能过多粗 | scan_device_fit |
| 要一张三维图 | render_viewer |
| 结节良恶性 / 是否该活检 / 器械型号推荐 / 风险概率 | 不调工具，直接说明能力边界 |

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
3. 每条路径结论必须完整附上 reachability 分级与 warnings 里的全部提醒，
   尤其是不利信息（路径太窄、依赖修复气道、靶点偏离气道）不得省略或淡化。
4. 工具返回 ok=false 时，读 error 判断原因再决定换参数重试还是换目标，
   不要用完全相同的参数反复重试。
5. 规划失败时主动给出可执行的替代方案：换更细的器械、换代价配置、或改用其他结节候选。
   可用 scan_device_fit 查清「最粗能过多粗的器械」再给建议。
6. 解释「为什么选这条」时，必须调 explain_route_choice 拿分项代价，不能自己解释分数。
   报占比而非只有总分 —— 同一个总分下「长而宽」与「短而窄」的临床含义完全不同。
7. 不属于本系统能力范围的问题（结节良恶性、是否该活检、器械型号推荐、风险概率），
   直接说明边界后转到你能提供的几何信息，不要给临床判断。

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


@dataclass
class AgentStep:
    """循环中的一步。"""

    index: int
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.content,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
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
        verbose: bool = False,
        system_prompt: str | None = None,
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
        # 真实存在的工具名，用于识别「宣告要调用某工具」这种文本
        # ⚠️ names 是方法不是属性 —— 写成 `set(registry.names)` 会得到
        # TypeError: 'method' object is not iterable，而且是在构造函数里炸，
        # 整个 Agent 一行都跑不了。eval harness 会把它记成「运行异常」而不是
        # 静默跳过，所以能立刻暴露；但命令行单跑更容易看出来。
        self._tool_names = set(registry.names())
        self.intent_corrections = 0
        self.verbose = verbose
        self._system_prompt = system_prompt or SYSTEM_PROMPT.format(
            tools=registry.describe()
        )

    # ------------------------------------------------------------ 主循环

    def run(self, question: str) -> AgentRun:
        started = time.time()
        self.registry.reset_trace()
        self.intent_corrections = 0
        run = AgentRun(question=question, backend=self.client.label)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]

        for step_index in range(1, self.max_steps + 1):
            step = AgentStep(index=step_index)

            try:
                reply = self.client.chat(
                    messages, tools=self.registry.openai_tools()
                )
            except LLMError as error:
                run.stop_reason = f"llm_error: {error}"
                run.answer = f"模型调用失败：{error}"
                break

            _merge_usage(run.usage, reply.usage)
            step.content = reply.content

            if not reply.wants_tools:
                text = reply.content.strip()
                if text:
                    # 先分辨这是「最终回答」还是「只在宣告下一步动作」。
                    # 后者绝不能当答案交出去 —— 医生拿到的会是一句承诺。
                    announcing = _looks_like_intent_to_act(text, self._tool_names)
                    if announcing and self.intent_corrections < self.max_intent_corrections:
                        self.intent_corrections += 1
                        run.intent_corrections = self.intent_corrections
                        step.note = (
                            f"第 {self.intent_corrections} 次「宣告动作但未调用工具」，"
                            "已注入纠偏提示"
                        )
                        if self.verbose:
                            print(f"  [step {step_index}] !! {step.note}")
                        # 把它自己的话放回上下文，模型才知道我们在纠正什么
                        messages.append({"role": "assistant", "content": text})
                        messages.append({"role": "user", "content": _INTENT_CORRECTION})
                        run.steps.append(step)
                        continue

                    run.answer = text
                    # 纠偏用尽仍只有计划：如实记为未完成，不要假报 answered
                    run.stop_reason = (
                        "unresolved_intent" if announcing else "answered"
                    )
                    run.steps.append(step)
                    break

                # 空回复：既没内容也没工具调用，纠偏一次
                step.note = "空回复，已注入纠偏提示"
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
            for call in reply.tool_calls:
                arguments_preview = call.parsed()
                if self.verbose:
                    print(f"  [step {step_index}] -> {call.name}({arguments_preview})")

                result = self.registry.execute(
                    call.name, call.arguments, context=self.session, step=step_index
                )
                step.tool_calls.append(
                    {"id": call.id, "name": call.name, "arguments": arguments_preview}
                )
                step.tool_results.append(
                    {"name": call.name, "ok": result.get("ok", False)}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": _pack(result),
                    }
                )

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
