"""Agent 循环控制流的针对性验证（不需要 LLM）。

针对的是一类**很隐蔽**的失败：模型输出「接下来，我将调用 plan_route 重新规划路径。」
这样的**计划**，既非空、又没有工具调用，循环如果不加分辨就会把这段文字
当成最终答案交出去 —— 医生拿到的是一句承诺，不是规划结论。

实测（gpu41 / qwen2.5:14b）评测用例 E01、E07 就是这么挂的。

这个脚本分四层验：

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

from agent.agent_loop import NavAgent, _looks_like_intent_to_act  # noqa: E402
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
    failures += check_correction_happens()
    failures += check_gives_up_honestly()
    failures += check_normal_run_untouched()
    failures += check_dependency_deferral()

    print("\n" + "=" * 72)
    if failures:
        print(f"未通过，共 {len(failures)} 项：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(
        f"通过：识别器召回 {len(SHOULD_DETECT)}/{len(SHOULD_DETECT)}、"
        f"精度 {len(SHOULD_NOT_DETECT)}/{len(SHOULD_NOT_DETECT)}，"
        "纠偏 / 如实记账 / 不干扰正常轨迹 / 依赖延后四条控制流全部符合预期"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
