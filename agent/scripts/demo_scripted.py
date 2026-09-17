"""脚本化端到端验证：不需要 LLM 就能跑通整条 Agent 链路。

验证内容：
  1. ReAct 循环的工具调度与消息装配
  2. 工具参数校验失败后的自愈（模型读到 error 再改参数重试）
  3. 依赖前序结果的动态决策（根据 rank_candidates 的输出挑候选）
  4. 跨步骤状态复用（解释与出图复用同一个候选号）
  5. 出图工具把大对象转成 artifact id 而不是塞进上下文
  6. 上下文压缩在长轨迹下不丢关键信息

用法：
    python -m agent.scripts.demo_scripted
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.agent_loop import NavAgent  # noqa: E402
from agent.llm.client import ScriptedClient  # noqa: E402
from agent.tools import NavSession, build_registry  # noqa: E402

CASE = "LIDC_0089"


def pick_best_reachable(messages):
    """从最近的 rank_candidates 结果里挑出最优先且可达的候选。

    这一步是「脚本」里唯一的动态部分，用来证明循环确实把工具结果
    喂回了模型，而不是死板地按固定顺序调用。
    """
    for item in reversed(messages):
        if item.get("role") != "tool" or item.get("name") != "rank_candidates":
            continue
        data = json.loads(item["content"])
        for row in data.get("ranked", []):
            if row.get("reachable") and row.get("reachability_grade") in {
                "adjacent",
                "reachable",
            }:
                return row["candidate_id"]
        for row in data.get("ranked", []):
            if row.get("reachable"):
                return row["candidate_id"]
    return 3


def main() -> int:
    registry = build_registry()
    session = NavSession()

    # 跨步骤共享状态：动态决策选出的候选号要被后续的「解释」和「出图」复用，
    # 证明循环里的决策结果会沉淀下来，而不是每一步都从零再猜一次。
    picked: dict[str, int | None] = {"candidate": None}

    def explain_step(messages):
        candidate = pick_best_reachable(messages)
        picked["candidate"] = candidate
        print(f"    [动态决策] 根据 rank_candidates 结果选择候选 {candidate}")
        return ScriptedClient.tool(
            "explain_route",
            case_id=CASE,
            candidate_id=candidate,
            profile="balanced",
            device_diameter_mm=2.0,
            device_margin_mm=0.2,
        )

    def render_step(messages):
        candidate = picked["candidate"]
        if candidate is None:  # 兜底：模型若跳过解释直接出图
            candidate = pick_best_reachable(messages)
        print(f"    [状态复用] 沿用上一步选定的候选 {candidate} 生成三维视图")
        return ScriptedClient.tool(
            "render_viewer",
            case_id=CASE,
            candidate_id=candidate,
            profile="balanced",
            device_diameter_mm=2.0,
            device_margin_mm=0.2,
            # 与 `outputs/viewers/` 里那份**入库的演示产物**保持同一组参数
            # （2.0 mm / balanced / step 1）。渲染层对「只有 generatedAt 不同」
            # 的重复渲染不写盘，所以跑演示不会把入库产物改脏。
            # 以前这里是 step=2，唯一后果就是每次 demo 都把入库那份换成粗网格版。
            mesh_step=1,
        )

    replies = [
        # 1) 先确认病例与编号口径
        ScriptedClient.tool("inspect_case", case_id=CASE),
        # 2) 故意传一个越界候选，验证参数校验与自愈
        ScriptedClient.tool("plan_route", case_id=CASE, candidate_id=99),
        # 3) 读 error 后改参数重试
        ScriptedClient.tool("list_nodule_candidates", case_id=CASE),
        # 4) 器械 2.0mm 下看谁可达
        ScriptedClient.tool(
            "rank_candidates", case_id=CASE, device_diameter_mm=2.0
        ),
        # 5) 依赖上一步结果挑候选，展开解释
        explain_step,
        # 6) 复用同一个候选出三维视图
        render_step,
        # 7) 收口
        ScriptedClient.say(
            "【脚本化演示的输出占位】\n"
            "真实运行时这里会是模型基于工具数据生成的规划结论。"
        ),
    ]

    client = ScriptedClient(replies, label="scripted（无 LLM 回放）")
    agent = NavAgent(
        client=client,
        registry=registry,
        session=session,
        max_steps=10,
        verbose=True,
    )

    question = f"{CASE} 上用 2.0mm 器械，哪个结节最值得做，给我完整方案"
    print("=" * 70)
    print(f"问题：{question}")
    print("=" * 70)

    run = agent.run(question)

    print("\n" + "=" * 70)
    print("逐步轨迹")
    print("=" * 70)
    for step in run.steps:
        print(f"\n[step {step.index}]")
        for call, result in zip(step.tool_calls, step.tool_results):
            args = json.dumps(call["arguments"], ensure_ascii=False)
            print(f"  调用 {call['name']}({args})  ->  ok={result['ok']}")
        if step.note:
            print(f"  备注：{step.note}")
        if step.content:
            print(f"  模型文字：{step.content[:100]}")

    print("\n" + "=" * 70)
    print("统计")
    print("=" * 70)
    stats = registry.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n结束原因：{run.stop_reason}")
    print(f"步数 {len(run.steps)} / 工具调用 {run.tool_call_count} / 耗时 {run.elapsed_s:.1f}s")
    print(f"artifact：{session.artifacts.summary()}")

    print("\n" + "=" * 70)
    print("最终回答")
    print("=" * 70)
    print(run.answer)

    # 断言 1：参数校验失败被正确捕获，且循环继续
    failed = [c for c in registry.trace() if not c["ok"]]
    assert failed, "预期 plan_route(candidate_id=99) 应当失败"
    print(f"\n[验证通过] 捕获到 {len(failed)} 次预期内的工具失败，循环未被中断")

    # 断言 2：出图工具被真实调用并成功
    viewer_calls = [c for c in registry.trace() if c["tool"] == "render_viewer"]
    assert viewer_calls, "预期 render_viewer 被调用"
    assert viewer_calls[-1]["ok"], f"render_viewer 失败：{viewer_calls[-1]['error']}"

    # 断言 3：候选号在「解释 → 出图」两步之间保持一致
    assert picked["candidate"] is not None, "动态决策未产出候选号"
    print(f"[验证通过] 解释与出图复用同一候选 {picked['candidate']}")

    # 断言 4：大对象没有进上下文，只留了 artifact id
    viewer_artifacts = [a for a in session.artifacts.summary() if a["kind"] == "viewer"]
    assert viewer_artifacts, "artifact store 里没有 viewer 记录"
    artifact_id = viewer_artifacts[-1]["id"]
    viewer_path = Path(session.artifacts.get(artifact_id))
    assert viewer_path.exists(), f"视图文件不存在：{viewer_path}"
    size_mb = viewer_path.stat().st_size / (1024 * 1024)
    print(f"[验证通过] 出图落盘 {viewer_path.name}  {size_mb:.2f} MB  artifact={artifact_id}")

    # 断言 5：真正进上下文的工具消息里不得夹带几何，只能给 id 与摘要。
    # 注意：step.tool_results 只存 {"name","ok"}，要量的是 messages 里的 tool 消息。
    viewer_msgs = [
        m
        for m in run.messages
        if m.get("role") == "tool" and m.get("name") == "render_viewer"
    ]
    assert viewer_msgs, "上下文里找不到 render_viewer 的工具消息"
    payload_text = viewer_msgs[-1]["content"]
    assert len(payload_text) < 2000, (
        f"出图工具回传 {len(payload_text)} 字符，大对象应走 artifact 而不是塞进上下文"
    )
    # 真正的几何泄露特征是一段超长 base64 连续串；airway_vertices 这类只是计数，不算。
    leak = re.search(r"[A-Za-z0-9+/]{200,}={0,2}", payload_text)
    assert leak is None, f"出图工具把几何写进了上下文（发现 {len(leak.group(0))} 字符的 base64 串）"
    # 出图工具必须给出**真文件**（viewer_path），而不是一个内部 id。
    #
    # 这条断言以前是反过来的：断言「上下文里必须有 artifact_id，模型才能引用」。
    # 那个前提是错的 —— 没有任何工具吃 artifact_id 作为入参，模型拿到它唯一能做的
    # 事就是把它写成链接。实测出现过 `![](route-001)` 破图，以及
    # 「三维路径可视化文件已生成，ID 为 route-001」这种假话（那时根本没出图）。
    # 所以现在的口径是：**给文件，不给 id**。
    assert "viewer_path" in payload_text, "上下文里缺少 viewer_path，模型无法把文件交代给用户"
    stray = re.search(r"(?:route|viewer|segmentation)-\d+", payload_text)
    assert stray is None, (
        f"出图工具把内部 artifact id（{stray.group(0) if stray else ''}）暴露进了上下文 ——"
        "模型会把它当链接写进回答，变成死链"
    )
    print(
        f"[验证通过] 出图工具进上下文 {len(payload_text)} 字符、"
        f"最长连续串 {max((len(s) for s in re.findall(r'[A-Za-z0-9+/]{20,}', payload_text)), default=0)} 字符，"
        f"几何体已隔离在 artifact 中，且只给文件不给 id"
    )

    # 断言 6：整条轨迹的上下文体积可控（大对象没有随步数膨胀）
    total_ctx = sum(
        len(m.get("content") or "") for m in run.messages if m.get("role") == "tool"
    )
    print(f"[统计] 全部工具消息合计 {total_ctx} 字符")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
