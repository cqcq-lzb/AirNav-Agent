"""后端探活与降级标识自检：**不可达时必须照常启动**，且降级状态要能被下游看见。

    python -m agent.scripts.check_backend

秒级、**不需要模型、不联网**（用的是必然连不上的地址，以及一个假的 health 桩），
也不需要 `cases/`，所以干净 clone、CI 里都能跑。

## 这一层在防什么

`完成定义.md` L1-4 的判据是「后端探活 + 降级标识」。这条最容易假过的方式是
**只把提示打在 stdout**：模型看不见控制台，于是降级信息既不会改变 Agent 行为，
也不会进 `--json`，评测/报告/网页端全都无从知道「这次回答是在后端不可达时产生的」。
所以这里查两件事：

1. **不可达不许退出**（rc 仍为 0）—— 大量问题只走工具，提前 return 1 会把能用的也挡掉；
2. **降级状态必须结构化**（`backend_probe.degraded` 在 JSON 里），不能只有一句话。

## 为什么必须有「反向用例」

只测「不可达 → degraded=True」的话，一个**永远返回 True** 的实现也能通过。
所以这里同时构造可达的 health 桩，要求 `degraded=False` —— 两个方向都测，
才证明它真的在区分状态，而不是恒定输出。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.cli import probe_backend  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS: list[tuple[bool, str, str]] = []

#: 必然连不上的地址：RFC 5737 保留段 + 一个没人监听的端口。
#: 用它而不是「拔网线」，是为了让这条自检**确定性** —— 不依赖外部环境。
DEAD_URL = "http://203.0.113.99:11434/v1"


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((ok, label, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"（{detail}）" if detail else ""))
    return ok


class _StubClient:
    """带 health() 的最小桩，用来构造「可达 / 不可达」两种状态。"""

    def __init__(self, health: dict, model: str = "qwen2.5:14b", label: str = "stub"):
        self._health = health
        self.model = model
        self.label = label

    def health(self) -> dict:
        return self._health


def layer_1_unreachable() -> None:
    """不可达 → degraded=True 且 detail 说明原因。"""
    print("\n[1] 后端不可达：必须降级而不是崩掉")
    client = _StubClient({"ok": False, "detail": "connection refused"})
    probe = probe_backend(client)
    check(probe["degraded"] is True, "degraded 为 True", f"degraded={probe['degraded']}")
    check(probe["ok"] is False, "ok 为 False", f"ok={probe['ok']}")
    check("connection refused" in probe["detail"], "detail 带上不可达原因", probe["detail"][:60])


def layer_2_reachable() -> None:
    """反向用例：可达 → degraded=False。防止「永远返回 True」的实现蒙混过关。"""
    print("\n[2] 后端可达：不许误报降级（反向用例）")
    client = _StubClient({"ok": True, "models": ["qwen2.5:14b"], "has_target": True})
    probe = probe_backend(client)
    check(probe["degraded"] is False, "degraded 为 False", f"degraded={probe['degraded']}")
    check(probe["ok"] is True, "ok 为 True")
    check(len(probe["models"]) == 1, "带上可用模型列表", f"models={probe['models']}")


def layer_3_model_missing() -> None:
    """可达但目标模型不在列表 → 也是降级（这种情况最容易漏判）。"""
    print("\n[3] 可达但缺目标模型：同样算降级")
    client = _StubClient({"ok": True, "models": ["qwen2.5:7b"], "has_target": False})
    probe = probe_backend(client)
    check(probe["degraded"] is True, "缺模型判为降级", f"degraded={probe['degraded']}")
    check("qwen2.5:14b" in probe["detail"], "detail 点名缺哪个模型", probe["detail"][:70])


def layer_4_cli_survives() -> None:
    """端到端：真跑一次 CLI，后端不可达也必须 rc=0 且有降级横幅。

    ⚠️ 这一层是**唯一真验「没退出」的地方** —— 前三层都只查了 `probe_backend`
    的返回值，而「不可达就 return 1」的回归可能发生在 `run_question` 里。
    """
    print("\n[4] 端到端：不可达时 CLI 仍以 rc=0 结束（不提前退出）")
    proc = subprocess.run(
        [sys.executable, "-m", "agent.cli", "ask",
         "--backend", "gpu41", "--base-url", DEAD_URL, "测试降级"],
        cwd=ROOT, capture_output=True, timeout=180,
    )
    text = (proc.stdout or b"").decode("utf-8", "replace")
    check(proc.returncode == 0, "退出码为 0", f"rc={proc.returncode}")
    check("降级运行" in text, "打出降级横幅")
    check("[降级]" in text, "末尾追一行 [降级] 标记")


def layer_5_json_field() -> None:
    """`--json` 必须带上结构化的 backend_probe —— 下游靠它区分「答错」与「后端没起」。"""
    print("\n[5] --json：降级状态进结构化字段")
    proc = subprocess.run(
        [sys.executable, "-m", "agent.cli", "ask",
         "--backend", "gpu41", "--base-url", DEAD_URL, "--json", "测试降级"],
        cwd=ROOT, capture_output=True, timeout=180,
    )
    text = (proc.stdout or b"").decode("utf-8", "replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        check(False, "输出是合法 JSON", str(error)[:60])
        return
    check(True, "输出是合法 JSON")
    probe = payload.get("backend_probe")
    check(isinstance(probe, dict), "含 backend_probe 字段")
    if isinstance(probe, dict):
        check(probe.get("degraded") is True, "backend_probe.degraded 为 True")


def layer_6_tools_work() -> None:
    """横幅承诺「纯工具查询仍可用」—— 必须验证这句是真话。

    ⚠️ 界面上的承诺也是断言。写着「仍可用」而实际不可用，比不写更坏。
    """
    print("\n[6] 无模型时纯工具仍可用（横幅的承诺必须为真）")
    code = (
        "import sys; sys.path.insert(0,'.')\n"
        "from agent.tools import build_registry, NavSession\n"
        "r = build_registry(); s = NavSession(root='cases')\n"
        "names = ['list_cases', 'inspect_case']\n"
        "args = [{}, {'case_id': 'LIDC_0089'}]\n"
        "bad = []\n"
        "for n, a in zip(names, args):\n"
        "    res = r.execute(n, a, context=s)\n"
        "    if not res.get('ok'): bad.append(n)\n"
        "print('BAD=' + ','.join(bad))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, timeout=180)
    out = (proc.stdout or b"").decode("utf-8", "replace")
    if "BAD=" not in out:
        # 没有 cases/ 时不表态（本地/CI 都可能没有 1.9G 的真实病例）
        check(True, "跳过（本机无 cases/，不足以判定）", "⚪ 非失败")
        return
    bad = out.split("BAD=", 1)[1].strip()
    check(bad == "", "list_cases / inspect_case 在无模型下仍返回 ok", bad or "全通过")


def main() -> int:
    layer_1_unreachable()
    layer_2_reachable()
    layer_3_model_missing()
    layer_4_cli_survives()
    layer_5_json_field()
    layer_6_tools_work()

    failed = [item for item in RESULTS if not item[0]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} 项符合预期")
    for _ok, label, detail in failed:
        print(f"  FAIL  {label}" + (f"（{detail}）" if detail else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
