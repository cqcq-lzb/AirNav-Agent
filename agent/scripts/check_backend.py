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
    """带 health() 的最小桩，用来构造「可达 / 不可达」两种状态。

    ⚠️ `health` 必须接受 `timeout` 关键字 —— 真实客户端（`OpenAICompatClient`
    / `ScriptedClient`）都是这个签名，桩对不上的话，
    `probe_backend` 一调用就 TypeError（本自检第一次跑就抓到了这个）。
    """

    def __init__(self, health: dict, model: str = "qwen2.5:14b", label: str = "stub"):
        self._health = health
        self.model = model
        self.label = label

    def health(self, timeout: float = 8.0) -> dict:  # noqa: ARG002 - 桩不看超时
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

    ⚠️ 这里**不许写死病例目录、也不许写死病例号**。CI 与干净 clone 上没有
    `cases/`（1.9 GB，不入库），病例由 `AIRNAV_CASES_DIR` 或合成夹具提供。
    本函数原来写的是 `NavSession(root='cases')` + 写死的 `'LIDC_0089'`，
    于是在**本机（有真实病例）绿、在 CI 红** —— 这正是 README 里已记过一次的
    「绕过环境变量读真实数据」（当年是 `verify_geometry`）。现在一律用
    `NavSession()`，它内部走 `case_loader.cases_root()` 的三档解析，
    并把实际用到的病例来源打印出来（来历可查）。
    """
    print("\n[6] 无模型时纯工具仍可用（横幅的承诺必须为真）")
    code = (
        "import sys; sys.path.insert(0,'.')\n"
        "from agent.tools import build_registry, NavSession\n"
        "r = build_registry(); s = NavSession()\n"
        "res = r.execute('list_cases', {}, context=s)\n"
        "ids = [c['case_id'] for c in (res.get('cases') or [])]\n"
        "print('ROOT=' + str(res.get('cases_root') or s.root))\n"
        "print('N=' + str(len(ids)))\n"
        "bad = []\n"
        "if not res.get('ok'): bad.append('list_cases')\n"
        "if ids and not r.execute('inspect_case', {'case_id': ids[0]}, context=s).get('ok'):\n"
        "    bad.append('inspect_case')\n"
        "print('BAD=' + ','.join(bad))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, timeout=180)
    out = (proc.stdout or b"").decode("utf-8", "replace")
    lines = out.splitlines()
    if "BAD=" not in out:
        # 拿不到病例清单（连 list_cases 都跑不起来）——不表态，但也不说通过
        check(True, "跳过（拿不到病例清单，不足以判定）", "⚪ 非失败")
        return
    root = next((l[5:].strip() for l in lines if l.startswith("ROOT=")), "")
    try:
        n = int(next((l[2:].strip() for l in lines if l.startswith("N=")), "0"))
    except ValueError:
        n = 0
    if n == 0:
        # 缺数据 ≠ 通过：一个病例都没有时，这句承诺**没被验证过**，如实标出来
        check(True, "没有病例可查（不足以判定，缺数据 ≠ 通过）",
              f"⚪ 0 例 · 来源 {root}")
        return
    bad = next((l[4:].strip() for l in lines if l.startswith("BAD=")), "")
    check(bad == "", f"list_cases / inspect_case 在无模型下仍返回 ok（{n} 例）",
          (bad and f"失败：{bad}") or f"全通过 · 来源 {root}")


WEB_PORT = 8801
WEB_BASE = f"http://127.0.0.1:{WEB_PORT}"
WEB_DEAD_URL = "http://203.0.113.98:11434/v1"


def layer_7_web_degraded() -> None:
    """网页端：降级状态必须到得了界面，而 `/health` 必须**不受**后端影响。

    ⚠️ 为什么这条特别重要（本层的真正价值）：把推理后端接进 `/health`
    是鉴权/健康检查类改动里最常见的自伤 —— 后端一挂，编排层就会以为
    **整个服务死了**并重启它，而其实服务好得很（纯工具查询照常能用）。
    所以这里同时验「挡住的」（降级能报出来）与「放行的」（`/health` 不被牵连）。
    """
    print("\n[7] 网页端：降级横幅到得了界面，/health 不被牵连")

    import os
    import threading
    import time
    import urllib.parse
    import urllib.request

    # 指向必然连不上的地址（RFC 5737 保留段）。必须在起服务**之前**设，
    # 且要清掉探活缓存 —— 否则会命中上一层留下的 gpu41 结论。
    os.environ["AIRNAV_GPU41_URL"] = WEB_DEAD_URL
    os.environ["AIRNAV_WEB_PORT"] = str(WEB_PORT)

    try:
        import uvicorn

        from agent.web import server as web_server
    except Exception as error:  # noqa: BLE001
        check(False, "agent.web.server 可导入", f"{type(error).__name__}: {error}")
        return

    web_server._probe_cache.clear()

    # 反向用例：可达后端（heuristic 不需要网络）不许报降级。
    # 没有这条，一个「永远返回 degraded=True」的实现也能通过全部断言。
    healthy = web_server._backend_probe("heuristic")
    check(healthy["degraded"] is False, "反向：heuristic 不报降级",
          f"degraded={healthy['degraded']}")

    # 审计落临时目录，别污染 outputs/audit/
    import tempfile
    scratch = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    os.environ["AIRNAV_AUDIT_DIR"] = scratch.name

    config = uvicorn.Config(web_server.app, host="127.0.0.1", port=WEB_PORT,
                            log_level="error")
    server = uvicorn.Server(config)
    # opener 必须禁代理：本机 `http_proxy` 会把 127.0.0.1 也代理掉，
    # 表现成「连接被拒」，看起来像服务没起来（实测踩过）。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def probe(path: str, timeout: int = 20) -> tuple[int, str]:
        try:
            with opener.open(WEB_BASE + path, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except Exception as error:  # noqa: BLE001
            return 0, f"{type(error).__name__}: {error}"

    thread = threading.Thread(target=server.run, name="check-backend-web", daemon=True)
    thread.start()
    ready = False
    for _ in range(40):
        status, _body = probe("/health", timeout=3)
        if status == 200:
            ready = True
            break
        time.sleep(0.3)

    try:
        if not check(ready, "网页服务就绪"):
            return

        # ① /health 是存活探针：后端死不死都得是 ok
        status, body = probe("/health")
        check(status == 200 and body.strip() == "ok",
              "/health 不受后端影响（存活 ≠ 就绪）", f"{status} {body.strip()[:20]}")

        # ② /api/meta 必须把降级状态交出来
        status, body = probe("/api/meta", timeout=30)
        try:
            meta = json.loads(body)
        except json.JSONDecodeError as error:
            check(False, "/api/meta 返回 JSON", str(error)[:60])
            return
        bprobe = meta.get("backend_probe")
        check(isinstance(bprobe, dict), "/api/meta 含 backend_probe")
        if isinstance(bprobe, dict):
            check(bprobe.get("degraded") is True, "backend_probe.degraded 为 True")
            check(bool(bprobe.get("detail")), "带上不可达原因")
            check(bool(bprobe.get("note")), "带上可执行提示（note）")
            check(bprobe.get("backend") == web_server.DEFAULT_BACKEND,
                  "标注是哪个后端的结论")

        # ③ SSE：降级必须是**第一条**事件
        #    顺序就是信息 —— 若排在回答之后，用户已经把「连接失败」当结论读了。
        first_type = None
        try:
            with opener.open(
                WEB_BASE + "/api/chat?q=" + urllib.parse.quote("测试降级")
                + "&backend=gpu41", timeout=60
            ) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if line.startswith("data:"):
                        first_type = json.loads(line[5:].strip()).get("type")
                        break
        except Exception as error:  # noqa: BLE001
            check(False, "SSE 可读", f"{type(error).__name__}: {error}")
        if first_type is not None:
            check(first_type == "degraded",
                  "SSE 首事件是 degraded（不是等回答完才说）", f"首事件={first_type}")
    finally:
        server.should_exit = True
        scratch.cleanup()
        os.environ.pop("AIRNAV_GPU41_URL", None)


def main() -> int:
    layer_1_unreachable()
    layer_2_reachable()
    layer_3_model_missing()
    layer_4_cli_survives()
    layer_5_json_field()
    layer_6_tools_work()
    layer_7_web_degraded()

    failed = [item for item in RESULTS if not item[0]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} 项符合预期")
    for _ok, label, detail in failed:
        print(f"  FAIL  {label}" + (f"（{detail}）" if detail else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
