"""回归门禁：一条命令跑完所有常驻自检，并拿评测基线与当前结果对比。

## 为什么要有个统一入口

自检脚本已经攒到十二套（几何对拍 / 规划数值一致性 / 循环控制流 / 工具返回体契约 /
网页端 / 界面路径 / MCP 协议 / 打分器自检 / 评测基线 / 审计日志 / PHI 守卫 / 运行报告），
但**没人会记着全跑一遍**。门禁的价值不在「多了一个脚本」，而在把「改动前先过回归」
从一个习惯变成一条命令 —— 习惯会忘，命令不会。

退出码：`0` = 全绿；`1` = 有失败项（或基线倒退）；`2` = 门禁自身环境有问题。

## 两个必须照顾的现实

1. **`cases/` 不入库**（1.9GB 真实病例 + 隐私，见 `.gitignore`）。因此门禁分两档：
   - `--profile local`（默认）：全部检查，依赖病例数据的也能跑；
   - `--profile ci`：只跑**不依赖病例数据**的子集，适合 CI 或干净 clone。
   计划里本该有的「CI 上跑全量」需要病例 fixture，这是已知缺口，写在报告里而不是假装没有。

2. **自检渲染不能污染入库产物**。门禁把 `AIRNAV_VIEWER_DIR` 指到临时目录 ——
   和 `verify_agent_loop` / `run_eval` 里的 `use_scratch_output()` 是同一个原则：
   自检产生的东西是证据，不是交付物，不该混进 `outputs/viewers/`。

## 用法

    python -m agent.scripts.gate                    # 全量
    python -m agent.scripts.gate --profile ci       # 干净环境子集
    python -m agent.scripts.gate --fast             # 跳过评测（省 ~50s）
    python -m agent.scripts.gate --only planner,geometry
    python -m agent.scripts.gate --list             # 看检查项清单
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "gate"
BASELINE = ROOT / "agent" / "eval" / "baseline.json"

# 最后一次运行的报告（固定文件名，避免堆积）
REPORT_JSON = REPORT_DIR / "gate_report.json"
REPORT_MD = REPORT_DIR / "gate_report.md"


# ------------------------------------------------------------------ 检查项定义


@dataclass
class Check:
    """一个检查项。"""

    key: str
    title: str
    argv: list[str]
    needs: str = "python"  # python | node
    needs_cases: bool = False  # 依赖 cases/ 里的真实病例数据
    hard: bool = False  # 数值类硬门禁：错了就是错了，不允许商量
    no_user_site: bool = False  # 子进程屏蔽 user-site（只对 MCP 项开，见下方注释）
    timeout: int = 900
    category: str = ""
    # 运行后由 runner 回填
    ok: bool = False
    seconds: float = 0.0
    detail: str = ""
    output_tail: list[str] = field(default_factory=list)


def _checks() -> list[Check]:
    """门禁清单。顺序有意为之：便宜的排前面，早失败早停。"""
    return [
        Check(
            key="graders",
            title="打分器自检（植入缺陷必须被抓住）",
            argv=["-m", "agent.eval.selftest"],
            category="评测",
        ),
        Check(
            key="geometry",
            title="几何对拍（cKDTree 与 scipy EDT 逐位一致）",
            argv=["-m", "agent.scripts.verify_geometry"],
            hard=True,
            category="数值",
        ),
        Check(
            key="planner",
            title="规划数值一致性（与 V1 逐位一致）",
            argv=["-m", "agent.scripts.verify_planner"],
            needs_cases=True,
            hard=True,
            category="数值",
        ),
        Check(
            key="loop",
            title="ReAct 循环控制流",
            argv=["-m", "agent.scripts.verify_agent_loop"],
            needs_cases=True,
            category="Agent",
        ),
        Check(
            key="payloads",
            title="工具返回体契约",
            argv=["-m", "agent.scripts.verify_payloads"],
            needs_cases=True,
            category="Agent",
        ),
        Check(
            key="web",
            title="网页端（含鉴权、产物 URL 与目录穿越防护）",
            argv=["-m", "agent.scripts.check_web"],
            category="界面",
        ),
        Check(
            key="audit",
            title="审计日志（哈希链 · 篡改必被检出）",
            argv=["-m", "agent.scripts.check_audit"],
            category="治理",
        ),
        Check(
            key="phi",
            title="PHI 守卫（数据面不含患者可识别信息）",
            argv=["-m", "agent.scripts.scan_phi"],
            # 扫的是 cases/ 里的真实病例包，所以干净 clone 跑不了这一项
            needs_cases=True,
            category="治理",
        ),
        Check(
            key="report",
            title="运行报告（缺数据不说通过 · 篡改必被检出）",
            argv=["-m", "agent.scripts.check_report"],
            # 全程用临时审计目录与临时门禁文件，不依赖 cases/，CI 档也能跑
            needs_cases=False,
            category="治理",
        ),
        Check(
            key="clinical",
            title="临床报告（参数与页面逐位一致 · 打印不留白框）",
            argv=["-m", "agent.scripts.check_clinical"],
            # 夹具是 outputs/viewers/ 里入库的 sidecar（JSON），不需要 cases/，
            # 但**没有它们就没法验「参数与页面所见一致」**，所以自检会明确报错而不是静默跳过
            needs_cases=False,
            category="临床",
        ),
        Check(
            key="ui_paths",
            title="界面路径替换（真回答 → 可点按钮）",
            argv=["agent/scripts/check_ui_paths.js"],
            needs="node",
            timeout=300,
            category="界面",
        ),
        Check(
            key="mcp",
            title="MCP 协议四层自检",
            argv=["-m", "agent.scripts.check_mcp"],
            # MCP 依赖闭包是「不读 user-site」锁定出来的（见 requirements_mcp.txt 与
            # lock_mcp_deps 的来历）。**只有这一项**该屏蔽 user-site：屏蔽它才能证明
            # MCP 那套依赖真在自己的环境里齐活。
            # ⚠️ 千万别把它设成全局 —— 本机 `requests` 这类包只在 user-site，
            # 全局屏蔽的结果是 4 套自检一起 ModuleNotFoundError（2026-09-20 实测踩过）。
            no_user_site=True,
            category="协议",
        ),
        Check(
            key="eval",
            title="评测基线（15 用例 / 6 打分器）",
            argv=[
                "-m",
                "agent.scripts.run_eval",
                "--backend",
                "heuristic",
                "--stem",
                "gate_eval",
            ],
            needs_cases=True,
            hard=True,
            timeout=1200,
            category="评测",
        ),
    ]


# ------------------------------------------------------------------ 环境探测


def _decode(raw: bytes) -> str:
    """按 utf-8 → mbcs → gbk → latin-1 降级解码。

    Windows 原生工具（tasklist/netstat 之类）输出的是 OEM 代码页（中文机 = GBK），
    直接按 UTF-8 解码会在 reader 线程抛 UnicodeDecodeError，外层表现成「命令没输出」——
    这个坑本项目踩过，见 `agent/cli.py:_console_text`。
    """
    for encoding in ("utf-8", "mbcs", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _resolve_python() -> str:
    """优先项目 venv —— 门禁的结论只在项目解释器下有约束力。"""
    override = os.environ.get("AIRNAV_PYTHON")
    if override and Path(override).is_file():
        return override
    for candidate in (
        ROOT / ".venv-mcp" / "Scripts" / "python.exe",
        ROOT / ".venv-mcp" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _resolve_node() -> str | None:
    override = os.environ.get("AIRNAV_NODE")
    if override and Path(override).is_file():
        return override
    found = shutil.which("node")
    if found:
        return found
    managed = Path.home() / ".workbuddy" / "binaries" / "node"
    if managed.is_dir():
        for version in sorted(managed.glob("versions/*/node.exe"), reverse=True):
            return str(version)
        for version in sorted(managed.glob("versions/*/bin/node"), reverse=True):
            return str(version)
    return None


def _child_env(scratch: Path, check: Check) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if check.no_user_site:
        env["PYTHONNOUSERSITE"] = "1"
    else:
        # 显式不带这个开关：本机项目解释器看得见 user-site，
        # 而 `requests` 之类只在那里。默认屏蔽会静默制造 ImportError。
        env.pop("PYTHONNOUSERSITE", None)
    # 自检渲染走临时目录：证据归证据，交付物归交付物
    env["AIRNAV_VIEWER_DIR"] = str(scratch)
    # 审计同理：自检写下的 run/auth 记录属于**证据**，不该混进 outputs/audit/
    # （否则「跑了多少次门禁」会污染真正需要保留的运行留痕）
    env["AIRNAV_AUDIT_DIR"] = str(scratch / "audit")
    return env


# ------------------------------------------------------------------ 运行


def run_check(check: Check, python: str, node: str | None, scratch: Path) -> Check:
    if check.needs == "node":
        if not node:
            check.ok = False
            check.detail = "跳过：找不到 node"
            return check
        argv = [node, *check.argv]
    else:
        argv = [python, *check.argv]

    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(ROOT),
            capture_output=True,
            timeout=check.timeout,
            env=_child_env(scratch, check),
        )
        raw = (proc.stdout or b"") + (proc.stderr or b"")
        text = _decode(raw)
        check.ok = proc.returncode == 0
        check.detail = f"exit={proc.returncode}"
    except subprocess.TimeoutExpired:
        text = f"超时（>{check.timeout}s）"
        check.ok = False
        check.detail = "timeout"
    except OSError as error:  # 解释器不存在之类
        text = f"{type(error).__name__}: {error}"
        check.ok = False
        check.detail = "launch failed"
    check.seconds = round(time.time() - started, 1)

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    check.output_tail = lines[-12:]

    # 从输出里再捞一层信息：不少脚本用 print 报告「N/N 通过」
    for line in lines:
        if "通过" in line and ("/" in line or "%" in line):
            check.detail = f"{check.detail} · {line.strip()[:60]}"
            break
    return check


def parse_eval(python: str, scratch: Path) -> dict | None:
    """读门禁那次评测的 JSON，用来和基线比。"""
    path = ROOT / "outputs" / "eval" / "gate_eval.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return {
        "backend": data.get("backend"),
        "total": data.get("total"),
        "passed": data.get("passed"),
        "pass_rate": data.get("pass_rate"),
        "all_graders_pass": data.get("all_graders_pass"),
        "cases": [
            {
                "case_id": case.get("case_id"),
                "passed": case.get("passed"),
                "elapsed_s": case.get("elapsed_s"),
            }
            for case in data.get("cases", [])
        ],
    }


def compare_baseline(eval_summary: dict | None, update: bool) -> tuple[bool, list[str]]:
    """拿当前评测结果和基线比。返回 (是否通过, 说明行)。"""
    if eval_summary is None:
        return True, ["未产生评测结果（被 --fast/--only 跳过），本次不做基线对比"]

    notes: list[str] = []
    if update or not BASELINE.is_file():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": eval_summary["backend"],
            "total": eval_summary["total"],
            "passed": eval_summary["passed"],
            "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": (
                "规则基线（无 LLM），确定性可复现。"
                "门槛规则：passed 不得低于本文件，total 不得减少。"
                "提分或降分都必须更新本文件并写清原因（尺子变了要说，不能偷偷变）。"
            ),
        }
        BASELINE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        notes.append(
            f"{'已更新' if update else '首次运行，已写入'}评测基线："
            f"{payload['passed']}/{payload['total']}"
        )
        return True, notes

    try:
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return False, [f"基线文件损坏：{type(error).__name__}: {error}"]

    notes.append(
        f"基线 {base.get('passed')}/{base.get('total')}（{base.get('recorded_at')}）"
        f" → 本次 {eval_summary['passed']}/{eval_summary['total']}"
    )
    ok = True
    if (eval_summary["passed"] or 0) < (base.get("passed") or 0):
        ok = False
        notes.append("❌ 通过数低于基线 —— 是模型/机制退步，还是打分器变松？必须写清原因")
    if (eval_summary["total"] or 0) < (base.get("total") or 0):
        ok = False
        notes.append("❌ 用例总数少于基线 —— 用例被删了，评测尺子变短了")
    if eval_summary.get("all_graders_pass") is False:
        ok = False
        notes.append("❌ 有打分器自检未通过（植入缺陷未被抓住）")
    if ok:
        notes.append("✅ 未低于基线")
    return ok, notes


# ------------------------------------------------------------------ 报告


def write_reports(checks: list[Check], eval_summary, baseline_ok, baseline_notes, profile: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    failed = [c for c in checks if not c.ok]
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile,
        "passed": not failed and baseline_ok,
        "total_checks": len(checks),
        "failed_checks": [c.key for c in failed],
        "checks": [
            {
                "key": c.key,
                "title": c.title,
                "category": c.category,
                "hard": c.hard,
                "needs_cases": c.needs_cases,
                "ok": c.ok,
                "seconds": c.seconds,
                "detail": c.detail,
                "tail": c.output_tail,
            }
            for c in checks
        ],
        "eval": eval_summary,
        "baseline_ok": baseline_ok,
        "baseline_notes": baseline_notes,
    }
    REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 回归门禁报告",
        "",
        f"- 时间：{payload['generated_at']}",
        f"- 档位：`{profile}`",
        f"- 结论：**{'通过' if payload['passed'] else '未通过'}**"
        f"（{len(checks) - len(failed)}/{len(checks)} 项通过）",
        "",
        "| 检查 | 类别 | 硬门禁 | 结果 | 耗时 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for check in checks:
        mark = "✅" if check.ok else "❌"
        lines.append(
            f"| {check.title} | {check.category} | {'是' if check.hard else '—'} | {mark} | "
            f"{check.seconds}s | {check.detail} |"
        )
    if eval_summary:
        lines += [
            "",
            "## 评测基线",
            "",
            f"- 后端：`{eval_summary['backend']}`",
            f"- 通过：**{eval_summary['passed']}/{eval_summary['total']}**"
            f"（{round((eval_summary['pass_rate'] or 0) * 100, 1)}%）",
        ]
    lines += ["", "## 基线与门槛", ""] + [f"- {note}" for note in baseline_notes]
    if failed:
        lines += ["", "## 失败明细", ""]
        for check in failed:
            lines += [f"### {check.title}", "", f"```text", *check.output_tail, "```", ""]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description="AirNav-Agent 回归门禁")
    parser.add_argument("--profile", choices=("local", "ci"), default="local",
                        help="local=全量；ci=只跑不依赖病例数据的子集")
    parser.add_argument("--only", default="", help="只跑这些 key（逗号分隔）")
    parser.add_argument("--skip", default="", help="跳过这些 key（逗号分隔）")
    parser.add_argument("--fast", action="store_true", help="跳过评测（省时间，不做基线对比）")
    parser.add_argument("--update-baseline", action="store_true", help="把本次评测结果写成新基线")
    parser.add_argument("--list", action="store_true", help="列出检查项")
    args = parser.parse_args()

    checks = _checks()
    if args.list:
        for check in checks:
            tags = []
            if check.hard:
                tags.append("硬门禁")
            if check.needs_cases:
                tags.append("需病例数据")
            print(f"{check.key:<10} {check.title}  [{'/'.join(tags) or '常规'}]")
        return 0

    if args.profile == "ci":
        checks = [c for c in checks if not c.needs_cases]
    if args.fast:
        checks = [c for c in checks if c.key != "eval"]
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        checks = [c for c in checks if c.key in wanted]
    if args.skip:
        dropped = {item.strip() for item in args.skip.split(",") if item.strip()}
        checks = [c for c in checks if c.key not in dropped]
    if not checks:
        print("没有要跑的检查项", file=sys.stderr)
        return 2

    python = _resolve_python()
    node = _resolve_node()
    print("=" * 70)
    print(f"AirNav-Agent 回归门禁 · profile={args.profile} · {len(checks)} 项")
    print(f"  解释器  {python}")
    print(f"  node    {node or '(未找到，界面检查会被跳过)'}")
    print("=" * 70)

    scratch = Path(tempfile.mkdtemp(prefix="airnav_gate_"))
    passed, failed = [], []
    try:
        for check in checks:
            print(f"▶ {check.title} …", end="", flush=True)
            run_check(check, python, node, scratch)
            print(f" {'✅' if check.ok else '❌'} {check.seconds}s  {check.detail}")
            (passed if check.ok else failed).append(check)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    eval_summary = None
    if any(c.key == "eval" for c in checks):
        eval_summary = parse_eval(python, scratch)
    baseline_ok, baseline_notes = compare_baseline(eval_summary, args.update_baseline)

    write_reports(checks, eval_summary, baseline_ok, baseline_notes, args.profile)

    print("-" * 70)
    for note in baseline_notes:
        print(note)
    print("-" * 70)
    verdict = not failed and baseline_ok
    print(f"结论：{'✅ 全绿' if verdict else '❌ 未通过'}  ({len(passed)}/{len(checks)} 项)")
    if failed:
        print("失败项：" + "、".join(c.key for c in failed))
        for check in failed:
            print(f"\n--- {check.title} 末尾输出 ---")
            for line in check.output_tail:
                print("  " + line)
    print(f"报告：{REPORT_MD}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
