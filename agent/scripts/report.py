"""合规与运行报告：把审计日志、评测基线与门禁结果汇成一份能直接交出去的 HTML。

## 为什么需要它

审计日志是**证据**，但证据得能给人看。合规审查问的是几个很具体的问题：
「这段时间谁用过、问了什么病例、系统给了什么结论、有没有人试图越权进来、
数据面有没有患者信息、评测有没有退步」。这些散在 JSONL 与 JSON 里，
需要一份能直接打印、能直接转发给评审方的汇总。

这是 P2 里最快见效的一项 —— 它不需要新能力，只是把已经落地的证据重新组织一遍。

## 一条设计原则：**缺数据 ≠ 通过**

报告最危险的失效模式不是算错，而是**在信息不足时仍然说「一切正常」**。
一份永远说「正常」的报告比没有报告更糟：它会替代人的判断。

所以每一项结论都是三态：

| 状态 | 含义 |
|---|---|
| `ok`   | 有数据，且符合预期 |
| `bad`  | 有数据，且不符合预期 |
| `none` | **没有数据** —— 明说「未执行 / 无记录」，不计入通过 |

后果直接落在数字上：**0 次运行时成功率显示 `—`，不显示 100%**；
一次 PHI 扫描都没跑过时写「未执行扫描」，不写「未发现患者信息」。
任一 `bad` → 结论标红且退出码 1（可进 CI）；出现 `none` → 单独列成「未能判定」，
让人知道这份报告的结论覆盖面到哪为止。

## 不进报告的东西

问答原文**默认不入报告**，只给字数与答案哈希。报告是要被转发、打印、归档的，
把原文抄进去等于把审计日志的脱敏又还原一次。要带原文得显式加 `--include-text`，
并自己确认接收方可信。

## 用法

    python -m agent.scripts.report                       # 最近 7 天 → outputs/reports/
    python -m agent.scripts.report --days 30 --out D:\\tmp\\r.html
    python -m agent.scripts.report --since 2026-09-01 --until 2026-09-20
    python -m agent.scripts.report --format json         # 只出机器可读的汇总
    python -m agent.scripts.report --include-text        # ⚠️ 报告将含问答原文

退出码：`0` = 已生成且无 bad；`1` = 有 bad（链被破坏 / PHI 命中 / 评测倒退 / 门禁未通过）；
`2` = 环境问题（审计目录不可读）。
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import audit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "reports"
GATE_JSON = ROOT / "outputs" / "gate" / "gate_report.json"
BASELINE = ROOT / "agent" / "eval" / "baseline.json"

# 三态。用字符串而不是 bool：`None` 表达不了「没数据」和「数据说不行」的区别
OK, BAD, NONE = "ok", "bad", "none"

STATE_LABEL = {OK: "符合预期", BAD: "不符合预期", NONE: "未能判定"}
STATE_COLOR = {OK: ("#0f766e", "#ccfbf1"), BAD: ("#a32d2d", "#fee2e2"), NONE: ("#78786f", "#f1f0ec")}

# 门禁结果超过这个小时数就当「陈旧」，不作为本次结论依据
GATE_STALE_HOURS = 24


# ------------------------------------------------------------------ 采集


def _select_logs(since: datetime | None, until: datetime | None) -> list[Path]:
    """按**文件名日期**粗筛日志文件（`audit-YYYY-MM-DD.jsonl`）。

    粗筛的代价写在报告的「方法与边界」里：一天的文件会整份纳入，
    文件内某条记录的 `ts` 若落在区间外，仍会参与统计。选粗筛是因为
    分片边界天然对齐到天，而精确到条要扫全文 —— 天粒度是这台机器的真实切分单位。
    """
    picked: list[Path] = []
    for path in audit.list_logs():
        day = path.stem.replace("audit-", "")
        if since and day < since.strftime("%Y-%m-%d"):
            continue
        if until and day > until.strftime("%Y-%m-%d"):
            continue
        picked.append(path)
    return picked


def _in_range(ts: str | None, since: datetime | None, until: datetime | None) -> bool:
    """按记录自身的 `ts` 精筛（粗筛之外的第二道）。解析不了就放行，宁可多算不漏算。"""
    if not ts:
        return True
    try:
        moment = datetime.fromisoformat(ts)
    except ValueError:
        return True
    if moment.tzinfo:
        moment = moment.replace(tzinfo=None)
    if since and moment < since:
        return False
    if until and moment > until + timedelta(days=1):
        return False
    return True


def _qa_view(record: dict, side: str, include_text: bool) -> dict:
    """问答字段的呈现形态。默认只留字数与哈希 —— 报告不是脱敏日志的第二个副本。"""
    node = record.get(side)
    if isinstance(node, str):  # 兼容早期/外部写入的裸字符串
        node = {"redacted": False, "text": node}
    if not isinstance(node, dict):
        return {"chars": 0, "redacted": None, "text": None, "sha256": None}
    text = node.get("text")
    chars = node.get("chars") if node.get("redacted") else len(text or "")
    return {
        "chars": chars or 0,
        "redacted": bool(node.get("redacted")),
        "sha256": node.get("sha256"),
        "text": text if (include_text and not node.get("redacted")) else None,
    }


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _age_hours(stamp: str | None) -> float | None:
    if not stamp:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return (datetime.now() - datetime.strptime(stamp[:19], fmt)).total_seconds() / 3600
        except ValueError:
            continue
    return None


def collect(*, since: datetime | None, until: datetime | None, include_text: bool) -> dict:
    """把三处证据读成一份汇总。**只读，不写任何东西。**"""
    audit_dir = audit.audit_dir()
    logs = _select_logs(since, until)

    # ⚠️ 这里不能把空列表直接交给 verify_chain：在 store 修掉 falsy 陷阱之前，
    # 空列表会被当成「没指定」→ 拿全量去校验 → 报告期内无记录却报「链完整」。
    # store 已改为显式区分，这里仍保留早返回，避免依赖那个修复被回退。
    if not logs:
        return {
            "has_data": False,
            "audit_dir": str(audit_dir),
            "logs": [],
            "records": 0,
            "stats": _empty_stats(),
            "chain": {"state": NONE, "problems": [], "tail_hash": "", "per_file": []},
            "runs": [],
            "denied": [],
            "auths": [],
            "phi": [],
            "tools": [],
            "artifacts": [],
            "gate": _load_gate(),
            "baseline": _read_json(BASELINE) if BASELINE.is_file() else None,
        }

    raw = list(audit.iter_records(logs))
    broken_lines = [r for r in raw if r.get("_broken")]
    records = [r for r in raw if not r.get("_broken")]
    records = [r for r in records if _in_range(r.get("ts"), since, until)]

    chain_ok, problems = audit.verify_chain(logs)
    per_file = []
    for path in logs:
        file_records = [r for r in audit.iter_records([path]) if not r.get("_broken")]
        per_file.append({
            "name": path.name,
            "records": len(file_records),
            "first": file_records[0].get("ts") if file_records else None,
            "last": file_records[-1].get("ts") if file_records else None,
            "tail": (file_records[-1].get("hash") if file_records else None),
        })

    runs = [r for r in records if r.get("event") == "run"]
    auths = [r for r in records if r.get("event") == "auth"]
    phis = [r for r in records if r.get("event") == "phi_scan"]

    # 工具调用统计（含失败数：只看「调了几次」会掩盖「调了但都失败」）
    tool_total: Counter[str] = Counter()
    tool_fail: Counter[str] = Counter()
    for record in runs:
        for step in record.get("steps") or []:
            name = step.get("tool")
            if not name:
                continue
            tool_total[name] += 1
            if step.get("ok") is False:
                tool_fail[name] += 1

    artifacts = []
    for record in runs:
        for item in record.get("artifacts") or []:
            if item.get("file"):
                artifacts.append({
                    "file": item.get("file"),
                    "kind": item.get("kind"),
                    "bytes": item.get("bytes"),
                    "sha256": item.get("sha256"),
                    "ts": record.get("ts"),
                    "run_id": record.get("run_id"),
                })

    # 链的结论只看「文件是否被改动」，链尾哈希单独打印 —— 因为整份日志被换掉
    # 是哈希链唯一挡不住的，只能靠链尾哈希在别处留过底
    chain_state = BAD if (problems or broken_lines) else OK
    tail_hash = per_file[-1]["tail"] if per_file else None

    return {
        "has_data": True,
        "audit_dir": str(audit_dir),
        "logs": [p.name for p in logs],
        "records": len(records),
        "broken_lines": broken_lines[:20],
        "stats": {
            "runs": len(runs),
            "runs_ok": sum(1 for r in runs if r.get("verdict") == "ok"),
            "auths": len(auths),
            "auth_denied": sum(1 for r in auths if not r.get("allowed")),
            "phi_scans": len(phis),
            "phi_hits": sum(int(r.get("confirmed") or 0) for r in phis),
            "artifacts": len(artifacts),
        },
        "chain": {
            "state": chain_state,
            "problems": problems[:20],
            "problem_count": len(problems),
            "tail_hash": tail_hash,
            "per_file": per_file,
        },
        "runs": [
            {
                "ts": r.get("ts"),
                "run_id": r.get("run_id"),
                "actor": r.get("actor"),
                "client_ip": r.get("client_ip"),
                "case_ids": r.get("case_ids") or [],
                "backend": r.get("backend"),
                "model": r.get("model"),
                "verdict": r.get("verdict"),
                "error": r.get("error"),
                "elapsed_s": r.get("elapsed_s"),
                "params": r.get("params") or {},
                "tools": [s.get("tool") for s in (r.get("steps") or []) if s.get("tool")],
                "question": _qa_view(r, "question", include_text),
                "answer": _qa_view(r, "answer", include_text),
                "answer_sha256": r.get("answer_sha256"),
            }
            for r in runs
        ],
        "auths": [
            {
                "ts": r.get("ts"),
                "allowed": bool(r.get("allowed")),
                "path": r.get("path"),
                "actor": r.get("actor"),
                "client_ip": r.get("client_ip"),
                "user_agent": r.get("user_agent"),
            }
            for r in auths
        ],
        "denied": [
            {
                "ts": r.get("ts"),
                "path": r.get("path"),
                "actor": r.get("actor"),
                "client_ip": r.get("client_ip"),
                "user_agent": r.get("user_agent"),
            }
            for r in auths
            if not r.get("allowed")
        ],
        "phi": [
            {
                "ts": r.get("ts"),
                "roots": r.get("roots") or [],
                "scanned": r.get("scanned") or {},
                "confirmed": r.get("confirmed"),
                "suspicious": r.get("suspicious"),
                "clean": r.get("clean"),
                "selftest_failures": r.get("selftest_failures"),
                "hits": r.get("hits") or [],
            }
            for r in phis
        ],
        "tools": [
            {"tool": name, "calls": count, "failed": tool_fail.get(name, 0)}
            for name, count in tool_total.most_common()
        ],
        "artifacts": artifacts,
        "gate": _load_gate(),
        "baseline": _read_json(BASELINE) if BASELINE.is_file() else None,
    }


def _empty_stats() -> dict:
    return {"runs": 0, "runs_ok": 0, "auths": 0, "auth_denied": 0,
            "phi_scans": 0, "phi_hits": 0, "artifacts": 0}


def _load_gate() -> dict | None:
    """读最近一次门禁报告；超过 `GATE_STALE_HOURS` 就当没有（不拿陈旧数据当现状）。"""
    payload = _read_json(GATE_JSON) if GATE_JSON.is_file() else None
    if not payload:
        return None
    age = _age_hours(payload.get("generated_at"))
    payload = dict(payload)
    payload["age_hours"] = age
    payload["stale"] = age is not None and age > GATE_STALE_HOURS
    return payload


# ------------------------------------------------------------------ 结论


def verdicts(data: dict) -> list[dict]:
    """四项三态结论。这是整份报告的头部，也是退出码的来源。

    每个 `none` 都带一条 `fix` —— 「未判定」不能是终点，得说清补哪一条命令就能判定。
    一份只说「证据不足」的报告，读的人下一步还是不知道干什么。
    """
    items: list[dict] = []

    chain = data["chain"]
    if not data["has_data"]:
        detail = "报告期内没有审计日志，无法校验"
        fix = "拉长报告期看全量：--days 3650"
    elif chain["state"] == OK:
        detail = f"{len(chain['per_file'])} 份日志 / {data['records']} 条记录，链完整"
        fix = None
    else:
        detail = f"发现 {chain['problem_count']} 处问题：日志被改动过，需人工比对"
        fix = "用 `agent.scripts.audit_verify` 逐文件核对，确认是改动还是归档操作"
    items.append({"name": "审计链完整性", "state": chain["state"],
                  "detail": detail, "fix": fix})

    phis = data["phi"]
    if not phis:
        items.append({
            "name": "PHI 守卫", "state": NONE,
            "detail": "报告期内未执行 PHI 扫描 —— 不构成「数据面是干净的」",
            "fix": "python -m agent.scripts.scan_phi --root cases --audit",
        })
    else:
        last = phis[-1]
        confirmed = sum(int(p.get("confirmed") or 0) for p in phis)
        selftest_bad = sum(int(p.get("selftest_failures") or 0) for p in phis)
        state = BAD if (confirmed or selftest_bad) else OK
        detail = (f"最近一次 {last.get('ts')}：确认 {last.get('confirmed')} / 疑似 "
                  f"{last.get('suspicious')}；期间累计命中 {confirmed}")
        fix = None
        if selftest_bad:
            detail += f"；⚠️ 守卫自检失败 {selftest_bad} 项 —— 尺子本身没通过，扫描结论不可信"
            fix = "先修守卫自检（python -m agent.scripts.scan_phi --selftest-only）再重扫"
        items.append({"name": "PHI 守卫", "state": state, "detail": detail, "fix": fix})

    gate = data["gate"]
    if not gate:
        items.append({"name": "评测基线对比", "state": NONE,
                      "detail": "没有门禁报告，无法对比评测基线",
                      "fix": "python -m agent.scripts.gate"})
    elif gate.get("stale"):
        items.append({"name": "评测基线对比", "state": NONE,
                      "detail": f"最近一次门禁在 {gate.get('age_hours'):.1f} 小时前，已陈旧，"
                                f"不作为本次结论依据",
                      "fix": "python -m agent.scripts.gate"})
    else:
        summary = gate.get("eval") or {}
        ok = bool(gate.get("baseline_ok"))
        detail = (f"{summary.get('passed')}/{summary.get('total')}"
                  f"（后端 {summary.get('backend')}），门禁判定 "
                  f"{'未低于基线' if ok else '低于基线'}")
        items.append({"name": "评测基线对比", "state": OK if ok else BAD, "detail": detail,
                      "fix": None if ok else "跑 `eval_diff_report` 逐用例对照，分清末知是模型退步还是尺子变松"})

    if not gate:
        items.append({"name": "回归门禁", "state": NONE, "detail": "没有门禁报告",
                      "fix": "python -m agent.scripts.gate"})
    elif gate.get("stale"):
        items.append({"name": "回归门禁", "state": NONE,
                      "detail": f"最近一次门禁已陈旧（{gate.get('age_hours'):.1f} 小时前）",
                      "fix": "python -m agent.scripts.gate"})
    else:
        checks = gate.get("checks") or []
        failed = [c for c in checks if not c.get("ok")]
        items.append({
            "name": "回归门禁",
            "state": OK if gate.get("passed") else BAD,
            "detail": (f"{len(checks) - len(failed)}/{len(checks)} 项通过"
                       f"（{gate.get('generated_at')}）"
                       + (f"；失败：{'、'.join(c.get('key', '?') for c in failed)}" if failed else "")),
            "fix": None if not failed else "看 outputs/gate/gate_report.md 的失败明细",
        })
    return items


def overall(items: list[dict]) -> str:
    """总判定：有 bad → bad；否则有 none → none（覆盖面不足）；否则 ok。"""
    if any(item["state"] == BAD for item in items):
        return BAD
    if any(item["state"] == NONE for item in items):
        return NONE
    return OK


# ------------------------------------------------------------------ 渲染

CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 34px 28px 60px; background: #f7f7f5; color: #23231f;
  font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
  font-size: 14px; line-height: 1.65; }
.wrap { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 6px; letter-spacing: -.2px; }
h2 { font-size: 16px; font-weight: 600; margin: 38px 0 12px;
  padding-bottom: 8px; border-bottom: 1px solid #e2e2dd; }
h3 { font-size: 14px; font-weight: 600; margin: 20px 0 8px; }
.sub { color: #6b6b64; font-size: 13px; margin: 0 0 4px; }
.card { background: #fff; border: 1px solid #e6e6e1; border-radius: 12px;
  padding: 18px 20px; margin-bottom: 14px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
  gap: 12px; margin: 16px 0 4px; }
.metric { background: #fff; border: 1px solid #e6e6e1; border-radius: 10px; padding: 14px 16px; }
.metric .k { font-size: 12px; color: #78786f; margin: 0 0 6px; }
.metric .v { font-size: 25px; font-weight: 600; margin: 0; letter-spacing: -.5px; }
.metric .n { font-size: 12px; color: #8a8a81; margin: 4px 0 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; font-weight: 600; font-size: 12px; color: #6b6b64;
  padding: 8px 10px; border-bottom: 1px solid #e2e2dd; white-space: nowrap; }
td { padding: 9px 10px; border-bottom: 1px solid #f0f0ec; vertical-align: top; }
tr:last-child td { border-bottom: none; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 20px;
  font-size: 12px; font-weight: 500; white-space: nowrap; }
code { font-family: Consolas, "Courier New", monospace; font-size: 12px;
  background: #f2f2ee; padding: 1px 5px; border-radius: 4px; word-break: break-all; }
.mono { font-family: Consolas, "Courier New", monospace; font-size: 12px; color: #5c5c55; }
.why { color: #6b6b64; font-size: 12.5px; }
.banner { border-radius: 12px; padding: 16px 20px; margin: 14px 0 18px; border: 1px solid; }
.banner .t { font-size: 15px; font-weight: 600; margin: 0 0 4px; }
.banner .d { font-size: 13px; margin: 0; }
.banner.ok   { background: #f0fdfa; border-color: #a7d9d0; color: #0b5b53; }
.banner.bad  { background: #fef2f2; border-color: #e8b4b4; color: #8f2626; }
.banner.none { background: #fbfaf7; border-color: #e0ddd3; color: #6b6656; }
.note { background: #fffdf5; border: 1px solid #ece3c9; border-radius: 10px;
  padding: 14px 18px; font-size: 13px; color: #4a4230; }
.note.bad { background: #fef6f6; border-color: #ecc6c6; color: #7d2b2b; }
.note b { font-weight: 600; }
.empty { color: #8a8a81; font-size: 13px; font-style: normal; padding: 10px 0; }
ol, ul { margin: 8px 0; padding-left: 22px; }
li { margin: 5px 0; }
.bar { height: 8px; border-radius: 4px; background: #e8e8e2; overflow: hidden; display: block; }
.bar > i { display: block; height: 100%; background: #0f766e; }
@media print {
  body { background: #fff; padding: 0; font-size: 11.5px; }
  h2 { page-break-after: avoid; } table { page-break-inside: auto; }
  tr { page-break-inside: avoid; } .card, .banner, .note { break-inside: avoid; }
}
"""


def _esc(value) -> str:
    """一切来自日志的文本都要转义 —— IP / UA / 病例 ID 都是外部可控的。"""
    return html.escape("" if value is None else str(value))


def _tag(state: str) -> str:
    color, bg = STATE_COLOR[state]
    label = {"ok": "通过", "bad": "异常", "none": "未判定"}[state]
    return f'<span class="tag" style="color:{color};background:{bg}">{label}</span>'


def _metric(key: str, value: str, note: str = "") -> str:
    extra = f'<p class="n">{_esc(note)}</p>' if note else ""
    return f'<div class="metric"><p class="k">{_esc(key)}</p><p class="v">{value}</p>{extra}</div>'


def _table(headers: list[str], rows: list[str], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{_esc(empty)}</p>'
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_html(data: dict, *, period: str, include_text: bool) -> str:
    items = verdicts(data)
    state = overall(items)
    stats = data["stats"]
    chain = data["chain"]

    pending = [item for item in items if item["state"] == NONE]
    banner = {
        OK: ("结论：在可判定的范围内未发现问题", "#0f766e",
             "四项结论全部有据可依。"),
        BAD: ("结论：发现异常，需人工介入", "#a32d2d",
              "有项目与预期不符。报告只呈现事实，不代替处置决定。"),
        NONE: ("结论：不足以判定", "#78786f",
               f"有 {len(pending)} 项没有可用数据（"
               + "、".join(item["name"] for item in pending)
               + "），因此不能给出「正常」的结论。补齐后重新生成本报告。"),
    }[state]

    # --- 结论表
    verdict_rows = []
    for item in items:
        cell = _esc(item["detail"])
        if item.get("fix"):
            cell += f'<br><span class="why">补齐：<code>{_esc(item["fix"])}</code></span>'
        verdict_rows.append(
            f'<tr><td>{_esc(item["name"])}</td><td>{_tag(item["state"])}</td>'
            f'<td class="why">{cell}</td></tr>'
        )

    # --- 概览
    runs, runs_ok = stats["runs"], stats["runs_ok"]
    # 0 次运行的成功率必须是 "—"。写 100% 是最典型的「缺数据被当成通过」
    rate = f"{round(100 * runs_ok / runs, 1)}%" if runs else "—"
    rate_note = f"{runs_ok}/{runs} 判定为 answered" if runs else "报告期内无运行记录"
    cards = "".join([
        _metric("运行次数", str(runs), rate_note),
        _metric("成功率", rate, "0 次运行不构成 100%"),
        _metric("鉴权拒绝", str(stats["auth_denied"]),
                f"共 {stats['auths']} 次鉴权事件"),
        _metric("PHI 扫描", str(stats["phi_scans"]),
                f"累计命中 {stats['phi_hits']}" if stats["phi_scans"] else "未执行"),
        _metric("产物", str(stats["artifacts"]), "含 sha256 指纹"),
        _metric("审计记录", str(data["records"]), f"{len(data['logs'])} 份日志"),
    ])

    # --- 链明细
    chain_rows = [
        f'<tr><td class="mono">{_esc(item["name"])}</td>'
        f'<td style="text-align:right">{item["records"]}</td>'
        f'<td class="mono">{_esc(item["first"] or "—")}</td>'
        f'<td class="mono">{_esc(item["last"] or "—")}</td>'
        f'<td class="mono">{_esc((item["tail"] or "—")[:16])}…</td></tr>'
        for item in chain["per_file"]
    ]
    chain_problems = ""
    if chain["problems"]:
        chain_problems = (
            '<div class="note bad"><b>链校验发现的问题（最多列 20 条）：</b><ol>'
            + "".join(f"<li><code>{_esc(p)}</code></li>" for p in chain["problems"])
            + "</ol></div>"
        )
    if data.get("broken_lines"):
        chain_problems += (
            '<div class="note bad"><b>无法解析的行：</b><ul>'
            + "".join(f'<li><code>{_esc(b.get("_file"))}:{_esc(b.get("_line"))}</code> '
                      f'{_esc(b.get("_error"))}</li>' for b in data["broken_lines"])
            + "</ul></div>"
        )

    # --- 运行明细
    run_rows = []
    for run in data["runs"]:
        q, a = run["question"], run["answer"]
        qa = f"问 {q['chars']} 字" + ("（脱敏）" if q["redacted"] else "")
        qa += f"<br>答 {a['chars']} 字 / <span class=\"mono\">{_esc((run.get('answer_sha256') or '')[:12])}…</span>"
        if q["text"]:
            qa += f'<div class="why">问题：{_esc(q["text"][:200])}</div>'
        if a["text"]:
            qa += f'<div class="why">回答：{_esc(a["text"][:300])}</div>'
        params = run["params"]
        run_rows.append(
            f'<tr><td class="mono">{_esc(run["ts"] or "—")}</td>'
            f'<td class="mono">{_esc((run["run_id"] or "")[:8])}</td>'
            f'<td>{_esc(run["actor"] or "—")}<div class="why">{_esc(run["client_ip"] or "")}</div></td>'
            f'<td>{_esc("、".join(run["case_ids"]) or "—")}</td>'
            f'<td>{_esc(run["backend"] or "—")}<div class="why">{_esc(run["model"] or "")}</div></td>'
            f'<td class="mono">{_esc(" → ".join(run["tools"]) or "（无工具调用）")}</td>'
            f'<td>{_esc(run["verdict"] or "")}'
            + (f'<div class="why">{_esc(params.get("stop_reason"))}</div>' if params.get("stop_reason") else "")
            + "</td>"
            f'<td style="text-align:right">{_esc(run["elapsed_s"] if run["elapsed_s"] is not None else "—")}</td>'
            f'<td class="why">{qa}</td></tr>'
        )

    # --- 工具统计
    tool_rows = []
    for item in data["tools"]:
        calls, failed = item["calls"], item["failed"]
        mark = f'<span style="color:#a32d2d">{failed}</span>' if failed else "0"
        tool_rows.append(
            f'<tr><td class="mono">{_esc(item["tool"])}</td>'
            f'<td style="text-align:right">{calls}</td>'
            f'<td style="text-align:right">{mark}</td>'
            f'<td><span class="bar"><i style="width:'
            f'{round(100 * calls / max(t["calls"] for t in data["tools"]))}%"></i></span></td></tr>'
        )

    # --- 产物
    artifact_rows = [
        f'<tr><td class="mono">{_esc(item.get("file"))}</td>'
        f'<td>{_esc(item.get("kind") or "—")}</td>'
        f'<td style="text-align:right">{_esc(item.get("bytes") if item.get("bytes") is not None else "—")}</td>'
        f'<td class="mono">{_esc((item.get("sha256") or "（未计算）")[:16])}…</td>'
        f'<td class="mono">{_esc(item.get("ts") or "—")}</td>'
        f'<td class="mono">{_esc((item.get("run_id") or "")[:8])}</td></tr>'
        for item in data["artifacts"]
    ]

    # --- 鉴权
    denied_rows = [
        f'<tr><td class="mono">{_esc(item["ts"] or "—")}</td>'
        f'<td class="mono">{_esc(item["path"] or "—")}</td>'
        f'<td class="mono">{_esc(item["client_ip"] or "—")}</td>'
        f'<td class="why">{_esc((item["user_agent"] or "")[:110])}</td></tr>'
        for item in data["denied"]
    ]
    auth_rows = [
        f'<tr><td class="mono">{_esc(item["ts"] or "—")}</td>'
        f'<td>{"放行" if item["allowed"] else "拒绝"}</td>'
        f'<td class="mono">{_esc(item["path"] or "—")}</td>'
        f'<td class="mono">{_esc(item["client_ip"] or "—")}</td></tr>'
        for item in data["auths"]
    ]

    # --- PHI
    phi_rows = []
    for item in data["phi"]:
        scanned = item.get("scanned") or {}
        summary = "、".join(f"{_esc(k)} {_esc(v)}" for k, v in list(scanned.items())[:4]) or "—"
        hits = item.get("hits") or []
        hit_note = ("；".join(f'{_esc(h.get("where"))}' for h in hits[:5]) or "—")
        phi_rows.append(
            f'<tr><td class="mono">{_esc(item["ts"] or "—")}</td>'
            f'<td>{_esc("、".join(item.get("roots") or []) or "—")}</td>'
            f'<td class="why">{summary}</td>'
            f'<td style="text-align:right">{_esc(item.get("confirmed"))} / {_esc(item.get("suspicious"))}</td>'
            f'<td>{"干净" if item.get("clean") else "<b>有命中</b>"}</td>'
            f'<td class="mono">{hit_note}</td></tr>'
        )

    # --- 门禁 / 基线
    gate = data["gate"]
    baseline = data["baseline"]
    if gate:
        check_rows = [
            f'<tr><td>{_esc(c.get("title"))}</td><td>{_esc(c.get("category"))}</td>'
            f'<td>{"是" if c.get("hard") else "—"}</td>'
            f'<td>{"✅" if c.get("ok") else "❌"}</td>'
            f'<td style="text-align:right">{_esc(c.get("seconds"))}s</td>'
            f'<td class="why">{_esc(c.get("detail"))}</td></tr>'
            for c in (gate.get("checks") or [])
        ]
        gate_block = (
            f'<p class="sub">最近一次门禁：<code>{_esc(gate.get("generated_at"))}</code>　'
            f'档位 <code>{_esc(gate.get("profile"))}</code>　'
            f'结论 <b>{"通过" if gate.get("passed") else "未通过"}</b>'
            + (f'　⚠️ 已陈旧（{gate.get("age_hours"):.1f} 小时前）' if gate.get("stale") else "")
            + "</p>"
            + _table(["检查项", "类别", "硬门禁", "结果", "耗时", "说明"], check_rows, "门禁报告里没有检查项")
            + (f'<p class="sub" style="margin-top:10px">基线说明：'
               + "；".join(_esc(n) for n in (gate.get("baseline_notes") or [])) + "</p>"
               if gate.get("baseline_notes") else "")
        )
    else:
        gate_block = ('<p class="empty">没有找到门禁报告 '
                      '（<code>outputs/gate/gate_report.json</code>）—— 跑一次 '
                      '<code>python -m agent.scripts.gate</code> 后再生成报告。</p>')

    baseline_block = ""
    if baseline:
        baseline_block = (
            f'<p class="sub">评测基线文件记录：通过 <b>{_esc(baseline.get("passed"))}/'
            f'{_esc(baseline.get("total"))}</b>　'
            f'后端 <code>{_esc(baseline.get("backend"))}</code>　'
            f'记录于 {_esc(baseline.get("recorded_at"))}</p>'
        )

    # --- 方法与边界
    #
    # 一律先用变量接出来，不在 f-string 表达式里写 `data["key"]`：
    # 本机解释器是 Python 3.10，`"""` 里嵌同种引号要 3.12 才放开。
    audit_dir_text = _esc(data["audit_dir"])
    log_count = len(data["logs"])
    record_count = data["records"]
    tail_text = _esc((chain["tail_hash"] or "—")[:32])
    denied_count = stats["auth_denied"]
    text_in_report = ("<b>已包含</b>在本报告中（--include-text）" if include_text
                      else "不在本报告中")
    text_how = ("⚠️ 报告含问答原文，转发前请确认接收方可信。" if include_text
                else "要带原文请显式加 <code>--include-text</code>。")
    text_flag = "<b>已包含</b>" if include_text else "不含（默认脱敏）"

    t_verdict = _table(["项目", "状态", "依据"], verdict_rows, "没有可判定的项目")
    t_chain = _table(["日志文件", "记录数", "首条", "末条", "链尾哈希"], chain_rows,
                     "报告期内没有审计日志文件")
    t_runs = _table(
        ["时间", "run", "操作者", "病例", "后端 / 模型", "工具链", "结论", "耗时(s)", "问答摘要"],
        run_rows, "报告期内没有 Agent 运行记录")
    t_tools = _table(["工具", "调用", "失败", "相对频次"], tool_rows, "没有工具调用记录")
    t_artifacts = _table(["文件", "类型", "字节", "sha256", "时间", "run"], artifact_rows,
                         "没有产物记录")
    t_denied = _table(["时间", "路径", "来源 IP", "User-Agent"], denied_rows,
                      "报告期内没有被拒绝的访问")
    t_auths = _table(["时间", "结果", "路径", "来源 IP"], auth_rows, "报告期内没有鉴权事件")
    t_phi = _table(["时间", "扫描范围", "规模", "确认 / 疑似", "结论", "命中位置"], phi_rows,
                   "报告期内未执行 PHI 扫描 —— 不构成「数据面是干净的」")

    boundary = f"""
    <h2>方法与边界</h2>
    <div class="card">
      <h3>这份报告的数据从哪来</h3>
      <ul>
        <li>审计日志 <code>{audit_dir_text}</code>，共 {log_count} 份、{record_count} 条记录
            （按记录自身 <code>ts</code> 精筛）。</li>
        <li>报告期：{_esc(period)}。日志按<b>文件名日期</b>粗筛；文件内某条记录的
            <code>ts</code> 若落在区间外，仍会参与统计 —— 天是本系统审计分片的真实粒度。</li>
        <li>门禁与评测：<code>outputs/gate/gate_report.json</code> 与
            <code>agent/eval/baseline.json</code>。门禁结果超过 {GATE_STALE_HOURS} 小时
            视为陈旧，<b>不作为本次结论依据</b>。</li>
      </ul>

      <h3>哈希链能证明什么、不能证明什么</h3>
      <ul>
        <li><b>能证明</b>：这份日志没有被改动过 —— 改任何一条的任何一个字段，
            从它开始往后每一条都对不上。</li>
        <li><b>不能证明</b>：整份日志被替换。链是从 <code>genesis</code> 自洽重算的，
            能改日志的人也能把整条链重算一遍。所以链尾哈希要<b>抄到别处留底</b>
            （本次链尾 <code>{tail_text}…</code>），下次校验时它若对不上，
            就说明中间被动过。</li>
        <li>本报告<b>不构成法律意义上的签名</b>；抗抵赖需要「链尾定期外发 + 时间戳」，属 P1。</li>
      </ul>

      <h3>PHI 守卫覆盖到哪</h3>
      <ul>
        <li>扫的是「能承载 PHI 的位置」：NIfTI 头部 4 个自由文本槽、文本键名与值、
            zip 成员名、DICOM 标签。<b>不是全文扫描</b>，也不做像素级分析。</li>
        <li>清除路径（<code>--apply-out</code>）只在合成数据上验证过，仓库内没有真实阳性样本
            —— 这一条是已知缺口，不当作已解决。</li>
        <li>豁免项逐条留了理由（<code>PHI_EXEMPT</code>）；豁免不静默。</li>
      </ul>

      <h3>需要人为判断的部分</h3>
      <ul>
        <li>「鉴权拒绝 {denied_count} 次」<b>没有设阈值</b> —— 拒绝本身通常说明防线在工作，
            是否算异常要结合来源 IP 与时间分布判断，本报告只呈现事实。</li>
        <li>问答原文{text_in_report}，只有字数与答案哈希。{text_how}</li>
        <li>结论栏的「未判定」不等于「有问题」，只表示证据不足；<b>不要把它读成通过</b>。</li>
      </ul>
    </div>
    """

    title = f"AirNav-Agent 运行与合规报告　{period}"
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner_title = banner[0].replace("**", "")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>AirNav-Agent 运行与合规报告</h1>
  <p class="sub">报告期：{_esc(period)}　·　生成时间：{_esc(now_text)}</p>
  <p class="sub">数据源：<code>{audit_dir_text}</code>　·　问答原文：{text_flag}</p>

  <div class="banner {state}">
    <p class="t">{banner_title}</p>
    <p class="d">{_esc(banner[2])}</p>
  </div>

  <h2>1　结论清单</h2>
  {t_verdict}
  {chain_problems}

  <h2>2　概览</h2>
  <div class="cards">{cards}</div>

  <h2>3　审计链明细</h2>
  {t_chain}

  <h2>4　运行明细</h2>
  {t_runs}

  <h2>5　工具调用统计</h2>
  {t_tools}

  <h2>6　产物指纹</h2>
  <p class="sub">sha256 是运行当时算的；把手上文件重算一遍比对，就能确认「当时看的确实是这一份」。</p>
  {t_artifacts}

  <h2>7　鉴权事件</h2>
  <h3>7.1　被拒绝的访问</h3>
  {t_denied}
  <h3>7.2　全部鉴权事件</h3>
  {t_auths}

  <h2>8　PHI 守卫</h2>
  {t_phi}

  <h2>9　回归门禁与评测基线</h2>
  {gate_block}
  {baseline_block}

  {boundary}

  <p class="sub" style="margin-top:32px">
    本报告由 <code>python -m agent.scripts.report</code> 生成，只读审计数据，不修改任何日志。
  </p>
</div>
</body>
</html>
"""


# ------------------------------------------------------------------ main


def _period_label(since: datetime | None, until: datetime | None) -> str:
    if since and until:
        return f"{since:%Y-%m-%d} ~ {until:%Y-%m-%d}"
    if since:
        return f"{since:%Y-%m-%d} 起至今"
    if until:
        return f"截至 {until:%Y-%m-%d}"
    return "全部记录"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 AirNav-Agent 运行与合规报告")
    parser.add_argument("--dir", help="审计目录（默认 outputs/audit 或 $AIRNAV_AUDIT_DIR）")
    parser.add_argument("--days", type=int, default=7, help="报告期＝最近 N 天（默认 7）")
    parser.add_argument("--since", help="起始日期 YYYY-MM-DD（给定时忽略 --days）")
    parser.add_argument("--until", help="结束日期 YYYY-MM-DD（含当天）")
    parser.add_argument("--out", help="输出 HTML 路径（默认 outputs/reports/report-<日期>.html）")
    parser.add_argument("--format", choices=("html", "json", "both"), default="html",
                        help="输出形态（默认 html）")
    parser.add_argument("--include-text", action="store_true",
                        help="⚠️ 报告将包含问答原文（默认只给字数与哈希）")
    parser.add_argument("--quiet", action="store_true", help="只输出路径与结论行")
    args = parser.parse_args(argv)

    if args.dir:
        # `--dir` 走环境变量，和 audit_verify / audit_query 保持同一套口径
        import os
        os.environ["AIRNAV_AUDIT_DIR"] = args.dir

    if args.since:
        try:
            since = datetime.strptime(args.since, "%Y-%m-%d")
        except ValueError:
            print(f"--since 需要 YYYY-MM-DD，收到 {args.since!r}", file=sys.stderr)
            return 2
    else:
        since = datetime.now() - timedelta(days=max(args.days, 0))
    until = None
    if args.until:
        try:
            until = datetime.strptime(args.until, "%Y-%m-%d")
        except ValueError:
            print(f"--until 需要 YYYY-MM-DD，收到 {args.until!r}", file=sys.stderr)
            return 2

    if args.include_text:
        print("⚠️ 已启用 --include-text：报告将包含问答原文，转发前请确认接收方可信。",
              file=sys.stderr)

    data = collect(since=since, until=until, include_text=args.include_text)
    items = verdicts(data)
    state = overall(items)
    period = _period_label(since, until)

    if args.format in ("json", "both"):
        payload = dict(data)
        payload["period"] = period
        payload["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload["verdicts"] = items
        payload["overall"] = state
        target = Path(args.out).with_suffix(".json") if args.out else (
            REPORT_DIR / f"report-{datetime.now():%Y-%m-%d}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Python 写 .json = 明文（本机 DLP 矩阵），不需要补丁脚本
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"已写入 JSON：{target}")

    if args.format in ("html", "both"):
        page = build_html(data, period=period, include_text=args.include_text)
        target = Path(args.out) if args.out else (
            REPORT_DIR / f"report-{datetime.now():%Y-%m-%d}.html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page, encoding="utf-8")
        if not args.quiet:
            print(f"已写入报告：{target}（{len(page.encode('utf-8'))} 字节）")

    if not args.quiet:
        print("-" * 60)
        for item in items:
            mark = {OK: "✅", BAD: "❌", NONE: "⬜"}[item["state"]]
            print(f"  {mark} {item['name']}：{item['detail']}")
            if item.get("fix"):
                print(f"       ↳ 补齐：{item['fix']}")
        print("-" * 60)
        if not data["has_data"]:
            print("报告期内没有审计记录 —— 报告已生成，但结论覆盖面有限。")
        label = {OK: "✅ 符合预期", BAD: "❌ 发现异常", NONE: "⬜ 不足以判定"}[state]
        print(f"总判定：{label}")

    return 1 if state == BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
