"""报告自检：**缺数据不许说通过**、篡改必被检出、脱敏不漏、HTML 自包含。

    python -m agent.scripts.check_report

秒级、不联网、不要模型。全程在临时目录里跑（`AIRNAV_AUDIT_DIR` + 临时门禁/基线文件），
**绝不碰 `outputs/audit/`、`outputs/reports/`、`outputs/gate/`**。

## 为什么要专门测「缺数据」

报告这类东西和哈希链一样，平时**永远返回「一切正常」**，所以不测就等于没有。
但它比哈希链更阴：哈希链坏了顶多不报警，而报告会在**没有证据时替你下结论**。
最容易骗过所有自动化检查的一种实现是：0 次运行时成功率算 `0/0` → 除以零崩掉，
或者更糟 —— 写成 `100%`。两种都不会报错，只会让拿到报告的人误判。

所以第三层的第一条断言就是：**0 次运行时必须显示 `—`，且总判定必须是「未判定」而不是「通过」**。

## 为什么要专门测「篡改」

报告的头号用途是给出「合规结论」。如果它读了一份被改过的审计日志还说「链完整」，
那它就变成了替篡改背书的东西 —— 比不做报告更危险。
因此这里改一条记录，然后确认报告**标红 + 退出码 1**（CI 能拦住）。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import audit  # noqa: E402
from agent.scripts import report as report_mod  # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []

# 哨兵：问答原文里埋这个串，默认报告里出现即视为脱敏失效
SENTINEL = "ZZSENTINELQ9X"


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((ok, label, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"（{detail}）" if detail else ""))
    return ok


# ------------------------------------------------------------------ 夹具


def _dir() -> Path:
    return Path(os.environ["AIRNAV_AUDIT_DIR"])


def _logs() -> list[Path]:
    return sorted(_dir().glob("audit-*.jsonl"))


def _reset() -> None:
    for path in _logs():
        path.unlink()


def _write_gate(hours_ago: float = 0, passed: bool = True, baseline_ok: bool = True) -> None:
    """写一份门禁报告夹具。`hours_ago` 用来做「陈旧」那种情况。

    为什么要放进标准夹具：报告的四项结论里有两项依赖门禁文件。
    不写它的话，「四项齐备 → ok」这条断言永远只能看到 `none`，
    测的其实是「缺文件的降级路径」而不是「全绿路径」。
    """
    stamp = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
    report_mod.GATE_JSON.write_text(json.dumps({
        "generated_at": stamp, "profile": "local", "passed": passed,
        "checks": [{"key": "planner", "title": "规划数值一致性（与 V1 逐位一致）", "ok": passed}],
        "eval": {"passed": 15 if baseline_ok else 14, "total": 15, "backend": "heuristic"},
        "baseline_ok": baseline_ok,
        "baseline_notes": ["基线 15/15 → 本次 15/15", "✅ 未低于基线"],
    }, ensure_ascii=False), encoding="utf-8")


def _seed(artifact_path: Path) -> None:
    """标准夹具：3 次运行（2 ok / 1 unresolved_intent）、2 次鉴权（1 拒绝）、1 次 PHI 扫描。

    同时写一份**新鲜且通过**的门禁报告，让「四项齐备」这条路径可测。
    """
    audit.record_run(
        run_id="run-aaa", question=f"{SENTINEL} 的 3 号候选能过 1.5mm 吗？",
        answer=f"可以，代价 1.42。{SENTINEL}",
        backend="gpu41", model="qwen2.5:14b", verdict="ok",
        actor="wcq", client_ip="192.168.8.10", user_agent="selfcheck/1.0",
        elapsed_s=9.13, params={"max_steps": 8, "stop_reason": "answered"},
        steps=[{"step": 1, "tool": "plan_route", "ok": True},
               {"step": 1, "tool": "render_viewer", "ok": True}],
        artifacts=[{"kind": "viewer", "filename": artifact_path.name,
                    "path": str(artifact_path)}],
        case_ids=["LIDC_0089"])
    audit.record_run(
        run_id="run-bbb", question="LNDB_0196 有哪些候选？", answer="共 3 个。",
        backend="gpu41", model="qwen2.5:14b", verdict="ok",
        actor="wcq", client_ip="192.168.8.10", elapsed_s=6.2,
        steps=[{"step": 1, "tool": "list_nodule_candidates", "ok": True},
               {"step": 2, "tool": "plan_route", "ok": False}],
        case_ids=["LNDB_0196"])
    audit.record_run(
        run_id="run-ccc", question="随便说说", answer=None,
        backend="heuristic", model="rules", verdict="unresolved_intent",
        actor="guest", client_ip="10.0.0.7", elapsed_s=1.1)
    audit.record_auth(allowed=False, path="/api/chat", client_ip="203.0.113.9",
                      user_agent="curl/8.0", actor="anonymous")
    audit.record_auth(allowed=True, path="/health", client_ip="127.0.0.1")
    audit.record_phi_scan(roots=["cases"], scanned={"nifti": 78, "text": 64, "zip": 12},
                          confirmed=0, suspicious=0)
    _write_gate()


def _collect(**kw) -> dict:
    kw.setdefault("since", None)
    kw.setdefault("until", None)
    kw.setdefault("include_text", False)
    return report_mod.collect(**kw)


def _html(**kw) -> str:
    data = _collect(**kw)
    return report_mod.build_html(data, period="自检", include_text=kw.get("include_text", False))


# ------------------------------------------------------------------ 各层


def layer_1_empty() -> int:
    """最危险的一种：没有证据时不许下"正常"的结论。"""
    print("\n[1] 空数据（缺数据 ≠ 通过）")
    failures = 0
    _reset()

    data = _collect()
    if not check(data["has_data"] is False, "空目录：has_data=False"):
        failures += 1

    items = report_mod.verdicts(data)
    chain = next(i for i in items if i["name"] == "审计链完整性")
    if not check(chain["state"] == report_mod.NONE, "空目录：审计链结论为「未判定」", chain["state"]):
        failures += 1
    if not check(report_mod.overall(items) == report_mod.NONE,
                 "空目录：总判定为「未判定」而不是「通过」", report_mod.overall(items)):
        failures += 1

    phi = next(i for i in items if i["name"] == "PHI 守卫")
    if not check("未执行" in phi["detail"], "空目录：PHI 措辞是「未执行」而不是「未发现」",
                 phi["detail"][:40]):
        failures += 1
    if not check(bool(phi.get("fix")), "空目录：未判定项给出补齐命令", phi.get("fix", "")):
        failures += 1

    page = _html()
    # 0/0 一旦写成 100%，拿到报告的人会以为系统跑得好好的
    if not check('>—<' in page and ">100.0%<" not in page,
                 "空目录：成功率显示「—」而不是 100%",
                 "含 —" if '>—<' in page else "缺 —"):
        failures += 1
    if not check("无运行记录" in page, "空目录：明确写出「无运行记录」"):
        failures += 1

    out = _dir().parent / "empty.html"
    code = report_mod.main(["--out", str(out), "--quiet"])
    if not check(out.is_file(), "空目录：报告仍然生成（不崩）"):
        failures += 1
    if not check(code == 0, "空目录：退出码 0（未判定不算失败，但也不算通过）", f"exit={code}"):
        failures += 1
    return failures


def layer_2_stats() -> int:
    print("\n[2] 统计口径")
    failures = 0
    _reset()
    artifact = _dir().parent / "viewer_LIDC_0089_c1.html"
    artifact.write_text("<html>art</html>", encoding="utf-8")
    _seed(artifact)

    data = _collect()
    stats = data["stats"]
    if not check(stats["runs"] == 3, "运行次数 = 3", str(stats["runs"])):
        failures += 1
    if not check(stats["runs_ok"] == 2, "其中判定为 ok 的 = 2", str(stats["runs_ok"])):
        failures += 1
    if not check(stats["auths"] == 2 and stats["auth_denied"] == 1,
                 "鉴权 2 次、其中拒绝 1 次", f"{stats['auths']}/{stats['auth_denied']}"):
        failures += 1
    if not check(stats["phi_scans"] == 1, "PHI 扫描 1 次", str(stats["phi_scans"])):
        failures += 1
    if not check(len(data["denied"]) == 1, "被拒绝的访问单独成表", f"{len(data['denied'])} 条"):
        failures += 1
    if not check(data["denied"][0]["client_ip"] == "203.0.113.9",
                 "拒绝记录带来源 IP（审计要回答「从哪来」）"):
        failures += 1

    tools = {item["tool"]: item for item in data["tools"]}
    if not check(tools.get("plan_route", {}).get("calls") == 2,
                 "工具计数：plan_route 调用 2 次", str(tools.get("plan_route"))):
        failures += 1
    if not check(tools.get("plan_route", {}).get("failed") == 1,
                 "工具计数：plan_route 失败 1 次（「调了但都失败」不能掩盖）",
                 str(tools.get("plan_route"))):
        failures += 1

    expected = audit.hash_file(artifact)
    if not check(data["artifacts"] and data["artifacts"][0]["sha256"] == expected,
                 "产物 sha256 与实测一致", (expected or "")[:16]):
        failures += 1

    items = report_mod.verdicts(data)
    chain = next(i for i in items if i["name"] == "审计链完整性")
    if not check(chain["state"] == report_mod.OK, "链完整 → 结论 ok", chain["state"]):
        failures += 1
    if not check(report_mod.overall(items) == report_mod.OK,
                 "四项齐备且正常 → 总判定 ok", report_mod.overall(items)):
        failures += 1

    page = _html()
    if not check("66.7%" in page, "成功率按 2/3 渲染成 66.7%"):
        failures += 1
    if not check("LIDC_0089" in page and "LNDB_0196" in page, "病例 ID 出现在报告里"):
        failures += 1
    if not check(bool(data["chain"]["tail_hash"]) and data["chain"]["tail_hash"][:16] in page,
                 "链尾哈希被打印出来（唯一能对抗「整份日志被换」的手段）"):
        failures += 1
    if not check("203.0.113.9" in page, "被拒绝的访问出现在报告里"):
        failures += 1
    return failures


def layer_3_tamper() -> int:
    """头号用途是给合规结论 —— 读着被改过的日志还说"链完整"比不做报告更危险。"""
    print("\n[3] 篡改必被检出")
    failures = 0
    _reset()
    _seed(_dir().parent / "art.html")
    logs = _logs()
    if not check(len(logs) == 1, "夹具只产生一个日志文件", f"{len(logs)}"):
        return failures + 1

    original = [ln for ln in logs[0].read_text(encoding="utf-8").splitlines() if ln.strip()]

    def _verify() -> tuple[dict, str, int]:
        data = _collect()
        items = report_mod.verdicts(data)
        state = report_mod.overall(items)
        out = _dir().parent / "tampered.html"
        code = report_mod.main(["--out", str(out), "--quiet"])
        return data, state, code

    # ① 改内容、不动哈希
    record = json.loads(original[0])
    record["verdict"] = "ok-被篡改"
    lines = [json.dumps(record, ensure_ascii=False)] + original[1:]
    logs[0].write_text("\n".join(lines) + "\n", encoding="utf-8")
    data, state, code = _verify()
    if not check(data["chain"]["state"] == report_mod.BAD, "改一行的内容 → 链结论 bad"):
        failures += 1
    if not check(bool(data["chain"]["problems"]), "并且给出了具体问题行",
                 (data["chain"]["problems"] or ["（空）"])[0][:60]):
        failures += 1
    if not check(state == report_mod.BAD, "总判定 bad（不被其它项的 ok 冲淡）", state):
        failures += 1
    if not check(code == 1, "退出码 1（CI 能拦住）", f"exit={code}"):
        failures += 1

    page = (_dir().parent / "tampered.html").read_text(encoding="utf-8")
    if not check("内容被改" in page or "链校验发现的问题" in page, "报告里明确写出被改过"):
        failures += 1
    if not check('<div class="banner bad">' in page,
                 "横幅走「发现异常」分支，不混成「未判定」"):
        failures += 1

    # ② 写一半就断电：截断最后一行
    logs[0].write_text("\n".join(original) + "\n" + original[-1][:40], encoding="utf-8")
    data, state, code = _verify()
    if not check(data["chain"]["state"] == report_mod.BAD, "截断一行 → 链结论 bad（写一半的现场）"):
        failures += 1
    if not check(bool(data.get("broken_lines")), "无法解析的行被单列出来",
                 f"{len(data.get('broken_lines') or [])} 行"):
        failures += 1

    # 复原
    logs[0].write_text("\n".join(original) + "\n", encoding="utf-8")
    data, state, _ = _verify()
    if not check(state == report_mod.OK, "复原后重新变为 ok（不是粘住的红）", state):
        failures += 1
    return failures


def layer_4_redaction() -> int:
    print("\n[4] 脱敏：问答原文默认不进报告")
    failures = 0
    _reset()
    _seed(_dir().parent / "art2.html")

    page = _html()
    if not check(SENTINEL not in page, "默认报告里搜不到问答原文"):
        failures += 1
    if not check("字" in page, "但保留了字数（不是干脆没写这段）"):
        failures += 1
    if not check("题：" not in page and "问题：" not in page, "默认不出现问题正文小节"):
        failures += 1

    with_text = _html(include_text=True)
    if not check(SENTINEL in with_text, "--include-text 时才带出原文（说明是开关而非没实现）"):
        failures += 1
    if not check("已包含" in with_text, "报告头部如实标注「问答原文已包含」"):
        failures += 1

    data = _collect(include_text=True)
    if not check(data["runs"][0]["question"]["text"] is not None, "collect 层同步生效"):
        failures += 1
    return failures


def layer_5_html() -> int:
    print("\n[5] HTML 形态")
    failures = 0
    page = _html()
    raw = page.encode("utf-8")

    if not check("http://" not in page and "https://" not in page,
                 "没有任何外部 URL（单文件，可离线转发）"):
        failures += 1
    if not check("@import" not in page, "没有 CSS @import"):
        failures += 1
    if not check("@media print" in page, "带打印样式（报告是要打印归档的）"):
        failures += 1
    if not check(page.startswith("<!DOCTYPE html>"), "是完整 HTML 文档"):
        failures += 1
    for anchor in ("结论清单", "审计链明细", "运行明细", "工具调用统计", "产物指纹",
                   "鉴权事件", "PHI 守卫", "回归门禁与评测基线", "方法与边界"):
        if not check(anchor in page, f"小节「{anchor}」存在"):
            failures += 1
    if not check("哈希链能证明什么" in page and "不构成法律意义上的签名" in page,
                 "写明了哈希链能证明与不能证明的边界"):
        failures += 1
    if not check(len(raw) > 3000, "报告非空且规模合理", f"{len(raw)} 字节"):
        failures += 1
    return failures


def layer_6_gate() -> int:
    print("\n[6] 门禁结果的缺失与陈旧（陈旧数据不许当现状）")
    failures = 0
    _reset()
    _seed(_dir().parent / "art3.html")

    # ⚠️ 「缺失」用「把路径指到一个不存在的文件」来表达，**不去删**旧文件 ——
    # 本机（Windows + 沙箱/shim）会拒绝删除 Temp 下的文件（WinError 5），
    # 删不成会让自检**假红**：断言全过、rc 却是 1（见 PROJECT-NOTES §十三 13.18）。
    # 语义上也更直白：缺失 = 这个路径下没有文件，本来就不该依赖删除权限。
    report_mod.GATE_JSON = _dir().parent / "gate-does-not-exist.json"

    # ① 缺失
    items = report_mod.verdicts(_collect())
    item = next(i for i in items if i["name"] == "回归门禁")
    if not check(item["state"] == report_mod.NONE, "门禁文件缺失 → 未判定", item["state"]):
        failures += 1
    if not check("gate" in (item.get("fix") or ""), "并给出补齐命令", item.get("fix", "")):
        failures += 1
    if not check(report_mod.overall(items) == report_mod.NONE,
                 "缺门禁时总判定降级为未判定（不被其它 ok 冲淡）",
                 report_mod.overall(items)):
        failures += 1

    # ② 陈旧：昨天通过过，不等于现在通过
    _write_gate(report_mod.GATE_STALE_HOURS + 24, True, True)
    items = report_mod.verdicts(_collect())
    item = next(i for i in items if i["name"] == "回归门禁")
    if not check(item["state"] == report_mod.NONE, "陈旧门禁 → 未判定（不拿旧结果当现状）",
                 item["detail"][:50]):
        failures += 1
    if not check("陈旧" in _html(), "报告里明写「陈旧」"):
        failures += 1

    # ③ 新鲜
    _write_gate(0, True, True)
    if not check(report_mod.overall(report_mod.verdicts(_collect())) == report_mod.OK,
                 "门禁新鲜且通过 → 四项齐备为 ok"):
        failures += 1

    # ④ 门禁未通过
    _write_gate(0, False, False)
    items = report_mod.verdicts(_collect())
    if not check(all(i["state"] == report_mod.BAD
                     for i in items if i["name"] in ("回归门禁", "评测基线对比")),
                 "门禁未通过 → 门禁与评测两项都 bad"):
        failures += 1
    return failures


def layer_7_path_semantics() -> int:
    """`paths or list_logs()` 会把「筛完一条不剩」当成「没指定」→ 拿全量去跑。

    这类 falsy 陷阱在调用方完全看不出来，必须钉住语义。
    """
    print("\n[7] 空集合 ≠ 全量（store 的路径语义）")
    failures = 0
    _reset()
    _seed(_dir().parent / "art4.html")
    logs = _logs()

    if not check(len(list(audit.iter_records([]))) == 0,
                 "iter_records([]) 返回 0 条（不是全部）"):
        failures += 1

    ok, _ = audit.verify_chain([])
    if not check(ok, "verify_chain([]) 为 True（没选任何文件）"):
        failures += 1

    # 把日志改坏：此时「空集合」仍应为 True，而「指定该文件」应为 False。
    # 若实现退化成 `paths or list_logs()`，空集合会回退到全量 → 也变 False，被这条抓住。
    lines = [ln for ln in logs[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
    record = json.loads(lines[0])
    record["actor"] = "tampered-actor"
    lines[0] = json.dumps(record, ensure_ascii=False)
    logs[0].write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok_empty, _ = audit.verify_chain([])
    ok_file, problems = audit.verify_chain(logs)
    if not check(ok_empty, "日志被改坏后 verify_chain([]) 仍为 True（证明没有回退到全量）"):
        failures += 1
    if not check(not ok_file and problems, "verify_chain([该文件]) 为 False 并给出问题",
                 (problems or ["（空）"])[0][:50]):
        failures += 1
    return failures


def main() -> int:
    real_gate, real_baseline = report_mod.GATE_JSON, report_mod.BASELINE
    # ignore_cleanup_errors：临时目录是草稿，删不掉不算自检失败（§十三 13.18）
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["AIRNAV_AUDIT_DIR"] = str(Path(tmp) / "audit")
        report_mod.GATE_JSON = Path(tmp) / "gate_report.json"
        report_mod.BASELINE = Path(tmp) / "baseline.json"
        print("=" * 68)
        print("AirNav-Agent 报告自检（临时目录，不碰 outputs/）")
        print(f"审计目录：{os.environ['AIRNAV_AUDIT_DIR']}")
        print("=" * 68)

        failures = 0
        try:
            failures += layer_1_empty()
            failures += layer_2_stats()
            failures += layer_3_tamper()
            failures += layer_4_redaction()
            failures += layer_5_html()
            failures += layer_6_gate()
            failures += layer_7_path_semantics()
        finally:
            report_mod.GATE_JSON, report_mod.BASELINE = real_gate, real_baseline

    total = len(RESULTS)
    passed = sum(1 for ok, _, _ in RESULTS if ok)
    print("\n" + "=" * 68)
    if failures:
        print(f"未通过：{passed}/{total} 项通过，{failures} 项失败")
        for ok, label, detail in RESULTS:
            if not ok:
                print(f"  FAIL  {label}（{detail}）")
        return 1
    print(f"自检通过：{passed}/{total} 项全部符合预期")
    print("覆盖：缺数据≠通过 · 统计口径 · 篡改必被检出 · 脱敏 · HTML 自包含 · 陈旧门禁 · 空集合语义")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
