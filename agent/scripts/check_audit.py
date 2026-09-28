"""审计日志自检：写 → 读 → **篡改必被检出** → 脱敏 → 观测层不伤主流程。

    python -m agent.scripts.check_audit

秒级、不联网、不要模型。全程在临时目录里跑（`AIRNAV_AUDIT_DIR`），
**绝不碰 `outputs/audit/`** —— 和自检渲染走 `AIRNAV_VIEWER_DIR` 是同一条原则：
自检产生的是证据，不是交付物。

## 为什么「篡改必被检出」要专门测

哈希链这种东西，**不测就等于没有** —— 它平时永远返回「一切正常」，
只有被人改过时才该说话。所以这里不满足于「刚写完能校验通过」，
而是主动制造四种篡改，逐一确认能抓到：

    ① 改内容、不动哈希        → 该行「内容被改」
    ② 改内容并重算该行哈希    → 下一行「链接断裂」（攻击者会做的事）
    ③ 删掉中间一整行          → 下一行「链接断裂」
    ④ 把最后一行截断一半      → JSON 解析失败（写一半就断电的现场）

②是关键：只查「行内哈希对不对」是防不住有意的攻击者的 —— 他改完会顺手重算。
真正拦住他的是**下一行记着上一行的哈希**，而那一行他改不到（改了就轮到再下一行）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import audit  # noqa: E402
from agent.audit import store as store_mod  # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((ok, label, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"（{detail}）" if detail else ""))
    return ok


def _logs() -> list[Path]:
    return audit.list_logs()


def _lines() -> list[str]:
    """当日日志的原始行（保留空行之外的全部内容），用于篡改实验。"""
    paths = _logs()
    if not paths:
        return []
    return [ln for ln in paths[0].read_text(encoding="utf-8").splitlines() if ln.strip()]


def _rewrite(lines: list[str]) -> None:
    _logs()[0].write_text("\n".join(lines) + "\n", encoding="utf-8")


def _problems_text(problems: list[str]) -> str:
    return " | ".join(problems[:2])


# ------------------------------------------------------------------ 各层


def layer_1_write_and_shape() -> int:
    print("\n[1] 写入与字段结构")
    failures = 0

    record = audit.record_run(
        run_id="run-selfcheck-0001",
        started_at="2026-09-20T14:03:11+08:00",
        elapsed_s=9.13,
        question="LIDC_0089 的 3 号候选能过 1.5mm 器械吗？",
        answer="可以，路径代价 1.42。",
        backend="heuristic",
        model="rules",
        verdict="ok",
        actor="selfcheck",
        client_ip="127.0.0.1",
        user_agent="check_audit/1.0",
        params={"max_steps": 8, "device_diameter_mm": 1.5},
        steps=[{"step": 1, "tool": "plan_route", "args": {}, "ok": True, "elapsed_ms": 8123}],
        case_ids=["LIDC_0089", "LIDC_0089"],
    )
    if not check(record is not None, "record_run 返回了记录"):
        return failures + 1

    required = (
        "event", "run_id", "started_at", "elapsed_s", "actor", "client_ip", "user_agent",
        "backend", "model", "params", "case_ids", "question", "steps", "artifacts",
        "answer_sha256", "answer", "verdict", "ts", "prev_hash", "hash",
    )
    missing = [k for k in required if k not in record]
    if not check(not missing, f"记录含全部 {len(required)} 个必需字段", f"缺 {missing}"):
        failures += 1

    if not check(record["event"] == "run", "event 标记为 run", record["event"]):
        failures += 1
    if not check(record["case_ids"] == ["LIDC_0089"], "case_ids 去重并排序", f"{record['case_ids']}"):
        failures += 1
    if not check(record["prev_hash"] == store_mod.GENESIS, "首条记录挂在创世哈希上",
                 record["prev_hash"]):
        failures += 1

    if not check(len(_logs()) == 1, "当日日志恰好一个文件", f"{[p.name for p in _logs()]}"):
        failures += 1
    if not check(len(_lines()) == 1, "一条记录一行 JSONL", f"{len(_lines())} 行"):
        failures += 1

    # 大对象绝不进审计：几何数组只留类型与长度
    shrunk = json.dumps(record, ensure_ascii=False)
    if not check(len(shrunk) < 4000, "记录体积可控（没把几何塞进来）", f"{len(shrunk)} 字符"):
        failures += 1
    return failures


def layer_2_hash_chain() -> int:
    print("\n[2] 哈希链")
    failures = 0

    before = _logs()[0].read_bytes()
    for index in range(2):
        audit.append({"event": "run", "run_id": f"run-chain-{index}", "verdict": "ok"})
    after = _logs()[0].read_bytes()

    # append-only 的硬性质：新内容必须**以旧内容为前缀**，旧字节一个都不许动
    if not check(after.startswith(before), "追加式写入：旧字节一个都没动",
                 f"{len(before)} → {len(after)} 字节"):
        failures += 1

    lines = _lines()
    if not check(len(lines) == 3, "共 3 条记录", f"{len(lines)} 条"):
        failures += 1

    records = [json.loads(ln) for ln in lines]
    if not check(records[1]["prev_hash"] == records[0]["hash"],
                 "第 2 条记着第 1 条的哈希", f"{records[1]['prev_hash'][:12]}…"):
        failures += 1
    if not check(records[2]["prev_hash"] == records[1]["hash"],
                 "第 3 条记着第 2 条的哈希", f"{records[2]['prev_hash'][:12]}…"):
        failures += 1

    # 同样的内容必须永远算出同样的哈希（键序 / 中文都不能影响）
    shuffled = dict(reversed(list(records[0].items())))
    same = store_mod._digest(records[0]["prev_hash"], records[0]) == \
        store_mod._digest(shuffled["prev_hash"], shuffled)
    if not check(same, "规范化序列化：字段顺序不影响哈希"):
        failures += 1

    ok, problems = audit.verify_chain()
    if not check(ok, "未篡改时链校验通过", _problems_text(problems)):
        failures += 1
    return failures


def layer_3_tamper_must_be_caught() -> int:
    print("\n[3] 篡改必被检出（哈希链的存在意义）")
    failures = 0
    pristine = _lines()

    # ① 改内容、不动哈希
    lines = list(pristine)
    payload = json.loads(lines[1])
    payload["verdict"] = "篡改"   # 改一个业务字段
    lines[1] = json.dumps(payload, ensure_ascii=False)
    _rewrite(lines)
    ok, problems = audit.verify_chain()
    if not check(not ok and any("内容被改" in p for p in problems),
                 "① 改内容不动哈希 → 检出「内容被改」", _problems_text(problems)):
        failures += 1

    # ② 改内容**并重算该行哈希** —— 只查行内哈希是拦不住这种的
    lines = list(pristine)
    payload = json.loads(lines[1])
    payload["verdict"] = "tampered"
    payload.pop("hash", None)
    payload["hash"] = store_mod._digest(payload["prev_hash"], payload)
    lines[1] = json.dumps(payload, ensure_ascii=False)
    _rewrite(lines)
    ok, problems = audit.verify_chain()
    if not check(not ok and any("链接断裂" in p for p in problems),
                 "② 改内容并重算该行哈希 → 下一行「链接断裂」", _problems_text(problems)):
        failures += 1

    # ③ 删掉中间一整行（想「洗掉」一次不光彩的运行）
    lines = [pristine[0], pristine[2]]
    _rewrite(lines)
    ok, problems = audit.verify_chain()
    if not check(not ok and any("链接断裂" in p for p in problems),
                 "③ 删掉中间一行 → 检出「链接断裂」", _problems_text(problems)):
        failures += 1

    # ④ 最后一行被截断（写一半就断电 / 进程被杀）
    lines = list(pristine)
    lines[-1] = lines[-1][: len(lines[-1]) // 2]
    _rewrite(lines)
    ok, problems = audit.verify_chain()
    if not check(not ok and any("JSON 解析失败" in p for p in problems),
                 "④ 最后一行截断 → 检出「JSON 解析失败」", _problems_text(problems)):
        failures += 1

    # 复原，后面的层还要用
    _rewrite(pristine)
    ok, problems = audit.verify_chain()
    if not check(ok, "复原后重新通过校验", _problems_text(problems)):
        failures += 1
    return failures


def layer_4_artifacts_and_redaction() -> int:
    print("\n[4] 产物指纹与脱敏")
    failures = 0

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        blob = Path(tmp) / "viewer_selfcheck_c3.html"
        blob.write_bytes(b"<html>viewer payload</html>")
        record = audit.record_run(
            run_id="run-artifact-0001",
            question="出图",
            answer="已生成",
            backend="heuristic",
            model="rules",
            verdict="ok",
            artifacts=[{"kind": "viewer", "path": str(blob), "filename": blob.name}],
        )
        entry = (record or {}).get("artifacts", [{}])[0]
        expected = hashlib.sha256(blob.read_bytes()).hexdigest()
        if not check(entry.get("sha256") == expected,
                     "产物指纹与文件内容一致", f"{str(entry.get('sha256'))[:16]}…"):
            failures += 1
        if not check(entry.get("bytes") == len(blob.read_bytes()),
                     "记录了产物字节数", f"{entry.get('bytes')}"):
            failures += 1
        if not check(entry.get("file") == blob.name, "记录了产物文件名", f"{entry.get('file')}"):
            failures += 1

        # 产物事后被替换 → 指纹对不上（这正是 sha256 要回答的问题）
        blob.write_bytes(b"<html>DIFFERENT payload</html>")
        if not check(audit.hash_file(blob) != expected,
                     "产物内容变化后指纹随之改变（可证明「当时看的是哪一版」）"):
            failures += 1

    original = "LIDC_0089 的 3 号候选能过 1.5mm 器械吗？"
    previous = os.environ.get("AIRNAV_AUDIT_REDACT")
    os.environ["AIRNAV_AUDIT_REDACT"] = "1"
    try:
        record = audit.record_run(
            run_id="run-redact-0001",
            question=original,
            answer="可以",
            backend="heuristic",
            model="rules",
            verdict="ok",
        )
    finally:
        if previous is None:
            os.environ.pop("AIRNAV_AUDIT_REDACT", None)
        else:
            os.environ["AIRNAV_AUDIT_REDACT"] = previous

    question = (record or {}).get("question") or {}
    if not check(question.get("redacted") is True, "脱敏模式：question 标记 redacted",
                 f"{question}"):
        failures += 1
    if not check("text" not in question, "脱敏模式：不留问题原文"):
        failures += 1
    if not check(question.get("sha256") == hashlib.sha256(original.encode("utf-8")).hexdigest(),
                 "脱敏模式：保留 sha256 以便事后比对"):
        failures += 1
    if not check(question.get("chars") == len(original), "脱敏模式：保留字数",
                 f"{question.get('chars')}"):
        failures += 1
    if not check("answer_sha256" in (record or {}), "脱敏模式：答案哈希仍在"):
        failures += 1
    return failures


def layer_5_never_break_the_flow() -> int:
    print("\n[5] 观测层不得弄坏主流程")
    failures = 0

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        blocker = Path(tmp) / "not-a-dir.txt"
        blocker.write_text("x", encoding="utf-8")
        previous = os.environ.get("AIRNAV_AUDIT_DIR")
        os.environ["AIRNAV_AUDIT_DIR"] = str(blocker)   # 指向一个文件 → 必然写不进去
        try:
            result = audit.record_run(
                run_id="run-broken-dir",
                question="审计目录坏掉时也必须能提问",
                answer="照常回答",
                backend="heuristic",
                model="rules",
                verdict="ok",
            )
        except Exception as error:  # noqa: BLE001 —— 这里抛异常就是失败本身
            check(False, "审计写入失败时不抛异常", f"{type(error).__name__}: {error}")
            failures += 1
            result = "raised"
        finally:
            if previous is None:
                os.environ.pop("AIRNAV_AUDIT_DIR", None)
            else:
                os.environ["AIRNAV_AUDIT_DIR"] = previous

        if result != "raised":
            if not check(result is None, "审计写入失败时安静返回 None（调用方不用 try）"):
                failures += 1

    ok, problems = audit.verify_chain()
    if not check(ok, "环境复原后链仍然完整", _problems_text(problems)):
        failures += 1
    return failures


def layer_6_query() -> int:
    print("\n[6] 查询")
    failures = 0

    hits = audit.query(run_id="run-selfcheck-0001")
    if not check(len(hits) == 1, "按 run_id 精确命中 1 条", f"{len(hits)} 条"):
        failures += 1

    hits = audit.query(case_id="LIDC_0089")
    if not check(len(hits) == 1, "按 case_id 命中", f"{len(hits)} 条"):
        failures += 1

    hits = audit.query(actor="selfcheck")
    if not check(len(hits) == 1, "按 actor 命中", f"{len(hits)} 条"):
        failures += 1

    hits = audit.query(run_id="不存在的 run")
    if not check(len(hits) == 0, "查不到时返回空列表（不是报错）", f"{len(hits)} 条"):
        failures += 1

    hits = audit.query(since_days=1)
    if not check(len(hits) >= 3, "按时间窗取出全部记录", f"{len(hits)} 条"):
        failures += 1
    return failures


def main() -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["AIRNAV_AUDIT_DIR"] = tmp
        print("=" * 68)
        print("AirNav-Agent 审计日志自检（临时目录，不碰 outputs/audit/）")
        print(f"审计目录：{tmp}")
        print("=" * 68)

        failures = 0
        failures += layer_1_write_and_shape()
        failures += layer_2_hash_chain()
        failures += layer_3_tamper_must_be_caught()
        failures += layer_4_artifacts_and_redaction()
        failures += layer_5_never_break_the_flow()
        failures += layer_6_query()

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
    print("覆盖：字段结构 · 哈希链 · 四种篡改必被检出 · 产物指纹 · 脱敏 · 写失败不伤主流程 · 查询")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
