"""审计日志：追加式 JSONL + 哈希链。

## 为什么要自己写而不是上数据库

P0 的目标是**可追溯**，不是查询性能：单机并发低，审计要的是「不可改 + 可回放」。
一行一条 JSONL、每行带上一行的哈希，就同时拿到了这两个性质 ——
零依赖、可 grep、可 diff、迁移时按 schema 灌进 SQLite 即可。

## 哈希链怎么起作用

第 n 条记录记 `prev_hash` = 第 n-1 条的 `hash`，而
`hash = sha256(prev_hash + 规范化 JSON(本条内容))`。
于是改任何一条的任何一个字段，从它开始往后的每一条都对不上 —— 篡改无法局部完成。
`verify_chain()` 就是干这个的，已进回归门禁。

## 三条纪律

1. **写入失败绝不影响主流程**。审计是观测层，和 `on_event` 同一条原则：
   观测层不该有能力弄坏控制流。所有异常在这里被吞掉并打 stderr。
2. **不存密码、不存私钥**；患者侧只出现病例 ID（本身就是假名）。
   `AIRNAV_AUDIT_REDACT=1` 时连问题原文都不留，只留哈希与长度。
3. **目录不入库**（`outputs/audit/`），审计数据属于运行现场，不属于源码仓库。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]

# 写指针串行化：同一进程内多个 Agent 线程可能同时收尾
_LOCK = threading.Lock()

GENESIS = "genesis"


# ------------------------------------------------------------------ 路径


def audit_dir() -> Path:
    """审计根目录，可用 `AIRNAV_AUDIT_DIR` 改写（自检就靠它落到临时目录）。"""
    override = os.environ.get("AIRNAV_AUDIT_DIR")
    if override:
        return Path(override)
    return ROOT / "outputs" / "audit"


def log_path(day: datetime | None = None) -> Path:
    """当天切片。按天切是为了「保留策略」能按日期做，而不是删一个巨型文件。"""
    day = day or datetime.now()
    return audit_dir() / f"audit-{day.strftime('%Y-%m-%d')}.jsonl"


def list_logs() -> list[Path]:
    root = audit_dir()
    if not root.is_dir():
        return []
    return sorted(root.glob("audit-*.jsonl"))


# ------------------------------------------------------------------ 哈希链


def _canonical(record: dict[str, Any]) -> bytes:
    """规范化序列化：键排序 + 不转义中文，保证同样的内容永远算出同样的哈希。"""
    body = {k: v for k, v in record.items() if k != "hash"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(prev_hash: str, record: dict[str, Any]) -> str:
    return hashlib.sha256(prev_hash.encode("utf-8") + _canonical(record)).hexdigest()


def _last_hash(path: Path) -> str:
    """读文件最后一行拿 prev_hash 起点。文件不存在或最后一行坏了就从创世哈希起。"""
    if not path.is_file():
        return GENESIS
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        if not lines:
            return GENESIS
        return json.loads(lines[-1]).get("hash") or GENESIS
    except (OSError, json.JSONDecodeError):
        return GENESIS


def hash_file(path: Path, chunk: int = 1 << 20) -> str | None:
    """产物指纹：事后能证明「当时看的确实是这一份文件」。

    viewer 有 1.7~3MB，分块读，别整个吞进内存。
    """
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(chunk):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


# ------------------------------------------------------------------ 写入


def append(record: dict[str, Any], day: datetime | None = None) -> dict[str, Any] | None:
    """追加一条审计记录（自动补 ts / prev_hash / hash）。

    失败返回 None 并打 stderr —— 调用方不需要 try/except，也不该因为审计而失败。
    """
    path = log_path(day)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            record = dict(record)
            record.setdefault("ts", datetime.now().astimezone().isoformat(timespec="seconds"))
            record["prev_hash"] = _last_hash(path)
            record["hash"] = _digest(record["prev_hash"], record)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return record
    except Exception as error:  # noqa: BLE001 —— 审计不许反过来影响主流程
        print(f"[audit] 写入失败（已忽略）：{type(error).__name__}: {error}", flush=True)
        return None


def _maybe_redact(text: str | None) -> dict[str, Any] | None:
    """`AIRNAV_AUDIT_REDACT=1` 时只留哈希与长度，不留原文。"""
    if text is None:
        return None
    if os.environ.get("AIRNAV_AUDIT_REDACT") == "1":
        return {
            "redacted": True,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chars": len(text),
        }
    return {"redacted": False, "text": text}


def record_run(
    *,
    run_id: str,
    question: str,
    answer: str | None,
    backend: str,
    model: str | None,
    verdict: str,
    actor: str = "anonymous",
    client_ip: str | None = None,
    user_agent: str | None = None,
    params: dict[str, Any] | None = None,
    steps: Iterable[dict[str, Any]] | None = None,
    artifacts: Iterable[dict[str, Any]] | None = None,
    case_ids: Iterable[str] | None = None,
    error: str | None = None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
) -> dict[str, Any] | None:
    """一次 Agent 运行的完整留痕。"""
    artifact_list: list[dict[str, Any]] = []
    for item in artifacts or []:
        entry: dict[str, Any] = {
            "kind": item.get("kind"),
            "file": item.get("filename") or (Path(item["path"]).name if item.get("path") else None),
        }
        path = item.get("path")
        if path and Path(path).is_file():
            entry["sha256"] = hash_file(Path(path))
            entry["bytes"] = Path(path).stat().st_size
        artifact_list.append(entry)

    return append(
        {
            "event": "run",
            "run_id": run_id,
            "started_at": started_at,
            "elapsed_s": elapsed_s,
            "actor": actor,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "backend": backend,
            "model": model,
            "params": params or {},
            "case_ids": sorted(set(case_ids or [])),
            "question": _maybe_redact(question),
            "steps": list(steps or []),
            "artifacts": artifact_list,
            "answer_sha256": hashlib.sha256((answer or "").encode("utf-8")).hexdigest(),
            "answer": _maybe_redact(answer),
            "verdict": verdict,
            "error": error,
        }
    )


def record_auth(
    *,
    allowed: bool,
    path: str,
    client_ip: str | None = None,
    user_agent: str | None = None,
    actor: str = "anonymous",
) -> dict[str, Any] | None:
    """鉴权事件。**失败也要记** —— 谁在什么时候试图进来，本身就是审计要回答的问题。"""
    return append(
        {
            "event": "auth",
            "allowed": allowed,
            "path": path,
            "actor": actor,
            "client_ip": client_ip,
            "user_agent": user_agent,
        }
    )


def record_phi_scan(
    *,
    roots: Iterable[str],
    scanned: dict[str, Any],
    confirmed: int,
    suspicious: int,
    hits: Iterable[dict[str, Any]] | None = None,
    selftest_failures: int = 0,
    actor: str = "phi-guard",
) -> dict[str, Any] | None:
    """PHI 扫描留痕。

    为什么要留：「我们扫过了、当时是干净的」本身是要能举证的结论。
    只记**命中的位置（路径 + 字段）**，不记命中内容 —— 审计日志不是第二个 PHI 副本，
    把患者信息抄进日志等于自己制造泄漏面。这一点和 `record_run` 里
    `_maybe_redact` 是同一条原则。
    """
    return append(
        {
            "event": "phi_scan",
            "actor": actor,
            "roots": [str(item) for item in roots],
            "scanned": dict(scanned),
            "confirmed": confirmed,
            "suspicious": suspicious,
            "clean": confirmed == 0 and selftest_failures == 0,
            "selftest_failures": selftest_failures,
            # 只留位置，不留内容
            "hits": [{"path": str(h.get("path")), "where": h.get("where")} for h in (hits or [])],
        }
    )


# ------------------------------------------------------------------ 读取与校验


def _resolve_paths(paths: Iterable[Path] | None) -> list[Path]:
    """``None`` = 全部日志；**空集合 = 一条都不选**。

    别写成 `paths or list_logs()`：那样「筛完一条不剩」会被当成「没指定」，
    于是拿**全量**去跑 —— 静默地把「报告期内无记录」变成「报告期内有全部记录」。
    这类 falsy 陷阱在调用方完全看不出来，故显式区分。
    """
    if paths is None:
        return list_logs()
    return [Path(item) for item in paths]


def iter_records(paths: Iterable[Path] | None = None) -> Iterator[dict[str, Any]]:
    for path in _resolve_paths(paths):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    yield {"_broken": True, "_file": str(path), "_line": number, "_error": str(error)}


def verify_chain(paths: Iterable[Path] | None = None) -> tuple[bool, list[str]]:
    """逐条重算哈希链。返回 (是否完整, 问题列表)。

    每个文件独立成链（从 `GENESIS` 起），因为日志按天切片 —— 某天删掉整份文件
    不会让后面的天全部对不上。**代价**：删掉某天的一整份日志是看不出来的，
    这正是 `audit_verify` 文档里说的「链尾哈希要另存」的原因。
    """
    problems: list[str] = []
    prev = GENESIS
    for path in _resolve_paths(paths):
        if not path.is_file():
            continue
        prev = GENESIS
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    problems.append(f"{path.name}:{number} JSON 解析失败：{error}")
                    continue
                if record.get("prev_hash") != prev:
                    problems.append(
                        f"{path.name}:{number} 链接断裂：prev_hash={str(record.get('prev_hash'))[:12]}… "
                        f"但上一条是 {prev[:12]}…"
                    )
                expected = _digest(record.get("prev_hash", GENESIS), record)
                if record.get("hash") != expected:
                    problems.append(f"{path.name}:{number} 内容被改（哈希不匹配）")
                prev = record.get("hash") or prev
    return not problems, problems


def query(
    *,
    case_id: str | None = None,
    actor: str | None = None,
    run_id: str | None = None,
    since_days: int | None = None,
) -> list[dict[str, Any]]:
    """按条件捞记录。数据量按天切片，全量扫足够用（P1 迁库时换这里即可）。"""
    floor = None
    if since_days is not None:
        floor = (datetime.now() - timedelta(days=since_days)).strftime("%Y-%m-%d")
    paths = [p for p in list_logs() if floor is None or p.stem >= f"audit-{floor}"]

    hits: list[dict[str, Any]] = []
    for record in iter_records(paths):
        if record.get("_broken"):
            continue
        if case_id and case_id not in (record.get("case_ids") or []):
            continue
        if actor and record.get("actor") != actor:
            continue
        if run_id and record.get("run_id") != run_id:
            continue
        hits.append(record)
    return hits
