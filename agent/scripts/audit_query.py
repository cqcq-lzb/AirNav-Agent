"""按日期 / 病例 / 操作者查询审计记录，可导出 CSV。

    python -m agent.scripts.audit_query --case LIDC_0089
    python -m agent.scripts.audit_query --days 7 --csv D:\\tmp\\runs.csv
    python -m agent.scripts.audit_query --run-id 9f2c…  --json

## 用在哪

- **复盘**：「上周三对 LIDC_0089 那次给了什么结论？」→ `--case` 捞出全部运行；
- **对账**：「这一版三维视图当时算出来的指纹是多少？」→ 记录里的 `artifacts[].sha256`
  和手上的文件一对比就知道有没有被换过；
- **交差**：医疗场景里审计常常要**导出**给人看，所以 CSV 是内建能力而不是附加功能。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import audit  # noqa: E402

COLUMNS = [
    "ts", "event", "run_id", "actor", "client_ip",
    "verdict", "backend", "model", "elapsed_s",
    "case_ids", "tools", "artifacts", "question_chars", "answer_sha256",
]


def _tool_names(record: dict) -> str:
    return ",".join(str(s.get("tool")) for s in (record.get("steps") or []) if s.get("tool"))


def _artifact_names(record: dict) -> str:
    return ",".join(str(a.get("file")) for a in (record.get("artifacts") or []) if a.get("file"))


def _question_chars(record: dict) -> int:
    """原文明文与脱敏两种形态都给出「多长」，方便对账。"""
    question = record.get("question")
    if isinstance(question, dict):
        return question.get("chars") if question.get("redacted") else len(question.get("text") or "")
    return len(question or "") if isinstance(question, str) else 0


def to_row(record: dict) -> dict:
    return {
        "ts": record.get("ts") or record.get("started_at") or "",
        "event": record.get("event") or "",
        "run_id": record.get("run_id") or "",
        "actor": record.get("actor") or "",
        "client_ip": record.get("client_ip") or "",
        "verdict": record.get("verdict") or ("allowed" if record.get("allowed") else ""),
        "backend": record.get("backend") or "",
        "model": record.get("model") or "",
        "elapsed_s": record.get("elapsed_s") if record.get("elapsed_s") is not None else "",
        "case_ids": ",".join(record.get("case_ids") or []),
        "tools": _tool_names(record),
        "artifacts": _artifact_names(record),
        "question_chars": _question_chars(record) if record.get("event") == "run" else "",
        "answer_sha256": (record.get("answer_sha256") or "")[:16],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="查询审计日志")
    parser.add_argument("--dir", help="审计目录（默认 outputs/audit 或 $AIRNAV_AUDIT_DIR）")
    parser.add_argument("--case", help="按病例 ID（子串匹配 case_ids）")
    parser.add_argument("--actor", help="按操作者")
    parser.add_argument("--run-id", dest="run_id", help="按一次运行")
    parser.add_argument("--days", type=int, help="只查最近 N 天")
    parser.add_argument("--limit", type=int, default=50, help="最多输出多少条（默认 50）")
    parser.add_argument("--csv", dest="csv_path", help="导出 CSV 到指定路径")
    parser.add_argument("--json", action="store_true", help="原样输出 JSON（含完整字段）")
    args = parser.parse_args(argv)

    if args.dir:
        os.environ["AIRNAV_AUDIT_DIR"] = args.dir

    records = audit.query(
        case_id=args.case,
        actor=args.actor,
        run_id=args.run_id,
        since_days=args.days,
    )
    if not records:
        print(f"没有命中记录（目录 {audit.audit_dir()}）")
        return 0

    records = records[-args.limit:]
    rows = [to_row(r) for r in records]

    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        print(f"命中 {len(rows)} 条（目录 {audit.audit_dir()}）\n")
        width = {"ts": 20, "event": 5, "actor": 10, "verdict": 12, "case_ids": 22}
        header = (f"{'时间':<{width['ts']}} {'事件':<{width['event']}} "
                  f"{'操作者':<{width['actor']}} {'结论':<{width['verdict']}} "
                  f"{'病例':<{width['case_ids']}} 工具 / 产物")
        print(header)
        print("-" * len(header))
        for row in rows:
            tail = row["tools"] or row["artifacts"] or ""
            print(f"{row['ts']:<{width['ts']}} {row['event']:<{width['event']}} "
                  f"{row['actor']:<{width['actor']}} {row['verdict']:<{width['verdict']}} "
                  f"{row['case_ids']:<{width['case_ids']}} {tail}")

    if args.csv_path:
        target = Path(args.csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # utf-8-sig：Excel 打开中文 CSV 不装 utf-8-sig 会全是乱码
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n已导出 CSV：{target}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
