"""校验审计日志的哈希链完整性。退出码 0 = 完整，1 = 有问题（可进 CI / 门禁）。

    python -m agent.scripts.audit_verify                    # 校验默认审计目录
    python -m agent.scripts.audit_verify --dir D:\\audit     # 指定目录（归档盘）
    python -m agent.scripts.audit_verify --quiet            # 只给退出码

## 它到底证明了什么

**证明了「这份日志没有被改动过」**，前提是链尾的哈希被另外留存过
（打印出来抄进月报、或写进另一台机器）。如果有人能改日志、也能改这里，那他改完
整条链重算一遍就看不出来了 —— 这是哈希链的边界，得说清楚，别当成签名用。
真正需要抗抵赖时上的是「链尾定期外发 + 时间戳」，属 P1。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import audit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验审计日志哈希链")
    parser.add_argument("--dir", help="审计目录（默认 outputs/audit 或 $AIRNAV_AUDIT_DIR）")
    parser.add_argument("--quiet", action="store_true", help="只在失败时输出")
    args = parser.parse_args(argv)

    if args.dir:
        os.environ["AIRNAV_AUDIT_DIR"] = args.dir

    logs = audit.list_logs()
    if not logs:
        print(f"没有找到审计日志（{audit.audit_dir()}）")
        return 0

    print("=" * 68)
    print(f"审计日志哈希链校验 —— {audit.audit_dir()}")
    print("=" * 68)

    total_records = 0
    total_bad = 0
    tail_hash = ""
    for path in logs:
        records = [r for r in audit.iter_records([path])]
        bad = [r for r in records if r.get("_broken")]
        ok, problems = audit.verify_chain([path])
        total_records += len(records)
        total_bad += len(problems)
        last = records[-1] if records else {}
        tail_hash = last.get("hash") or tail_hash
        mark = "OK  " if ok and not bad else "BAD "
        print(f"  {mark} {path.name}  {len(records)} 条  "
              f"链尾 {str(last.get('hash'))[:16]}…  "
              f"({records[0].get('ts') if records else '-'} → {last.get('ts', '-')})")
        for problem in problems:
            print(f"        ! {problem}")

    print("-" * 68)
    print(f"共 {len(logs)} 个文件 / {total_records} 条记录；问题 {total_bad} 处")
    print(f"链尾哈希：{tail_hash}")
    if total_bad:
        print("\n结论：**链被破坏或文件被改动过**，需要人工比对。")
        return 1
    print("\n结论：链完整。把上面的链尾哈希抄下来存到别处，"
          "下次校验时它的值若对不上，就说明中间被动过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
