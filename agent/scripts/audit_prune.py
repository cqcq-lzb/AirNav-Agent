"""按保留策略归档旧的审计日志。**默认只演算，加 `--apply` 才真的动文件。**

    python -m agent.scripts.audit_prune --days 1825              # 看会动哪些（不删不改）
    python -m agent.scripts.audit_prune --days 1825 --apply      # 真归档
    python -m agent.scripts.audit_prune --days 1825 --apply --archive-dir E:\\audit-archive

## 三个刻意的设计

1. **默认不动手**。审计数据的删除是不可逆的（这正是它作为证据的价值），
   所以默认只演算，必须显式 `--apply`。和本项目其它「危险动作要另加开关」一致。

2. **先验链，后归档**。归档前先校验这份日志的哈希链 —— 链坏了就不动它，
   并如实报告。否则「归档」就成了替人**藏起被篡改的证据**的操作。

3. **归档不是删除**。默认移到 `<审计目录>/archive/`（`list_logs()` 不递归，
   所以归档后自动退出日常查询范围），而不是 `rm`。

## 保留期为什么默认 5 年

医疗数据的追溯期通常按 3~5 年要求（各地规定不同，按你们院的来）。
默认 1825 天 = 5 年，宁可留着 —— 日志按天切片，一年的量也就是几 MB，
为了省几 MB 去删掉证据不划算。

## 为什么按天切片让归档很安全

哈希链是**按文件独立**的（每天从创世哈希起链）。所以整份文件搬走不会让
任何一条链断裂 —— 这也是当初选「按天切片 + 每文件独立起链」而不是「一条全局长链」的原因。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import audit  # noqa: E402


def _day_of(path: Path) -> datetime | None:
    """从 audit-YYYY-MM-DD.jsonl 文件名里取日期。"""
    stem = path.stem
    if not stem.startswith("audit-"):
        return None
    try:
        return datetime.strptime(stem[len("audit-"):], "%Y-%m-%d")
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按保留策略归档旧审计日志")
    parser.add_argument("--dir", help="审计目录（默认 outputs/audit 或 $AIRNAV_AUDIT_DIR）")
    parser.add_argument("--days", type=int, default=1825, help="保留天数（默认 1825 = 5 年）")
    parser.add_argument("--archive-dir", dest="archive_dir", help="归档目录（默认 <审计目录>/archive）")
    parser.add_argument("--apply", action="store_true", help="真的执行归档（不加则只演算）")
    args = parser.parse_args(argv)

    if args.dir:
        os.environ["AIRNAV_AUDIT_DIR"] = args.dir

    root = audit.audit_dir()
    archive = Path(args.archive_dir) if args.archive_dir else root / "archive"
    cutoff = (datetime.now() - timedelta(days=args.days)).date()
    logs = audit.list_logs()

    print("=" * 68)
    print(f"审计日志保留策略 —— 目录 {root}")
    print(f"保留 {args.days} 天；早于 {cutoff} 的归档到 {archive}")
    print(f"模式：{'执行归档' if args.apply else '只演算（加 --apply 才动手）'}")
    print("=" * 68)

    if not logs:
        print("没有审计日志。")
        return 0

    # 只有**完整落在保留期之外**的文件才归档；跨界的当天文件不动
    # （否则会把今天正在写入的日志搬走，正在跑的进程会继续往旧句柄写）
    candidates: list[Path] = []
    kept = 0
    for path in logs:
        day = _day_of(path)
        if day is None:
            print(f"  SKIP {path.name}（文件名不是 audit-YYYY-MM-DD.jsonl，不动它）")
            kept += 1
            continue
        if day.date() < cutoff:
            candidates.append(path)
        else:
            kept += 1

    if not candidates:
        print(f"\n无需归档：{kept} 个文件都在保留期内。")
        return 0

    moved = 0
    refused = 0
    for path in candidates:
        ok, problems = audit.verify_chain([path])
        if not ok:
            # 链坏了先别搬 —— 归档一份被篡改的日志等于帮人把证据收进抽屉
            refused += 1
            print(f"  REFUSE {path.name}：哈希链不完整，先查清再归档")
            for problem in problems[:2]:
                print(f"         ! {problem}")
            continue
        size_mb = path.stat().st_size / (1 << 20)
        if not args.apply:
            print(f"  WILL   {path.name}  {size_mb:.2f} MB  → {archive / path.name}")
            moved += 1
            continue
        archive.mkdir(parents=True, exist_ok=True)
        target = archive / path.name
        if target.exists():
            # 目标已存在说明之前归档过一次；加时间戳后缀，绝不覆盖审计数据
            target = archive / f"{path.stem}.{datetime.now():%H%M%S}{path.suffix}"
        shutil.move(str(path), str(target))
        moved += 1
        print(f"  MOVED  {path.name}  {size_mb:.2f} MB  → {target}")

    print("-" * 68)
    print(f"{'已归档' if args.apply else '待归档'} {moved} 个文件；保留 {kept} 个；"
          f"拒绝 {refused} 个（链不完整）")
    if not args.apply and moved:
        print("这是演算结果，没有任何文件被移动。确认无误后加 --apply。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
