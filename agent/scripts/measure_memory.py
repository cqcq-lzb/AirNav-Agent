"""测量「加载病例 + 完整规划」的峰值内存占用。

用于判断 Agent 在 32 GB 机器上跑真实本地模型时，
留给病例处理的内存余量够不够。

用法：
    python -m agent.scripts.measure_memory [病例名] [候选编号]
"""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_memory() -> tuple[float, float]:
    """返回 (本进程已提交 MB, 系统可用物理 MB)。"""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return (
        (status.ullTotalPageFile - status.ullAvailPageFile) / 1024**2,
        status.ullAvailPhys / 1024**2,
    )


def main(argv: list[str]) -> int:
    from agent.core import case_loader
    from agent.core.planner import plan_candidate

    case_id = argv[0] if argv else "LIDC_0089"
    candidate_id = int(argv[1]) if len(argv) > 1 else 3

    case_dir = Path(case_loader.cases_root()) / case_id / "stage4_package"
    if not case_dir.is_dir():
        print(f"找不到病例目录：{case_dir}")
        return 2

    base_commit, base_phys = system_memory()
    print("病例  : %s" % case_dir)
    print("起始  : commit %8.1f MB  可用物理 %8.1f MB" % (base_commit, base_phys))

    t0 = time.time()
    ctx = case_loader.load_case(case_dir, case_id=case_id)
    c1, p1 = system_memory()
    print(
        "载入后: commit %8.1f MB (Δ%+7.1f)  可用物理 %8.1f MB   用时 %.2fs（缓存=%s）"
        % (c1, c1 - base_commit, p1, time.time() - t0, ctx.from_cache)
    )

    t0 = time.time()
    plan, _alternatives = plan_candidate(
        ctx, candidate_id, profile_name="balanced", device_diameter_mm=2.0
    )
    c2, p2 = system_memory()
    print(
        "规划后: commit %8.1f MB (Δ%+7.1f)  可用物理 %8.1f MB   用时 %.2fs"
        % (c2, c2 - base_commit, p2, time.time() - t0)
    )

    print()
    print("总峰值 commit 增量: %+.1f MB" % (c2 - base_commit))
    print("路由长度: %s mm" % getattr(plan, "route_length_mm", "n/a"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
