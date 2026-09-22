"""测量导入指定模块前后，系统 commit（提交内存）与可用物理内存的变化。

用途：量化 GUI 依赖（vtk / PySide6）对 Agent 进程的固定内存开销，
说明为什么无界面规划路径应当彻底绕过它们。

用法：
    python -m agent.scripts._memprobe numpy SimpleITK vtk PySide6.QtWidgets
"""
from __future__ import annotations

import sys

from agent.core.sysmem import read_memory


def probe() -> tuple[float, float]:
    """返回 (当前进程已提交 MB, 系统可用物理内存 MB)。

    读数来自 `agent.core.sysmem` —— 本仓库唯一一份内存读数实现（本文件原先自带的
    那份 `MEMORYSTATUSEX` 与 `measure_memory.py`、`verify_geometry.py` 逐字节相同，
    已合并，见该模块文档）。

    读不到时返回 `(nan, nan)`，**不伪装成 0**：0 会让上面的差值显示读成
    「内存耗尽」，而真相是「这个平台读不到」。
    """
    reading = read_memory()
    committed = reading.committed_mib
    available = reading.available_mib
    return (
        float("nan") if committed is None else committed,
        float("nan") if available is None else available,
    )


def main(argv: list[str]) -> int:
    if not argv:
        print("用法: python -m agent.scripts._memprobe <模块名> [模块名 ...]")
        return 2

    before_commit, before_phys = probe()
    loaded: list[str] = []
    for name in argv:
        try:
            __import__(name)
            loaded.append(name)
        except Exception as error:  # noqa: BLE001 - 探针工具，导入失败要报出来
            print(f"  导入 {name} 失败：{type(error).__name__}: {error}")
    after_commit, after_phys = probe()

    print("  已导入  : " + ", ".join(loaded))
    print(
        "  commit  : %8.1f MB -> %8.1f MB  (Δ %+7.1f MB)"
        % (before_commit, after_commit, after_commit - before_commit)
    )
    print(
        "  可用物理: %8.1f MB -> %8.1f MB  (Δ %+7.1f MB)"
        % (before_phys, after_phys, after_phys - before_phys)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
