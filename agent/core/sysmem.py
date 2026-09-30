"""系统内存读数：可用物理内存 / 本进程已提交。

## 为什么要有这个模块

本仓库曾有**三份逐字节相同**的 `_MemoryStatus` ctypes 结构体
（`agent/scripts/_memprobe.py`、`agent/scripts/measure_memory.py`、
`agent/scripts/verify_geometry.py`），而它们要的其实是同一个数。

危险的地方不在于重复，而在于 `GlobalMemoryStatusEx` **不会因为结构体字段写错而报错** ——
它只会把数**填进错位的字段**。少写一个字段，`ullAvailPhys` 读到就可能是别的量，
而调用方拿到一个「看起来合理」的数字继续往下判断。这类错最难查。
所以统一在这里，只留一份。

## 跨平台

- Windows：`GlobalMemoryStatusEx`
- Linux：`/proc/meminfo`
- 其它平台：返回 `source="unavailable"`，各字段为 `None`

**调用方必须处理 `None`** ——「读不到」既不是「内存充足」也不是「内存不足」。
拿默认值 0 去比较，会把「不知道」误判成「内存耗尽」。
"""
from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

MIB = 1024**2


class _MemoryStatusEx(ctypes.Structure):
    """Windows `MEMORYSTATUSEX`。

    ⚠️ 字段顺序就是 ABI：一个都不能挪、不能省。少一个字段，其后所有字段的偏移
    都会错位，而 API 仍然返回成功 —— 于是拿到一个错位数。
    """

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


@dataclass(frozen=True)
class MemoryReading:
    """一次内存读数。字段为 `None` ＝ 这个平台/这次调用拿不到。

    `committed` 是**本进程已提交**（页文件口径），用来量「导入某模块要多少内存」；
    与 `available_phys`（**系统级**可用物理内存）不是一回事，别混用。
    """

    total_phys: int | None = None
    available_phys: int | None = None
    committed: int | None = None
    # 🔴 **可用提交**（页文件还能扩多少）—— 与 `available_phys` 是**两个不同的上限**。
    # 2026-09-30 实测踩的坑：物理 31.7 GB、可用物理 12.1 GB，但**已提交 38.9 GB**
    # （超过物理，全靠页文件顶）。此时要 882 MiB **连续已提交**空间的 numpy 分配
    # 照样失败（`_ArrayMemoryError`），而「可用物理」看着还剩 12 GB ——
    # 于是守卫放行、真分配时崩掉，把「机器腾不出资源」写成了 ❌（＝回归）。
    # 判据：**申请连续大块内存时，看 `min(available_phys, available_commit)`**。
    available_commit: int | None = None
    source: str = "unavailable"

    @property
    def available_mib(self) -> float | None:
        return None if self.available_phys is None else self.available_phys / MIB

    @property
    def committed_mib(self) -> float | None:
        return None if self.committed is None else self.committed / MIB

    @property
    def available_commit_mib(self) -> float | None:
        return None if self.available_commit is None else self.available_commit / MIB

    @property
    def headroom_bytes(self) -> int | None:
        """**还能安全申请多少连续内存** = min(可用物理, 可用提交)。

        申请大块连续内存时要看这个，而不是只看 `available_phys` —— 见字段注释。
        两者任一取不到就退回另一个（不知道的那一侧不参与限制）。
        """
        candidates = [v for v in (self.available_phys, self.available_commit) if v is not None]
        return min(candidates) if candidates else None


def _read_windows() -> MemoryReading | None:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except (OSError, AttributeError):
        return None
    return MemoryReading(
        total_phys=int(status.ullTotalPhys),
        available_phys=int(status.ullAvailPhys),
        committed=int(status.ullTotalPageFile - status.ullAvailPageFile),
        available_commit=int(status.ullAvailPageFile),
        source="GlobalMemoryStatusEx",
    )


def _read_linux() -> MemoryReading | None:
    wanted = {"MemTotal": "total", "MemAvailable": "avail"}
    got: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in wanted and wanted[key] not in got:
                    got[wanted[key]] = int(line.split()[1]) * 1024
                if len(got) == len(wanted):
                    break
    except (OSError, ValueError, IndexError):
        return None
    if "avail" not in got:
        return None
    return MemoryReading(
        total_phys=got.get("total"),
        available_phys=got["avail"],
        committed=None,          # Linux 侧没有等价物（要看得另读 /proc/self/status）
        available_commit=None,   # 同上；Linux 的提交口径是 CommitLimit − Committed_AS
        source="/proc/meminfo",
    )


def read_memory() -> MemoryReading:
    """读一次系统内存。拿不到返回 `source="unavailable"` 的读数（**不抛异常**）。"""
    if sys.platform == "win32":
        reading = _read_windows()
    elif sys.platform.startswith("linux"):
        reading = _read_linux()
    else:
        reading = None
    return reading if reading is not None else MemoryReading()


def available_bytes() -> int | None:
    """可用物理内存（字节）。读不到返回 `None` —— **不要当成 0，也不要当成无穷**。"""
    return read_memory().available_phys


def available_mib() -> float | None:
    """可用物理内存（MiB）。读不到返回 `None`。"""
    return read_memory().available_mib


def committed_mib() -> float | None:
    """本进程已提交内存（MiB）。读不到返回 `None`。"""
    return read_memory().committed_mib


__all__ = [
    "MemoryReading",
    "available_bytes",
    "available_mib",
    "committed_mib",
    "read_memory",
]
