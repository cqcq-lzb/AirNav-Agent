"""对拍 `agent/core/geometry.py` 的零分配实现与 scipy 的 `distance_transform_edt`。

背景：scipy 的 EDT 在内部必须分配 `(ndim, *体素数)` 的 float64 特征变换数组，
在本项目的 CT 上就是 882MB / 次（见 core/geometry.py 模块文档）。
我们把两处只用「少数查询点的取值」的调用换成了 KD 树实现。
换完必须证明**结果没有变化** —— 规划数值一旦漂移，V1 与 Agent 就不可比了。

本脚本做两类检查：

1. **合成体积穷举**（`--only synthetic`）
   构造一个带非各向同性 spacing 的小体积，对**全部体素**逐点比较新实现与 EDT。
   穷举而非抽样，因为这是算法正确性的证明，不是性能测试。
   额外覆盖三个退化场景：全 True 掩膜、全 False 掩膜、空目标掩膜。

2. **真实病例抽样**（`--only radius` / `--only distance`）
   在 LIDC_0089 的真实掩膜上，随机抽一批体素坐标（掩膜内外都有）比较。
   等价性是逐点的，因此抽样足以证伪；这里不抽取骨架，是为了让覆盖更广。

用法：
    python -m agent.scripts.verify_geometry                  # 全部
    python -m agent.scripts.verify_geometry --only synthetic  # 只跑合成

退出码（三态，与门禁的 ok/bad/none 对应）：
    0  → 通过
    1  → 未通过（对拍不一致 —— 这是真回归）
    2  → **不足以判定**：整卷 EDT 要 882MB 特征数组，这台机器此刻腾不出来。
         这不是几何实现的问题，因此**不能**记成失败（在门禁里长得跟真回归一样），
         也**绝不能**记成通过。按「缺数据 ≠ 通过」单列为 ⚪ 并给出补齐命令。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.core import geometry  # noqa: E402
from agent.core.sysmem import available_bytes as _sysmem_available_bytes  # noqa: E402

CASE_ID = "LIDC_0089"
SAMPLE_POINTS = 200_000
SEED = 20260916


# ---------------------------------------------------------------- 内存预算
#
# scipy 的 EDT 要一次性分配 `(3, *体素数)` 的 float64 特征变换数组 —— 本项目
# 真实病例上是 882MB。这台机器（32GB）平时够用，但会话里多开几个 python 进程 /
# 病例缓存 / Docker 之后就未必了。
#
# ⚠️ 为什么要**提前**算而不是 `except MemoryError`：
#   捕获异常的话，「机器腾不出内存」与「几何实现漂了」都会走进同一个 except，
#   于是一个偶发的环境抖动被写成 ❌ —— 上一轮就是这样，害得以为发生了回归。
#   反过来把它写成 ✅ 更糟。所以：分配前先算需求，看可用量，不够就明确报 ⚪。

EDT_SAFETY = 1.3      # 安全余量：EDT 内部除特征数组外还有输入掩膜与返回的距离场


def _available_bytes() -> int | None:
    """当前可用物理内存（字节）。取不到返回 None（＝不知道，不拦）。

    读数来自 `agent.core.sysmem` —— 本仓库唯一一份内存读数实现。
    ⚠️ **不要**在这里再抄一份 `ctypes` 结构体：本仓库曾经有三份逐字节相同的
    `MEMORYSTATUSEX`，而字段写错时 `GlobalMemoryStatusEx` 不会报错，
    只会把数填进错位的字段，于是「可用内存」悄悄变成别的量。
    这层包装是为了让测试能替换本函数（monkeypatch `_available_bytes`）。
    """
    return _sysmem_available_bytes()


def _edt_memory_need(size: int) -> int:
    """`ndi.distance_transform_edt` 在一卷 `size` 体素上的峰值内存估算（字节）。

    主开销是 3 个 float64 特征数组，另有返回的距离场（float64）与输入掩膜（bool）。
    """
    return int(EDT_SAFETY * (3 * size * 8 + size * 8 + size))


def _memory_gate(only: str, size: int) -> bool:
    """够不够跑整卷 EDT。True ＝ 可以跑；False ＝ 调用方应返回 None（不足以判定）。"""
    need = _edt_memory_need(size)
    avail = _available_bytes()
    if avail is None:
        print(f"         ⚠️  可用内存取不到（{sys.platform}）—— 不拦，直接尝试")
        return True
    print(
        f"         内存预算：整卷 EDT 峰值 ≈{need / 2**20:,.0f} MB"
        f"（其中特征数组 {3 * size * 8 / 2**20:,.0f} MB），"
        f"当前可用 {avail / 2**20:,.0f} MB"
    )
    if avail >= need:
        return True
    gap = need - avail
    gap_text = f"{gap / 2**20:,.1f} MB" if gap >= 2**20 else f"{gap:,} 字节"
    print(f"         ⚪ 可用内存不足（还差 {gap_text}）—— 本项**不足以判定**")
    print("            这不是几何实现的问题，是这台机器此刻腾不出整卷 EDT 的内存。")
    print("            补齐：关掉占内存的进程（浏览器 / 其它 python / Docker）后单跑：")
    print(f"              python -m agent.scripts.verify_geometry --only {only}")
    return False


# ------------------------------------------------------------------ 比对

def _report(
    label: str,
    gt: np.ndarray,
    got: np.ndarray,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> bool:
    """比较两组距离值，允许浮点末位误差。

    为什么不要求逐位相同：EDT 内部按轴累加平方再开方，
    而 KD 树给出的是 `sqrt(sum((Δ·spacing)²))`，两者的结合顺序不同。
    典型表现是 `sqrt(0.7²)` 得到 0.6999999999999993 而 EDT 直接写 0.7，
    绝对量级 1e-15 —— 比一个原子核半径还小 10 个数量级，物理上无意义。
    真正的验收标准是下游规划数值与 V1 一致，那由 verify_planner 把关。
    """
    gt = np.asarray(gt, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    both_inf = np.isinf(gt) & np.isinf(got)
    diff = np.abs(gt - got)
    diff[both_inf] = 0.0
    worst = float(diff.max()) if diff.size else 0.0
    exact = int(np.count_nonzero(diff == 0.0))
    ok = bool(np.allclose(gt[~both_inf], got[~both_inf], rtol=rtol, atol=atol)) if (
        ~both_inf
    ).any() else True
    flag = "PASS" if ok else "FAIL"
    print(
        f"  {flag}  {label:<44} 最大差异 {worst:.3e}  "
        f"逐位相同 {exact}/{diff.size}"
        + (f"，inf 一致 {int(both_inf.sum())}" if both_inf.any() else "")
    )
    if not ok:
        bad = np.flatnonzero(diff > atol)[:5]
        for index in bad:
            print(f"        索引 {index}: EDT={gt[index]!r}  new={got[index]!r}")
    return ok


# ------------------------------------------------------------------ 合成

def _synthetic_volume() -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    rng = np.random.default_rng(SEED)
    shape = (24, 32, 40)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = ((zz - 12) ** 2 / 64.0 + (yy - 16) ** 2 / 100.0 + (xx - 20) ** 2 / 144.0) <= 1.0
    mask |= (yy > 24) & (xx > 28)          # 一段贴边的结构
    mask |= rng.random(shape) < 0.02       # 零散体素

    target = np.zeros(shape, dtype=bool)
    target[18:22, 8:12, 6:10] = True
    target |= rng.random(shape) < 0.005

    spacing = (1.5, 0.75, 0.7)             # 故意非各向同性
    return mask, target, spacing


def check_synthetic() -> bool:
    print("\n[1] 合成体积穷举（全部体素，非各向同性 spacing）")
    mask, target, spacing = _synthetic_volume()
    coords = np.argwhere(np.ones(mask.shape, dtype=bool))
    if coords.shape[0] != int(np.prod(mask.shape)):
        print("  FAIL  查询点没有覆盖全部体素")
        return False
    print(f"         体积 {mask.shape}，spacing={spacing}，查询点 {coords.shape[0]}")

    ok = True

    gt_radius = ndi.distance_transform_edt(mask, sampling=spacing)
    ok &= _report(
        "半径场：radius_at_points vs EDT(mask)",
        gt_radius[tuple(coords.T)],
        geometry.radius_at_points(mask, coords, spacing),
    )

    gt_distance = ndi.distance_transform_edt(~target, sampling=spacing)
    ok &= _report(
        "靶点距离：distance_to_mask_at vs EDT(~target)",
        gt_distance[tuple(coords.T)],
        geometry.distance_to_mask_at(target, coords, spacing),
    )

    # 退化场景 1：掩膜占满整卷（一个背景体素都没有）
    #
    # 这里**不能**要求与 scipy 等价。实测 scipy 在无零值体素时给出的是伪值：
    # 单位 spacing 下 (0,0,0)=1.0、(0,0,1)=√2、(0,0,2)=√5、(2,3,3)=√27，
    # 等价于「在角上凭空存在一个零值点」——那是内部特征变换数组未初始化
    # 的产物，不是定义好的距离。本项目里气道掩膜不可能占满整卷，
    # 因此这里只要求行为明确（返回 inf）并记录差异。
    full = np.ones(mask.shape, dtype=bool)
    got_full = geometry.radius_at_points(full, coords, spacing)
    if np.isinf(got_full).all():
        print("  PASS  退化：全 True 掩膜返回 inf（无背景可测，明确而非伪值）")
    else:
        print("  FAIL  退化：全 True 掩膜应返回 inf")
        ok = False

    # 退化场景 2：掩膜全 False（没有前景体素）
    empty = np.zeros(mask.shape, dtype=bool)
    gt_empty = ndi.distance_transform_edt(empty, sampling=spacing)
    ok &= _report(
        "退化：全 False 掩膜",
        gt_empty[tuple(coords.T)],
        geometry.radius_at_points(empty, coords, spacing),
    )

    # 退化场景 3：目标掩膜为空 → 必须明确报错而不是给个假数字
    try:
        geometry.distance_to_mask_at(np.zeros(mask.shape, dtype=bool), coords, spacing)
    except RuntimeError:
        print("  PASS  退化：空目标掩膜抛 RuntimeError")
    else:
        print("  FAIL  退化：空目标掩膜没有报错")
        ok = False

    return ok


# ------------------------------------------------------------------ 真实病例

def _case_arrays():
    package = ROOT / "cases" / CASE_ID / "stage4_package"
    airway_img = sitk.ReadImage(str(package / "airway_mask.nii.gz"))
    nodule_img = sitk.ReadImage(str(package / "nodule_selected.nii.gz"))
    airway = sitk.GetArrayFromImage(airway_img).astype(bool)
    nodule = sitk.GetArrayFromImage(nodule_img).astype(bool)
    spacing = tuple(reversed(airway_img.GetSpacing()))
    return airway, nodule, spacing


def _sample_coords(shape, rng) -> np.ndarray:
    return np.column_stack(
        [rng.integers(0, shape[axis], SAMPLE_POINTS) for axis in range(3)]
    ).astype(np.int64)


def check_real_radius() -> bool | None:
    """真实病例半径场对拍。None ＝ 内存不够，不足以判定（不是失败）。"""
    print("\n[2] 真实病例 · 半径场")
    airway, _nodule, spacing = _case_arrays()
    voxels = int(airway.sum())
    print(f"         气道掩膜 {airway.shape}，体素 {voxels:,}，spacing={tuple(round(s, 4) for s in spacing)}")
    shell = geometry.mask_shell(airway)
    shell_px = int(shell.sum())
    edt_bytes = 3 * airway.size * 8          # EDT 内部的特征变换数组
    tree_bytes = shell_px * 3 * 8            # KD 树要持有的坐标数组
    print(
        f"         外表面壳 {shell_px:,} 体素（整卷 {airway.size:,}），"
        f"占整卷 {100.0 * shell_px / airway.size:.4f}%"
    )
    print(
        f"         内存：EDT 特征数组 {edt_bytes / 2**20:,.0f} MB"
        f"  →  KD 树坐标 {tree_bytes / 2**20:,.2f} MB"
        f"（{edt_bytes / max(tree_bytes, 1):,.0f}×）"
    )

    if not _memory_gate("radius", int(airway.size)):
        return None

    rng = np.random.default_rng(SEED)
    coords = _sample_coords(airway.shape, rng)

    started = time.time()
    gt = ndi.distance_transform_edt(airway, sampling=spacing)
    edt_ms = (time.time() - started) * 1000
    gt_values = gt[tuple(coords.T)].astype(np.float64)
    del gt

    started = time.time()
    got = geometry.radius_at_points(airway, coords, spacing)
    new_ms = (time.time() - started) * 1000

    ok = _report("radius_at_points vs EDT(airway_mask)", gt_values, got)
    print(f"         耗时：EDT {edt_ms:,.0f} ms（且需 882MB 特征数组）  →  KD 树 {new_ms:,.0f} ms")
    return ok


def check_real_distance() -> bool | None:
    """真实病例靶点距离对拍。None ＝ 内存不够，不足以判定（不是失败）。"""
    print("\n[3] 真实病例 · 靶点距离")
    _airway, nodule, spacing = _case_arrays()
    print(f"         结节掩膜 {nodule.shape}，体素 {int(nodule.sum()):,}")

    if not _memory_gate("distance", int(nodule.size)):
        return None

    rng = np.random.default_rng(SEED + 1)
    coords = _sample_coords(nodule.shape, rng)

    started = time.time()
    gt = ndi.distance_transform_edt(~nodule, sampling=spacing)
    edt_ms = (time.time() - started) * 1000
    gt_values = gt[tuple(coords.T)].astype(np.float64)
    del gt

    started = time.time()
    got = geometry.distance_to_mask_at(nodule, coords, spacing)
    new_ms = (time.time() - started) * 1000

    ok = _report("distance_to_mask_at vs EDT(~nodule)", gt_values, got)
    print(f"         耗时：EDT {edt_ms:,.0f} ms（且需 882MB 特征数组）  →  KD 树 {new_ms:,.0f} ms")
    return ok


# ------------------------------------------------------------------ 入口

def main() -> int:
    parser = argparse.ArgumentParser(description="geometry 与 scipy EDT 对拍")
    parser.add_argument(
        "--only",
        choices=("all", "synthetic", "radius", "distance"),
        default="all",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("geometry.py 与 scipy distance_transform_edt 等价性对拍")
    print("=" * 78)

    results: list[tuple[str, bool | None]] = []
    if args.only in ("all", "synthetic"):
        results.append(("合成体积穷举", check_synthetic()))
    if args.only in ("all", "radius"):
        results.append(("真实病例 · 半径场", check_real_radius()))
    if args.only in ("all", "distance"):
        results.append(("真实病例 · 靶点距离", check_real_distance()))

    print("\n" + "=" * 78)
    word = {True: "通过", False: "未通过", None: "不足以判定"}
    for label, ok in results:
        print(f"  {label}：{word.get(ok, '未知')}")

    failed = [label for label, ok in results if ok is False]
    undecided = [label for label, ok in results if ok is None]

    if not failed and not undecided:
        print(
            "\n对拍通过：新实现与 scipy EDT 在本项目使用的场景下等价"
            "（差异 ≤ 1e-12 mm，为双精度末位舍入）"
        )
        print("下游规划数值是否与 V1 一致，由 verify_planner 回归把关。")
        return 0
    if not failed:
        print(f"\n⚪ 不足以判定：{len(undecided)} 项没跑成 —— {'、'.join(undecided)}")
        print("   按「缺数据 ≠ 通过」：本次既不能说等价，也不能说不等价。")
        print("   每项的补齐命令已在上面给出。")
        return 2
    print(f"\n对拍未通过：{'、'.join(failed)}")
    if undecided:
        print(f"（另有 {len(undecided)} 项不足以判定：{'、'.join(undecided)}）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
