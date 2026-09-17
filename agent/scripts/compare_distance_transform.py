"""对照验证：全卷 EDT 与"结节体素 KD 树"是否给出同一个距离。

背景
----
`choose_target_nodes` 里有一句
`ndi.distance_transform_edt(~target_mask, sampling=spacing)`，
它在语义上是「每个体素到最近结节体素的距离」，输出形状是整卷。

问题是 scipy 的 EDT 内部必须分配一个 `(ndim, *体素数)` 的 float64
特征变换数组：147×512×512 的 CT 上 = 3×147×512×512×8 B ≈ **882 MB**。
本机页面文件固定 8GB 时，这一步会直接 MemoryError（真实发生过）。

替代方案：只在掩膜内的体素上建 KD 树，查询中心线节点。
语义等价、结果精确，内存从 O(全卷) 降到 O(掩膜体素数)。

本脚本对全部候选逐个比较两种算法的距离数组，确认差异在浮点误差内。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.core import case_loader as cl  # noqa: E402

CASE_DIR = Path(cl.cases_root()) / "LIDC_0089" / "stage4_package"


def edt_values(case, target_mask: np.ndarray) -> np.ndarray:
    """现有实现：全卷 EDT 后在中心线节点上取值。"""
    distance = ndi.distance_transform_edt(~target_mask, sampling=case.spacing_zyx)
    return distance[tuple(case.centerline_coords.T)].astype(np.float64)


def tree_values(case, target_mask: np.ndarray) -> np.ndarray:
    """候选实现：掩膜体素建 KD 树后查询中心线节点。"""
    coords = np.argwhere(target_mask)
    if coords.size == 0:
        raise RuntimeError("掩膜为空")
    scaled = coords.astype(np.float64) * case.spacing_zyx
    tree = cKDTree(scaled)
    points = case.centerline_coords.astype(np.float64) * case.spacing_zyx
    values, _ = tree.query(points)
    return np.asarray(values, dtype=np.float64)


def main() -> int:
    case = cl.load_case(CASE_DIR)
    print(f"病例 {case.case_id}  体积 {case.airway_mask.shape}  "
          f"体素 mm {tuple(round(float(v), 4) for v in case.spacing_zyx)}")
    print(f"中心线节点 {case.centerline_voxels}")
    print()

    worst = 0.0
    for info in case.candidates:
        mask = case.candidate_mask(info.candidate_id)
        voxels = int(np.count_nonzero(mask))

        started = time.time()
        a = edt_values(case, mask)
        t_edt = time.time() - started

        started = time.time()
        b = tree_values(case, mask)
        t_tree = time.time() - started

        diff = float(np.max(np.abs(a - b)))
        worst = max(worst, diff)

        edt_bytes = 3 * int(np.prod(case.airway_mask.shape)) * 8
        print(
            f"候选 {info.candidate_id}（{voxels} 体素）"
            f"  EDT {t_edt:5.2f}s   KD树 {t_tree:5.3f}s"
            f"  最大差异 {diff:.3e} mm"
            f"  最近距离 EDT {a.min():.4f} / KD {b.min():.4f}"
        )
        print(
            f"          EDT 峰值工作数组 {edt_bytes / 1024**2:.0f} MB"
            f"  vs  KD 树 {voxels * 3 * 8 / 1024:.1f} KB"
        )

    print()
    if worst < 1e-6:
        print(f"结论：两种算法等价（最大差异 {worst:.3e} mm < 1e-6），可安全替换")
        return 0
    print(f"结论：差异过大（{worst:.3e} mm），不能替换")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
