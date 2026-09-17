"""零分配的几何距离计算。

为什么需要这个模块
------------------
scipy 的 `ndi.distance_transform_edt` 在内部**必须**分配一个
`(ndim, *体素数)` 的 float64 特征变换数组，然后才能填出距离场。
在本项目的 CT 上（147 × 512 × 512）那就是：

    3 × 147 × 512 × 512 × 8 B ≈ 882 MB

单次调用还要 4.6 ~ 6.6 秒。本机页面文件固定 8GB，规划路径里接连调用几次
就会直接抛：

    MemoryError: Unable to allocate 882. MiB for an array with shape
    (3, 147, 512, 512) and data type float64

而且这些调用往往**只需要中心线上少数几个节点处的取值**，
为几个点付整卷 882MB 的代价完全不划算。

本模块用 KD 树把这件事降到「只对真正需要的体素建索引」，并且**结果与 EDT
逐位相同**，因此下游的靶点选择、可达性分级、半径场、规划数值全部不变
（由 agent/scripts/verify_planner.py 回归确认）。

两种降维各自的依据
------------------

1. **到掩膜的距离**（`distance_to_mask_at`）
   只需要「若干个查询点到掩膜内最近体素」的距离，
   直接对**掩膜内体素**建 KD 树即可，与 EDT 定义完全一致。

2. **掩膜内的半径场**（`radius_at_points`）
   `EDT(mask)` 在点 p 处的值 = p 到最近**背景**体素的距离。
   直接对背景体素建树是不可行的（背景有 3800 万个体素），
   关键观察是：

       对掩膜内的点 p，离它最近的背景体素 b 必定紧邻某个掩膜体素。

   否则 b 的六个邻居全是背景，其中朝 p 方向的那一个距离更小，与 b 的
   最近性矛盾。所以只需对**掩膜外侧紧邻的一层壳**（`mask_shell`）建 KD 树，
   就能给出精确解 —— 壳的规模取决于气道表面积，比整卷小三个数量级。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

__all__ = ["mask_shell", "radius_at_points", "distance_to_mask_at"]


def _spacing_array(spacing_zyx: Any) -> np.ndarray:
    return np.asarray(spacing_zyx, dtype=np.float64)


def mask_shell(mask: np.ndarray) -> np.ndarray:
    """掩膜外侧紧邻的一层背景体素（6 邻域）。

    对掩膜内的点而言，这一层就足以给出与整卷 EDT 完全一致的最近背景距离，
    详见模块文档的论证。
    """
    data = np.asarray(mask).astype(bool, copy=False)
    return ndi.binary_dilation(data) & ~data


def radius_at_points(
    mask: np.ndarray,
    coords: np.ndarray,
    spacing_zyx: Any,
) -> np.ndarray:
    """掩膜内给定位点处的等效半径（mm），等价于 `EDT(mask)[coords]`。

    Parameters
    ----------
    mask
        三维布尔掩膜（气道）。
    coords
        `(N, 3)` 的 zyx 体素坐标。
    spacing_zyx
        体素物理尺寸，用于把体素距离换算成 mm。

    Notes
    -----
    掩膜**外**的点按 EDT 的定义返回 0（到最近零值体素的距离为零）。
    掩膜占满整卷时没有背景，EDT 的退化行为是 `inf`，此处保持一致。
    """
    data = np.asarray(mask).astype(bool, copy=False)
    points = np.asarray(coords)
    if points.size == 0:
        return np.zeros(0, dtype=np.float64)

    spacing = _spacing_array(spacing_zyx)
    inside = data[tuple(points.astype(np.int64).T)]

    shell = np.argwhere(mask_shell(data))
    values = np.empty(points.shape[0], dtype=np.float64)
    if shell.size == 0:
        values[:] = np.inf
    else:
        tree = cKDTree(shell.astype(np.float64) * spacing)
        values[:] = tree.query(points.astype(np.float64) * spacing)[0]

    values[~inside] = 0.0
    return values


def distance_to_mask_at(
    target_mask: np.ndarray,
    coords: np.ndarray,
    spacing_zyx: Any,
) -> np.ndarray:
    """给定位点到目标掩膜的最近距离（mm），等价于 `EDT(~target_mask)[coords]`。

    Parameters
    ----------
    target_mask
        三维布尔掩膜（结节）。
    coords
        `(N, 3)` 的 zyx 体素坐标。
    spacing_zyx
        体素物理尺寸。
    """
    voxels = np.argwhere(np.asarray(target_mask).astype(bool, copy=False))
    if voxels.size == 0:
        raise RuntimeError("结节掩膜为空，无法计算靶点距离")

    spacing = _spacing_array(spacing_zyx)
    tree = cKDTree(voxels.astype(np.float64) * spacing)
    points = np.asarray(coords, dtype=np.float64) * spacing
    values, _ = tree.query(points)
    return np.asarray(values, dtype=np.float64)
