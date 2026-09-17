"""复用 Airway 导航 V1 规划内核的导入桥。

V1 的 interactive_navigation_stage4_v3.py 里有完整的
气道图构建 / 分支拓扑 / 多代价 A* 规划实现，但它是一个 GUI 文件
（模块级只 import Qt/VTK，没有副作用），因此可以在无界面环境下直接 import。

本模块做两件事：

1. 把 V1 根目录放进 sys.path 并导出该模块（`nav`）。
2. 对**唯一一处会拖垮内存的 V1 调用**做零分配覆盖
   （`choose_entry_node_no_edt`，原因见该函数文档）。

Agent 侧调用的 V1 函数共 13 个，其中只有一个落在「内部使用整卷 EDT」的
函数集合里（V1 中含 EDT 的函数为 choose_entry_node / load_case /
choose_target_nodes / prepare_baseline_planning_data /
compare_with_baseline_route，后四者 Agent 都不调用）。
所以覆盖这一处即可让整条链路摆脱 882MB 的分配。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# agent/core/navbridge.py -> parents[2] 即 AirNav-Agent 根目录
NAV_ROOT = Path(__file__).resolve().parents[2]

if str(NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(NAV_ROOT))

import interactive_navigation_stage4_v3 as nav  # noqa: E402

from . import geometry  # noqa: E402

__all__ = ["nav", "NAV_ROOT", "choose_entry_node_no_edt"]


def choose_entry_node_no_edt(
    coords_zyx: np.ndarray,
    points_xyz: np.ndarray,
    radii_mm: np.ndarray,
    entry_mask: np.ndarray | None,
    spacing_zyx: np.ndarray,
) -> tuple[int, str]:
    """与 `nav.choose_entry_node` 行为一致，但入口掩膜分支不做整卷 EDT。

    V1 在这个分支里调
    `ndi.distance_transform_edt(~entry_mask, sampling=spacing)`，
    在 147×512×512 的 CT 上要依次分配
    `(3, 147, 512, 512)` 的 int32 特征数组（441MB）与 float64 距离场（882MB）。
    而它随后只取**中心线节点处**的取值并 `argmin`，所以真正需要的只是
    「各中心线节点到入口掩膜内最近体素的距离」——
    这与 `geometry.distance_to_mask_at` 完全等价（已对拍：
    真实病例 20 万随机点逐位相同，见 agent/scripts/verify_geometry.py）。

    没有入口掩膜时仍原样交给 V1 实现，避免在自动入口点分支上引入偏差。
    """
    if entry_mask is not None and entry_mask.any():
        values = geometry.distance_to_mask_at(entry_mask, coords_zyx, spacing_zyx)
        return int(np.argmin(values)), "固定入口点"
    return nav.choose_entry_node(
        coords_zyx, points_xyz, radii_mm, entry_mask, spacing_zyx
    )
