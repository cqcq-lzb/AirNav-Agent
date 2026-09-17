"""把病例 + 规划结果装配成 viewer 需要的 JSON 负载。

设计要点
--------
1. **单位统一**：所有坐标在进入负载前就转成 Three.js 世界坐标（mm），
   前端不再做任何坐标变换 —— 免得两处各写一遍、哪天改了一处对不上。
2. **按需裁剪**：只放 viewer 真正要画的东西，不放 CT 体数据。
3. **语义着色**：路径不只画一条线，而是画成「气道管腔 + 器械」两根同轴管，
   器械管按余量着色。肉眼就能看出哪一段紧 —— 这比读数字直观得多。

工程约定
--------
本文件只允许用编辑工具（Write / Edit）写入。
用本机的 python.exe 覆写会被 DLP 透明加密，此后 Read / Edit 只能读到密文；
而加密层在 GBK 往返中还会丢掉 ▶ ⏸ − ³ 这类非 GBK 字符。
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np

from ..core.case_loader import CaseContext, NoduleCandidateInfo
from ..core.planner import RoutePlan, classify_reachability
from .mesh import encode_geometry, extract_surface, lps_to_world

# 器械管身的最小绘制半径（mm）。真实器械细到 1 mm，按原尺寸画会看不见。
MIN_DEVICE_DRAW_RADIUS_MM = 1.2

# 余量着色阈值（mm）：>= safe 判为宽松，<= tight 判为贴壁。
CLEARANCE_SAFE_MM = 2.0
CLEARANCE_TIGHT_MM = 0.5

# 表面平滑尺度（mm，物理空间）。
# CT 层厚 2.5 mm 而层内只有 0.82 mm，直接对二值掩膜做 marching cubes
# 会得到明显的梯田状条纹。按物理尺度抹一层很轻的高斯模糊再取等值面，
# 观感提升明显。只影响显示网格，不影响任何规划数值
# （管腔半径来自距离变换，与网格无关）。
AIRWAY_SMOOTHING_MM = 1.35

# 结节绘制半径下限（mm）：低于这个尺寸肉眼看不出是个球。
MIN_NODULE_DRAW_RADIUS_MM = 1.6


def _centerline_edges(adjacency: Any, node_count: int) -> list[list[int]]:
    """把 V1 的邻接表转成去重后的边表。

    注意：V1 的 centerline_points 是**图的节点数组**，不是一条有序折线。
    按数组顺序连成 Line 会把整棵气道树连成乱麻（这是第一个 viewer 版本的
    真实 bug），必须按邻接关系画。
    """
    seen: set[tuple[int, int]] = set()
    edges: list[list[int]] = []
    for source in range(node_count):
        for neighbour in adjacency[source]:
            target = (
                int(neighbour[0])
                if isinstance(neighbour, (tuple, list))
                else int(neighbour)
            )
            if target == source or not (0 <= target < node_count):
                continue
            key = (source, target) if source < target else (target, source)
            if key in seen:
                continue
            seen.add(key)
            edges.append([key[0], key[1]])
    return edges


def _node_radius_mm(voxel_count: int, spacing_zyx: np.ndarray) -> float:
    """按体积换算等效球半径，作为结节的绘制大小。"""
    voxel_volume = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    volume = max(voxel_count, 1) * voxel_volume
    return float((3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0))


def _nodule_payload(
    case: CaseContext,
    candidate: NoduleCandidateInfo,
    selected_id: int | None,
) -> dict[str, Any]:
    position = lps_to_world(np.asarray([candidate.center_xyz_mm], dtype=np.float64))[0]
    equivalent_radius = _node_radius_mm(candidate.voxel_count, case.spacing_zyx)
    return {
        "clientId": candidate.candidate_id,
        "serverId": candidate.server_candidate_id,
        "position": [round(float(v), 3) for v in position],
        "volumeMm3": round(float(candidate.volume_mm3), 2),
        "voxelCount": int(candidate.voxel_count),
        "equivalentRadiusMm": round(equivalent_radius, 3),
        "drawRadiusMm": round(max(equivalent_radius, MIN_NODULE_DRAW_RADIUS_MM), 3),
        "selected": candidate.candidate_id == selected_id,
    }


def _route_payload(plan: RoutePlan, device_diameter_mm: float) -> dict[str, Any]:
    """路径负载：世界坐标折线 + 每点余量，供前端画同轴双管。"""
    world = lps_to_world(plan.points_xyz_mm)
    radii = np.asarray(plan.radii_mm, dtype=np.float64)
    device_radius = device_diameter_mm / 2.0
    clearance = radii - device_radius - plan.device_margin_mm

    return {
        "positions": [[round(float(v), 3) for v in row] for row in world],
        "lumenRadiiMm": [round(float(v), 3) for v in radii],
        "clearanceMm": [round(float(v), 3) for v in clearance],
        "cumulativeMm": [round(float(v), 2) for v in plan.cumulative_mm],
        "turnAnglesDeg": [round(float(v), 2) for v in plan.turn_angles_deg],
        "waypointCount": plan.waypoint_count,
        "deviceDrawRadiusMm": round(max(device_radius, MIN_DEVICE_DRAW_RADIUS_MM), 3),
    }


def build_payload(
    case: CaseContext,
    plan: RoutePlan,
    device_diameter_mm: float,
    warnings: list[str] | None = None,
    mesh_step: int = 1,
    include_repaired: bool = True,
) -> dict[str, Any]:
    """装配 viewer 负载。"""
    image = case.ct_image
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64)

    airway = extract_surface(
        case.airway_mask,
        case.spacing_zyx,
        origin,
        direction,
        step=mesh_step,
        smoothing_mm=AIRWAY_SMOOTHING_MM,
    )

    payload: dict[str, Any] = {
        "meta": {
            "caseId": case.case_id,
            "candidateId": plan.candidate_id,
            "serverCandidateId": next(
                (
                    c.server_candidate_id
                    for c in case.candidates
                    if c.candidate_id == plan.candidate_id
                ),
                None,
            ),
            "profile": plan.profile_name,
            "profileTitle": plan.profile_title,
            "entryMode": plan.entry_mode,
            "deviceDiameterMm": device_diameter_mm,
            "deviceMarginMm": plan.device_margin_mm,
            "topologySignature": plan.topology_signature,
            "generatedAt": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "engine": "AirNav-Agent 无界面规划内核",
            "spacingZyx": [round(float(v), 6) for v in case.spacing_zyx],
            "imageSize": list(image.GetSize()),
            "meshStep": mesh_step,
        },
        "metrics": plan.metrics,
        "warnings": list(warnings or []),
        "reachability": classify_reachability(
            float(plan.metrics.get("target_distance_mm") or 0.0)
        ),
        "airway": encode_geometry(airway.vertices, airway.faces, airway.normals),
        "airwayStats": {
            "vertexCount": airway.vertex_count,
            "triangleCount": airway.triangle_count,
        },
        "route": _route_payload(plan, device_diameter_mm),
        "nodules": [
            _nodule_payload(case, candidate, plan.candidate_id)
            for candidate in case.candidates
        ],
        "clearanceScale": {
            "safeMm": CLEARANCE_SAFE_MM,
            "tightMm": CLEARANCE_TIGHT_MM,
        },
    }

    # 中心线：节点 + 邻接边表（不是有序折线，见 _centerline_edges 的说明）
    node_total = int(case.centerline_points.shape[0])
    world_centerline = lps_to_world(np.asarray(case.centerline_points, dtype=np.float64))
    edges = _centerline_edges(case.adjacency, node_total)
    payload["centerline"] = {
        "positions": [[round(float(v), 2) for v in row] for row in world_centerline],
        "edges": edges,
        "radiiMm": [
            round(float(v), 3)
            for v in np.asarray(case.centerline_radii, dtype=np.float64)
        ],
        "nodeCount": node_total,
        "edgeCount": len(edges),
        "entryNode": int(case.entry_node),
        "junctionCount": int(np.count_nonzero(np.asarray(case.degrees) >= 3)),
    }

    # 入口点
    entry_world = lps_to_world(np.asarray([case.centerline_points[case.entry_node]]))[0]
    payload["entryPoint"] = [round(float(v), 3) for v in entry_world]

    # 最窄处
    bottleneck_index = int(np.argmin(np.asarray(plan.radii_mm, dtype=np.float64)))
    payload["bottleneck"] = {
        "index": bottleneck_index,
        "position": payload["route"]["positions"][bottleneck_index],
        "diameterMm": round(float(plan.radii_mm[bottleneck_index]) * 2.0, 3),
        "clearanceMm": payload["route"]["clearanceMm"][bottleneck_index],
        "cumulativeMm": payload["route"]["cumulativeMm"][bottleneck_index],
    }

    # 最大转角
    turn_index = int(np.argmax(np.asarray(plan.turn_angles_deg, dtype=np.float64)))
    payload["sharpestTurn"] = {
        "index": turn_index,
        "position": payload["route"]["positions"][turn_index],
        "angleDeg": round(float(plan.turn_angles_deg[turn_index]), 2),
    }

    # 分割修复段（如果这条路径依赖了被修出来的气道，必须让人一眼看见）
    repaired: dict[str, Any] = {}
    if include_repaired:
        for key, mask, label in (
            ("recovered", case.airway_recovered_mask, "形态学修复"),
            ("added", case.airway_added_mask, "连通性补桥"),
        ):
            if mask is None or not np.any(mask):
                continue
            # 修复段本身体积小，平滑过头会糊掉，用更轻的一档
            mesh = extract_surface(
                mask,
                case.spacing_zyx,
                origin,
                direction,
                step=mesh_step,
                smoothing_mm=AIRWAY_SMOOTHING_MM * 0.6,
            )
            if mesh.triangle_count == 0:
                continue
            repaired[key] = {
                "label": label,
                "geometry": encode_geometry(mesh.vertices, mesh.faces, mesh.normals),
                "voxelCount": int(np.count_nonzero(mask)),
                "triangleCount": mesh.triangle_count,
            }
    payload["repaired"] = repaired

    # 场景包围盒，前端据此定相机
    route_world = lps_to_world(plan.points_xyz_mm)
    all_points = np.vstack(
        [
            airway.vertices,
            world_centerline,
            route_world,
            np.asarray([n["position"] for n in payload["nodules"]], dtype=np.float64),
        ]
    )
    payload["bounds"] = {
        "min": [round(float(v), 2) for v in all_points.min(axis=0)],
        "max": [round(float(v), 2) for v in all_points.max(axis=0)],
    }
    # 气道自身的包围盒：初始取景和参考网格用它，避免被远处的结节把画面拉散
    payload["airwayBounds"] = {
        "min": [round(float(v), 2) for v in airway.vertices.min(axis=0)],
        "max": [round(float(v), 2) for v in airway.vertices.max(axis=0)],
    }
    return payload


__all__ = ["build_payload"]
