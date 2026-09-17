"""生成自包含的三维 viewer HTML。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.case_loader import CaseContext, cases_root
from ..core.planner import RoutePlan, plan_candidate
from .payload import build_payload
from .template import render_html

# 默认输出目录（相对项目根）
DEFAULT_OUTPUT_DIR = "outputs/viewers"


@dataclass
class ViewerOutput:
    """一次 viewer 渲染的产物。"""

    path: Path
    sidecar: Path
    payload: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / 1024 / 1024

    @property
    def airway_triangles(self) -> int:
        return int(self.payload.get("airwayStats", {}).get("triangleCount", 0))

    @property
    def airway_vertices(self) -> int:
        return int(self.payload.get("airwayStats", {}).get("vertexCount", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewer_path": str(self.path),
            "sidecar_path": str(self.sidecar),
            "viewer_size_mb": round(self.size_mb, 2),
            "airway_triangles": self.airway_triangles,
            "airway_vertices": self.airway_vertices,
            "centerline_nodes": self.payload.get("centerline", {}).get("nodeCount"),
            "waypoints": self.payload.get("route", {}).get("waypointCount"),
            "warnings": list(self.warnings),
        }


def _output_root() -> Path:
    return Path(__file__).resolve().parents[2] / DEFAULT_OUTPUT_DIR


def warnings_for(case: CaseContext, plan: RoutePlan, device_diameter_mm: float) -> list[str]:
    """把规划数字翻译成给医生看的注意事项。

    与工具层 agent/tools/imaging.py 的口径保持一致。
    """
    metrics = plan.metrics or {}
    notes: list[str] = []

    minimum = metrics.get("minimum_diameter_mm")
    required = device_diameter_mm + 2 * float(plan.device_margin_mm or 0.0)
    if minimum is not None:
        if minimum < required:
            notes.append(
                f"路径最窄处直径 {minimum:.2f} mm，小于「器械外径 + 双侧安全余量」"
                f"要求的 {required:.2f} mm，当前器械不建议通过这一支。"
            )
        elif minimum < required * 1.35:
            notes.append(
                f"路径最窄处直径 {minimum:.2f} mm，仅比要求值 {required:.2f} mm 略宽，"
                "通过时需留意阻力与贴壁。"
            )

    turn = metrics.get("maximum_turn_angle_deg")
    if turn is not None and turn > 90.0:
        notes.append(
            f"路径最大转角 {turn:.1f}°，超过 90°，实际进镜时可能需要在分叉处反复调整。"
        )

    if metrics.get("bridge_voxels_on_route"):
        notes.append(
            f"路径有 {metrics['bridge_voxels_on_route']} 个体素落在「连通性补桥」区域，"
            "该段气道来自分割后处理而非原始影像，需人工复核。"
        )
    if metrics.get("recovered_voxels_on_route"):
        notes.append(
            f"路径有 {metrics['recovered_voxels_on_route']} 个体素落在「形态学修复」区域，"
            "该段气道为重建结果，需人工复核。"
        )

    distance = metrics.get("target_distance_mm")
    if distance is not None:
        if distance > 30.0:
            notes.append(
                f"靶点距气道中心线 {distance:.2f} mm，明显偏离气道，"
                "提示当前分割对这一支的覆盖可能不足。"
            )
        elif distance > 15.0:
            notes.append(
                f"靶点距气道中心线 {distance:.2f} mm，位于气道邻近区域，"
                "远端细小气道的连通性需要确认。"
            )

    degenerate = metrics.get("degenerate_routes_rejected")
    if degenerate:
        notes.append(
            f"规划过程中拦截了 {degenerate} 条退化路径（起点即终点），"
            "相关靶点已被排除，未参与最优方案比较。"
        )

    notes.append("本视图为工程研究演示结果，不用于临床诊断与治疗决策。")
    return notes


def render_case_viewer(
    case: CaseContext,
    plan: RoutePlan,
    device_diameter_mm: float,
    output_dir: str | Path | None = None,
    mesh_step: int = 1,
    warnings: list[str] | None = None,
    filename: str | None = None,
) -> ViewerOutput:
    """把一个病例的规划结果渲染成单文件 HTML。

    返回 ViewerOutput —— 调用方（尤其是工具层）需要 viewer 的统计信息
    （三角面数、实际使用的告警），从返回值里拿比去重新解析 HTML 干净得多。
    """
    notes = (
        warnings
        if warnings is not None
        else warnings_for(case, plan, device_diameter_mm)
    )
    payload = build_payload(
        case,
        plan,
        device_diameter_mm,
        warnings=notes,
        mesh_step=mesh_step,
    )

    html = render_html(payload)

    target_dir = Path(output_dir) if output_dir else _output_root()
    target_dir.mkdir(parents=True, exist_ok=True)
    name = filename or f"viewer_{case.case_id}_c{plan.candidate_id}.html"
    path = target_dir / name
    path.write_text(html, encoding="utf-8")

    sidecar = _write_sidecar(payload, path)
    return ViewerOutput(path=path, sidecar=sidecar, payload=payload, warnings=notes)


def _write_sidecar(payload: dict[str, Any], html_path: Path) -> Path:
    """同时落一份纯 JSON，便于脚本或其它前端复用同一批数据。"""
    data_path = html_path.with_suffix(".json")
    data_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return data_path


def render_from_case_id(
    case_id: str,
    candidate_id: int,
    device_diameter_mm: float = 0.0,
    profile_name: str = "balanced",
    mesh_step: int = 1,
    output_dir: str | Path | None = None,
) -> ViewerOutput:
    """便捷入口：按病例名 + 候选号直接出图。"""
    from ..core import case_loader

    case_dir = Path(cases_root()) / case_id / "stage4_package"
    if not case_dir.is_dir():
        raise FileNotFoundError(f"找不到病例目录：{case_dir}")

    case = case_loader.load_case(case_dir, case_id=case_id)
    plan, _alternatives = plan_candidate(
        case, candidate_id, profile_name=profile_name, device_diameter_mm=device_diameter_mm
    )
    return render_case_viewer(
        case,
        plan,
        device_diameter_mm,
        output_dir=output_dir,
        mesh_step=mesh_step,
    )


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "ViewerOutput",
    "render_case_viewer",
    "render_from_case_id",
    "warnings_for",
]
