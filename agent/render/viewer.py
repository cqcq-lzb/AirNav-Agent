"""生成自包含的三维 viewer HTML。"""
from __future__ import annotations

import json
import os
import re
import tempfile
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

# 只匹配**字面量**里的时间戳。
# 模板里还有一处 `P.meta.generatedAt`，那是读对象的 JS 引用，不带引号，不会被误伤。
_TS_RE = re.compile(r'("generatedAt"\s*:\s*)"[^"]*"')


@dataclass
class ViewerOutput:
    """一次 viewer 渲染的产物。"""

    path: Path
    sidecar: Path
    payload: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # 目标文件已存在、且内容与本次渲染**除时间戳外完全相同**，于是没有重写。
    reused: bool = False

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
    """viewer 落盘根目录，可用 `AIRNAV_VIEWER_DIR` 改写。

    `outputs/viewers` 里那几份 HTML 是**入了库的演示产物**（.gitignore 明确放行）。
    自检与评测也走 render_viewer，但它们只是要验证链路通不通 —— 参数与演示那几份
    不同（器械外径、mesh_step），于是每跑一次回归就把入库的产物重渲染一遍，
    在工作区留下一处与源码无关的改动。所以自检/评测一律改成写临时目录，
    见 `use_scratch_output()`。
    """
    override = os.environ.get("AIRNAV_VIEWER_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / DEFAULT_OUTPUT_DIR


def viewers_dir() -> Path:
    """产物的落盘根目录 —— **服务端与渲染端必须用同一个来源**。

    `agent/web/server.py` 的 `/artifacts/<name>` 就靠它定位文件。以前那里硬编码了
    `outputs/viewers`，于是当渲染端被 `AIRNAV_VIEWER_DIR` 改到临时目录时
    （门禁与自检就是这么做的，为的是不碰入库的演示产物），
    服务端还在老地方找 —— 结果是「渲染成功但 URL 404」。

    这不是测试的问题，是**两处各自维护了一份真相**。2026-09-20 修：
    服务端改为调用本函数，两边永远一致。
    """
    return _output_root()


def use_scratch_output(tag: str = "selftest") -> Path:
    """把**本进程**的 viewer 输出改到临时目录，并返回该目录。

    自检与评测入口应当在开始渲染之前调用它。这样 `outputs/viewers` 里那份
    已入库的演示产物就只会在**人手跑演示**时被写。
    """
    root = Path(tempfile.mkdtemp(prefix=f"airnav_{tag}_"))
    os.environ["AIRNAV_VIEWER_DIR"] = str(root)
    return root


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


# ------------------------------------------------------------------ 产物命名


def param_tag(
    device_diameter_mm: float | None, profile: str | None, mesh_step: int
) -> str:
    """把**会影响视图内容**的参数压成文件名后缀。

    只编码这三个：器械外径（决定器械管与余量着色）、代价配置（决定走的是哪条路）、
    网格步长（决定气道表面精度）。安全余量这类固定值不入名，免得名字失控。

    公开而不是私有：`agent/scripts/clinical_report.py` 也要按同一口径给报告产物起名
    —— 命名只有一套，否则「这份报告对应的就是那份视图」只能靠人去认。
    """
    parts: list[str] = []
    device = float(device_diameter_mm or 0.0)
    if device > 0:
        parts.append(f"d{device:g}")
    if profile and str(profile) != "balanced":
        parts.append(str(profile))
    if int(mesh_step) != 1:
        parts.append(f"s{int(mesh_step)}")
    return "_" + "_".join(parts) if parts else ""


def _auto_filename(
    case_id: str,
    candidate_id: int,
    device_diameter_mm: float | None,
    profile: str | None,
    mesh_step: int,
    target_dir: Path,
) -> str:
    """`viewer_<病例>_c<候选>[_d<外径>][_<档>][_s<步长>].html`

    为什么参数要进文件名：旧口径 `viewer_LIDC_0089_c3.html` 只编码「病例 + 候选」，
    于是同一个结节用 1.5 mm 和 2.0 mm 各渲染一次会**互相覆盖** ——
    后渲染的顶掉先渲染的，用户点开看到的器械尺寸未必是他问的那个。

    为什么还认旧名字：`outputs/viewers/` 里入库的那两份演示产物就是旧口径。
    统一改名等于在 git 里删两个、再加两个 1.8 MB 的文件（仓库会胖一圈），
    为了命名整齐不值得。所以旧文件只要 sidecar 里记的参数与本次一致就继续用，
    参数对不上才启用带后缀的新名字。
    """
    base = f"viewer_{case_id}_c{candidate_id}"
    tag = param_tag(device_diameter_mm, profile, mesh_step)
    if not tag:
        return f"{base}.html"
    if _legacy_params_match(
        target_dir / f"{base}.json", device_diameter_mm, profile, mesh_step
    ):
        return f"{base}.html"
    return f"{base}{tag}.html"


def _legacy_params_match(
    sidecar: Path,
    device_diameter_mm: float | None,
    profile: str | None,
    mesh_step: int,
) -> bool:
    """旧命名的 sidecar 里记的参数，是否与本次渲染完全一致。"""
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8")).get("meta") or {}
        return (
            abs(
                float(meta.get("deviceDiameterMm", -1.0))
                - float(device_diameter_mm or 0.0)
            )
            < 1e-6
            and str(meta.get("profile") or "balanced") == str(profile or "balanced")
            and int(meta.get("meshStep", 1)) == int(mesh_step)
        )
    except (OSError, ValueError, AttributeError, TypeError):
        return False


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
    name = filename or _auto_filename(
        case.case_id,
        plan.candidate_id,
        device_diameter_mm,
        payload.get("meta", {}).get("profile"),
        mesh_step,
        target_dir,
    )
    path = target_dir / name

    # 两重去重叠加，各自解决一个问题：
    # ① 命名（见 `_auto_filename`）—— 参数进了文件名，不同器械外径不再互相覆盖；
    # ② 内容（下面这几行）—— 同一组参数重渲染时只有 generatedAt 会变，
    #    那就干脆不写盘，免得每跑一次演示就在 `git status` 里留一条「只差一个时间戳」。
    # 代价是 generatedAt 的语义变成「这份视图的**内容**最后一次变化的时刻」，
    # 而不是「最后一次渲染的时刻」（工具层会用 viewer_reused 如实告诉模型）。
    reused = _same_content(html, path)
    if not reused:
        path.write_text(html, encoding="utf-8")

    sidecar = _write_sidecar(payload, path)
    return ViewerOutput(
        path=path, sidecar=sidecar, payload=payload, warnings=notes, reused=reused
    )


def _same_content(new_text: str, old_path: Path) -> bool:
    """旧文件与本次要写的内容是否等价（忽略 generatedAt 时间戳）。"""
    try:
        old_text = old_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return _TS_RE.sub(r"\1", new_text) == _TS_RE.sub(r"\1", old_text)


def _write_sidecar(payload: dict[str, Any], html_path: Path) -> Path:
    """同时落一份纯 JSON，便于脚本或其它前端复用同一批数据。"""
    data_path = html_path.with_suffix(".json")
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if not _same_content(text, data_path):
        data_path.write_text(text, encoding="utf-8")
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
    "param_tag",
    "render_case_viewer",
    "render_from_case_id",
    "use_scratch_output",
    "warnings_for",
]
