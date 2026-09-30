"""无 GUI 的病例加载。

严格复刻 interactive_navigation_stage4_v3.InteractiveNavigationWindow.load_case
的前半段（读文件 -> 骨架化 -> 建图 -> 分支拓扑 -> 入口点），
剥掉所有 Qt / VTK 渲染相关的部分。

两点与原版的差异（都是有意为之）：
1. 增加加载缓存，跳过耗时的 skeletonize / distance_transform_edt
2. 结节候选同时暴露两种编号口径：
   - candidate_id        客户端顺序（连通域标签升序），与 V1 GUI 一致
   - server_candidate_id 服务端顺序（stage4_manifest.json，按体积降序）
   原版只有前者，导致「结节 1」在客户端与服务端指代的是不同结节。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from . import cache as _cache
from . import geometry
from .navbridge import choose_entry_node_no_edt, nav

REQUIRED_FILES = ("ct.nii.gz", "airway_mask.nii.gz", "nodule_raw.nii.gz")

# 合成夹具根目录（随仓库入库，体积 ~26KB/例，实测 26730 B）。
# 只在 `<repo>/cases` 缺席时兜底，且来历会被显式标出来 —— 见 cases_root()。
FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "cases"

DEFAULT_MIN_NODULE_VOXELS = 8
DEFAULT_MAX_TARGET_NODES = 6
DEFAULT_TARGET_DISTANCE_SLACK_MM = 12.0


@dataclass
class NoduleCandidateInfo:
    """一个结节候选。"""

    candidate_id: int
    component_label: int
    voxel_count: int
    volume_mm3: float
    center_xyz_mm: list[float]
    bbox_min_zyx: list[int]
    bbox_max_zyx: list[int]
    server_candidate_id: int | None = None

    @property
    def equivalent_diameter_mm(self) -> float:
        return 2.0 * (3.0 * self.volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "server_candidate_id": self.server_candidate_id,
            "component_label": self.component_label,
            "voxel_count": self.voxel_count,
            "volume_mm3": round(self.volume_mm3, 3),
            "equivalent_diameter_mm": round(self.equivalent_diameter_mm, 3),
            "center_xyz_mm": [round(v, 3) for v in self.center_xyz_mm],
        }


@dataclass
class CaseContext:
    """一个病例的全部规划状态（无 GUI 版本）。"""

    case_id: str
    case_dir: Path
    entry_mode: str
    spacing_zyx: np.ndarray
    ct_image: sitk.Image
    airway_mask: np.ndarray
    nodule_raw_mask: np.ndarray
    nodule_labels: np.ndarray
    candidates: list[NoduleCandidateInfo]
    centerline_coords: np.ndarray
    centerline_points: np.ndarray
    centerline_radii: np.ndarray
    adjacency: Any
    degrees: np.ndarray
    junction_labels: Any
    segment_labels: Any
    entry_node: int
    airway_baseline_mask: np.ndarray | None = None
    airway_recovered_mask: np.ndarray | None = None
    airway_added_mask: np.ndarray | None = None
    entry_mask: np.ndarray | None = None
    load_seconds: float = 0.0
    from_cache: bool = False
    available_files: list[str] = field(default_factory=list)

    # ---- 便捷访问 ----

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def centerline_voxels(self) -> int:
        return int(self.centerline_coords.shape[0])

    def candidate(self, candidate_id: int) -> NoduleCandidateInfo:
        for item in self.candidates:
            if item.candidate_id == candidate_id:
                return item
        raise KeyError(
            f"病例 {self.case_id} 没有候选 {candidate_id}"
            f"（可选范围 1..{self.candidate_count}）"
        )

    def candidate_by_server_id(self, server_candidate_id: int) -> NoduleCandidateInfo:
        for item in self.candidates:
            if item.server_candidate_id == server_candidate_id:
                return item
        raise KeyError(
            f"病例 {self.case_id} 没有服务端编号 {server_candidate_id} 的候选"
        )

    def id_mapping(self) -> list[dict[str, Any]]:
        """两种编号口径的对照表 —— 供 Agent 明示，避免歧义。"""
        return [
            {
                "client_candidate_id": item.candidate_id,
                "server_candidate_id": item.server_candidate_id,
                "component_label": item.component_label,
                "voxel_count": item.voxel_count,
            }
            for item in self.candidates
        ]

    def candidate_mask(self, candidate_id: int) -> np.ndarray:
        """与 V1 的 candidate_mask 完全一致。"""
        if not 1 <= candidate_id <= self.candidate_count:
            raise KeyError(
                f"候选编号 {candidate_id} 越界，有效范围 1..{self.candidate_count}"
            )
        return self.nodule_labels == candidate_id

    def merge_state(self) -> np.ndarray:
        """供 A* 使用的气道状态体：主掩膜 + 修复体素 + 桥接体素。"""
        state = self.airway_mask.copy()
        if self.airway_recovered_mask is not None:
            state |= self.airway_recovered_mask
        if self.airway_added_mask is not None:
            state |= self.airway_added_mask
        return state

    def summary(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_dir": str(self.case_dir),
            # ⚠️ 合成夹具必须在返回体里自报家门 —— 否则「这个病例的数字哪来的」
            # 就没法从工具输出本身回答，报告也可能把夹具当成真实病例。
            "fixture": is_fixture(self.case_dir),
            "voxel_size_mm": [round(float(v), 4) for v in self.spacing_zyx],
            "volume_shape_zyx": list(self.airway_mask.shape),
            "airway_voxels": int(self.airway_mask.sum()),
            "centerline_voxels": self.centerline_voxels,
            "entry_mode": self.entry_mode,
            "entry_node": self.entry_node,
            "candidate_count": self.candidate_count,
            "has_baseline_airway": self.airway_baseline_mask is not None,
            "has_recovered_voxels": self.airway_recovered_mask is not None,
            "has_bridge_voxels": self.airway_added_mask is not None,
            "has_entry_point": self.entry_mask is not None,
            "available_files": self.available_files,
            "from_cache": self.from_cache,
            "load_seconds": round(self.load_seconds, 3),
        }


# ---------------------------------------------------------------- 文件解析


def resolve_case_files(case_dir: Path) -> dict[str, Path | None]:
    """与 V1 的 resolve_case_files 一致的病例文件解析。"""
    case_dir = Path(case_dir)
    mapping: dict[str, Path | None] = {
        "ct": case_dir / "ct.nii.gz",
        "airway": case_dir / "airway_mask.nii.gz",
        "airway_baseline": case_dir / "airway_raw_baseline.nii.gz",
        "airway_recovered": case_dir / "airway_recovered_voxels.nii.gz",
        "airway_bridge": case_dir / "airway_bridge_voxels.nii.gz",
        "nodule_raw": case_dir / "nodule_raw.nii.gz",
        "entry": case_dir / "entry_point.nii.gz",
    }
    missing = [k for k in REQUIRED_FILES if not (case_dir / k).is_file()]
    if missing:
        raise FileNotFoundError(
            f"病例目录 {case_dir} 缺少必要文件：{', '.join(missing)}"
        )
    for key in ("airway_baseline", "airway_recovered", "airway_bridge", "entry"):
        if not mapping[key].is_file():
            mapping[key] = None
    return mapping


def _server_numbering(case_dir: Path) -> dict[int, int]:
    """从 stage4_manifest.json 读出 连通域标签 -> 服务端候选编号 的映射。"""
    manifest = Path(case_dir) / "stage4_manifest.json"
    if not manifest.is_file():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[int, int] = {}
    for item in data.get("candidates", []):
        label = item.get("source_component_label")
        cid = item.get("candidate_id")
        if isinstance(label, int) and isinstance(cid, int):
            out[label] = cid
    return out


def _extract_nodule_candidates(
    nodule_raw_mask: np.ndarray,
    ct_image: sitk.Image,
    min_nodule_voxels: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """与 V1 的 extract_nodule_candidates 一致（26 邻域连通域）。"""
    labels, count = ndi.label(
        nodule_raw_mask,
        structure=ndi.generate_binary_structure(3, 3),
    )
    sizes = np.bincount(labels.ravel())
    voxel_volume_mm3 = float(np.prod(ct_image.GetSpacing()))

    accepted_labels = np.zeros_like(labels, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    candidate_id = 0

    for component_label in range(1, count + 1):
        voxel_count = int(sizes[component_label])
        if voxel_count < min_nodule_voxels:
            continue

        component = labels == component_label
        coords = np.argwhere(component)

        candidate_id += 1
        accepted_labels[component] = candidate_id

        center_zyx = coords.mean(axis=0)
        center_xyz = nav.physical_points(center_zyx[None, :], ct_image)[0]

        rows.append(
            {
                "candidate_id": candidate_id,
                "component_label": int(component_label),
                "voxel_count": voxel_count,
                "volume_mm3": voxel_count * voxel_volume_mm3,
                "center_xyz_mm": [float(v) for v in center_xyz],
                "bbox_min_zyx": [int(v) for v in coords.min(axis=0)],
                "bbox_max_zyx": [int(v) for v in coords.max(axis=0)],
            }
        )

    return accepted_labels, rows


# ---------------------------------------------------------------- 主加载


def _build_derived_state(case_dir: Path, min_nodule_voxels: int) -> dict[str, Any]:
    """耗时部分的计算：读掩膜、骨架化、距离变换、建图、拓扑、入口点。"""
    files = resolve_case_files(case_dir)

    ct_image, _ = nav.read_image(files["ct"])
    airway_image, airway_array = nav.read_image(files["airway"])
    nodule_image, nodule_array = nav.read_image(files["nodule_raw"])

    nav.check_geometry(ct_image, airway_image, "airway")
    nav.check_geometry(ct_image, nodule_image, "nodule_raw")

    airway_mask = airway_array > 0
    nodule_raw_mask = nodule_array > 0
    if not airway_mask.any():
        raise RuntimeError(f"{files['airway']} 为空")
    if not nodule_raw_mask.any():
        raise RuntimeError(f"{files['nodule_raw']} 为空")

    spacing_zyx = np.asarray(ct_image.GetSpacing()[::-1], dtype=np.float64)

    def optional_mask(key: str, label: str) -> np.ndarray | None:
        path = files[key]
        if path is None:
            return None
        image, array = nav.read_image(path)
        nav.check_geometry(ct_image, image, label)
        return array > 0

    airway_baseline_mask = optional_mask("airway_baseline", "airway_raw_baseline")
    airway_recovered_mask = optional_mask("airway_recovered", "airway_recovered_voxels")
    airway_added_mask = optional_mask("airway_bridge", "airway_bridge_voxels")
    entry_mask = optional_mask("entry", "entry_point")

    nodule_labels, candidate_rows = _extract_nodule_candidates(
        nodule_raw_mask, ct_image, min_nodule_voxels
    )

    centerline_mask = skeletonize(airway_mask, method="lee").astype(bool)
    centerline_coords = np.argwhere(centerline_mask).astype(np.int32)
    if centerline_coords.shape[0] == 0:
        raise RuntimeError("气道骨架为空，无法规划")

    centerline_points = nav.physical_points(centerline_coords, ct_image)

    # 只取骨架节点处的半径，避免整卷 882MB 的 EDT 特征变换数组
    # （等价性与内存论证见 core/geometry.py）
    centerline_radii = geometry.radius_at_points(
        airway_mask, centerline_coords, spacing_zyx
    )

    adjacency, degrees = nav.build_graph(
        centerline_coords, airway_mask.shape, spacing_zyx
    )
    junction_labels, segment_labels = nav.build_branch_topology(adjacency, degrees)
    entry_node, entry_mode = choose_entry_node_no_edt(
        centerline_coords,
        centerline_points,
        centerline_radii,
        entry_mask,
        spacing_zyx,
    )

    return {
        "spacing_zyx": spacing_zyx,
        "airway_mask": airway_mask,
        "nodule_raw_mask": nodule_raw_mask,
        "nodule_labels": nodule_labels,
        "candidate_rows": candidate_rows,
        "centerline_coords": centerline_coords,
        "centerline_points": centerline_points,
        "centerline_radii": centerline_radii,
        "adjacency": adjacency,
        "degrees": degrees,
        "junction_labels": junction_labels,
        "segment_labels": segment_labels,
        "entry_node": int(entry_node),
        "entry_mode": str(entry_mode),
        "airway_baseline_mask": airway_baseline_mask,
        "airway_recovered_mask": airway_recovered_mask,
        "airway_added_mask": airway_added_mask,
        "entry_mask": entry_mask,
        "available_files": sorted(p.name for p in Path(case_dir).glob("*.nii.gz")),
    }


def load_case(
    case_dir: str | Path,
    case_id: str | None = None,
    min_nodule_voxels: int = DEFAULT_MIN_NODULE_VOXELS,
    use_cache: bool = True,
) -> CaseContext:
    """加载一个 stage4_package 病例目录，返回可规划的 CaseContext。"""
    started = time.time()
    case_dir = Path(case_dir).resolve()
    suffix = f"__min{min_nodule_voxels}"

    state: dict[str, Any] | None = None
    from_cache = False
    if use_cache:
        state = _cache.load(case_dir, suffix)
        from_cache = state is not None

    if state is None:
        state = _build_derived_state(case_dir, min_nodule_voxels)
        if use_cache:
            # ct_image 不进缓存（SimpleITK 对象不保证可序列化），只存派生数据
            _cache.save(case_dir, state, suffix)

    ct_image, _ = nav.read_image(resolve_case_files(case_dir)["ct"])

    server_map = _server_numbering(case_dir)
    candidates = [
        NoduleCandidateInfo(
            candidate_id=row["candidate_id"],
            component_label=row["component_label"],
            voxel_count=row["voxel_count"],
            volume_mm3=row["volume_mm3"],
            center_xyz_mm=row["center_xyz_mm"],
            bbox_min_zyx=row["bbox_min_zyx"],
            bbox_max_zyx=row["bbox_max_zyx"],
            server_candidate_id=server_map.get(row["component_label"]),
        )
        for row in state["candidate_rows"]
    ]

    return CaseContext(
        case_id=case_id or case_dir.parent.name,
        case_dir=case_dir,
        entry_mode=state["entry_mode"],
        spacing_zyx=state["spacing_zyx"],
        ct_image=ct_image,
        airway_mask=state["airway_mask"],
        nodule_raw_mask=state["nodule_raw_mask"],
        nodule_labels=state["nodule_labels"],
        candidates=candidates,
        centerline_coords=state["centerline_coords"],
        centerline_points=state["centerline_points"],
        centerline_radii=state["centerline_radii"],
        adjacency=state["adjacency"],
        degrees=state["degrees"],
        junction_labels=state["junction_labels"],
        segment_labels=state["segment_labels"],
        entry_node=state["entry_node"],
        airway_baseline_mask=state["airway_baseline_mask"],
        airway_recovered_mask=state["airway_recovered_mask"],
        airway_added_mask=state["airway_added_mask"],
        entry_mask=state["entry_mask"],
        load_seconds=time.time() - started,
        from_cache=from_cache,
        available_files=state["available_files"],
    )


# ---------------------------------------------------------------- 病例发现


def cases_root() -> Path:
    """病例根目录。三档来源，且**来历必须可查**（`cases_root_provenance()`）：

    1. `AIRNAV_CASES_DIR` 环境变量 —— 显式指定，优先（给测试/自检隔离用，
       与 `agent/render/viewer.py` 的 `AIRNAV_VIEWER_DIR` 同一套办法）；
    2. `<repo>/cases` —— 真实病例（不入库，见 .gitignore）；
    3. `<repo>/fixtures/cases` —— **合成夹具**，只在真实病例缺席时兜底。

    ⚠️ 第 3 档是**兜底而不是默认**：它让「全新 clone + 无真实数据」的机器也能跑
    依赖病例的自检（见 `docs/合成夹具.md`）。但绝不能静默发生 ——
    调到夹具时 `list_cases` / `inspect_case` 的返回体里会带上
    `fixture: true` 与来历说明，报告里也会写出来。缺数据 ≠ 可以假装有数据。
    """
    override = os.environ.get("AIRNAV_CASES_DIR")
    if override:
        return Path(override)
    from .navbridge import NAV_ROOT

    real = NAV_ROOT / "cases"
    if _has_cases(real):
        return real
    return FIXTURE_ROOT


def _has_cases(root: Path) -> bool:
    """目录存在且真的装了至少一个病例（空目录不算）。"""
    if not root.is_dir():
        return False
    return any(_looks_like_case(entry) for entry in root.iterdir())


def _looks_like_case(entry: Path) -> bool:
    if not entry.is_dir():
        return False
    for candidate in (entry, entry / "stage4_package"):
        if all((candidate / name).is_file() for name in REQUIRED_FILES):
            return True
    return False


def cases_root_provenance(root: str | Path | None = None) -> dict[str, Any]:
    """这份病例数据是哪来的 —— 给报告与工具返回体用。"""
    override = os.environ.get("AIRNAV_CASES_DIR")
    resolved = Path(root) if root else cases_root()
    try:
        under_fixture = resolved.resolve() == FIXTURE_ROOT.resolve()
    except OSError:
        under_fixture = False

    if override:
        kind = "env"
    elif under_fixture:
        kind = "fixture"
    else:
        kind = "real"
    return {
        "cases_root": str(resolved),
        "kind": kind,
        "fixture": kind == "fixture" or under_fixture,
        "note": {
            "real": "真实病例目录",
            "fixture": "**合成夹具**：几何由 agent/scripts/make_fixture_case.py 生成，"
                       "不含任何患者信息；数字不可用于临床或对外结论",
            "env": "由 AIRNAV_CASES_DIR 指定",
        }[kind],
    }


def is_fixture(case_dir: str | Path | None = None) -> bool:
    """这个病例目录是不是合成夹具。"""
    if case_dir is None:
        return cases_root_provenance()["fixture"]
    try:
        target = Path(case_dir).resolve()
    except OSError:
        return False
    try:
        target.relative_to(FIXTURE_ROOT.resolve())
        return True
    except ValueError:
        return False


def list_local_cases(root: str | Path | None = None) -> list[dict[str, Any]]:
    """枚举本地已下载的 stage4_package 病例目录。"""
    root = Path(root) if root else cases_root()
    if not root.is_dir():
        return []

    found: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        pkg = entry if (entry / "ct.nii.gz").is_file() else entry / "stage4_package"
        if not (pkg / "ct.nii.gz").is_file():
            continue

        info: dict[str, Any] = {
            "case_id": entry.name,
            "package_dir": str(pkg),
            "required_ok": all((pkg / n).is_file() for n in REQUIRED_FILES),
            "cached": _cache.cache_path(
                pkg, f"__min{DEFAULT_MIN_NODULE_VOXELS}"
            ).is_file(),
        }
        manifest = pkg / "stage4_manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                info["server_candidate_count"] = data.get("candidate_count")
                if data.get("warnings"):
                    info["warnings"] = data["warnings"]
            except Exception:
                pass
        found.append(info)
    return found
