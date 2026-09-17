#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
第四阶段：交互式经支气管肺结节导航 v3
================================

实现功能
--------
1. 显示全部结节候选；
2. 在结节表格、二维切片或三维视图中点击选择目标；
3. 不重新分割，直接利用既有气道和结节分割重新规划；
4. 同步二维轴位、三维全局和虚拟支气管镜视图；
5. 显示支气管分叉序列；
6. 沿路径实时显示局部气道直径、器械余量、累计距离和局部转角；
7. 支持器械外径和安全余量约束；
8. 默认输出一条综合最优路径。

病例目录至少需要
----------------
ct.nii.gz
airway_mask.nii.gz
nodule_raw.nii.gz
entry_point.nii.gz（可选，但建议提供）

运行
----
python interactive_navigation_stage4.py

或：
python interactive_navigation_stage4.py --case-dir D:\...\stage4_package

依赖
----
PySide6
vtk
numpy
scipy
scikit-image
SimpleITK

说明
----
本程序为工程研究演示，不用于临床诊断和治疗。
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import vtk
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import skeletonize
from vtkmodules.qt.QVTKRenderWindowInteractor import (
    QVTKRenderWindowInteractor,
)


WINDOW_TITLE = "经支气管肺结节导航：第一版腔内三维箭头 v3.6"

NODULE_COLORS = [
    (1.00, 0.35, 0.10),
    (0.95, 0.75, 0.10),
    (0.65, 0.30, 1.00),
    (0.10, 0.85, 0.95),
    (0.95, 0.25, 0.70),
    (0.40, 0.95, 0.30),
    (0.95, 0.55, 0.15),
    (0.30, 0.60, 1.00),
]

SELECTED_NODULE_COLOR = (1.00, 0.90, 0.05)
AIRWAY_COLOR = (0.18, 0.62, 0.92)
AIRWAY_RECOVERED_COLOR = (0.05, 0.95, 0.72)
AIRWAY_ADDED_COLOR = (1.00, 0.72, 0.05)
ROUTE_COLOR = (0.10, 1.00, 0.42)
CURRENT_POINT_COLOR = (0.10, 0.95, 1.00)
BOTTLENECK_COLOR = (1.00, 0.12, 0.08)


@dataclass
class NoduleCandidate:
    candidate_id: int
    component_label: int
    voxel_count: int
    volume_mm3: float
    center_zyx: np.ndarray
    center_xyz_mm: np.ndarray
    minimum_zyx: np.ndarray
    maximum_zyx: np.ndarray


@dataclass(frozen=True)
class CostProfile:
    name: str
    title: str
    weight_length: float
    weight_radius: float
    weight_curvature: float
    weight_branch: float


@dataclass
class RouteResult:
    nodes: list[int]
    coords_zyx: np.ndarray
    points_xyz_mm: np.ndarray
    radii_mm: np.ndarray
    diameters_mm: np.ndarray
    cumulative_mm: np.ndarray
    smooth_turn_angles_deg: np.ndarray
    topology_tokens: tuple[str, ...]
    topology_signature: str
    target_distance_mm: float
    target_candidate_id: int
    profile_name: str
    profile_title: str
    score: float
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第四阶段交互式导航界面"
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="本地 stage4_package 病例目录",
    )
    parser.add_argument(
        "--min-nodule-voxels",
        type=int,
        default=8,
        help="结节候选最小连通组件体素数",
    )
    parser.add_argument(
        "--max-target-nodes",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--target-distance-slack-mm",
        type=float,
        default=12.0,
    )
    return parser.parse_args()


def read_image(path: Path) -> tuple[sitk.Image, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    if array.ndim != 3:
        raise ValueError(f"只支持三维图像：{path}")

    return image, array


def check_geometry(
    reference: sitk.Image,
    other: sitk.Image,
    name: str,
) -> None:
    if reference.GetSize() != other.GetSize():
        raise ValueError(f"{name} 尺寸与 CT 不一致")

    if not np.allclose(
        reference.GetSpacing(),
        other.GetSpacing(),
        atol=1e-5,
    ):
        raise ValueError(f"{name} spacing 与 CT 不一致")

    if not np.allclose(
        reference.GetOrigin(),
        other.GetOrigin(),
        atol=1e-4,
    ):
        raise ValueError(f"{name} origin 与 CT 不一致")

    if not np.allclose(
        reference.GetDirection(),
        other.GetDirection(),
        atol=1e-5,
    ):
        raise ValueError(f"{name} direction 与 CT 不一致")


def physical_points(
    coords_zyx: np.ndarray,
    image: sitk.Image,
) -> np.ndarray:
    coords_zyx = np.asarray(
        coords_zyx,
        dtype=np.float64,
    )

    spacing_xyz = np.asarray(
        image.GetSpacing(),
        dtype=np.float64,
    )
    origin_xyz = np.asarray(
        image.GetOrigin(),
        dtype=np.float64,
    )
    direction = np.asarray(
        image.GetDirection(),
        dtype=np.float64,
    ).reshape(3, 3)

    indices_xyz = coords_zyx[:, ::-1]

    return (
        origin_xyz
        + (
            direction
            @ (indices_xyz * spacing_xyz).T
        ).T
    )


def numpy_faces_to_vtk(
    points_xyz: np.ndarray,
    faces: np.ndarray,
) -> vtk.vtkPolyData:
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToFloat()
    vtk_points.SetNumberOfPoints(len(points_xyz))

    for index, point in enumerate(points_xyz):
        vtk_points.SetPoint(
            index,
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )

    cells = vtk.vtkCellArray()

    for face in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, int(face[0]))
        triangle.GetPointIds().SetId(1, int(face[1]))
        triangle.GetPointIds().SetId(2, int(face[2]))
        cells.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(cells)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(cleaner.GetOutputPort())
    normals.AutoOrientNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()

    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output


def mask_to_polydata(
    mask_zyx: np.ndarray,
    reference: sitk.Image,
    *,
    smooth_iterations: int,
    decimate_reduction: float,
) -> vtk.vtkPolyData:
    if not mask_zyx.any():
        raise RuntimeError("三维重建掩膜为空")

    spacing_xyz = np.asarray(
        reference.GetSpacing(),
        dtype=np.float64,
    )
    spacing_zyx = spacing_xyz[::-1]

    vertices_zyx_mm, faces, _normals, _values = (
        measure.marching_cubes(
            mask_zyx.astype(np.uint8),
            level=0.5,
            spacing=tuple(spacing_zyx),
            allow_degenerate=False,
        )
    )

    local_xyz_mm = vertices_zyx_mm[:, ::-1]

    origin_xyz = np.asarray(
        reference.GetOrigin(),
        dtype=np.float64,
    )
    direction = np.asarray(
        reference.GetDirection(),
        dtype=np.float64,
    ).reshape(3, 3)

    points_xyz = (
        origin_xyz
        + (
            direction
            @ local_xyz_mm.T
        ).T
    )

    polydata = numpy_faces_to_vtk(
        points_xyz,
        faces.astype(np.int64),
    )

    current_port: vtk.vtkAlgorithmOutput | None = None
    current_data: vtk.vtkPolyData | None = polydata

    if decimate_reduction > 0:
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(current_data)
        decimate.SetTargetReduction(
            float(
                np.clip(
                    decimate_reduction,
                    0.0,
                    0.95,
                )
            )
        )
        decimate.PreserveTopologyOn()
        decimate.BoundaryVertexDeletionOff()
        decimate.Update()

        current_port = decimate.GetOutputPort()
        current_data = None

    if smooth_iterations > 0:
        smoother = vtk.vtkWindowedSincPolyDataFilter()

        if current_port is not None:
            smoother.SetInputConnection(current_port)
        else:
            assert current_data is not None
            smoother.SetInputData(current_data)

        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.SetPassBand(0.08)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        output = vtk.vtkPolyData()
        output.DeepCopy(smoother.GetOutput())
        return output

    if current_port is not None:
        producer = current_port.GetProducer()
        producer.Update()
        output = vtk.vtkPolyData()
        output.DeepCopy(producer.GetOutputDataObject(0))
        return output

    assert current_data is not None
    return current_data


def make_surface_actor(
    polydata: vtk.vtkPolyData,
    color: tuple[float, float, float],
    opacity: float,
) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    prop.SetInterpolationToPhong()
    prop.SetAmbient(0.18)
    prop.SetDiffuse(0.78)
    prop.SetSpecular(0.28)
    prop.SetSpecularPower(25.0)

    return actor


def make_route_actor(
    points_xyz: np.ndarray,
    color: tuple[float, float, float],
    radius_mm: float,
) -> tuple[vtk.vtkActor, vtk.vtkPolyData]:
    vtk_points = vtk.vtkPoints()

    for point in points_xyz:
        vtk_points.InsertNextPoint(
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )

    lines = vtk.vtkCellArray()
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(
        len(points_xyz)
    )

    for index in range(len(points_xyz)):
        polyline.GetPointIds().SetId(index, index)

    lines.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(polydata)
    tube.SetRadius(radius_mm)
    tube.SetNumberOfSides(18)
    tube.CappingOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(1.0)
    prop.SetInterpolationToPhong()
    prop.SetAmbient(0.45)
    prop.SetDiffuse(0.72)
    prop.SetSpecular(0.55)
    prop.SetSpecularPower(35.0)

    return actor, polydata


def make_sphere_actor(
    center_xyz: np.ndarray,
    radius_mm: float,
    color: tuple[float, float, float],
) -> tuple[vtk.vtkActor, vtk.vtkSphereSource]:
    source = vtk.vtkSphereSource()
    source.SetCenter(
        float(center_xyz[0]),
        float(center_xyz[1]),
        float(center_xyz[2]),
    )
    source.SetRadius(radius_mm)
    source.SetThetaResolution(24)
    source.SetPhiResolution(24)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetInterpolationToPhong()

    return actor, source



def make_arrow_actor(
    color: tuple[float, float, float],
) -> tuple[vtk.vtkActor, vtk.vtkArrowSource]:
    """
    创建局部 +X 方向的长箭头。实际位置、方向和长度由
    set_arrow_pose 动态设置。
    """
    source = vtk.vtkArrowSource()
    source.SetTipLength(0.24)
    source.SetTipRadius(0.14)
    source.SetTipResolution(28)
    source.SetShaftRadius(0.045)
    source.SetShaftResolution(28)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(1.0)
    prop.SetInterpolationToPhong()
    prop.SetAmbient(0.62)
    prop.SetDiffuse(0.70)
    prop.SetSpecular(0.75)
    prop.SetSpecularPower(45.0)

    return actor, source


def set_arrow_pose(
    actor: vtk.vtkActor,
    start_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    length_mm: float,
    thickness_scale: float,
) -> None:
    """
    将 vtkArrowSource 的局部 +X 轴旋转到 direction_xyz，
    并把箭头起点放在 start_xyz。
    """
    direction = np.asarray(
        direction_xyz,
        dtype=np.float64,
    )

    norm = float(np.linalg.norm(direction))

    if norm <= 1e-8:
        direction = np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )
    else:
        direction = direction / norm

    reference_up = np.asarray(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    if abs(float(np.dot(direction, reference_up))) > 0.92:
        reference_up = np.asarray(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )

    local_y = np.cross(
        reference_up,
        direction,
    )

    local_y_norm = float(
        np.linalg.norm(local_y)
    )

    if local_y_norm <= 1e-8:
        local_y = np.asarray(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )
    else:
        local_y = local_y / local_y_norm

    local_z = np.cross(
        direction,
        local_y,
    )

    local_z_norm = float(
        np.linalg.norm(local_z)
    )

    if local_z_norm > 1e-8:
        local_z = local_z / local_z_norm

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()

    # 直接把长度和粗细写入基向量，避免 UserMatrix 与 Actor.Scale
    # 的组合顺序在不同 VTK 版本中产生差异。
    for row in range(3):
        matrix.SetElement(
            row,
            0,
            float(direction[row] * length_mm),
        )
        matrix.SetElement(
            row,
            1,
            float(local_y[row] * thickness_scale),
        )
        matrix.SetElement(
            row,
            2,
            float(local_z[row] * thickness_scale),
        )
        matrix.SetElement(
            row,
            3,
            float(start_xyz[row]),
        )

    actor.SetScale(1.0, 1.0, 1.0)
    actor.SetUserMatrix(matrix)


def build_graph(
    coords_zyx: np.ndarray,
    shape: tuple[int, int, int],
    spacing_zyx: np.ndarray,
) -> tuple[
    list[list[tuple[int, float]]],
    np.ndarray,
]:
    flat = np.ravel_multi_index(
        coords_zyx.T,
        shape,
    )

    lookup = {
        int(value): int(index)
        for index, value in enumerate(flat)
    }

    offsets: list[
        tuple[np.ndarray, float]
    ] = []

    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue

                offset = np.asarray(
                    [dz, dy, dx],
                    dtype=np.int32,
                )

                step_mm = float(
                    np.linalg.norm(
                        offset.astype(np.float64)
                        * spacing_zyx
                    )
                )

                offsets.append((offset, step_mm))

    adjacency: list[
        list[tuple[int, float]]
    ] = [
        []
        for _ in range(len(coords_zyx))
    ]

    shape_array = np.asarray(
        shape,
        dtype=np.int32,
    )

    for node, coord in enumerate(coords_zyx):
        for offset, step_mm in offsets:
            neighbor = coord + offset

            if (
                np.any(neighbor < 0)
                or np.any(neighbor >= shape_array)
            ):
                continue

            neighbor_flat = int(
                np.ravel_multi_index(
                    tuple(neighbor),
                    shape,
                )
            )

            neighbor_node = lookup.get(
                neighbor_flat
            )

            if neighbor_node is not None:
                adjacency[node].append(
                    (neighbor_node, step_mm)
                )

    degrees = np.asarray(
        [
            len(neighbors)
            for neighbors in adjacency
        ],
        dtype=np.int32,
    )

    return adjacency, degrees


def label_graph_components(
    adjacency: list[list[tuple[int, float]]],
    allowed: np.ndarray,
) -> np.ndarray:
    labels = np.zeros(
        len(adjacency),
        dtype=np.int32,
    )

    component_id = 0

    for start in range(len(adjacency)):
        if not allowed[start] or labels[start] != 0:
            continue

        component_id += 1
        labels[start] = component_id
        stack = [start]

        while stack:
            node = stack.pop()

            for neighbor, _step in adjacency[node]:
                if (
                    allowed[neighbor]
                    and labels[neighbor] == 0
                ):
                    labels[neighbor] = component_id
                    stack.append(neighbor)

    return labels


def build_branch_topology(
    adjacency: list[list[tuple[int, float]]],
    degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    junction_mask = degrees >= 3

    junction_labels = label_graph_components(
        adjacency,
        junction_mask,
    )

    segment_labels = label_graph_components(
        adjacency,
        ~junction_mask,
    )

    return junction_labels, segment_labels


def route_topology_signature(
    nodes: list[int],
    junction_labels: np.ndarray,
    segment_labels: np.ndarray,
) -> tuple[tuple[str, ...], str]:
    raw_tokens: list[str] = []

    for node in nodes:
        junction_id = int(
            junction_labels[node]
        )
        segment_id = int(
            segment_labels[node]
        )

        if junction_id > 0:
            raw_tokens.append(f"J{junction_id}")
        elif segment_id > 0:
            raw_tokens.append(f"S{segment_id}")
        else:
            raw_tokens.append(f"N{node}")

    compressed: list[str] = []

    for token in raw_tokens:
        if (
            not compressed
            or compressed[-1] != token
        ):
            compressed.append(token)

    tokens = tuple(compressed)
    return tokens, " > ".join(tokens)


def choose_entry_node(
    coords_zyx: np.ndarray,
    points_xyz: np.ndarray,
    radii_mm: np.ndarray,
    entry_mask: np.ndarray | None,
    spacing_zyx: np.ndarray,
) -> tuple[int, str]:
    if entry_mask is not None and entry_mask.any():
        distance = ndi.distance_transform_edt(
            ~entry_mask,
            sampling=spacing_zyx,
        )

        values = distance[
            tuple(coords_zyx.T)
        ]

        return (
            int(np.argmin(values)),
            "固定入口点",
        )

    z = points_xyz[:, 2]
    threshold = float(
        np.quantile(z, 0.88)
    )
    candidates = np.flatnonzero(
        z >= threshold
    )

    if len(candidates) == 0:
        candidates = np.arange(
            len(coords_zyx)
        )

    local_radii = radii_mm[candidates]
    local_z = z[candidates]

    radius_score = (
        local_radii - local_radii.min()
    ) / max(float(np.ptp(local_radii)), 1e-6)

    height_score = (
        local_z - local_z.min()
    ) / max(float(np.ptp(local_z)), 1e-6)

    score = (
        0.85 * radius_score
        + 0.15 * height_score
    )

    return (
        int(candidates[int(np.argmax(score))]),
        "自动入口点",
    )


def transition_angle_rad(
    previous_coord: np.ndarray | None,
    current_coord: np.ndarray,
    next_coord: np.ndarray,
    spacing_zyx: np.ndarray,
) -> float:
    if previous_coord is None:
        return 0.0

    incoming = (
        current_coord.astype(np.float64)
        - previous_coord.astype(np.float64)
    ) * spacing_zyx

    outgoing = (
        next_coord.astype(np.float64)
        - current_coord.astype(np.float64)
    ) * spacing_zyx

    denominator = float(
        np.linalg.norm(incoming)
        * np.linalg.norm(outgoing)
    )

    if denominator <= 1e-8:
        return 0.0

    cosine = float(
        np.clip(
            np.dot(incoming, outgoing)
            / denominator,
            -1.0,
            1.0,
        )
    )

    return float(math.acos(cosine))


def edge_cost(
    previous_node: int,
    current_node: int,
    next_node: int,
    step_mm: float,
    coords_zyx: np.ndarray,
    radii_mm: np.ndarray,
    degrees: np.ndarray,
    spacing_zyx: np.ndarray,
    profile: CostProfile,
    device_diameter_mm: float,
    device_margin_mm: float,
) -> float | None:
    local_radius = float(
        min(
            radii_mm[current_node],
            radii_mm[next_node],
        )
    )

    if device_diameter_mm > 0:
        required_radius = (
            device_diameter_mm / 2.0
            + device_margin_mm
        )

        if local_radius < required_radius:
            return None

    length_term = step_mm

    radius_term = (
        step_mm
        * (
            2.0
            / max(local_radius, 0.25)
        ) ** 2.0
    )

    previous_coord = (
        None
        if previous_node < 0
        else coords_zyx[previous_node]
    )

    angle = transition_angle_rad(
        previous_coord,
        coords_zyx[current_node],
        coords_zyx[next_node],
        spacing_zyx,
    )

    free_angle = math.radians(15.0)
    excess = max(0.0, angle - free_angle)

    curvature_term = (
        step_mm
        * (excess / math.pi) ** 2
    )

    branch_term = 0.0

    if degrees[current_node] >= 3:
        branch_term = (
            step_mm
            * (angle / math.pi) ** 2
        )

    return float(
        profile.weight_length * length_term
        + profile.weight_radius * radius_term
        + profile.weight_curvature * curvature_term
        + profile.weight_branch * branch_term
    )


def stateful_astar(
    start_node: int,
    target_node: int,
    adjacency: list[list[tuple[int, float]]],
    coords_zyx: np.ndarray,
    points_xyz: np.ndarray,
    radii_mm: np.ndarray,
    degrees: np.ndarray,
    spacing_zyx: np.ndarray,
    profile: CostProfile,
    device_diameter_mm: float,
    device_margin_mm: float,
) -> tuple[list[int], float]:
    start_state = (-1, start_node)

    distance: dict[
        tuple[int, int],
        float,
    ] = {
        start_state: 0.0
    }

    predecessor: dict[
        tuple[int, int],
        tuple[int, int] | None,
    ] = {
        start_state: None
    }

    target_point = points_xyz[target_node]

    def heuristic(node: int) -> float:
        return (
            profile.weight_length
            * float(
                np.linalg.norm(
                    points_xyz[node]
                    - target_point
                )
            )
        )

    heap: list[
        tuple[float, float, int, int]
    ] = [
        (
            heuristic(start_node),
            0.0,
            -1,
            start_node,
        )
    ]

    final_state: tuple[int, int] | None = None

    while heap:
        (
            _estimated,
            current_cost,
            previous_node,
            current_node,
        ) = heapq.heappop(heap)

        state = (
            previous_node,
            current_node,
        )

        if current_cost != distance.get(
            state,
            math.inf,
        ):
            continue

        if current_node == target_node:
            final_state = state
            break

        for next_node, step_mm in adjacency[
            current_node
        ]:
            if next_node == previous_node:
                continue

            transition = edge_cost(
                previous_node=previous_node,
                current_node=current_node,
                next_node=next_node,
                step_mm=step_mm,
                coords_zyx=coords_zyx,
                radii_mm=radii_mm,
                degrees=degrees,
                spacing_zyx=spacing_zyx,
                profile=profile,
                device_diameter_mm=(
                    device_diameter_mm
                ),
                device_margin_mm=(
                    device_margin_mm
                ),
            )

            if transition is None:
                continue

            next_state = (
                current_node,
                next_node,
            )

            candidate_cost = (
                current_cost + transition
            )

            if candidate_cost < distance.get(
                next_state,
                math.inf,
            ):
                distance[next_state] = (
                    candidate_cost
                )
                predecessor[next_state] = state

                heapq.heappush(
                    heap,
                    (
                        candidate_cost
                        + heuristic(next_node),
                        candidate_cost,
                        current_node,
                        next_node,
                    ),
                )

    if final_state is None:
        raise RuntimeError(
            "没有找到满足当前器械直径约束的路径"
        )

    states: list[tuple[int, int]] = []
    state: tuple[int, int] | None = final_state

    while state is not None:
        states.append(state)
        state = predecessor[state]

    states.reverse()

    nodes = [states[0][1]]
    nodes.extend(
        item[1]
        for item in states[1:]
    )

    return nodes, float(distance[final_state])


def cumulative_distance(
    points_xyz: np.ndarray,
) -> np.ndarray:
    cumulative = np.zeros(
        len(points_xyz),
        dtype=np.float64,
    )

    if len(points_xyz) > 1:
        cumulative[1:] = np.cumsum(
            np.linalg.norm(
                np.diff(points_xyz, axis=0),
                axis=1,
            )
        )

    return cumulative


def smoothed_turn_angles(
    points_xyz: np.ndarray,
    window_mm: float = 5.0,
) -> np.ndarray:
    angles = np.zeros(
        len(points_xyz),
        dtype=np.float64,
    )

    if len(points_xyz) < 3:
        return angles

    cumulative = cumulative_distance(
        points_xyz
    )

    for index in range(1, len(points_xyz) - 1):
        left = int(
            np.searchsorted(
                cumulative,
                cumulative[index] - window_mm,
                side="left",
            )
        )

        right = int(
            np.searchsorted(
                cumulative,
                cumulative[index] + window_mm,
                side="left",
            )
        )

        left = max(
            0,
            min(left, index - 1),
        )
        right = min(
            len(points_xyz) - 1,
            max(right, index + 1),
        )

        incoming = (
            points_xyz[index]
            - points_xyz[left]
        )

        outgoing = (
            points_xyz[right]
            - points_xyz[index]
        )

        denominator = float(
            np.linalg.norm(incoming)
            * np.linalg.norm(outgoing)
        )

        if denominator <= 1e-8:
            continue

        cosine = float(
            np.clip(
                np.dot(incoming, outgoing)
                / denominator,
                -1.0,
                1.0,
            )
        )

        angles[index] = float(
            np.degrees(np.arccos(cosine))
        )

    return angles


def profiles() -> list[CostProfile]:
    return [
        CostProfile(
            name="balanced",
            title="平衡型",
            weight_length=1.0,
            weight_radius=1.5,
            weight_curvature=0.35,
            weight_branch=0.80,
        ),
        CostProfile(
            name="wide_airway",
            title="宽气道优先",
            weight_length=0.90,
            weight_radius=3.30,
            weight_curvature=0.28,
            weight_branch=0.68,
        ),
        CostProfile(
            name="gentle_turn",
            title="平缓转弯优先",
            weight_length=1.05,
            weight_radius=1.65,
            weight_curvature=0.84,
            weight_branch=1.92,
        ),
    ]


class OrthogonalSliceWidget(QLabel):
    """
    轴位、冠状位和矢状位共用的二维切片控件。

    plane:
        axial    -> 固定 z，显示 y-x
        coronal  -> 固定 y，显示 z-x
        sagittal -> 固定 x，显示 z-y
    """

    voxelClicked = Signal(int, int, int)

    def __init__(
        self,
        plane: str,
    ) -> None:
        super().__init__()

        if plane not in {
            "axial",
            "coronal",
            "sagittal",
        }:
            raise ValueError(
                f"不支持的二维平面：{plane}"
            )

        self.plane = plane

        self.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.setMinimumSize(250, 220)
        self.setStyleSheet(
            "background-color: black;"
            "border: 1px solid #24516b;"
        )

        self.ct: np.ndarray | None = None
        self.airway: np.ndarray | None = None
        self.nodule_labels: np.ndarray | None = None
        self.route_mask: np.ndarray | None = None
        self.current_zyx: np.ndarray | None = None
        self.selected_component_label: int | None = None

        self.current_index = 0
        self.source_width = 0
        self.source_height = 0
        self.last_pixmap_size = (0, 0)
        self.last_offsets = (0, 0)

    def set_data(
        self,
        ct: np.ndarray,
        airway: np.ndarray,
        nodule_labels: np.ndarray,
    ) -> None:
        self.ct = ct
        self.airway = airway
        self.nodule_labels = nodule_labels

        if self.plane == "axial":
            self.current_index = ct.shape[0] // 2
        elif self.plane == "coronal":
            self.current_index = ct.shape[1] // 2
        else:
            self.current_index = ct.shape[2] // 2

        self.update_view()

    def set_route(
        self,
        route_mask: np.ndarray | None,
    ) -> None:
        self.route_mask = route_mask
        self.update_view()

    def set_current_point(
        self,
        coord_zyx: np.ndarray | None,
    ) -> None:
        self.current_zyx = (
            None
            if coord_zyx is None
            else np.asarray(
                coord_zyx,
                dtype=np.int32,
            )
        )

        if coord_zyx is not None:
            self.focus_zyx(coord_zyx)
        else:
            self.update_view()

    def focus_zyx(
        self,
        coord_zyx: np.ndarray,
    ) -> None:
        coord = np.asarray(
            coord_zyx,
            dtype=np.float64,
        )

        if self.plane == "axial":
            self.current_index = int(
                round(coord[0])
            )
        elif self.plane == "coronal":
            self.current_index = int(
                round(coord[1])
            )
        else:
            self.current_index = int(
                round(coord[2])
            )

        self.update_view()

    def set_selected_component(
        self,
        component_label: int | None,
    ) -> None:
        self.selected_component_label = (
            component_label
        )
        self.update_view()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.update_view()

    def _extract_slice(
        self,
        array: np.ndarray,
    ) -> np.ndarray:
        if self.plane == "axial":
            index = int(
                np.clip(
                    self.current_index,
                    0,
                    array.shape[0] - 1,
                )
            )
            return array[index, :, :]

        if self.plane == "coronal":
            index = int(
                np.clip(
                    self.current_index,
                    0,
                    array.shape[1] - 1,
                )
            )
            return array[:, index, :]

        index = int(
            np.clip(
                self.current_index,
                0,
                array.shape[2] - 1,
            )
        )
        return array[:, :, index]

    def _point_on_current_slice(
        self,
    ) -> tuple[int, int] | None:
        if self.current_zyx is None:
            return None

        z = int(self.current_zyx[0])
        y = int(self.current_zyx[1])
        x = int(self.current_zyx[2])

        if (
            self.plane == "axial"
            and z == self.current_index
        ):
            return y, x

        if (
            self.plane == "coronal"
            and y == self.current_index
        ):
            return z, x

        if (
            self.plane == "sagittal"
            and x == self.current_index
        ):
            return z, y

        return None

    def _display_to_voxel(
        self,
        row: int,
        column: int,
    ) -> tuple[int, int, int]:
        if self.plane == "axial":
            return (
                int(self.current_index),
                int(row),
                int(column),
            )

        if self.plane == "coronal":
            return (
                int(row),
                int(self.current_index),
                int(column),
            )

        return (
            int(row),
            int(column),
            int(self.current_index),
        )

    def update_view(self) -> None:
        if (
            self.ct is None
            or self.airway is None
            or self.nodule_labels is None
        ):
            return

        ct_slice = self._extract_slice(
            self.ct
        ).astype(np.float32)

        low_hu = -1000.0
        high_hu = 400.0

        normalized = np.clip(
            (ct_slice - low_hu)
            / (high_hu - low_hu),
            0.0,
            1.0,
        )

        gray = (
            normalized * 255.0
        ).astype(np.uint8)

        rgb = np.stack(
            [gray, gray, gray],
            axis=-1,
        ).astype(np.float32)

        airway_slice = self._extract_slice(
            self.airway
        )

        if airway_slice.any():
            airway_color = np.asarray(
                [30, 130, 245],
                dtype=np.float32,
            )
            alpha = 0.30

            rgb[airway_slice] = (
                (1.0 - alpha)
                * rgb[airway_slice]
                + alpha
                * airway_color
            )

        labels_slice = self._extract_slice(
            self.nodule_labels
        )

        for component_label in np.unique(
            labels_slice
        ):
            if component_label == 0:
                continue

            mask = (
                labels_slice
                == component_label
            )

            if (
                int(component_label)
                == self.selected_component_label
            ):
                color = np.asarray(
                    [255, 225, 15],
                    dtype=np.float32,
                )
                alpha = 0.88
            else:
                color_tuple = NODULE_COLORS[
                    (int(component_label) - 1)
                    % len(NODULE_COLORS)
                ]
                color = (
                    np.asarray(color_tuple)
                    * 255.0
                ).astype(np.float32)
                alpha = 0.56

            rgb[mask] = (
                (1.0 - alpha)
                * rgb[mask]
                + alpha * color
            )

        if self.route_mask is not None:
            route_slice = self._extract_slice(
                self.route_mask
            )

            if route_slice.any():
                route_color = np.asarray(
                    [25, 255, 90],
                    dtype=np.float32,
                )
                rgb[route_slice] = (
                    0.12 * rgb[route_slice]
                    + 0.88 * route_color
                )

        marker = self._point_on_current_slice()

        if marker is not None:
            row, column = marker

            for delta in range(-8, 9):
                current_column = column + delta
                current_row = row + delta

                if (
                    0 <= current_column
                    < rgb.shape[1]
                    and 0 <= row < rgb.shape[0]
                ):
                    rgb[row, current_column] = (
                        0,
                        255,
                        255,
                    )

                if (
                    0 <= current_row
                    < rgb.shape[0]
                    and 0 <= column < rgb.shape[1]
                ):
                    rgb[current_row, column] = (
                        0,
                        255,
                        255,
                    )

        rgb_uint8 = np.ascontiguousarray(
            np.clip(
                rgb,
                0,
                255,
            ).astype(np.uint8)
        )

        self.source_height = (
            rgb_uint8.shape[0]
        )
        self.source_width = (
            rgb_uint8.shape[1]
        )

        image = QImage(
            rgb_uint8.data,
            self.source_width,
            self.source_height,
            3 * self.source_width,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image)

        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.last_pixmap_size = (
            scaled.width(),
            scaled.height(),
        )

        self.last_offsets = (
            (
                self.width()
                - scaled.width()
            ) // 2,
            (
                self.height()
                - scaled.height()
            ) // 2,
        )

        self.setPixmap(scaled)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if (
            self.nodule_labels is None
            or self.source_width <= 0
            or self.source_height <= 0
        ):
            return

        click_x = float(
            event.position().x()
        )
        click_y = float(
            event.position().y()
        )

        offset_x, offset_y = (
            self.last_offsets
        )
        pixmap_width, pixmap_height = (
            self.last_pixmap_size
        )

        if (
            click_x < offset_x
            or click_y < offset_y
            or click_x
            >= offset_x + pixmap_width
            or click_y
            >= offset_y + pixmap_height
        ):
            return

        normalized_x = (
            click_x - offset_x
        ) / max(pixmap_width, 1)

        normalized_y = (
            click_y - offset_y
        ) / max(pixmap_height, 1)

        column = int(
            np.clip(
                normalized_x
                * self.source_width,
                0,
                self.source_width - 1,
            )
        )

        row = int(
            np.clip(
                normalized_y
                * self.source_height,
                0,
                self.source_height - 1,
            )
        )

        z, y, x = self._display_to_voxel(
            row,
            column,
        )

        self.voxelClicked.emit(
            x,
            y,
            z,
        )


class InteractiveNavigationWindow(QMainWindow):
    def __init__(
        self,
        initial_case_dir: Path | None,
        min_nodule_voxels: int,
        max_target_nodes: int,
        target_distance_slack_mm: float,
    ) -> None:
        super().__init__()

        self.project_dir = Path(
            __file__
        ).resolve().parent

        self.case_dir: Path | None = None

        self.min_nodule_voxels = (
            min_nodule_voxels
        )
        self.max_target_nodes = (
            max_target_nodes
        )
        self.target_distance_slack_mm = (
            target_distance_slack_mm
        )

        self.ct_image: sitk.Image | None = None
        self.ct_array: np.ndarray | None = None
        self.airway_mask: np.ndarray | None = None
        self.airway_baseline_mask: np.ndarray | None = None
        self.airway_recovered_mask: np.ndarray | None = None
        self.airway_added_mask: np.ndarray | None = None
        self.nodule_raw_mask: np.ndarray | None = None
        self.nodule_labels: np.ndarray | None = None
        self.entry_mask: np.ndarray | None = None

        self.spacing_zyx: np.ndarray | None = None
        self.centerline_mask: np.ndarray | None = None
        self.centerline_coords: np.ndarray | None = None
        self.centerline_points: np.ndarray | None = None
        self.radius_map: np.ndarray | None = None
        self.centerline_radii: np.ndarray | None = None
        self.adjacency: list[
            list[tuple[int, float]]
        ] | None = None
        self.degrees: np.ndarray | None = None
        self.junction_labels: np.ndarray | None = None
        self.segment_labels: np.ndarray | None = None
        self.entry_node: int | None = None
        self.entry_mode = ""

        self.nodule_candidates: list[
            NoduleCandidate
        ] = []

        self.selected_candidate_id: int | None = None
        self.route: RouteResult | None = None
        self.visual_localizer = None
        self.visual_model_case_id = "LIDC_0089"
        self.visual_model_path = (
            self.project_dir.parent
            / "bronchopose_localization"
            / "models"
            / "LIDC_0089_mapclassifier_v1"
            / "checkpoint_best.pt"
        )
        self.baseline_planning_data: dict[str, Any] | None = None
        self.baseline_comparison: dict[str, Any] | None = None

        self.airway_polydata: vtk.vtkPolyData | None = None
        self.airway_actor_3d: vtk.vtkActor | None = None
        self.airway_actor_virtual: vtk.vtkActor | None = None

        self.nodule_actors: dict[
            int,
            vtk.vtkActor,
        ] = {}

        self.actor_to_candidate: dict[
            int,
            int,
        ] = {}

        self.route_actor: vtk.vtkActor | None = None
        self.current_point_actor: vtk.vtkActor | None = None
        self.current_point_source: vtk.vtkSphereSource | None = None
        self.current_point_halo_actor: vtk.vtkActor | None = None
        self.current_point_halo_source: vtk.vtkSphereSource | None = None
        self.map_direction_actor: vtk.vtkActor | None = None
        self.map_direction_source: vtk.vtkArrowSource | None = None
        self.bottleneck_actor: vtk.vtkActor | None = None
        self.bottleneck_source: vtk.vtkSphereSource | None = None

        self.virtual_direction_source: vtk.vtkArrowSource | None = None
        self.virtual_direction_actor: vtk.vtkActor | None = None

        # 虚拟镜前景单箭头。使用 VTK Actor2D，而不是 Qt QLabel，
        # 避免 Windows 下被 OpenGL 原生窗口遮挡。
        self.virtual_nav_arrow_polydata: vtk.vtkPolyData | None = None
        self.virtual_nav_arrow_mapper: vtk.vtkPolyDataMapper2D | None = None
        self.virtual_nav_arrow_actor: vtk.vtkActor2D | None = None
        self.renderer_virtual_overlay: vtk.vtkRenderer | None = None

        # 保留旧字段仅用于兼容，实际不再使用 QLabel 绘制箭头。
        self.navigation_arrow_overlay: QLabel | None = None

        self.timer = QTimer(self)
        self.timer.setInterval(90)
        self.timer.timeout.connect(
            self.advance_route
        )

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1760, 980)
        self.setMinimumSize(1380, 820)

        self._build_ui()
        self._apply_style()

        if initial_case_dir is not None:
            resolved = initial_case_dir.resolve()

            QTimer.singleShot(
                80,
                lambda path=resolved: (
                    self.load_case(path)
                ),
            )
        else:
            QTimer.singleShot(
                120,
                self.choose_case_directory,
            )

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )
        root_layout.setSpacing(7)

        top_frame = QFrame()
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(
            14,
            9,
            14,
            9,
        )

        title_box = QVBoxLayout()

        title_label = QLabel(
            "经支气管肺结节交互导航系统"
        )
        title_label.setObjectName("TitleLabel")

        subtitle_label = QLabel(
            "全部结节候选 · 点击选靶 · "
            "无需重新分割即可规划 · "
            "2D/3D/虚拟镜同步"
        )
        subtitle_label.setObjectName(
            "SubtitleLabel"
        )

        title_box.addWidget(title_label)
        title_box.addWidget(subtitle_label)

        top_layout.addLayout(title_box)

        self.case_path_label = QLabel(
            "尚未加载病例"
        )
        self.case_path_label.setObjectName(
            "CasePathLabel"
        )
        self.case_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        top_layout.addWidget(
            self.case_path_label,
            1,
        )

        choose_button = QPushButton(
            "选择病例目录"
        )
        choose_button.clicked.connect(
            self.choose_case_directory
        )
        top_layout.addWidget(choose_button)

        reload_button = QPushButton(
            "重新加载"
        )
        reload_button.clicked.connect(
            self.reload_case
        )
        top_layout.addWidget(reload_button)

        root_layout.addWidget(top_frame)

        main_splitter = QSplitter(
            Qt.Orientation.Horizontal
        )
        main_splitter.setChildrenCollapsible(
            False
        )
        root_layout.addWidget(
            main_splitter,
            1,
        )

        left_panel = QWidget()
        left_panel.setMinimumWidth(400)
        left_panel.setMaximumWidth(510)

        left_layout = QVBoxLayout(
            left_panel
        )
        left_layout.setContentsMargins(
            0,
            0,
            6,
            0,
        )
        left_layout.setSpacing(7)

        nodule_group = QGroupBox(
            "全部结节候选"
        )
        nodule_layout = QVBoxLayout(
            nodule_group
        )

        self.nodule_table = QTableWidget()
        self.nodule_table.setColumnCount(5)
        self.nodule_table.setHorizontalHeaderLabels(
            [
                "候选",
                "体素",
                "体积\nmm³",
                "X\nmm",
                "Y/Z\nmm",
            ]
        )
        self.nodule_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.nodule_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.nodule_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.nodule_table.verticalHeader().setVisible(
            False
        )
        self.nodule_table.itemSelectionChanged.connect(
            self.on_nodule_table_selection
        )

        header = self.nodule_table.horizontalHeader()
        header.setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        header.setStretchLastSection(True)

        nodule_layout.addWidget(
            self.nodule_table
        )

        selection_hint = QLabel(
            "可点击表格、二维候选区域或三维结节。"
            "选择后自动重新规划，不重复分割。"
        )
        selection_hint.setWordWrap(True)
        selection_hint.setObjectName("HintLabel")
        nodule_layout.addWidget(selection_hint)

        left_layout.addWidget(
            nodule_group,
            2,
        )

        device_group = QGroupBox(
            "器械与规划"
        )
        device_layout = QGridLayout(
            device_group
        )

        device_layout.addWidget(
            QLabel("器械外径"),
            0,
            0,
        )

        self.device_diameter_spin = (
            QDoubleSpinBox()
        )
        self.device_diameter_spin.setRange(
            0.0,
            10.0,
        )
        self.device_diameter_spin.setDecimals(2)
        self.device_diameter_spin.setSingleStep(
            0.10
        )
        self.device_diameter_spin.setValue(
            0.0
        )
        self.device_diameter_spin.setSuffix(
            " mm"
        )
        device_layout.addWidget(
            self.device_diameter_spin,
            0,
            1,
        )

        device_layout.addWidget(
            QLabel("安全余量"),
            1,
            0,
        )

        self.device_margin_spin = (
            QDoubleSpinBox()
        )
        self.device_margin_spin.setRange(
            0.0,
            3.0,
        )
        self.device_margin_spin.setDecimals(2)
        self.device_margin_spin.setSingleStep(
            0.05
        )
        self.device_margin_spin.setValue(
            0.20
        )
        self.device_margin_spin.setSuffix(
            " mm"
        )
        device_layout.addWidget(
            self.device_margin_spin,
            1,
            1,
        )

        self.auto_replan_checkbox = QCheckBox(
            "选择目标后自动重新规划"
        )
        self.auto_replan_checkbox.setChecked(
            True
        )
        device_layout.addWidget(
            self.auto_replan_checkbox,
            2,
            0,
            1,
            2,
        )

        replan_button = QPushButton(
            "重新规划当前目标"
        )
        replan_button.clicked.connect(
            self.replan_selected_target
        )
        device_layout.addWidget(
            replan_button,
            3,
            0,
            1,
            2,
        )

        left_layout.addWidget(device_group)

        metrics_group = QGroupBox(
            "推荐路径指标"
        )
        metrics_layout = QGridLayout(
            metrics_group
        )

        metric_rows = [
            ("目标", "target"),
            ("策略", "profile"),
            ("路径长度", "length"),
            ("最小直径", "min_diameter"),
            ("平均直径", "mean_diameter"),
            ("最大转角", "max_turn"),
            ("目标距离", "target_distance"),
            ("器械判断", "device"),
            ("原始/优化对比", "ab_comparison"),
            ("修复区域依赖", "repair_dependency"),
        ]

        self.metric_labels: dict[
            str,
            QLabel,
        ] = {}

        for row, (title, key) in enumerate(
            metric_rows
        ):
            title_label = QLabel(title)
            value_label = QLabel("—")
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            value_label.setWordWrap(True)

            metrics_layout.addWidget(
                title_label,
                row,
                0,
                alignment=(
                    Qt.AlignmentFlag.AlignTop
                ),
            )
            metrics_layout.addWidget(
                value_label,
                row,
                1,
            )

            self.metric_labels[key] = (
                value_label
            )

        metrics_layout.setColumnStretch(1, 1)

        left_layout.addWidget(metrics_group)

        branch_group = QGroupBox(
            "分叉序列与当前位置"
        )
        branch_layout = QVBoxLayout(
            branch_group
        )

        self.current_local_label = QLabel(
            "尚未规划"
        )
        self.current_local_label.setWordWrap(
            True
        )
        self.current_local_label.setObjectName(
            "CurrentLocalLabel"
        )
        branch_layout.addWidget(
            self.current_local_label
        )

        self.branch_text = QTextEdit()
        self.branch_text.setReadOnly(True)
        self.branch_text.setMinimumHeight(120)
        branch_layout.addWidget(
            self.branch_text
        )

        left_layout.addWidget(
            branch_group,
            1,
        )

        main_splitter.addWidget(left_panel)

        # ============================================================
        # 中间：大尺寸虚拟支气管镜（与原来的气道地图位置互换）
        # ============================================================
        center_panel = QWidget()
        center_layout = QVBoxLayout(
            center_panel
        )
        center_layout.setContentsMargins(
            0,
            0,
            4,
            0,
        )
        center_layout.setSpacing(5)

        virtual_header = QHBoxLayout()

        self.virtual_title = QLabel(
            "虚拟支气管镜主视图"
        )
        self.virtual_title.setObjectName(
            "ViewLabel"
        )
        virtual_header.addWidget(
            self.virtual_title
        )
        virtual_header.addStretch(1)

        self.follow_camera_checkbox = QCheckBox(
            "镜头沿路径自动跟随"
        )
        self.follow_camera_checkbox.setChecked(
            True
        )
        virtual_header.addWidget(
            self.follow_camera_checkbox
        )

        center_layout.addLayout(
            virtual_header
        )

        self.vtk_virtual = (
            QVTKRenderWindowInteractor(
                center_panel
            )
        )
        center_layout.addWidget(
            self.vtk_virtual,
            1,
        )

        # 屏幕前景导航箭头。它是 QVTK 控件的透明子控件，
        # 始终完整显示在虚拟镜画面前方，不会再因为箭头与
        # 摄像机视线共线而只能看到尾部。
        # 箭头改由 VTK Actor2D 绘制。这里不再创建 Qt 覆盖面板，
        # 从根本上避免 QLabel 被 QVTK/OpenGL 窗口遮挡。
        self.navigation_arrow_overlay = None

        # ------------------------------------------------------------
        # 路径控制：起点、逐点、跨步、播放、速度、终点
        # ------------------------------------------------------------
        route_control = QHBoxLayout()

        self.start_button = QPushButton(
            "起点"
        )
        self.start_button.clicked.connect(
            self.jump_to_route_start
        )
        route_control.addWidget(
            self.start_button
        )

        self.back_ten_button = QPushButton(
            "后退10点"
        )
        self.back_ten_button.clicked.connect(
            lambda: self.step_route(-10)
        )
        route_control.addWidget(
            self.back_ten_button
        )

        self.previous_button = QPushButton(
            "上一点"
        )
        self.previous_button.clicked.connect(
            lambda: self.step_route(-1)
        )
        route_control.addWidget(
            self.previous_button
        )

        self.play_button = QPushButton(
            "播放"
        )
        self.play_button.clicked.connect(
            self.toggle_playback
        )
        route_control.addWidget(
            self.play_button
        )

        self.next_button = QPushButton(
            "下一点"
        )
        self.next_button.clicked.connect(
            lambda: self.step_route(1)
        )
        route_control.addWidget(
            self.next_button
        )

        self.forward_ten_button = QPushButton(
            "前进10点"
        )
        self.forward_ten_button.clicked.connect(
            lambda: self.step_route(10)
        )
        route_control.addWidget(
            self.forward_ten_button
        )

        self.end_button = QPushButton(
            "终点"
        )
        self.end_button.clicked.connect(
            self.jump_to_route_end
        )
        route_control.addWidget(
            self.end_button
        )

        route_control.addWidget(
            QLabel("速度")
        )

        self.speed_combo = QComboBox()
        for text_value, interval_ms in [
            ("0.25×", 360),
            ("0.5×", 180),
            ("1×", 90),
            ("2×", 45),
            ("4×", 25),
        ]:
            self.speed_combo.addItem(
                text_value,
                interval_ms,
            )
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(
            self.change_playback_speed
        )
        route_control.addWidget(
            self.speed_combo
        )

        self.loop_checkbox = QCheckBox(
            "循环"
        )
        self.loop_checkbox.setChecked(
            False
        )
        route_control.addWidget(
            self.loop_checkbox
        )

        center_layout.addLayout(
            route_control
        )

        visual_control = QHBoxLayout()
        self.load_visual_model_button = QPushButton("加载视觉定位模型")
        self.load_visual_model_button.clicked.connect(self.load_visual_localization_model)
        visual_control.addWidget(self.load_visual_model_button)
        self.test_visual_frame_button = QPushButton("测试镜头帧")
        self.test_visual_frame_button.clicked.connect(self.choose_visual_test_frame)
        self.test_visual_frame_button.setEnabled(False)
        visual_control.addWidget(self.test_visual_frame_button)
        self.reset_visual_tracking_button = QPushButton("视觉定位复位")
        self.reset_visual_tracking_button.clicked.connect(self.reset_visual_tracking)
        self.reset_visual_tracking_button.setEnabled(False)
        visual_control.addWidget(self.reset_visual_tracking_button)
        self.visual_tracking_label = QLabel("视觉定位：未加载")
        self.visual_tracking_label.setObjectName("ViewLabel")
        visual_control.addWidget(self.visual_tracking_label, 1)
        center_layout.addLayout(visual_control)

        slider_control = QHBoxLayout()

        self.route_slider = QSlider(
            Qt.Orientation.Horizontal
        )
        self.route_slider.setRange(0, 0)
        self.route_slider.valueChanged.connect(
            self.update_synchronized_views
        )
        slider_control.addWidget(
            self.route_slider,
            1,
        )

        self.route_position_label = QLabel(
            "0 / 0"
        )
        self.route_position_label.setMinimumWidth(
            110
        )
        slider_control.addWidget(
            self.route_position_label
        )

        center_layout.addLayout(
            slider_control
        )

        main_splitter.addWidget(
            center_panel
        )

        # ============================================================
        # 右侧：三方位二维图 + 小尺寸气道地图
        # ============================================================
        right_splitter = QSplitter(
            Qt.Orientation.Vertical
        )
        right_splitter.setChildrenCollapsible(
            False
        )

        orthogonal_container = QWidget()
        orthogonal_layout = QGridLayout(
            orthogonal_container
        )
        orthogonal_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        orthogonal_layout.setHorizontalSpacing(
            4
        )
        orthogonal_layout.setVerticalSpacing(
            4
        )

        self.axial_title = QLabel(
            "轴位"
        )
        self.axial_title.setObjectName(
            "ViewLabel"
        )
        self.axial_widget = (
            OrthogonalSliceWidget(
                "axial"
            )
        )
        self.axial_widget.voxelClicked.connect(
            self.on_slice_voxel_clicked
        )

        axial_box = QWidget()
        axial_box_layout = QVBoxLayout(
            axial_box
        )
        axial_box_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        axial_box_layout.setSpacing(2)
        axial_box_layout.addWidget(
            self.axial_title
        )
        axial_box_layout.addWidget(
            self.axial_widget,
            1,
        )

        self.coronal_title = QLabel(
            "冠状位"
        )
        self.coronal_title.setObjectName(
            "ViewLabel"
        )
        self.coronal_widget = (
            OrthogonalSliceWidget(
                "coronal"
            )
        )
        self.coronal_widget.voxelClicked.connect(
            self.on_slice_voxel_clicked
        )

        coronal_box = QWidget()
        coronal_box_layout = QVBoxLayout(
            coronal_box
        )
        coronal_box_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        coronal_box_layout.setSpacing(2)
        coronal_box_layout.addWidget(
            self.coronal_title
        )
        coronal_box_layout.addWidget(
            self.coronal_widget,
            1,
        )

        self.sagittal_title = QLabel(
            "矢状位"
        )
        self.sagittal_title.setObjectName(
            "ViewLabel"
        )
        self.sagittal_widget = (
            OrthogonalSliceWidget(
                "sagittal"
            )
        )
        self.sagittal_widget.voxelClicked.connect(
            self.on_slice_voxel_clicked
        )

        sagittal_box = QWidget()
        sagittal_box_layout = QVBoxLayout(
            sagittal_box
        )
        sagittal_box_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        sagittal_box_layout.setSpacing(2)
        sagittal_box_layout.addWidget(
            self.sagittal_title
        )
        sagittal_box_layout.addWidget(
            self.sagittal_widget,
            1,
        )

        orthogonal_layout.addWidget(
            axial_box,
            0,
            0,
        )
        orthogonal_layout.addWidget(
            coronal_box,
            0,
            1,
        )
        orthogonal_layout.addWidget(
            sagittal_box,
            1,
            0,
            1,
            2,
        )

        orthogonal_layout.setRowStretch(
            0,
            1,
        )
        orthogonal_layout.setRowStretch(
            1,
            1,
        )
        orthogonal_layout.setColumnStretch(
            0,
            1,
        )
        orthogonal_layout.setColumnStretch(
            1,
            1,
        )

        right_splitter.addWidget(
            orthogonal_container
        )

        airway_map_container = QWidget()
        airway_map_layout = QVBoxLayout(
            airway_map_container
        )
        airway_map_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        airway_map_layout.setSpacing(4)

        map_bar = QHBoxLayout()

        for button_text, callback in [
            ("等轴测", self.set_isometric_view),
            ("前", lambda: self.set_3d_camera("front")),
            ("后", lambda: self.set_3d_camera("back")),
            ("左", lambda: self.set_3d_camera("left")),
            ("右", lambda: self.set_3d_camera("right")),
            ("上", lambda: self.set_3d_camera("top")),
            ("适配", self.reset_3d_camera),
        ]:
            button = QPushButton(
                button_text
            )
            button.clicked.connect(
                callback
            )
            map_bar.addWidget(button)

        map_bar.addStretch(1)

        self.view_3d_label = QLabel(
            "气道地图 · 实时位置"
        )
        self.view_3d_label.setObjectName(
            "ViewLabel"
        )
        map_bar.addWidget(
            self.view_3d_label
        )

        airway_map_layout.addLayout(
            map_bar
        )

        self.vtk_3d = (
            QVTKRenderWindowInteractor(
                airway_map_container
            )
        )
        airway_map_layout.addWidget(
            self.vtk_3d,
            1,
        )

        right_splitter.addWidget(
            airway_map_container
        )

        right_splitter.setSizes(
            [590, 330]
        )

        main_splitter.addWidget(
            right_splitter
        )

        main_splitter.setStretchFactor(
            0,
            0,
        )
        main_splitter.setStretchFactor(
            1,
            2,
        )
        main_splitter.setStretchFactor(
            2,
            1,
        )
        main_splitter.setSizes(
            [430, 830, 650]
        )

        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(
            0.018,
            0.035,
            0.050,
        )
        self.renderer_3d.SetBackground2(
            0.055,
            0.095,
            0.125,
        )
        self.renderer_3d.GradientBackgroundOn()

        self.vtk_3d.GetRenderWindow().AddRenderer(
            self.renderer_3d
        )

        self.interactor_3d = (
            self.vtk_3d
            .GetRenderWindow()
            .GetInteractor()
        )

        self.interactor_3d.SetInteractorStyle(
            vtk.vtkInteractorStyleTrackballCamera()
        )

        self.interactor_3d.AddObserver(
            "LeftButtonPressEvent",
            self.on_3d_left_click,
            1.0,
        )

        self.renderer_virtual = (
            vtk.vtkRenderer()
        )
        self.renderer_virtual.SetBackground(
            0.03,
            0.015,
            0.012,
        )

        virtual_render_window = (
            self.vtk_virtual.GetRenderWindow()
        )

        virtual_render_window.SetNumberOfLayers(
            2
        )

        self.renderer_virtual.SetLayer(
            0
        )

        virtual_render_window.AddRenderer(
            self.renderer_virtual
        )

        self.renderer_virtual_overlay = (
            vtk.vtkRenderer()
        )
        self.renderer_virtual_overlay.SetLayer(
            1
        )
        self.renderer_virtual_overlay.SetInteractive(
            False
        )
        self.renderer_virtual_overlay.PreserveColorBufferOn()
        self.renderer_virtual_overlay.PreserveDepthBufferOff()
        self.renderer_virtual_overlay.SetViewport(
            0.0,
            0.0,
            1.0,
            1.0,
        )

        virtual_render_window.AddRenderer(
            self.renderer_virtual_overlay
        )

        self.interactor_virtual = (
            self.vtk_virtual
            .GetRenderWindow()
            .GetInteractor()
        )

        self.interactor_virtual.SetInteractorStyle(
            vtk.vtkInteractorStyleTrackballCamera()
        )

        self.statusBar().showMessage(
            "请选择第四阶段病例目录"
        )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #07131d;
                color: #d7ecf7;
                font-family: "Microsoft YaHei";
                font-size: 13px;
            }

            QFrame {
                background-color: #0a1b27;
                border: 1px solid #21445a;
                border-radius: 10px;
            }

            QGroupBox {
                background-color: #0a1b27;
                border: 1px solid #21445a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 700;
                color: #60d4ff;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }

            QLabel#TitleLabel {
                color: white;
                font-size: 23px;
                font-weight: 800;
            }

            QLabel#SubtitleLabel {
                color: #8bcbe5;
                font-size: 12px;
            }

            QLabel#CasePathLabel {
                color: #9bdcf4;
                background: transparent;
            }

            QLabel#HintLabel {
                color: #87a9bb;
                font-size: 11px;
            }

            QLabel#ViewLabel {
                color: #71dcff;
                font-size: 14px;
                font-weight: 800;
            }

            QLabel#CurrentLocalLabel {
                color: #fff2a8;
                background-color: #0a2637;
                border: 1px solid #24516b;
                border-radius: 6px;
                padding: 7px;
            }

            QPushButton {
                background-color: #12344a;
                border: 1px solid #2a6582;
                border-radius: 6px;
                padding: 7px 12px;
                color: #e8f8ff;
                font-weight: 700;
            }

            QPushButton:hover {
                background-color: #19506e;
                border-color: #44bdea;
            }

            QTableWidget {
                background-color: #07141e;
                alternate-background-color: #0c202e;
                gridline-color: #1c4054;
                selection-background-color: #1a617e;
                selection-color: white;
                border: 1px solid #21445a;
            }

            QHeaderView::section {
                background-color: #103047;
                color: #cbefff;
                border: 1px solid #21445a;
                padding: 5px;
                font-weight: 700;
            }

            QTextEdit {
                background-color: #06141e;
                border: 1px solid #21445a;
                color: #bfeaff;
            }

            QDoubleSpinBox {
                background-color: #06141e;
                border: 1px solid #2a6582;
                border-radius: 5px;
                padding: 5px;
            }

            QSlider::groove:horizontal {
                height: 6px;
                background: #19384b;
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                width: 16px;
                margin: -5px 0;
                background: #38c5f4;
                border-radius: 8px;
            }

            QStatusBar {
                background: #061019;
                color: #8ec7dc;
            }
            """
        )

    def choose_case_directory(self) -> None:
        start_dir = (
            str(self.case_dir)
            if self.case_dir is not None
            else str(self.project_dir)
        )

        selected = QFileDialog.getExistingDirectory(
            self,
            "选择 stage4_package 病例目录",
            start_dir,
        )

        if selected:
            self.load_case(
                Path(selected)
            )

    def reload_case(self) -> None:
        if self.case_dir is None:
            self.choose_case_directory()
            return

        self.load_case(self.case_dir)

    def resolve_case_files(
        self,
        case_dir: Path,
    ) -> dict[str, Path | None]:
        ct = case_dir / "ct.nii.gz"
        airway = case_dir / "airway_mask.nii.gz"
        airway_baseline = case_dir / "airway_raw_baseline.nii.gz"
        airway_recovered = case_dir / "airway_recovered_voxels.nii.gz"
        airway_bridge = case_dir / "airway_bridge_voxels.nii.gz"
        nodule_raw = (
            case_dir / "nodule_raw.nii.gz"
        )
        entry = case_dir / "entry_point.nii.gz"

        required = [
            ct,
            airway,
            nodule_raw,
        ]

        missing = [
            path
            for path in required
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "病例目录缺少必要文件：\n"
                + "\n".join(
                    str(path)
                    for path in missing
                )
            )

        return {
            "ct": ct,
            "airway": airway,
            "airway_baseline": airway_baseline if airway_baseline.is_file() else None,
            "airway_recovered": airway_recovered if airway_recovered.is_file() else None,
            "airway_bridge": airway_bridge if airway_bridge.is_file() else None,
            "nodule_raw": nodule_raw,
            "entry": (
                entry
                if entry.is_file()
                else None
            ),
        }

    def clear_case(self) -> None:
        self.timer.stop()
        self.play_button.setText("播放")

        self.renderer_3d.RemoveAllViewProps()
        self.renderer_virtual.RemoveAllViewProps()

        if self.renderer_virtual_overlay is not None:
            self.renderer_virtual_overlay.RemoveAllViewProps()

        self.nodule_actors.clear()
        self.actor_to_candidate.clear()

        self.route_actor = None
        self.current_point_actor = None
        self.current_point_source = None
        self.current_point_halo_actor = None
        self.current_point_halo_source = None
        self.map_direction_actor = None
        self.map_direction_source = None
        self.bottleneck_actor = None
        self.bottleneck_source = None
        self.virtual_direction_actor = None
        self.virtual_direction_source = None

        self.virtual_nav_arrow_polydata = None
        self.virtual_nav_arrow_mapper = None
        self.virtual_nav_arrow_actor = None

        if self.navigation_arrow_overlay is not None:
            self.navigation_arrow_overlay.clear()
            self.navigation_arrow_overlay.hide()

        self.nodule_table.clearContents()
        self.nodule_table.setRowCount(0)

        self.route_slider.setRange(0, 0)
        self.route_slider.setValue(0)

        self.branch_text.clear()

        for label in self.metric_labels.values():
            label.setText("—")

    def load_case(
        self,
        case_dir: Path,
    ) -> None:
        QApplication.setOverrideCursor(
            Qt.CursorShape.WaitCursor
        )

        try:
            case_dir = case_dir.resolve()
            files = self.resolve_case_files(
                case_dir
            )

            self.clear_case()
            self.case_dir = case_dir
            self.case_path_label.setText(
                str(case_dir)
            )

            self.statusBar().showMessage(
                "正在读取 CT、气道和全部结节候选……"
            )
            QApplication.processEvents()

            ct_image, ct_array = read_image(
                files["ct"]
            )
            airway_image, airway_array = (
                read_image(files["airway"])
            )
            nodule_image, nodule_array = (
                read_image(files["nodule_raw"])
            )

            check_geometry(
                ct_image,
                airway_image,
                "airway",
            )
            check_geometry(
                ct_image,
                nodule_image,
                "nodule_raw",
            )

            self.ct_image = ct_image
            self.ct_array = ct_array.astype(
                np.float32
            )
            self.airway_mask = airway_array > 0
            if files["airway_baseline"] is not None:
                baseline_image, baseline_array = read_image(files["airway_baseline"])
                check_geometry(ct_image, baseline_image, "airway_raw_baseline")
                self.airway_baseline_mask = baseline_array > 0
            else:
                self.airway_baseline_mask = None
            self.baseline_planning_data = None
            self.baseline_comparison = None
            if files["airway_recovered"] is not None:
                recovered_image, recovered_array = read_image(files["airway_recovered"])
                check_geometry(ct_image, recovered_image, "airway_recovered_voxels")
                self.airway_recovered_mask = recovered_array > 0
            else:
                self.airway_recovered_mask = None
            if files["airway_bridge"] is not None:
                bridge_image, bridge_array = read_image(files["airway_bridge"])
                check_geometry(ct_image, bridge_image, "airway_bridge_voxels")
                self.airway_added_mask = bridge_array > 0
            else:
                self.airway_added_mask = None
            self.nodule_raw_mask = nodule_array > 0

            self.spacing_zyx = np.asarray(
                ct_image.GetSpacing()[::-1],
                dtype=np.float64,
            )

            entry_path = files["entry"]

            if isinstance(entry_path, Path):
                entry_image, entry_array = (
                    read_image(entry_path)
                )
                check_geometry(
                    ct_image,
                    entry_image,
                    "entry_point",
                )
                self.entry_mask = entry_array > 0
            else:
                self.entry_mask = None

            if not self.airway_mask.any():
                raise RuntimeError(
                    "airway_mask.nii.gz 为空"
                )

            if not self.nodule_raw_mask.any():
                raise RuntimeError(
                    "nodule_raw.nii.gz 为空"
                )

            self.extract_nodule_candidates()

            self.statusBar().showMessage(
                "正在提取气道中心线并建立规划图……"
            )
            QApplication.processEvents()

            self.centerline_mask = skeletonize(
                self.airway_mask,
                method="lee",
            ).astype(bool)

            self.centerline_coords = np.argwhere(
                self.centerline_mask
            ).astype(np.int32)

            self.centerline_points = (
                physical_points(
                    self.centerline_coords,
                    ct_image,
                )
            )

            self.radius_map = (
                ndi.distance_transform_edt(
                    self.airway_mask,
                    sampling=self.spacing_zyx,
                )
            )

            self.centerline_radii = (
                self.radius_map[
                    tuple(
                        self.centerline_coords.T
                    )
                ].astype(np.float64)
            )

            (
                self.adjacency,
                self.degrees,
            ) = build_graph(
                self.centerline_coords,
                self.airway_mask.shape,
                self.spacing_zyx,
            )

            (
                self.junction_labels,
                self.segment_labels,
            ) = build_branch_topology(
                self.adjacency,
                self.degrees,
            )

            (
                self.entry_node,
                self.entry_mode,
            ) = choose_entry_node(
                self.centerline_coords,
                self.centerline_points,
                self.centerline_radii,
                self.entry_mask,
                self.spacing_zyx,
            )

            self.statusBar().showMessage(
                "正在重建三维气道和结节候选……"
            )
            QApplication.processEvents()

            self.build_3d_scene()
            self.build_virtual_scene()
            self.populate_nodule_table()

            for slice_widget in [
                self.axial_widget,
                self.coronal_widget,
                self.sagittal_widget,
            ]:
                slice_widget.set_data(
                    self.ct_array,
                    self.airway_mask,
                    self.nodule_labels,
                )

            self.vtk_3d.Initialize()
            self.vtk_virtual.Initialize()

            self.set_isometric_view()
            self.renderer_virtual.ResetCamera()

            self.statusBar().showMessage(
                f"病例加载完成，共发现 "
                f"{len(self.nodule_candidates)} "
                f"个结节候选"
            )

            if self.nodule_candidates:
                self.nodule_table.selectRow(0)

        except Exception as error:
            QMessageBox.critical(
                self,
                "病例加载失败",
                f"{type(error).__name__}: {error}\n\n"
                f"{traceback.format_exc()}",
            )
            self.statusBar().showMessage(
                "病例加载失败"
            )

        finally:
            QApplication.restoreOverrideCursor()

    def extract_nodule_candidates(
        self,
    ) -> None:
        assert self.nodule_raw_mask is not None
        assert self.ct_image is not None

        labels, count = ndi.label(
            self.nodule_raw_mask,
            structure=ndi.generate_binary_structure(
                3,
                3,
            ),
        )

        sizes = np.bincount(labels.ravel())

        voxel_volume_mm3 = float(
            np.prod(
                self.ct_image.GetSpacing()
            )
        )

        candidates: list[
            NoduleCandidate
        ] = []

        accepted_labels = np.zeros_like(
            labels,
            dtype=np.int32,
        )

        candidate_id = 0

        for component_label in range(
            1,
            count + 1,
        ):
            voxel_count = int(
                sizes[component_label]
            )

            if (
                voxel_count
                < self.min_nodule_voxels
            ):
                continue

            component = (
                labels == component_label
            )
            coords = np.argwhere(
                component
            )

            candidate_id += 1
            accepted_labels[component] = (
                candidate_id
            )

            center_zyx = coords.mean(
                axis=0
            )

            center_xyz = physical_points(
                center_zyx[None, :],
                self.ct_image,
            )[0]

            candidates.append(
                NoduleCandidate(
                    candidate_id=candidate_id,
                    component_label=(
                        candidate_id
                    ),
                    voxel_count=voxel_count,
                    volume_mm3=(
                        voxel_count
                        * voxel_volume_mm3
                    ),
                    center_zyx=center_zyx,
                    center_xyz_mm=center_xyz,
                    minimum_zyx=coords.min(axis=0),
                    maximum_zyx=coords.max(axis=0),
                )
            )

        candidates.sort(
            key=lambda item: (
                -item.voxel_count,
                item.candidate_id,
            )
        )

        # 重新编号，保证表格编号与 label 值一致。
        remapped = np.zeros_like(
            accepted_labels,
            dtype=np.int32,
        )

        normalized_candidates: list[
            NoduleCandidate
        ] = []

        for new_id, candidate in enumerate(
            candidates,
            start=1,
        ):
            old_id = candidate.component_label
            remapped[
                accepted_labels == old_id
            ] = new_id

            candidate.candidate_id = new_id
            candidate.component_label = new_id
            normalized_candidates.append(
                candidate
            )

        self.nodule_labels = remapped
        self.nodule_candidates = (
            normalized_candidates
        )

        if not self.nodule_candidates:
            raise RuntimeError(
                "没有达到最小体素阈值的结节候选"
            )

    def build_3d_scene(self) -> None:
        assert self.airway_mask is not None
        assert self.ct_image is not None
        assert self.nodule_labels is not None

        self.airway_polydata = (
            mask_to_polydata(
                self.airway_mask,
                self.ct_image,
                smooth_iterations=16,
                decimate_reduction=0.18,
            )
        )

        self.airway_actor_3d = (
            make_surface_actor(
                self.airway_polydata,
                AIRWAY_COLOR,
                0.22,
            )
        )

        self.renderer_3d.AddActor(
            self.airway_actor_3d
        )

        if self.airway_recovered_mask is not None and self.airway_recovered_mask.any():
            recovered_polydata = mask_to_polydata(
                self.airway_recovered_mask,
                self.ct_image,
                smooth_iterations=4,
                decimate_reduction=0.0,
            )
            self.airway_recovered_actor_3d = make_surface_actor(
                recovered_polydata,
                AIRWAY_RECOVERED_COLOR,
                0.82,
            )
            self.renderer_3d.AddActor(self.airway_recovered_actor_3d)

        if self.airway_added_mask is not None and self.airway_added_mask.any():
            added_polydata = mask_to_polydata(
                self.airway_added_mask,
                self.ct_image,
                smooth_iterations=4,
                decimate_reduction=0.0,
            )
            self.airway_added_actor_3d = make_surface_actor(
                added_polydata,
                AIRWAY_ADDED_COLOR,
                0.95,
            )
            self.renderer_3d.AddActor(self.airway_added_actor_3d)

        for candidate in self.nodule_candidates:
            component_mask = (
                self.nodule_labels
                == candidate.component_label
            )

            polydata = mask_to_polydata(
                component_mask,
                self.ct_image,
                smooth_iterations=14,
                decimate_reduction=0.0,
            )

            color = NODULE_COLORS[
                (candidate.candidate_id - 1)
                % len(NODULE_COLORS)
            ]

            actor = make_surface_actor(
                polydata,
                color,
                0.72,
            )

            self.renderer_3d.AddActor(actor)

            self.nodule_actors[
                candidate.candidate_id
            ] = actor

            self.actor_to_candidate[
                id(actor)
            ] = candidate.candidate_id

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(
            25.0,
            25.0,
            25.0,
        )
        self.renderer_3d.AddActor(axes)

    def build_virtual_scene(self) -> None:
        assert self.airway_polydata is not None

        self.airway_actor_virtual = (
            make_surface_actor(
                self.airway_polydata,
                (0.82, 0.48, 0.40),
                1.0,
            )
        )

        prop = (
            self.airway_actor_virtual
            .GetProperty()
        )
        prop.FrontfaceCullingOn()
        prop.SetAmbient(0.42)
        prop.SetDiffuse(0.68)
        prop.SetSpecular(0.18)

        self.renderer_virtual.AddActor(
            self.airway_actor_virtual
        )

        (
            self.virtual_direction_actor,
            self.virtual_direction_source,
        ) = make_arrow_actor(
            (0.78, 1.00, 1.00)
        )

        # 恢复第一版的腔内三维箭头外观。
        # 箭头由 vtkArrowSource 生成，直接位于气道腔内，
        # 不使用二维 HUD、文字或背景面板。
        self.virtual_direction_source.SetTipLength(
            0.30
        )
        self.virtual_direction_source.SetTipRadius(
            0.22
        )
        self.virtual_direction_source.SetTipResolution(
            40
        )
        self.virtual_direction_source.SetShaftRadius(
            0.065
        )
        self.virtual_direction_source.SetShaftResolution(
            40
        )
        self.virtual_direction_source.Update()

        virtual_arrow_property = (
            self.virtual_direction_actor
            .GetProperty()
        )
        virtual_arrow_property.SetColor(
            0.78,
            1.00,
            1.00,
        )
        virtual_arrow_property.SetOpacity(
            0.92
        )
        virtual_arrow_property.SetInterpolationToPhong()
        virtual_arrow_property.SetAmbient(
            0.64
        )
        virtual_arrow_property.SetDiffuse(
            0.76
        )
        virtual_arrow_property.SetSpecular(
            1.00
        )
        virtual_arrow_property.SetSpecularPower(
            80.0
        )

        self.virtual_direction_actor.SetVisibility(
            True
        )

        self.renderer_virtual.AddActor(
            self.virtual_direction_actor
        )

        # ------------------------------------------------------------
        # 虚拟镜前景单箭头
        # ------------------------------------------------------------
        self.virtual_nav_arrow_polydata = vtk.vtkPolyData()

        self.virtual_nav_arrow_mapper = vtk.vtkPolyDataMapper2D()
        self.virtual_nav_arrow_mapper.SetInputData(
            self.virtual_nav_arrow_polydata
        )

        self.virtual_nav_arrow_actor = vtk.vtkActor2D()
        self.virtual_nav_arrow_actor.SetMapper(
            self.virtual_nav_arrow_mapper
        )

        arrow_property = (
            self.virtual_nav_arrow_actor.GetProperty()
        )
        arrow_property.SetColor(
            1.0,
            1.0,
            1.0,
        )
        arrow_property.SetOpacity(0.82)
        arrow_property.SetDisplayLocationToForeground()

        # 第一版三维箭头恢复后，不再显示后来增加的二维前景箭头。
        self.virtual_nav_arrow_actor.SetVisibility(
            False
        )


    def populate_nodule_table(self) -> None:
        self.nodule_table.setRowCount(
            len(self.nodule_candidates)
        )

        for row, candidate in enumerate(
            self.nodule_candidates
        ):
            values = [
                str(candidate.candidate_id),
                str(candidate.voxel_count),
                f"{candidate.volume_mm3:.2f}",
                f"{candidate.center_xyz_mm[0]:.1f}",
                (
                    f"{candidate.center_xyz_mm[1]:.1f} / "
                    f"{candidate.center_xyz_mm[2]:.1f}"
                ),
            ]

            for column, value in enumerate(
                values
            ):
                item = QTableWidgetItem(
                    value
                )
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

                if column == 0:
                    color = NODULE_COLORS[
                        (
                            candidate.candidate_id
                            - 1
                        )
                        % len(NODULE_COLORS)
                    ]
                    item.setForeground(
                        QColor(
                            int(color[0] * 255),
                            int(color[1] * 255),
                            int(color[2] * 255),
                        )
                    )

                self.nodule_table.setItem(
                    row,
                    column,
                    item,
                )

    def on_nodule_table_selection(
        self,
    ) -> None:
        selected_rows = (
            self.nodule_table
            .selectionModel()
            .selectedRows()
        )

        if not selected_rows:
            return

        candidate_id = (
            selected_rows[0].row() + 1
        )

        self.select_nodule_candidate(
            candidate_id,
            source="table",
        )

    def on_slice_voxel_clicked(
        self,
        x: int,
        y: int,
        z: int,
    ) -> None:
        if self.nodule_labels is None:
            return

        component_label = int(
            self.nodule_labels[z, y, x]
        )

        if component_label > 0:
            self.select_nodule_candidate(
                component_label,
                source="2d",
            )

    def on_3d_left_click(
        self,
        _caller,
        _event,
    ) -> None:
        picker = vtk.vtkPropPicker()

        click_x, click_y = (
            self.interactor_3d
            .GetEventPosition()
        )

        picker.Pick(
            click_x,
            click_y,
            0,
            self.renderer_3d,
        )

        actor = picker.GetActor()

        if actor is None:
            return

        candidate_id: int | None = None

        for current_id, candidate_actor in (
            self.nodule_actors.items()
        ):
            if actor == candidate_actor:
                candidate_id = current_id
                break

        if candidate_id is not None:
            self.select_nodule_candidate(
                candidate_id,
                source="3d",
            )

    def select_nodule_candidate(
        self,
        candidate_id: int,
        source: str,
    ) -> None:
        if not (
            1
            <= candidate_id
            <= len(self.nodule_candidates)
        ):
            return

        self.selected_candidate_id = (
            candidate_id
        )

        for candidate in self.nodule_candidates:
            actor = self.nodule_actors[
                candidate.candidate_id
            ]

            prop = actor.GetProperty()

            if (
                candidate.candidate_id
                == candidate_id
            ):
                prop.SetColor(
                    *SELECTED_NODULE_COLOR
                )
                prop.SetOpacity(1.0)
                prop.SetAmbient(0.45)
                prop.SetSpecular(0.55)
            else:
                color = NODULE_COLORS[
                    (
                        candidate.candidate_id
                        - 1
                    )
                    % len(NODULE_COLORS)
                ]
                prop.SetColor(*color)
                prop.SetOpacity(0.35)
                prop.SetAmbient(0.18)
                prop.SetSpecular(0.18)

        for slice_widget in [
            self.axial_widget,
            self.coronal_widget,
            self.sagittal_widget,
        ]:
            slice_widget.set_selected_component(
                candidate_id
            )

        current_row = (
            self.nodule_table.currentRow()
        )

        if current_row != candidate_id - 1:
            self.nodule_table.blockSignals(
                True
            )
            self.nodule_table.selectRow(
                candidate_id - 1
            )
            self.nodule_table.blockSignals(
                False
            )

        candidate = self.nodule_candidates[
            candidate_id - 1
        ]

        for slice_widget in [
            self.axial_widget,
            self.coronal_widget,
            self.sagittal_widget,
        ]:
            slice_widget.focus_zyx(
                candidate.center_zyx
            )

        self.statusBar().showMessage(
            f"已选择结节候选 {candidate_id}，"
            f"来源：{source}"
        )

        self.vtk_3d.GetRenderWindow().Render()

        if self.auto_replan_checkbox.isChecked():
            self.replan_selected_target()

    def candidate_mask(
        self,
        candidate_id: int,
    ) -> np.ndarray:
        assert self.nodule_labels is not None

        return (
            self.nodule_labels
            == candidate_id
        )

    def choose_target_nodes(
        self,
        target_mask: np.ndarray,
    ) -> tuple[list[int], np.ndarray]:
        assert self.centerline_coords is not None
        assert self.spacing_zyx is not None
        assert self.centerline_points is not None
        assert self.centerline_radii is not None

        distance = ndi.distance_transform_edt(
            ~target_mask,
            sampling=self.spacing_zyx,
        )

        values = distance[
            tuple(
                self.centerline_coords.T
            )
        ].astype(np.float64)

        nearest = float(values.min())

        eligible = np.flatnonzero(
            values
            <= min(
                nearest
                + self.target_distance_slack_mm,
                60.0,
            )
        )

        ranking = (
            values[eligible]
            - 0.20
            * self.centerline_radii[
                eligible
            ]
        )

        ordered = eligible[
            np.argsort(ranking)
        ]

        selected: list[int] = []

        for raw_node in ordered:
            node = int(raw_node)

            if not selected:
                selected.append(node)
            else:
                separation = np.linalg.norm(
                    self.centerline_points[
                        selected
                    ]
                    - self.centerline_points[
                        node
                    ],
                    axis=1,
                )

                if float(
                    separation.min()
                ) >= 4.0:
                    selected.append(node)

            if (
                len(selected)
                >= self.max_target_nodes
            ):
                break

        if not selected:
            selected = [
                int(np.argmin(values))
            ]

        return selected, values

    def prepare_baseline_planning_data(self) -> dict[str, Any] | None:
        if self.airway_baseline_mask is None or not self.airway_baseline_mask.any():
            return None
        if self.baseline_planning_data is not None:
            return self.baseline_planning_data
        assert self.ct_image is not None
        assert self.spacing_zyx is not None

        centerline_mask = skeletonize(
            self.airway_baseline_mask,
            method="lee",
        ).astype(bool)
        coords = np.argwhere(centerline_mask).astype(np.int32)
        if len(coords) == 0:
            return None
        points = physical_points(coords, self.ct_image)
        radius_map = ndi.distance_transform_edt(
            self.airway_baseline_mask,
            sampling=self.spacing_zyx,
        )
        radii = radius_map[tuple(coords.T)].astype(np.float64)
        adjacency, degrees = build_graph(
            coords,
            self.airway_baseline_mask.shape,
            self.spacing_zyx,
        )
        entry_node, entry_mode = choose_entry_node(
            coords,
            points,
            radii,
            self.entry_mask,
            self.spacing_zyx,
        )
        self.baseline_planning_data = {
            "coords": coords,
            "points": points,
            "radii": radii,
            "adjacency": adjacency,
            "degrees": degrees,
            "entry_node": entry_node,
            "entry_mode": entry_mode,
        }
        return self.baseline_planning_data

    def compare_with_baseline_route(
        self,
        target_mask: np.ndarray,
        device_diameter: float,
        device_margin: float,
    ) -> dict[str, Any] | None:
        data = self.prepare_baseline_planning_data()
        if data is None:
            return None
        assert self.spacing_zyx is not None

        coords = data["coords"]
        points = data["points"]
        radii = data["radii"]
        distance = ndi.distance_transform_edt(
            ~target_mask,
            sampling=self.spacing_zyx,
        )
        target_distances = distance[tuple(coords.T)].astype(np.float64)
        nearest = float(target_distances.min())
        eligible = np.flatnonzero(
            target_distances
            <= min(nearest + self.target_distance_slack_mm, 60.0)
        )
        ranking = target_distances[eligible] - 0.20 * radii[eligible]
        ordered = eligible[np.argsort(ranking)]
        target_nodes: list[int] = []
        for raw_node in ordered:
            node = int(raw_node)
            if not target_nodes:
                target_nodes.append(node)
            else:
                separation = np.linalg.norm(
                    points[target_nodes] - points[node],
                    axis=1,
                )
                if float(separation.min()) >= 4.0:
                    target_nodes.append(node)
            if len(target_nodes) >= self.max_target_nodes:
                break
        if not target_nodes:
            target_nodes = [int(np.argmin(target_distances))]

        proposals: list[dict[str, Any]] = []
        for target_node in target_nodes:
            for profile in profiles():
                try:
                    nodes, search_cost = stateful_astar(
                        start_node=data["entry_node"],
                        target_node=target_node,
                        adjacency=data["adjacency"],
                        coords_zyx=coords,
                        points_xyz=points,
                        radii_mm=radii,
                        degrees=data["degrees"],
                        spacing_zyx=self.spacing_zyx,
                        profile=profile,
                        device_diameter_mm=device_diameter,
                        device_margin_mm=device_margin,
                    )
                except RuntimeError:
                    continue
                node_array = np.asarray(nodes, dtype=np.int64)
                route_points = points[node_array]
                route_radii = radii[node_array]
                cumulative = cumulative_distance(route_points)
                turns = smoothed_turn_angles(route_points)
                target_distance = float(target_distances[target_node])
                proposals.append({
                    "score": float(search_cost + 2.0 * target_distance),
                    "profile_title": profile.title,
                    "route_length_mm": float(cumulative[-1]),
                    "minimum_diameter_mm": float(2.0 * route_radii.min()),
                    "maximum_turn_angle_deg": float(turns.max()),
                    "target_distance_mm": target_distance,
                })
        if not proposals:
            return {
                "available": True,
                "route_found": False,
                "reason": "原始气道在当前器械约束下没有可行路径",
            }
        proposals.sort(key=lambda item: (item["score"], item["route_length_mm"]))
        result = proposals[0]
        result["available"] = True
        result["route_found"] = True
        return result

    def replan_selected_target(self) -> None:
        if self.selected_candidate_id is None:
            QMessageBox.information(
                self,
                "尚未选择目标",
                "请先选择一个结节候选。",
            )
            return

        required_objects = [
            self.adjacency,
            self.centerline_coords,
            self.centerline_points,
            self.centerline_radii,
            self.degrees,
            self.spacing_zyx,
            self.junction_labels,
            self.segment_labels,
        ]

        if any(
            item is None
            for item in required_objects
        ):
            return

        assert self.entry_node is not None

        QApplication.setOverrideCursor(
            Qt.CursorShape.WaitCursor
        )

        try:
            self.statusBar().showMessage(
                f"正在为结节候选 "
                f"{self.selected_candidate_id} "
                f"重新规划……"
            )
            QApplication.processEvents()

            target_mask = self.candidate_mask(
                self.selected_candidate_id
            )

            target_nodes, target_distances = (
                self.choose_target_nodes(
                    target_mask
                )
            )

            device_diameter = (
                self.device_diameter_spin.value()
            )
            device_margin = (
                self.device_margin_spin.value()
            )

            proposals: list[
                RouteResult
            ] = []

            for target_node in target_nodes:
                for profile in profiles():
                    try:
                        nodes, search_cost = (
                            stateful_astar(
                                start_node=(
                                    self.entry_node
                                ),
                                target_node=(
                                    target_node
                                ),
                                adjacency=(
                                    self.adjacency
                                ),
                                coords_zyx=(
                                    self.centerline_coords
                                ),
                                points_xyz=(
                                    self.centerline_points
                                ),
                                radii_mm=(
                                    self.centerline_radii
                                ),
                                degrees=(
                                    self.degrees
                                ),
                                spacing_zyx=(
                                    self.spacing_zyx
                                ),
                                profile=profile,
                                device_diameter_mm=(
                                    device_diameter
                                ),
                                device_margin_mm=(
                                    device_margin
                                ),
                            )
                        )
                    except RuntimeError:
                        continue

                    node_array = np.asarray(
                        nodes,
                        dtype=np.int64,
                    )

                    route_coords = (
                        self.centerline_coords[
                            node_array
                        ]
                    )

                    route_points = (
                        self.centerline_points[
                            node_array
                        ]
                    )

                    route_radii = (
                        self.centerline_radii[
                            node_array
                        ]
                    )

                    route_diameters = (
                        2.0 * route_radii
                    )

                    cumulative = (
                        cumulative_distance(
                            route_points
                        )
                    )

                    turns = smoothed_turn_angles(
                        route_points
                    )

                    (
                        topology_tokens,
                        topology_signature,
                    ) = route_topology_signature(
                        nodes,
                        self.junction_labels,
                        self.segment_labels,
                    )

                    target_distance = float(
                        target_distances[
                            target_node
                        ]
                    )

                    minimum_radius = float(
                        route_radii.min()
                    )

                    required_radius = (
                        device_diameter / 2.0
                        + device_margin
                        if device_diameter > 0
                        else 0.0
                    )

                    # 统一排序：平衡搜索代价 + 靶点距离。
                    score = (
                        search_cost
                        + 2.0 * target_distance
                    )

                    metrics = {
                        "route_length_mm": float(
                            cumulative[-1]
                        ),
                        "minimum_radius_mm": (
                            minimum_radius
                        ),
                        "minimum_diameter_mm": (
                            2.0 * minimum_radius
                        ),
                        "mean_radius_mm": float(
                            route_radii.mean()
                        ),
                        "mean_diameter_mm": float(
                            route_diameters.mean()
                        ),
                        "maximum_turn_angle_deg": (
                            float(turns.max())
                        ),
                        "target_distance_mm": (
                            target_distance
                        ),
                        "required_radius_mm": (
                            required_radius
                        ),
                        "minimum_clearance_mm": (
                            minimum_radius
                            - required_radius
                            if device_diameter > 0
                            else None
                        ),
                        "device_passable": (
                            bool(
                                minimum_radius
                                >= required_radius
                            )
                            if device_diameter > 0
                            else True
                        ),
                    }

                    proposals.append(
                        RouteResult(
                            nodes=nodes,
                            coords_zyx=route_coords,
                            points_xyz_mm=route_points,
                            radii_mm=route_radii,
                            diameters_mm=(
                                route_diameters
                            ),
                            cumulative_mm=(
                                cumulative
                            ),
                            smooth_turn_angles_deg=(
                                turns
                            ),
                            topology_tokens=(
                                topology_tokens
                            ),
                            topology_signature=(
                                topology_signature
                            ),
                            target_distance_mm=(
                                target_distance
                            ),
                            target_candidate_id=(
                                self.selected_candidate_id
                            ),
                            profile_name=(
                                profile.name
                            ),
                            profile_title=(
                                profile.title
                            ),
                            score=score,
                            metrics=metrics,
                        )
                    )

            if not proposals:
                raise RuntimeError(
                    "当前器械直径约束下没有可行路径。"
                )

            proposals.sort(
                key=lambda item: (
                    item.score,
                    item.metrics[
                        "route_length_mm"
                    ],
                )
            )

            self.route = proposals[0]

            route_coords = tuple(self.route.coords_zyx.T)
            bridge_voxels_on_route = (
                int(np.count_nonzero(self.airway_added_mask[route_coords]))
                if self.airway_added_mask is not None
                else 0
            )
            recovered_voxels_on_route = (
                int(np.count_nonzero(self.airway_recovered_mask[route_coords]))
                if self.airway_recovered_mask is not None
                else 0
            )
            self.route.metrics["bridge_voxels_on_route"] = bridge_voxels_on_route
            self.route.metrics["recovered_voxels_on_route"] = recovered_voxels_on_route
            self.baseline_comparison = self.compare_with_baseline_route(
                target_mask,
                device_diameter,
                device_margin,
            )
            if self.baseline_comparison is not None:
                self.route.metrics["baseline_comparison"] = self.baseline_comparison

            self.update_route_scene()
            self.update_route_summary()
            self.save_current_plan()

            self.statusBar().showMessage(
                f"结节候选 "
                f"{self.selected_candidate_id} "
                f"重新规划完成"
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "重新规划失败",
                f"{type(error).__name__}: {error}",
            )
            self.statusBar().showMessage(
                "重新规划失败"
            )

        finally:
            QApplication.restoreOverrideCursor()

    def update_route_scene(self) -> None:
        assert self.route is not None
        assert self.airway_mask is not None

        if self.visual_localizer is not None:
            self.visual_localizer.set_route(self.route.points_xyz_mm)
            class_id = self.visual_localizer.route_start_class_id
            offset = self.visual_localizer.route_start_offset_mm
            self.visual_tracking_label.setText(
                f"视觉定位：地图入口 P{class_id:05d}，与规划入口相差 {offset:.1f} mm"
                if class_id is not None and offset is not None
                else "视觉定位：路线已绑定"
            )

        for actor in [
            self.route_actor,
            self.current_point_actor,
            self.current_point_halo_actor,
            self.map_direction_actor,
            self.bottleneck_actor,
        ]:
            if actor is not None:
                self.renderer_3d.RemoveActor(
                    actor
                )

        (
            self.route_actor,
            _route_polydata,
        ) = make_route_actor(
            self.route.points_xyz_mm,
            ROUTE_COLOR,
            radius_mm=0.85,
        )

        self.renderer_3d.AddActor(
            self.route_actor
        )

        # 气道地图中的实时位置：亮青色实心球。
        (
            self.current_point_actor,
            self.current_point_source,
        ) = make_sphere_actor(
            self.route.points_xyz_mm[0],
            radius_mm=2.35,
            color=CURRENT_POINT_COLOR,
        )

        self.current_point_actor.GetProperty().SetAmbient(
            0.65
        )
        self.current_point_actor.GetProperty().SetSpecular(
            0.85
        )
        self.current_point_actor.GetProperty().SetSpecularPower(
            55.0
        )

        self.renderer_3d.AddActor(
            self.current_point_actor
        )

        # 半透明光环增强位置可见性。
        (
            self.current_point_halo_actor,
            self.current_point_halo_source,
        ) = make_sphere_actor(
            self.route.points_xyz_mm[0],
            radius_mm=4.2,
            color=CURRENT_POINT_COLOR,
        )

        self.current_point_halo_actor.GetProperty().SetOpacity(
            0.18
        )
        self.current_point_halo_actor.GetProperty().SetAmbient(
            0.85
        )

        self.renderer_3d.AddActor(
            self.current_point_halo_actor
        )

        # 气道地图中的实时前进方向箭头。
        (
            self.map_direction_actor,
            self.map_direction_source,
        ) = make_arrow_actor(
            (1.0, 1.0, 1.0)
        )
        self.map_direction_actor.GetProperty().SetOpacity(
            0.95
        )
        self.renderer_3d.AddActor(
            self.map_direction_actor
        )

        bottleneck_index = int(
            np.argmin(
                self.route.radii_mm
            )
        )

        (
            self.bottleneck_actor,
            self.bottleneck_source,
        ) = make_sphere_actor(
            self.route.points_xyz_mm[
                bottleneck_index
            ],
            radius_mm=1.8,
            color=BOTTLENECK_COLOR,
        )

        self.renderer_3d.AddActor(
            self.bottleneck_actor
        )

        route_mask = np.zeros_like(
            self.airway_mask,
            dtype=bool,
        )

        route_mask[
            tuple(
                self.route.coords_zyx.T
            )
        ] = True

        route_display = (
            ndi.binary_dilation(
                route_mask,
                iterations=1,
            )
        )

        for slice_widget in [
            self.axial_widget,
            self.coronal_widget,
            self.sagittal_widget,
        ]:
            slice_widget.set_route(
                route_display
            )

        self.route_slider.setRange(
            0,
            len(self.route.nodes) - 1,
        )
        self.route_slider.setValue(0)

        self.renderer_3d.ResetCameraClippingRange()
        self.vtk_3d.GetRenderWindow().Render()

        self.update_synchronized_views(0)

    def current_case_id(self) -> str:
        if self.case_dir is None:
            return ""
        return self.case_dir.parent.name if self.case_dir.name == "stage4_package" else self.case_dir.name

    def load_visual_localization_model(self) -> None:
        if self.current_case_id() != self.visual_model_case_id:
            QMessageBox.warning(
                self, "模型与病例不匹配",
                f"当前模型只适用于 {self.visual_model_case_id}，当前病例为 {self.current_case_id() or '未加载'}。",
            )
            return
        if not self.visual_model_path.is_file():
            QMessageBox.warning(self, "模型不存在", f"找不到模型：\n{self.visual_model_path}")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from visual_localization_bridge import VisualLocalizationBridge
            self.visual_localizer = VisualLocalizationBridge(
                self.visual_model_path, self.current_case_id()
            )
            if self.route is not None:
                self.visual_localizer.set_route(self.route.points_xyz_mm)
                class_id = self.visual_localizer.route_start_class_id
                offset = self.visual_localizer.route_start_offset_mm
                message = (
                    f"视觉定位：已绑定路线，入口测试帧 P{class_id:05d}_v06，"
                    f"入口偏差 {offset:.1f} mm"
                    if class_id is not None and offset is not None
                    else "视觉定位：模型已加载，已绑定当前路线"
                )
            else:
                message = "视觉定位：模型已加载，请先选择结节并规划路线"
            self.visual_tracking_label.setText(message)
            self.test_visual_frame_button.setEnabled(True)
            self.reset_visual_tracking_button.setEnabled(True)
            self.statusBar().showMessage(message)
        except Exception as error:
            self.visual_localizer = None
            QMessageBox.critical(
                self, "视觉定位模型加载失败",
                f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}",
            )
        finally:
            QApplication.restoreOverrideCursor()

    def choose_visual_test_frame(self) -> None:
        initial_path = self.project_dir.parent
        if self.visual_localizer is not None and self.visual_localizer.route_start_class_id is not None:
            class_id = self.visual_localizer.route_start_class_id
            candidate = (
                self.project_dir.parent / "bronchopose_localization" / "data" / "generated"
                / "LIDC_0089_v1" / "images" / f"LIDC_0089_p{class_id:05d}_v06.png"
            )
            if candidate.is_file():
                initial_path = candidate
        path, _ = QFileDialog.getOpenFileName(
            self, "选择支气管镜测试帧", str(initial_path),
            "图像 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)",
        )
        if path:
            self.process_bronchoscope_frame(path)

    def reset_visual_tracking(self) -> None:
        if self.visual_localizer is None:
            return
        start = self.route.points_xyz_mm[0] if self.route is not None else None
        self.visual_localizer.reset(start)
        if self.route is not None:
            self.route_slider.setValue(0)
        class_id = self.visual_localizer.route_start_class_id
        self.visual_tracking_label.setText(
            f"视觉定位：已复位到地图入口 P{class_id:05d}"
            if class_id is not None else "视觉定位：已复位"
        )

    def process_bronchoscope_frame(self, frame_rgb):
        """Process one RGB frame supplied by a future camera/video adapter."""
        if self.visual_localizer is None:
            raise RuntimeError("视觉定位模型尚未加载")
        if self.route is None:
            raise RuntimeError("请先选择结节并完成路线规划")
        result = self.visual_localizer.update(frame_rgb)
        if result.accepted and result.route_index is not None:
            self.route_slider.setValue(int(result.route_index))
            jump_text = "起始帧" if result.jump_mm is None else f"移动 {result.jump_mm:.1f} mm"
            self.visual_tracking_label.setText(
                f"视觉定位：已跟踪 · 路线点 {result.route_index + 1} · "
                f"置信度 {result.confidence:.3f} · {jump_text}"
            )
        else:
            self.visual_tracking_label.setText(
                f"视觉定位：未更新（{result.status}，置信度 {result.confidence:.3f}）"
            )
            self.statusBar().showMessage("视觉定位不可靠，保持上一位置")
        return result


    def update_route_summary(self) -> None:
        assert self.route is not None

        metrics = self.route.metrics

        self.metric_labels[
            "target"
        ].setText(
            f"结节候选 "
            f"{self.route.target_candidate_id}"
        )

        self.metric_labels[
            "profile"
        ].setText(
            self.route.profile_title
        )

        self.metric_labels[
            "length"
        ].setText(
            f"{metrics['route_length_mm']:.2f} mm"
        )

        self.metric_labels[
            "min_diameter"
        ].setText(
            f"{metrics['minimum_diameter_mm']:.2f} mm"
        )

        self.metric_labels[
            "mean_diameter"
        ].setText(
            f"{metrics['mean_diameter_mm']:.2f} mm"
        )

        self.metric_labels[
            "max_turn"
        ].setText(
            f"{metrics['maximum_turn_angle_deg']:.2f}°"
        )

        self.metric_labels[
            "target_distance"
        ].setText(
            f"{metrics['target_distance_mm']:.2f} mm"
        )

        if (
            self.device_diameter_spin.value()
            > 0
        ):
            clearance = metrics[
                "minimum_clearance_mm"
            ]

            if metrics["device_passable"]:
                result_text = (
                    f"可通过；最小半径余量 "
                    f"{clearance:.2f} mm"
                )
            else:
                result_text = (
                    f"不可通过；最小半径余量 "
                    f"{clearance:.2f} mm"
                )
        else:
            result_text = "未启用硬约束"

        self.metric_labels[
            "device"
        ].setText(result_text)

        comparison = self.baseline_comparison
        if comparison is None:
            comparison_text = "无原始五折基线；需重新生成病例包"
        elif not comparison.get("route_found", False):
            comparison_text = (
                "原始气道无可行路径；优化气道已找到路径"
            )
        else:
            baseline_target = float(comparison["target_distance_mm"])
            optimized_target = float(metrics["target_distance_mm"])
            target_gain = baseline_target - optimized_target
            gain_word = "缩短" if target_gain >= 0 else "增加"
            comparison_text = (
                f"目标距离 {baseline_target:.2f} → {optimized_target:.2f} mm"
                f"（{gain_word} {abs(target_gain):.2f} mm）\n"
                f"路径长度 {comparison['route_length_mm']:.2f} → "
                f"{metrics['route_length_mm']:.2f} mm\n"
                f"最小直径 {comparison['minimum_diameter_mm']:.2f} → "
                f"{metrics['minimum_diameter_mm']:.2f} mm"
            )
        self.metric_labels["ab_comparison"].setText(comparison_text)

        bridge_count = int(metrics.get("bridge_voxels_on_route", 0))
        recovered_count = int(metrics.get("recovered_voxels_on_route", 0))
        if bridge_count > 0:
            dependency_text = (
                f"警告：路线经过 {bridge_count} 个桥接中心线点；"
                "需要逐层CT确认"
            )
        elif recovered_count > 0:
            dependency_text = (
                f"路线经过 {recovered_count} 个远端恢复中心线点；"
                "未经过人工桥接"
            )
        else:
            dependency_text = "路线完全位于原始五折气道内"
        self.metric_labels["repair_dependency"].setText(dependency_text)

        branch_lines = [
            "入口方式："
            + self.entry_mode,
            "",
            "完整分叉序列：",
            self.route.topology_signature,
            "",
            "标记说明：",
            "J = 分叉簇",
            "S = 两个分叉之间的连续气道段",
        ]

        self.branch_text.setPlainText(
            "\n".join(branch_lines)
        )

    def update_synchronized_views(
        self,
        route_index: int,
    ) -> None:
        if self.route is None:
            return

        route_index = int(
            np.clip(
                route_index,
                0,
                len(self.route.nodes) - 1,
            )
        )

        point_xyz = (
            self.route.points_xyz_mm[
                route_index
            ]
        )

        point_zyx = (
            self.route.coords_zyx[
                route_index
            ]
        )

        if self.current_point_source is not None:
            self.current_point_source.SetCenter(
                float(point_xyz[0]),
                float(point_xyz[1]),
                float(point_xyz[2]),
            )
            self.current_point_source.Modified()

        if self.current_point_halo_source is not None:
            self.current_point_halo_source.SetCenter(
                float(point_xyz[0]),
                float(point_xyz[1]),
                float(point_xyz[2]),
            )
            self.current_point_halo_source.Modified()

        tangent = self.route_tangent(
            route_index,
            lookahead=5,
        )

        if self.map_direction_actor is not None:
            set_arrow_pose(
                self.map_direction_actor,
                point_xyz,
                tangent,
                length_mm=10.0,
                thickness_scale=1.9,
            )

        for slice_widget in [
            self.axial_widget,
            self.coronal_widget,
            self.sagittal_widget,
        ]:
            slice_widget.set_current_point(
                point_zyx
            )

        self.axial_title.setText(
            f"轴位 · z={int(point_zyx[0])}"
        )
        self.coronal_title.setText(
            f"冠状位 · y={int(point_zyx[1])}"
        )
        self.sagittal_title.setText(
            f"矢状位 · x={int(point_zyx[2])}"
        )

        if self.follow_camera_checkbox.isChecked():
            self.update_virtual_camera(
                route_index
            )
        else:
            self.update_virtual_arrow_only(
                route_index
            )

        local_diameter = float(
            self.route.diameters_mm[
                route_index
            ]
        )

        local_radius = float(
            self.route.radii_mm[
                route_index
            ]
        )

        local_turn = float(
            self.route.smooth_turn_angles_deg[
                route_index
            ]
        )

        cumulative = float(
            self.route.cumulative_mm[
                route_index
            ]
        )

        required_radius = (
            self.device_diameter_spin.value()
            / 2.0
            + self.device_margin_spin.value()
            if self.device_diameter_spin.value()
            > 0
            else 0.0
        )

        clearance = (
            local_radius - required_radius
            if self.device_diameter_spin.value()
            > 0
            else None
        )

        node = self.route.nodes[
            route_index
        ]

        if (
            self.junction_labels is not None
            and self.segment_labels is not None
        ):
            junction_id = int(
                self.junction_labels[node]
            )
            segment_id = int(
                self.segment_labels[node]
            )

            if junction_id > 0:
                local_branch = (
                    f"分叉簇 J{junction_id}"
                )
            elif segment_id > 0:
                local_branch = (
                    f"气道段 S{segment_id}"
                )
            else:
                local_branch = "未分类节点"
        else:
            local_branch = "—"

        clearance_text = (
            f"{clearance:.2f} mm"
            if clearance is not None
            else "未启用"
        )

        self.current_local_label.setText(
            f"当前位置：{route_index + 1} / "
            f"{len(self.route.nodes)}\n"
            f"距入口：{cumulative:.2f} mm\n"
            f"局部气道直径：{local_diameter:.2f} mm\n"
            f"局部转角：{local_turn:.2f}°\n"
            f"器械半径余量：{clearance_text}\n"
            f"当前分支：{local_branch}"
        )

        self.route_position_label.setText(
            f"{route_index + 1} / "
            f"{len(self.route.nodes)}"
        )

        self.virtual_title.setText(
            f"虚拟支气管镜主视图 · "
            f"局部直径 {local_diameter:.2f} mm"
        )

        self.view_3d_label.setText(
            f"气道地图 · 实时位置 "
            f"{route_index + 1}/{len(self.route.nodes)}"
        )

        self.vtk_3d.GetRenderWindow().Render()
        self.vtk_virtual.GetRenderWindow().Render()

    def route_tangent(
        self,
        route_index: int,
        *,
        lookahead: int,
    ) -> np.ndarray:
        assert self.route is not None

        points = self.route.points_xyz_mm
        last_index = len(points) - 1

        current_index = int(
            np.clip(
                route_index,
                0,
                last_index,
            )
        )

        forward_index = min(
            current_index + lookahead,
            last_index,
        )

        if forward_index > current_index:
            tangent = (
                points[forward_index]
                - points[current_index]
            )
        else:
            previous_index = max(
                current_index - lookahead,
                0,
            )
            tangent = (
                points[current_index]
                - points[previous_index]
            )

        norm = float(
            np.linalg.norm(tangent)
        )

        if norm <= 1e-8:
            return np.asarray(
                [1.0, 0.0, 0.0],
                dtype=np.float64,
            )

        return tangent / norm



    def update_navigation_arrow_hud(
        self,
        route_index: int,
    ) -> None:
        """
        在虚拟支气管镜最上层显示一个标准导航箭头。

        箭头由明确的三角形单元组成，并放入独立的 VTK
        前景渲染层，避免 Windows/OpenGL 环境中消失。
        """
        if (
            self.route is None
            or self.virtual_nav_arrow_polydata is None
            or self.virtual_nav_arrow_actor is None
            or self.renderer_virtual_overlay is None
        ):
            return

        route_points = self.route.points_xyz_mm

        current_index = int(
            np.clip(
                route_index,
                0,
                len(route_points) - 1,
            )
        )

        if len(route_points) < 2:
            self.virtual_nav_arrow_actor.SetVisibility(
                False
            )
            return

        current_xyz = route_points[
            current_index
        ]

        forward_xyz = self.route_tangent(
            current_index,
            lookahead=5,
        )

        world_up = np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )

        if abs(
            float(
                np.dot(
                    forward_xyz,
                    world_up,
                )
            )
        ) > 0.92:
            world_up = np.asarray(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            )

        screen_right_xyz = np.cross(
            forward_xyz,
            world_up,
        )

        right_norm = float(
            np.linalg.norm(
                screen_right_xyz
            )
        )

        if right_norm <= 1e-8:
            screen_right_xyz = np.asarray(
                [1.0, 0.0, 0.0],
                dtype=np.float64,
            )
        else:
            screen_right_xyz /= right_norm

        future_index = min(
            len(route_points) - 1,
            current_index + 20,
        )

        if future_index > current_index:
            future_delta = (
                route_points[
                    future_index
                ]
                - current_xyz
            )
        else:
            previous_index = max(
                current_index - 5,
                0,
            )
            future_delta = (
                current_xyz
                - route_points[
                    previous_index
                ]
            )

        forward_component = max(
            float(
                np.dot(
                    future_delta,
                    forward_xyz,
                )
            ),
            1.0,
        )

        lateral_component = float(
            np.dot(
                future_delta,
                screen_right_xyz,
            )
        )

        turn_angle = math.atan2(
            lateral_component,
            forward_component,
        )

        turn_angle = float(
            np.clip(
                turn_angle,
                math.radians(-40.0),
                math.radians(40.0),
            )
        )

        render_size = (
            self.vtk_virtual
            .GetRenderWindow()
            .GetSize()
        )

        render_width = float(
            max(
                int(render_size[0]),
                int(self.vtk_virtual.width()),
                400,
            )
        )

        render_height = float(
            max(
                int(render_size[1]),
                int(self.vtk_virtual.height()),
                400,
            )
        )

        tail_center = np.asarray(
            [
                render_width * 0.50,
                render_height * 0.08,
            ],
            dtype=np.float64,
        )

        arrow_length = float(
            np.clip(
                render_height * 0.34,
                180.0,
                245.0,
            )
        )

        direction = np.asarray(
            [
                math.sin(
                    turn_angle
                ),
                math.cos(
                    turn_angle
                ),
            ],
            dtype=np.float64,
        )

        direction /= max(
            float(
                np.linalg.norm(
                    direction
                )
            ),
            1e-8,
        )

        normal = np.asarray(
            [
                -direction[1],
                direction[0],
            ],
            dtype=np.float64,
        )

        arrow_tip = (
            tail_center
            + direction
            * arrow_length
        )

        head_length = float(
            np.clip(
                arrow_length * 0.28,
                52.0,
                70.0,
            )
        )

        neck_center = (
            arrow_tip
            - direction
            * head_length
        )

        shaft_tail_half_width = float(
            np.clip(
                render_width * 0.011,
                10.0,
                15.0,
            )
        )

        shaft_neck_half_width = float(
            np.clip(
                render_width * 0.008,
                7.0,
                11.0,
            )
        )

        head_half_width = float(
            np.clip(
                render_width * 0.035,
                32.0,
                46.0,
            )
        )

        tail_left = (
            tail_center
            + normal
            * shaft_tail_half_width
        )
        tail_right = (
            tail_center
            - normal
            * shaft_tail_half_width
        )

        neck_left = (
            neck_center
            + normal
            * shaft_neck_half_width
        )
        neck_right = (
            neck_center
            - normal
            * shaft_neck_half_width
        )

        head_left = (
            neck_center
            + normal
            * head_half_width
        )
        head_right = (
            neck_center
            - normal
            * head_half_width
        )

        point_values = [
            tail_left,   # 0
            neck_left,   # 1
            head_left,   # 2
            arrow_tip,   # 3
            head_right,  # 4
            neck_right,  # 5
            tail_right,  # 6
        ]

        vtk_points = vtk.vtkPoints()

        for point in point_values:
            vtk_points.InsertNextPoint(
                float(point[0]),
                float(point[1]),
                0.0,
            )

        triangle_cells = vtk.vtkCellArray()

        # 显式使用逆时针三角形，避免多边形朝向与三角化问题。
        triangle_indices = [
            (0, 6, 5),
            (0, 5, 1),
            (2, 4, 3),
        ]

        for ids in triangle_indices:
            triangle = vtk.vtkTriangle()

            for local_id, point_id in enumerate(
                ids
            ):
                triangle.GetPointIds().SetId(
                    local_id,
                    point_id,
                )

            triangle_cells.InsertNextCell(
                triangle
            )

        self.virtual_nav_arrow_polydata.SetPoints(
            vtk_points
        )
        self.virtual_nav_arrow_polydata.SetPolys(
            triangle_cells
        )
        self.virtual_nav_arrow_polydata.Modified()

        arrow_property = (
            self.virtual_nav_arrow_actor
            .GetProperty()
        )

        arrow_property.SetColor(
            1.0,
            1.0,
            1.0,
        )
        arrow_property.SetOpacity(
            0.90
        )
        arrow_property.SetDisplayLocationToForeground()

        self.virtual_nav_arrow_actor.SetVisibility(
            True
        )
        self.virtual_nav_arrow_actor.Modified()

    def update_virtual_camera(
        self,
        route_index: int,
    ) -> None:
        if self.route is None:
            return

        points = self.route.points_xyz_mm
        last_index = len(points) - 1

        current_index = int(
            np.clip(
                route_index,
                0,
                last_index,
            )
        )

        tangent = self.route_tangent(
            current_index,
            lookahead=6,
        )

        global_up = np.asarray(
            [0.0, 0.0, 1.0]
        )

        if abs(
            float(
                np.dot(
                    tangent,
                    global_up,
                )
            )
        ) > 0.92:
            global_up = np.asarray(
                [0.0, 1.0, 0.0]
            )

        right = np.cross(
            tangent,
            global_up,
        )

        right_norm = float(
            np.linalg.norm(right)
        )

        if right_norm > 1e-8:
            right /= right_norm

        view_up = np.cross(
            right,
            tangent,
        )

        camera = (
            self.renderer_virtual
            .GetActiveCamera()
        )

        camera.SetPosition(
            *points[current_index]
        )

        focal_point = (
            points[current_index]
            + 20.0 * tangent
        )

        camera.SetFocalPoint(
            *focal_point
        )
        camera.SetViewUp(
            *view_up
        )
        camera.SetViewAngle(78.0)
        self.update_virtual_arrow_only(
            current_index
        )

        self.renderer_virtual.ResetCameraClippingRange()
        camera.SetClippingRange(
            0.10,
            220.0,
        )

    def update_virtual_arrow_only(
        self,
        route_index: int,
    ) -> None:
        """
        更新第一版腔内三维箭头。

        箭头放在当前镜头前方，并指向后续路径点：
        - 使用真实三维 vtkArrowSource；
        - 没有文字；
        - 没有背景框；
        - 没有二维覆盖路线；
        - 箭头会随规划路径转向；
        - 在直行段可能看到箭头头部的正面，这是正常的三维透视。
        """
        if (
            self.route is None
            or self.virtual_direction_actor is None
            or self.virtual_direction_source is None
        ):
            return

        route_points = (
            self.route.points_xyz_mm
        )

        current_index = int(
            np.clip(
                route_index,
                0,
                len(route_points) - 1,
            )
        )

        current_point = route_points[
            current_index
        ]

        current_tangent = (
            self.route_tangent(
                current_index,
                lookahead=5,
            )
        )

        # 按实际路径距离寻找前方约 18 mm 的目标点。
        cumulative_mm = (
            self.route.cumulative_mm
        )

        current_distance_mm = float(
            cumulative_mm[current_index]
        )

        desired_lookahead_mm = 18.0

        guide_index = int(
            np.searchsorted(
                cumulative_mm,
                current_distance_mm
                + desired_lookahead_mm,
                side="left",
            )
        )

        guide_index = int(
            np.clip(
                guide_index,
                current_index + 1,
                len(route_points) - 1,
            )
        )

        guide_vector = (
            route_points[guide_index]
            - current_point
        )

        guide_norm = float(
            np.linalg.norm(
                guide_vector
            )
        )

        if guide_norm <= 1e-8:
            guide_direction = (
                current_tangent
            )
        else:
            guide_direction = (
                guide_vector
                / guide_norm
            )

        local_diameter_mm = float(
            self.route.diameters_mm[
                current_index
            ]
        )

        # 放在镜头前方，避免箭头与相机原点重合。
        start_offset_mm = float(
            np.clip(
                4.0
                + 0.08
                * local_diameter_mm,
                4.0,
                6.0,
            )
        )

        arrow_start = (
            current_point
            + current_tangent
            * start_offset_mm
        )

        # 第一版箭头较粗、较亮，长度随局部管径轻微变化。
        arrow_length_mm = float(
            np.clip(
                10.5
                + 0.42
                * local_diameter_mm,
                12.0,
                17.0,
            )
        )

        arrow_thickness = float(
            np.clip(
                2.7
                + 0.07
                * local_diameter_mm,
                2.7,
                3.8,
            )
        )

        set_arrow_pose(
            self.virtual_direction_actor,
            arrow_start,
            guide_direction,
            length_mm=arrow_length_mm,
            thickness_scale=arrow_thickness,
        )

        self.virtual_direction_actor.SetVisibility(
            True
        )

        # 明确关闭所有后来增加的二维箭头。
        if (
            self.virtual_nav_arrow_actor
            is not None
        ):
            self.virtual_nav_arrow_actor.SetVisibility(
                False
            )

        if (
            self.navigation_arrow_overlay
            is not None
        ):
            self.navigation_arrow_overlay.hide()

    def change_playback_speed(
        self,
        _index: int,
    ) -> None:
        interval = self.speed_combo.currentData()

        if interval is None:
            interval = 90

        self.timer.setInterval(
            int(interval)
        )

    def jump_to_route_start(self) -> None:
        if self.route is None:
            return

        self.timer.stop()
        self.play_button.setText("播放")
        self.route_slider.setValue(0)

    def jump_to_route_end(self) -> None:
        if self.route is None:
            return

        self.timer.stop()
        self.play_button.setText("播放")
        self.route_slider.setValue(
            self.route_slider.maximum()
        )

    def step_route(
        self,
        step: int,
    ) -> None:
        if self.route is None:
            return

        target = int(
            np.clip(
                self.route_slider.value()
                + int(step),
                self.route_slider.minimum(),
                self.route_slider.maximum(),
            )
        )

        self.route_slider.setValue(
            target
        )

    def toggle_playback(self) -> None:
        if self.route is None:
            return

        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText(
                "播放"
            )
        else:
            if (
                self.route_slider.value()
                >= self.route_slider.maximum()
            ):
                self.route_slider.setValue(0)

            self.change_playback_speed(
                self.speed_combo.currentIndex()
            )
            self.timer.start()
            self.play_button.setText(
                "暂停"
            )

    def advance_route(self) -> None:
        if self.route is None:
            self.timer.stop()
            return

        next_value = (
            self.route_slider.value()
            + 1
        )

        if next_value > self.route_slider.maximum():
            if self.loop_checkbox.isChecked():
                self.route_slider.setValue(0)
                return

            self.timer.stop()
            self.play_button.setText(
                "播放"
            )
            return

        self.route_slider.setValue(
            next_value
        )


    def set_isometric_view(self) -> None:
        self.renderer_3d.ResetCamera()
        camera = (
            self.renderer_3d
            .GetActiveCamera()
        )
        camera.Azimuth(35.0)
        camera.Elevation(25.0)
        camera.OrthogonalizeViewUp()
        self.renderer_3d.ResetCameraClippingRange()
        self.vtk_3d.GetRenderWindow().Render()

    def reset_3d_camera(self) -> None:
        self.renderer_3d.ResetCamera()
        self.renderer_3d.ResetCameraClippingRange()
        self.vtk_3d.GetRenderWindow().Render()

    def set_3d_camera(
        self,
        direction: str,
    ) -> None:
        bounds = (
            self.renderer_3d
            .ComputeVisiblePropBounds()
        )

        center = np.asarray(
            [
                (
                    bounds[0]
                    + bounds[1]
                ) / 2.0,
                (
                    bounds[2]
                    + bounds[3]
                ) / 2.0,
                (
                    bounds[4]
                    + bounds[5]
                ) / 2.0,
            ]
        )

        size = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
            1.0,
        )

        mapping = {
            "front": (
                center
                + np.asarray(
                    [0.0, -3.0 * size, 0.0]
                ),
                (0.0, 0.0, 1.0),
            ),
            "back": (
                center
                + np.asarray(
                    [0.0, 3.0 * size, 0.0]
                ),
                (0.0, 0.0, 1.0),
            ),
            "left": (
                center
                + np.asarray(
                    [-3.0 * size, 0.0, 0.0]
                ),
                (0.0, 0.0, 1.0),
            ),
            "right": (
                center
                + np.asarray(
                    [3.0 * size, 0.0, 0.0]
                ),
                (0.0, 0.0, 1.0),
            ),
            "top": (
                center
                + np.asarray(
                    [0.0, 0.0, 3.0 * size]
                ),
                (0.0, 1.0, 0.0),
            ),
        }

        position, view_up = mapping[
            direction
        ]

        camera = (
            self.renderer_3d
            .GetActiveCamera()
        )
        camera.SetFocalPoint(*center)
        camera.SetPosition(*position)
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()

        self.renderer_3d.ResetCameraClippingRange()
        self.vtk_3d.GetRenderWindow().Render()

    def save_current_plan(self) -> None:
        if (
            self.case_dir is None
            or self.route is None
        ):
            return

        output_dir = (
            self.case_dir
            / "interactive_plans"
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidate_id = (
            self.route.target_candidate_id
        )

        json_path = (
            output_dir
            / f"nodule_{candidate_id:02d}_plan.json"
        )

        csv_path = (
            output_dir
            / f"nodule_{candidate_id:02d}_route.csv"
        )

        payload = {
            "target_candidate_id": (
                candidate_id
            ),
            "profile": (
                self.route.profile_name
            ),
            "profile_title": (
                self.route.profile_title
            ),
            "topology_signature": (
                self.route.topology_signature
            ),
            "metrics": (
                self.route.metrics
            ),
            "entry_mode": self.entry_mode,
            "device_diameter_mm": (
                self.device_diameter_spin.value()
            ),
            "device_margin_mm": (
                self.device_margin_spin.value()
            ),
        }

        json_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        with csv_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "index",
                    "x_mm",
                    "y_mm",
                    "z_mm",
                    "cumulative_mm",
                    "radius_mm",
                    "diameter_mm",
                    "turn_angle_deg",
                ]
            )

            for index, point in enumerate(
                self.route.points_xyz_mm
            ):
                writer.writerow(
                    [
                        index,
                        f"{point[0]:.6f}",
                        f"{point[1]:.6f}",
                        f"{point[2]:.6f}",
                        f"{self.route.cumulative_mm[index]:.6f}",
                        f"{self.route.radii_mm[index]:.6f}",
                        f"{self.route.diameters_mm[index]:.6f}",
                        f"{self.route.smooth_turn_angles_deg[index]:.6f}",
                    ]
                )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.timer.stop()
            self.vtk_3d.Finalize()
            self.vtk_virtual.Finalize()
        except Exception:
            pass

        super().closeEvent(event)


def main() -> None:
    args = parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName(
        "InteractiveNavigationStage4"
    )

    window = InteractiveNavigationWindow(
        initial_case_dir=args.case_dir,
        min_nodule_voxels=(
            args.min_nodule_voxels
        ),
        max_target_nodes=(
            args.max_target_nodes
        ),
        target_distance_slack_mm=(
            args.target_distance_slack_mm
        ),
    )

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
