"""气道表面网格提取。

从二值气道掩膜生成三角网格，供三维 viewer 使用。

设计要点
--------
1. **裁剪**：气道体素在 512×512×147 的 CT 里只占很小一块，
   先按包围盒 +padding 裁掉空白，marching cubes 的开销能降一个数量级。
2. **降采样**：viewer 只做展示，步长 2 时三角形数约为 1/4，
   视觉上几乎无损，但 HTML 体积小很多。
3. **量化 + base64**：顶点坐标压到 int16 网格单位，法线压到 int8。
   直接写 JSON 浮点数体积是量化后的 6~8 倍。
4. **坐标变换**：输出统一的「世界坐标」——医学 LPS 转 Three.js 的 Y-up 右手系，
   即 (x, y, z)_LPS -> (x, z, -y)。
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

# viewer 默认降采样步长。1 = 原始分辨率，2 = 每两格取一格。
DEFAULT_STEP = 2
# 裁剪时在包围盒外扩的体素数量，避免把靠近边界的表面切平。
CROP_PADDING = 2


@dataclass
class SurfaceMesh:
    """一个三角网格，顶点为世界坐标（mm）。"""

    vertices: np.ndarray  # (N, 3) float32，世界坐标 mm
    faces: np.ndarray  # (M, 3) int32
    normals: np.ndarray  # (N, 3) float32，单位向量

    @property
    def triangle_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


def lps_to_world(points_xyz: np.ndarray) -> np.ndarray:
    """医学 LPS 坐标 -> Three.js 世界坐标（Y 轴向上）。

    LPS 的 z 是「头->脚」方向（上方为负），取反后 y 轴向上，符合直觉。
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = pts[:, 2]
    out[:, 2] = -pts[:, 1]
    return out


def voxel_to_world_zyx(
    coords_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_xyz: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """体素索引 (z,y,x) -> 世界坐标 (X, Z, -Y)，直接给 Three.js 用。"""
    coords = np.asarray(coords_zyx, dtype=np.float64)
    spacing_xyz = np.asarray(spacing_zyx, dtype=np.float64)[::-1]
    indices_xyz = coords[:, ::-1]
    physical = np.asarray(origin_xyz, dtype=np.float64) + (
        np.asarray(direction, dtype=np.float64).reshape(3, 3)
        @ (indices_xyz * spacing_xyz).T
    ).T
    return lps_to_world(physical)


def _crop_bounds(mask: np.ndarray, padding: int = CROP_PADDING) -> tuple[slice, ...]:
    """返回把非零体素包住的最小切片（外扩 padding）。"""
    filled = np.argwhere(mask)
    if filled.size == 0:
        return tuple(slice(0, size) for size in mask.shape)
    lower = np.maximum(filled.min(axis=0) - padding, 0)
    upper = np.minimum(filled.max(axis=0) + 1 + padding, np.asarray(mask.shape))
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def extract_surface(
    mask: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_xyz: np.ndarray,
    direction: np.ndarray,
    step: int = DEFAULT_STEP,
    level: float = 0.5,
    smoothing_mm: float = 0.0,
) -> SurfaceMesh:
    """从二值掩膜提取表面网格。

    step > 1 时先对掩膜做块状降采样（max 池化），保住细支气管不被吃掉。
    smoothing_mm 是在**物理空间**下的高斯平滑尺度，按体素间距折算成各向异性的
    sigma —— CT 的层厚通常是层内间距的 3 倍（这里是 2.5 mm vs 0.82 mm），
    二值掩膜的等值面在层向上是 2.5 mm 一跳，直接提取会得到很明显的「梯田」
    条纹。等物理尺度的模糊正好把这些条纹抹平，而不会把层内细节一起糊掉。
    """
    volume = np.asarray(mask).astype(np.uint8)
    bounds = _crop_bounds(volume)
    volume = volume[bounds]

    # 裁剪后每个轴的实际起始体素索引
    starts = np.array([b.start for b in bounds], dtype=np.float64)

    if step > 1:
        # 用 maximum 池化而不是抽样：细支气管只有 1~2 体素粗，
        # 直接抽样会让它们整段消失。
        target = tuple(
            int(np.ceil(size / step)) * step for size in volume.shape
        )
        padded = np.zeros(target, dtype=np.uint8)
        padded[: volume.shape[0], : volume.shape[1], : volume.shape[2]] = volume
        trimmed = padded.reshape(
            target[0] // step, step, target[1] // step, step, target[2] // step, step
        )
        volume = trimmed.max(axis=(1, 3, 5))

    volume = volume.astype(np.float32)
    spacing = np.asarray(spacing_zyx, dtype=np.float64) * step
    if smoothing_mm > 0 and float(volume.max()) > 0:
        sigma = tuple(float(smoothing_mm / s) for s in spacing)
        smoothed = ndi.gaussian_filter(volume, sigma=sigma, mode="nearest")
        # 平滑会压低峰值。体积很小的掩膜（比如只有几十个体素的修复段）
        # 峰值可能被压到阈值以下，导致等值面整体消失。
        # 把峰值归一化回原值：等值面仍然落在「半密度」处，语义不变。
        peak = float(smoothed.max())
        if peak > 0:
            volume = smoothed * (float(volume.max()) / peak)

    if volume.max() <= level:
        raise ValueError("气道掩膜裁剪后没有足够体素，无法生成表面")

    vertices_zyx, faces, normals_zyx, _ = measure.marching_cubes(
        volume, level=level, spacing=spacing
    )

    # marching_cubes 的输出原点在裁剪块的第一格，换算回原始体素索引
    voxel_zyx = vertices_zyx / spacing + starts
    world = voxel_to_world_zyx(voxel_zyx, spacing_zyx, origin_xyz, direction)

    # 法线跟着做同一套旋转（仅取方向部分，无平移）
    normals_world = lps_to_world(normals_zyx)

    return SurfaceMesh(
        vertices=world.astype(np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        normals=normals_world.astype(np.float32),
    )


# ------------------------------------------------------------------ 量化


def _quantize(values: np.ndarray, fixed_scale: float | None = None):
    """把浮点数组线性映射到 int16（或 int8）整数域。

    返回 (整数数组, 最小值, 缩放系数)。定点缩放让还原时无需再存 min/max。
    """
    arr = np.asarray(values, dtype=np.float64)
    span = float(arr.max() - arr.min())
    scale = fixed_scale if fixed_scale else (span / 65534.0 if span > 0 else 1.0)
    if scale <= 0:
        scale = 1.0
    quantized = np.round((arr - arr.min()) / scale).astype(np.int32)
    quantized = np.clip(quantized, -32768, 32767).astype(np.int16)
    return quantized, float(arr.min()), scale


def encode_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> dict[str, object]:
    """把网格编码成可嵌进 HTML 的紧凑字典。

    顶点走「相对首顶点的 int16 量化」+ base64；法线压到 int8。
    相比直接 dump JSON 浮点，体积大约降到 1/7。
    """
    verts = np.asarray(vertices, dtype=np.float64)
    origin = verts.min(axis=0)
    span = float((verts - origin).max())
    scale = span / 65534.0 if span > 0 else 1.0

    vq = np.round((verts - origin) / scale).astype(np.int32)
    vq = np.clip(vq, 0, 65535).astype(np.uint16)

    nq = np.round(np.clip(np.asarray(normals, dtype=np.float64), -1, 1) * 127).astype(
        np.int8
    )

    return {
        "vertexOrigin": [round(float(v), 4) for v in origin],
        "vertexScale": round(float(scale), 6),
        "vertexCount": int(verts.shape[0]),
        "faceCount": int(faces.shape[0]),
        # uint16 小端，每顶点 3 个分量
        "vertices": base64.b64encode(vq.astype("<u2").tobytes()).decode("ascii"),
        # int8，每顶点 3 个分量
        "normals": base64.b64encode(nq.tobytes()).decode("ascii"),
        # uint32 小端，每三角形 3 个索引
        "faces": base64.b64encode(
            np.asarray(faces, dtype="<u4").tobytes()
        ).decode("ascii"),
    }


__all__ = [
    "DEFAULT_STEP",
    "SurfaceMesh",
    "encode_geometry",
    "extract_surface",
    "lps_to_world",
    "voxel_to_world_zyx",
]
