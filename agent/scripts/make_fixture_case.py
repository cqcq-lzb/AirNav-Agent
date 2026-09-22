"""生成**合成病例夹具**：让门禁与评测在没有真实病例的机器上也能真跑。

## 为什么需要这个

`cases/`（真实病例，1.9GB）按隐私与体积不入库。于是任何全新 clone 上，
依赖病例数据的 5 项门禁（`planner` / `loop` / `payloads` / `phi` / `eval`）全都跑不了 ——
**其中 `eval` 是硬门禁**。也就是说：这套项目里最值钱的资产（30 条用例 + 区分度实验），
恰恰是**最不可被外人验证**的那一件。

本脚本生成一个**几何合法、体积极小、不含任何患者信息**的病例，让这些检查在
干净环境里有东西可跑。

## 设计约束（不是随便造一个）

夹具必须满足**已经被写死在各处自检里的期望**，否则「能跑」就变成了「换个法子跳过」。
逐条来自源码（改了这里就得同步改那边）：

| 约束 | 值 | 出处 |
|---|---|---|
| 候选数 | 恰好 4 个（`[1,2,3,4]`） | `verify_payloads.layer_5` 断言 `valid_candidate_id_range == [1,4]` |
| 候选 3 在 1.5mm / 2.0mm 下可规划 | ✅ | `LEAKY_TOOLS`、`layer_4` 都用 1.5mm；`layer_4` 还要 2.0mm 换文件 |
| 候选 3 在 6.0mm 下失败 | ✅ 且归因为 `device_too_thick` | `verify_payloads.layer_5` |
| 候选 3 的最大可通过外径 | 落在 (2.0, 6.0) 内 | 同上（`bound < 6.0` 且 2.0mm 必须能过） |
| 4 个候选在 0.1mm 下都有路 | ✅ | `verify_payloads.layer_5` ③ 的注释明写了这条前提 |
| 候选 4 靶距落在 marginal 带 | 30 < d ≤ 50 | `cases.E15` 判据要「复核」二字，而 `planner.classify_reachability` 只有 marginal 档的文案含「人工复核」 |
| 候选编号 = 连通域升序 | 按 z 递增控制 | `case_loader._extract_nodule_candidates` 用 `ndi.label`，标签按扫描序 |
| 服务端编号 | 与客户端不同（更严格） | `cases.E02` 考的就是两套编号的对应关系 |

## 几何怎么造

一根主干在 z=... 处依次分叉成 4 条末端枝，每条末端外放一个结节球。
候选 3 那条路的中段做一个**收窄段**（半径 2.2mm → 直径 4.4mm → 最大器械 4.0mm），
于是 `device_too_thick` 与二分搜索都有真实的几何含义，而不是一个凑出来的数。

⚠️ 收窄段两侧的粗管必须**离得足够远**（> 粗管半径），否则两个粗管的端帽会
把细腰填实，收窄段就白做了。本脚本里间距取 18.5mm，粗管半径 4.5mm。

用法：
    python -m agent.scripts.make_fixture_case                # 生成到 fixtures/cases
    python -m agent.scripts.make_fixture_case --report        # 生成后打印派生事实
    python -m agent.scripts.make_fixture_case --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CASE_ID = "LIDC_0089"          # ⚠️ 刻意沿用真病例名：各处自检都写死了它
DEFAULT_OUT = ROOT / "fixtures" / "cases"

# 体素间距 1mm 各向同性；体积 128×128×96（z,y,x）≈ 157 万体素
SHAPE_ZYX = (128, 128, 96)
SPACING_XYZ = (1.0, 1.0, 1.0)

# 气道：一串 (起点 zyx, 终点 zyx, 半径 mm) 的圆管
# 主干 -> J1 分成两支 -> 各再分成两枝 = 4 条末端
TUBES: tuple[tuple[tuple[float, float, float], tuple[float, float, float], float], ...] = (
    # 主干（最粗 + 顶端做入口）
    ((6, 64, 48), (40, 64, 48), 6.0),
    # J1 第一支：到 J2
    ((40, 64, 48), (70, 44, 40), 4.5),
    # J2 的两枝
    ((70, 44, 40), (92, 30, 30), 3.0),
    ((70, 44, 40), (92, 44, 62), 2.6),
    # J1 第二支：粗管 -> **收窄段** -> J3
    # 收窄段两端离粗管端点 18.5mm（>> 4.5mm 半径），细腰才不会被端帽填实
    ((40, 64, 48), (55, 74, 44), 4.5),
    ((55, 74, 44), (70, 84, 40), 2.2),      # ← 细腰：直径 4.4mm，最大器械 4.0mm
    # J3 的两枝
    ((70, 84, 40), (92, 98, 30), 3.0),
    ((70, 84, 40), (92, 84, 62), 2.6),
)

# 入口点：主干顶端
ENTRY_CENTER_ZYX = (6.0, 64.0, 48.0)
ENTRY_RADIUS_MM = 3.0

# 结节候选：4 个椭球。**z 递增决定候选编号**（连通域按扫描序标号）。
# 靶距（到中心线的最近距离，mm）按可达性带设计：
#   N1 ~6   -> adjacent      (≤15)
#   N2 ~12  -> adjacent      (≤15)
#   N3 ~20  -> reachable     (≤30)
#   N4 ~41  -> marginal      (≤50)  ← E15 要的「复核」只在这个档的文案里
NODULES: tuple[tuple[float, float, float, tuple[float, float, float]], ...] = (
    (97.0, 27.0, 28.0, (2.6, 2.6, 2.6)),    # 候选 1：S3 末端外侧
    (104.0, 101.0, 29.0, (2.4, 2.4, 2.4)),   # 候选 2：S5 末端外侧
    (111.0, 87.0, 68.0, (3.0, 3.0, 3.0)),    # 候选 3：S6 末端外侧，且路过细腰
    (122.0, 44.0, 90.0, (2.5, 2.5, 2.5)),    # 候选 4：S4 延长线上，靶距 ~41mm
)


# ------------------------------------------------------------------ 几何原语


def _segment_distance(
    shape_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    """点集到线段的最短距离 < radius 的体素掩膜（只在包围盒里算）。"""
    sp = np.asarray(spacing_xyz, dtype=np.float64)
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)

    lo = np.floor(np.minimum(a, b) - radius_mm).astype(int)
    hi = np.ceil(np.maximum(a, b) + radius_mm).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(shape_zyx) - 1)
    if np.any(hi < lo):
        return np.zeros(shape_zyx, dtype=bool)

    zz, yy, xx = np.meshgrid(
        np.arange(lo[0], hi[0] + 1),
        np.arange(lo[1], hi[1] + 1),
        np.arange(lo[2], hi[2] + 1),
        indexing="ij",
    )
    pts = np.stack([zz, yy, xx], axis=-1).astype(np.float64) * sp  # 物理坐标 (z,y,x)·mm
    seg = b - a
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 <= 0:
        dist = np.linalg.norm(pts - a, axis=-1)
    else:
        t = np.clip(((pts - a) @ seg) / seg_len2, 0.0, 1.0)
        proj = a + t[..., None] * seg
        dist = np.linalg.norm(pts - proj, axis=-1)

    out = np.zeros(shape_zyx, dtype=bool)
    out[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = dist < radius_mm
    return out


def _ellipsoid(
    shape_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    center_zyx: tuple[float, float, float],
    radii_zyx_mm: tuple[float, float, float],
) -> np.ndarray:
    """椭球掩膜（同样只在包围盒里算）。"""
    sp = np.asarray(spacing_xyz, dtype=np.float64)
    c = np.asarray(center_zyx, dtype=np.float64)
    r = np.asarray(radii_zyx_mm, dtype=np.float64)

    lo = np.maximum(np.floor(c - r).astype(int), 0)
    hi = np.minimum(np.ceil(c + r).astype(int), np.asarray(shape_zyx) - 1)
    zz, yy, xx = np.meshgrid(
        np.arange(lo[0], hi[0] + 1),
        np.arange(lo[1], hi[1] + 1),
        np.arange(lo[2], hi[2] + 1),
        indexing="ij",
    )
    pts = np.stack([zz, yy, xx], axis=-1).astype(np.float64) * sp
    d = ((pts - c) / r) ** 2
    out = np.zeros(shape_zyx, dtype=bool)
    out[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = d.sum(axis=-1) <= 1.0
    return out


# ------------------------------------------------------------------ 组装


def build_arrays() -> dict[str, np.ndarray]:
    airway = np.zeros(SHAPE_ZYX, dtype=bool)
    for p0, p1, radius in TUBES:
        airway |= _segment_distance(SHAPE_ZYX, SPACING_XYZ, p0, p1, radius)

    nodule = np.zeros(SHAPE_ZYX, dtype=bool)
    for center, radii in [(n[0:3], n[3]) for n in NODULES]:
        nodule |= _ellipsoid(SHAPE_ZYX, SPACING_XYZ, center, radii)

    entry = _ellipsoid(SHAPE_ZYX, SPACING_XYZ, ENTRY_CENTER_ZYX,
                       (ENTRY_RADIUS_MM,) * 3)

    # CT 只用来提供 spacing/origin（规划器只用它做物理坐标换算，不读 HU），
    # 但给一个有区分度的填充值，免得渲染出来是一片死黑。
    ct = np.full(SHAPE_ZYX, -900, dtype=np.int16)
    ct[airway] = -100
    ct[nodule] = 40
    return {"airway": airway, "nodule": nodule, "entry": entry, "ct": ct}


def _write(name: str, array: np.ndarray, out_dir: Path, dtype) -> Path:
    """写成 NIfTI，几何（尺寸/间距/原点/方向）四张图完全一致。"""
    image = sitk.GetImageFromArray(np.ascontiguousarray(array.astype(dtype)))
    image.SetSpacing(SPACING_XYZ)
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    path = out_dir / name
    sitk.WriteImage(image, str(path))
    return path


def write_case(out_root: Path) -> Path:
    pkg = out_root / CASE_ID / "stage4_package"
    pkg.mkdir(parents=True, exist_ok=True)

    arrays = build_arrays()
    _write("ct.nii.gz", arrays["ct"], pkg, np.int16)
    _write("airway_mask.nii.gz", arrays["airway"], pkg, np.uint8)
    _write("nodule_raw.nii.gz", arrays["nodule"], pkg, np.uint8)
    _write("entry_point.nii.gz", arrays["entry"], pkg, np.uint8)

    # 服务端编号（stage4_manifest.json）：按**体积降序**，刻意与客户端顺序不同，
    # 好让「两套编号的对应关系」这件事在夹具上仍然是有内容的（E02 考的就是它）。
    rows = [
        {
            "candidate_id": index,
            "source_component_label": index,
            "volume_mm3": float(np.count_nonzero(_ellipsoid(
                SHAPE_ZYX, SPACING_XYZ, n[0:3], n[3]
            ))),
        }
        for index, n in enumerate(NODULES, start=1)
    ]
    rows.sort(key=lambda row: -row["volume_mm3"])
    manifest = {
        "case_id": CASE_ID,
        "candidate_count": len(rows),
        # ⚠️ 措辞注意：不能出现「姓名 / 患者 / 病人」这三个词 ——
        # PHI 守卫（scan_phi）的规则是「这三个词后面跟中文就按姓名报」，
        # 第一版这里写「不含任何患者信息」，被守卫**正确地**判成命中。
        # 守卫没做错（它宁可误报），所以改的是本文件的措辞，不是守卫的规则。
        "note": "合成夹具，几何由 agent/scripts/make_fixture_case.py 生成；"
                "不含任何隐私内容，其中的数字不可用于临床结论",
        "candidates": [
            {
                "candidate_id": rank,
                "source_component_label": row["source_component_label"],
                "voxel_count": int(row["volume_mm3"]),
            }
            for rank, row in enumerate(rows, start=1)
        ],
    }
    (pkg / "stage4_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pkg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--report", action="store_true",
                        help="生成后加载一遍并打印派生事实（用于核对约束）")
    args = parser.parse_args()

    pkg = write_case(Path(args.out))
    total = sum(p.stat().st_size for p in pkg.iterdir())
    print(f"已生成合成病例：{pkg}")
    print(f"  文件 {len(list(pkg.iterdir()))} 个，合计 {total / 1024:.1f} KB")

    if not args.report:
        return 0

    from agent.core.case_loader import load_case
    from agent.core.planner import classify_reachability, plan_candidate

    case = load_case(pkg, case_id=CASE_ID, use_cache=False)
    print(f"\n加载成功：{case.candidate_count} 个候选，"
          f"中心线 {case.centerline_voxels} 节点，"
          f"入口节点 {case.entry_node}（{case.entry_mode}）")
    for item in case.candidates:
        info = item.to_dict()
        print(f"  候选 {item.candidate_id}（服务端 {item.server_candidate_id}）"
              f"  {item.voxel_count} 体素  等效直径 {info['equivalent_diameter_mm']}mm"
              f"  中心 {info['center_xyz_mm']}")

    print("\n各候选规划（默认器械 2.0mm）：")
    for item in case.candidates:
        try:
            plan = plan_candidate(case, item.candidate_id)[0]
            metrics = plan.metrics
            reach = classify_reachability(metrics["target_distance_mm"])
            print(f"  候选 {item.candidate_id}: 长度 {metrics['route_length_mm']:.2f}mm"
                  f"  最窄直径 {metrics['minimum_diameter_mm']:.3f}mm"
                  f"  靶距 {metrics['target_distance_mm']:.2f}mm"
                  f"  可达性 {reach['grade']}（{reach['label']}）")
        except RuntimeError as error:
            print(f"  候选 {item.candidate_id}: 规划失败 - {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
