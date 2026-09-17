#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
服务器端第四阶段一键流水线 v3.1
================================

修复内容
--------
1. 强制把当前 perm_test Python 所在 bin 目录加入 PATH；
2. 显式检查 nnUNetv2_predict；
3. 设置 CONDA_PREFIX 和 CONDA_DEFAULT_ENV；
4. 服务器失败时输出 run_full_pipeline.py 的真实错误；
5. 只要气道与结节分割已经生成，即使旧导航质控失败，也继续打包。

放置位置
--------
/data2/home/wcq/nnUNet/nav_project/engineering_demo/
run_stage4_server_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


DEFAULT_ENGINEERING_DIR = Path(
    "/data2/home/wcq/nnUNet/nav_project/engineering_demo"
)

DEFAULT_NNUNET_RAW = Path(
    "/data2/home/wcq/nnUNet/nnUNet_raw"
)
DEFAULT_NNUNET_PREPROCESSED = Path(
    "/data2/home/wcq/nnUNet/nnUNet_preprocessed"
)
DEFAULT_NNUNET_RESULTS = Path(
    "/data2/home/wcq/nnUNet/nnUNet_results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="服务器端第四阶段一键流水线 v3.1"
    )
    parser.add_argument(
        "--ct",
        type=Path,
        required=True,
        help="已经上传到服务器的 NIfTI CT",
    )
    parser.add_argument(
        "--case-id",
        required=True,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--engineering-dir",
        type=Path,
        default=DEFAULT_ENGINEERING_DIR,
    )
    parser.add_argument(
        "--min-nodule-voxels",
        type=int,
        default=8,
    )
    return parser.parse_args()


def sanitize_case_id(value: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value.strip(),
    )
    cleaned = cleaned.strip("._-")

    if not cleaned:
        raise ValueError("病例编号不能为空")

    return cleaned


def first_existing(
    paths: list[Path],
    *,
    label: str,
) -> Path:
    for path in paths:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"没有找到{label}，检查过：\n"
        + "\n".join(
            str(path)
            for path in paths
        )
    )


def optional_existing(
    paths: list[Path],
) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def read_json(
    path: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    return (
        value
        if isinstance(value, dict)
        else None
    )


def extract_error_text(
    payload: dict[str, Any] | None,
) -> str:
    if not payload:
        return ""

    parts: list[str] = []

    for key in [
        "error",
        "message",
        "error_type",
        "failed_stage",
        "stage",
    ]:
        value = payload.get(key)

        if value not in (
            None,
            "",
        ):
            parts.append(
                f"{key}: {value}"
            )

    traceback_text = payload.get(
        "traceback"
    )

    if traceback_text:
        parts.append(
            "traceback:\n"
            + str(traceback_text)
        )

    return "\n".join(parts)


def stream_process(
    command: list[str],
    *,
    environment: dict[str, str],
    cwd: Path,
) -> int:
    print()
    print("=" * 96)
    print("执行现有完整流水线")
    print("=" * 96)
    print("命令：")
    print(" ".join(command))
    print()

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(
            line.rstrip("\n"),
            flush=True,
        )

    return int(process.wait())


def count_nodule_candidates(
    nodule_raw_path: Path,
    min_voxels: int,
) -> list[dict[str, Any]]:
    image = sitk.ReadImage(
        str(nodule_raw_path)
    )
    array = (
        sitk.GetArrayFromImage(image) > 0
    )

    labels, count = ndi.label(
        array,
        structure=ndi.generate_binary_structure(
            3,
            3,
        ),
    )

    sizes = np.bincount(labels.ravel())
    voxel_volume_mm3 = float(
        np.prod(image.GetSpacing())
    )

    candidates: list[dict[str, Any]] = []

    for component_label in range(
        1,
        count + 1,
    ):
        voxel_count = int(
            sizes[component_label]
        )

        if voxel_count < min_voxels:
            continue

        coords = np.argwhere(
            labels == component_label
        )

        center_zyx = coords.mean(axis=0)

        center_xyz = image.TransformContinuousIndexToPhysicalPoint(
            (
                float(center_zyx[2]),
                float(center_zyx[1]),
                float(center_zyx[0]),
            )
        )

        candidates.append(
            {
                "source_component_label": int(
                    component_label
                ),
                "voxel_count": voxel_count,
                "volume_mm3": float(
                    voxel_count
                    * voxel_volume_mm3
                ),
                "center_xyz_mm": [
                    float(value)
                    for value in center_xyz
                ],
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["voxel_count"],
            item["source_component_label"],
        )
    )

    for candidate_id, item in enumerate(
        candidates,
        start=1,
    ):
        item["candidate_id"] = candidate_id

    return candidates


def write_status(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    start_time = time.time()
    case_id = sanitize_case_id(
        args.case_id
    )

    engineering_dir = (
        args.engineering_dir.resolve()
    )
    ct_path = args.ct.resolve()

    case_dir = (
        engineering_dir
        / "cases"
        / case_id
    )
    package_dir = (
        case_dir
        / "stage4_package"
    )
    status_path = (
        engineering_dir
        / "pipeline_status"
        / f"{case_id}_stage4.json"
    )

    initial_status: dict[str, Any] = {
        "case_id": case_id,
        "stage": "starting",
        "success": False,
        "ct": str(ct_path),
        "case_dir": str(case_dir),
        "gpu": int(args.gpu),
        "started_at_unix": start_time,
    }
    write_status(
        status_path,
        initial_status,
    )

    try:
        if not ct_path.is_file():
            raise FileNotFoundError(
                f"CT 文件不存在：{ct_path}"
            )

        full_pipeline_script = (
            engineering_dir
            / "run_full_pipeline.py"
        )

        if not full_pipeline_script.is_file():
            raise FileNotFoundError(
                "缺少现有完整流水线："
                f"{full_pipeline_script}"
            )

        # ----------------------------------------------------------
        # 关键修复：绝对 Python 路径并不会自动激活 conda 环境。
        # 必须把同目录下的 nnUNetv2_predict 加入 PATH。
        # ----------------------------------------------------------
        python_path = Path(
            sys.executable
        ).resolve()
        env_bin = python_path.parent
        conda_prefix = env_bin.parent

        existing_path = os.environ.get(
            "PATH",
            "",
        )

        environment = os.environ.copy()
        environment["PATH"] = (
            str(env_bin)
            + os.pathsep
            + existing_path
        )
        environment["CONDA_PREFIX"] = str(
            conda_prefix
        )
        environment["CONDA_DEFAULT_ENV"] = (
            conda_prefix.name
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = str(
            args.gpu
        )
        environment["nnUNet_raw"] = str(
            DEFAULT_NNUNET_RAW
        )
        environment[
            "nnUNet_preprocessed"
        ] = str(
            DEFAULT_NNUNET_PREPROCESSED
        )
        environment["nnUNet_results"] = str(
            DEFAULT_NNUNET_RESULTS
        )

        predictor = shutil.which(
            "nnUNetv2_predict",
            path=environment["PATH"],
        )

        direct_predictor = (
            env_bin
            / "nnUNetv2_predict"
        )

        if predictor is None and direct_predictor.is_file():
            predictor = str(
                direct_predictor
            )

        print("=" * 96)
        print("服务器运行环境检查")
        print("=" * 96)
        print("Python：", python_path)
        print("环境 bin：", env_bin)
        print("CONDA_PREFIX：", conda_prefix)
        print(
            "nnUNetv2_predict：",
            predictor,
        )
        print(
            "PATH 首项：",
            environment["PATH"].split(
                os.pathsep
            )[0],
        )
        print(
            "nnUNet_raw：",
            environment["nnUNet_raw"],
        )
        print(
            "nnUNet_preprocessed：",
            environment[
                "nnUNet_preprocessed"
            ],
        )
        print(
            "nnUNet_results：",
            environment[
                "nnUNet_results"
            ],
        )
        print(
            "CUDA_VISIBLE_DEVICES：",
            environment[
                "CUDA_VISIBLE_DEVICES"
            ],
        )

        if predictor is None:
            available = sorted(
                path.name
                for path in env_bin.glob(
                    "nnUNetv2_*"
                )
            )

            raise RuntimeError(
                "perm_test 环境中找不到 "
                "nnUNetv2_predict。\n"
                f"环境 bin：{env_bin}\n"
                f"发现的 nnUNet 命令："
                f"{available}"
            )

        command = [
            str(python_path),
            str(full_pipeline_script),
            "--ct",
            str(ct_path),
            "--case-id",
            case_id,
            "--gpu",
            str(args.gpu),
        ]

        if args.overwrite:
            command.append(
                "--overwrite"
            )

        pipeline_return_code = stream_process(
            command,
            environment=environment,
            cwd=engineering_dir,
        )

        print()
        print(
            "现有完整流水线退出代码：",
            pipeline_return_code,
        )

        optimization_warning = ""
        optimized_root = case_dir / "segmentation_v2"
        optimizer_runner = engineering_dir / "run_optimized_nnunet.py"
        optimizer_config = engineering_dir / "optimizer_config.json"
        if optimizer_runner.is_file():
            print()
            print("开始概率图驱动的气道/结节优化……")
            optimizer_command = [
                str(python_path),
                str(optimizer_runner),
                "--ct",
                str(ct_path),
                "--case-id",
                case_id,
                "--output-dir",
                str(optimized_root),
                "--gpu",
                str(args.gpu),
            ]
            if optimizer_config.is_file():
                optimizer_command.extend(
                    ["--optimizer-config", str(optimizer_config)]
                )
            if args.overwrite:
                optimizer_command.append("--overwrite")
            optimizer_return_code = stream_process(
                optimizer_command,
                environment=environment,
                cwd=engineering_dir,
            )
            if optimizer_return_code != 0:
                optimization_warning = (
                    "概率图分割优化失败，已安全回退到原始分割；"
                    f"退出代码={optimizer_return_code}。"
                )
                print(optimization_warning)
        else:
            optimization_warning = (
                "服务器未找到分割优化脚本，已使用原始分割。"
            )

        segmentation_dir = (
            case_dir
            / "segmentation"
        )
        input_dir = case_dir / "input"

        airway_candidates = [
            optimized_root
            / "optimized"
            / "airway_optimized.nii.gz",
            segmentation_dir
            / "airway_for_navigation.nii.gz",
            segmentation_dir
            / "airway_repaired.nii.gz",
            segmentation_dir
            / "airway_main.nii.gz",
            segmentation_dir
            / "airway_raw.nii.gz",
        ]

        nodule_candidates = [
            optimized_root
            / "optimized"
            / "nodule_candidates_ranked.nii.gz",
            segmentation_dir
            / "nodule_raw.nii.gz",
        ]

        airway_exists = any(
            path.is_file()
            for path in airway_candidates
        )
        nodule_exists = any(
            path.is_file()
            for path in nodule_candidates
        )

        if (
            pipeline_return_code != 0
            and not (
                airway_exists
                and nodule_exists
            )
        ):
            full_status = read_json(
                case_dir
                / "full_pipeline_status.json"
            )
            inference_status = read_json(
                case_dir
                / "status.json"
            )

            details = "\n\n".join(
                text
                for text in [
                    extract_error_text(
                        full_status
                    ),
                    extract_error_text(
                        inference_status
                    ),
                ]
                if text
            )

            raise RuntimeError(
                "run_full_pipeline.py 执行失败，"
                f"退出代码：{pipeline_return_code}"
                + (
                    "\n\n真实错误：\n"
                    + details
                    if details
                    else ""
                )
            )

        packaged_ct = first_existing(
            sorted(
                input_dir.glob(
                    "*.nii.gz"
                )
            )
            + sorted(
                input_dir.glob(
                    "*.nii"
                )
            )
            + [ct_path],
            label="病例 CT",
        )

        airway_path = first_existing(
            airway_candidates,
            label="气道分割",
        )

        nodule_raw_path = first_existing(
            nodule_candidates,
            label="全部结节候选分割",
        )

        nodule_selected_path = (
            optional_existing(
                [
                    optimized_root
                    / "optimized"
                    / "nodule_selected_optimized.nii.gz",
                    segmentation_dir
                    / "nodule_selected.nii.gz",
                ]
            )
        )

        entry_path = optional_existing(
            [
                case_dir
                / "navigation_branchaware_v4"
                / "entry_point.nii.gz",
                case_dir
                / "navigation_candidates"
                / "entry_point.nii.gz",
                case_dir
                / "navigation_multicost"
                / "entry_point.nii.gz",
                case_dir
                / "navigation"
                / "entry_point.nii.gz",
            ]
        )

        previous_summary = optional_existing(
            [
                case_dir
                / "navigation_branchaware_v4"
                / "case_summary.json",
                case_dir
                / "navigation"
                / "case_summary.json",
                case_dir
                / "full_pipeline_status.json",
            ]
        )

        if package_dir.exists():
            shutil.rmtree(
                package_dir
            )

        package_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        copy_map: dict[str, Path] = {
            "ct.nii.gz": packaged_ct,
            "airway_mask.nii.gz": (
                airway_path
            ),
            "nodule_raw.nii.gz": (
                nodule_raw_path
            ),
        }

        airway_raw_baseline = optional_existing(
            [optimized_root / "airway_prediction" / f"{case_id}.nii.gz"]
        )
        if airway_raw_baseline is not None:
            copy_map["airway_raw_baseline.nii.gz"] = airway_raw_baseline

        if nodule_selected_path is not None:
            copy_map[
                "nodule_selected.nii.gz"
            ] = nodule_selected_path

        if entry_path is not None:
            copy_map[
                "entry_point.nii.gz"
            ] = entry_path

        if previous_summary is not None:
            copy_map[
                "previous_pipeline_summary.json"
            ] = previous_summary

        segmentation_quality = optional_existing(
            [optimized_root / "optimized" / "segmentation_quality.json"]
        )
        if segmentation_quality is not None:
            copy_map["segmentation_quality.json"] = segmentation_quality

        optimized_inference_manifest = optional_existing(
            [optimized_root / "optimized_inference_manifest.json"]
        )
        if optimized_inference_manifest is not None:
            copy_map["optimized_inference_manifest.json"] = optimized_inference_manifest

        airway_added_voxels = optional_existing(
            [optimized_root / "optimized" / "airway_added_voxels.nii.gz"]
        )
        if airway_added_voxels is not None:
            copy_map["airway_added_voxels.nii.gz"] = airway_added_voxels

        airway_recovered_voxels = optional_existing(
            [optimized_root / "optimized" / "airway_recovered_voxels.nii.gz"]
        )
        if airway_recovered_voxels is not None:
            copy_map["airway_recovered_voxels.nii.gz"] = airway_recovered_voxels

        airway_bridge_voxels = optional_existing(
            [optimized_root / "optimized" / "airway_bridge_voxels.nii.gz"]
        )
        if airway_bridge_voxels is not None:
            copy_map["airway_bridge_voxels.nii.gz"] = airway_bridge_voxels

        for output_name, source_path in (
            copy_map.items()
        ):
            shutil.copy2(
                source_path,
                package_dir
                / output_name,
            )

        candidates = count_nodule_candidates(
            nodule_raw_path,
            args.min_nodule_voxels,
        )

        warnings: list[str] = []

        if optimization_warning:
            warnings.append(optimization_warning)

        if pipeline_return_code != 0:
            warnings.append(
                "现有完整流水线返回非零退出代码，"
                "但气道和结节分割结果完整，"
                "已继续生成第四阶段交互病例包。"
            )

        if entry_path is None:
            warnings.append(
                "未找到既有入口点，Windows "
                "交互系统将自动选择入口。"
            )

        manifest = {
            "case_id": case_id,
            "source_ct": str(ct_path),
            "server_case_dir": str(case_dir),
            "pipeline_return_code": (
                pipeline_return_code
            ),
            "candidate_count": len(
                candidates
            ),
            "min_nodule_voxels": int(
                args.min_nodule_voxels
            ),
            "candidates": candidates,
            "warnings": warnings,
            "environment": {
                "python": str(
                    python_path
                ),
                "env_bin": str(
                    env_bin
                ),
                "conda_prefix": str(
                    conda_prefix
                ),
                "nnunet_predict": str(
                    predictor
                ),
            },
            "files": {
                name: str(
                    package_dir / name
                )
                for name in copy_map
            },
        }

        (
            package_dir
            / "stage4_manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        zip_base = (
            case_dir
            / f"{case_id}_stage4_package"
        )

        old_zip = zip_base.with_suffix(
            ".zip"
        )

        if old_zip.exists():
            old_zip.unlink()

        zip_path = Path(
            shutil.make_archive(
                str(zip_base),
                "zip",
                root_dir=package_dir,
            )
        )

        duration = float(
            time.time() - start_time
        )

        final_status = {
            "case_id": case_id,
            "stage": "completed",
            "success": True,
            "pipeline_return_code": (
                pipeline_return_code
            ),
            "candidate_count": len(
                candidates
            ),
            "package_dir": str(
                package_dir
            ),
            "zip_path": str(zip_path),
            "warnings": warnings,
            "duration_seconds": duration,
            "completed_at_unix": (
                time.time()
            ),
        }

        write_status(
            status_path,
            final_status,
        )
        write_status(
            case_dir
            / "stage4_pipeline_status.json",
            final_status,
        )

        print()
        print("=" * 96)
        print("第四阶段服务器流水线完成")
        print("=" * 96)
        print("病例：", case_id)
        print(
            "结节候选数：",
            len(candidates),
        )
        print(
            "病例包目录：",
            package_dir,
        )
        print("ZIP：", zip_path)

        if warnings:
            print("警告：")
            for warning in warnings:
                print(" -", warning)

        return 0

    except Exception as error:
        failure = {
            "case_id": case_id,
            "stage": "failed",
            "success": False,
            "error_type": type(
                error
            ).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "duration_seconds": float(
                time.time() - start_time
            ),
        }

        write_status(
            status_path,
            failure,
        )

        print()
        print("=" * 96)
        print("第四阶段服务器流水线失败")
        print("=" * 96)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
