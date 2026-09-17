"""诊断：候选编号与靶点选择。

回答两个问题：
1. GUI 保存的 nodule_01_plan.json 到底对应哪个连通域？
2. 每个候选最近的中心线距离是多少（决定靶点是否可达）？
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.core.case_loader import load_case  # noqa: E402
from agent.core.navbridge import nav  # noqa: E402
from agent.core.planner import choose_target_nodes, plan_candidate  # noqa: E402

CASE = Path(__file__).resolve().parents[2] / "cases" / "LIDC_0089" / "stage4_package"


def main() -> None:
    case = load_case(CASE)
    print(f"病例 {case.case_id}  中心线节点 {case.centerline_voxels}")
    print(f"入口节点 {case.entry_node}  模式 {case.entry_mode}")
    print()

    saved = json.loads(
        (CASE / "interactive_plans" / "nodule_01_plan.json").read_text(encoding="utf-8")
    )
    print("GUI 保存结果：")
    print(f"  route_length_mm={saved['metrics']['route_length_mm']:.4f}")
    print(f"  target_distance_mm={saved['metrics']['target_distance_mm']:.4f}")
    print(f"  maximum_turn_angle_deg={saved['metrics']['maximum_turn_angle_deg']:.4f}")
    print()

    print(f"{'cand':>4} {'voxels':>7} {'nearest_mm':>11} {'targets':>8} "
          f"{'best_profile':>14} {'len_mm':>10} {'tgt_mm':>9} {'ok':>4}")
    print("-" * 78)

    for cand in case.candidates:
        mask = case.candidate_mask(cand.candidate_id)
        dist = ndi.distance_transform_edt(~mask, sampling=case.spacing_zyx)
        values = dist[tuple(case.centerline_coords.T)].astype(np.float64)
        nearest = float(values.min())
        nodes, _v, _n = choose_target_nodes(case, mask)

        row = f"{cand.candidate_id:>4} {cand.voxel_count:>7} {nearest:>11.3f} {len(nodes):>8} "
        try:
            best, allp = plan_candidate(
                case, cand.candidate_id, with_baseline=False
            )
            row += (
                f"{best.profile_name:>14} "
                f"{best.metrics['route_length_mm']:>10.4f} "
                f"{best.metrics['target_distance_mm']:>9.4f} "
                f"{len(allp):>4}"
            )
        except RuntimeError as error:
            row += f"{'FAIL':>14} {str(error)[:24]:>10}"
        print(row)

    print()
    print("GUI 目标值：route_length=%.4f  target_distance=%.4f"
          % (saved["metrics"]["route_length_mm"], saved["metrics"]["target_distance_mm"]))


if __name__ == "__main__":
    main()
