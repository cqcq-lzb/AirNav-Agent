"""验证无 GUI 规划内核与 V1 GUI 保存结果的一致性。

对照物：病例目录下 interactive_plans/nodule_XX_plan.json
（由 V1 的 InteractiveNavigationWindow.save_current_plan() 写出）

做法：对每个结节候选各规划一条路径，再与每个已保存的 GUI 结果做数值匹配，
自动找出「GUI 的 nodule_XX 对应客户端第几个候选」，并顺带暴露编号口径差异。

用法：
    python -m agent.scripts.verify_planner
    python -m agent.scripts.verify_planner --case-dir <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.core.case_loader import load_case  # noqa: E402
from agent.core.planner import plan_candidate  # noqa: E402

# 参与比对的数值型指标
METRIC_KEYS = [
    "route_length_mm",
    "minimum_radius_mm",
    "minimum_diameter_mm",
    "mean_radius_mm",
    "mean_diameter_mm",
    "maximum_turn_angle_deg",
    "target_distance_mm",
    "bridge_voxels_on_route",
    "recovered_voxels_on_route",
]
TOP_KEYS = ["topology_signature", "entry_mode", "device_diameter_mm", "device_margin_mm"]

REL_TOL = 1e-9


def numeric_match(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(a) * REL_TOL, 1e-9)


def diff(expected: dict, actual: dict, keys: list[str]) -> list[str]:
    out: list[str] = []
    for key in keys:
        want, got = expected.get(key), actual.get(key)
        if want is None and got is None:
            continue
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if not numeric_match(float(want), float(got)):
                out.append(f"    {key}: GUI={want!r}  headless={got!r}")
        elif want != got:
            out.append(f"    {key}: GUI={want!r}  headless={got!r}")
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", default=str(root / "cases" / "LIDC_0089" / "stage4_package"))
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    case = load_case(case_dir)

    print("=" * 72)
    print(f"病例 {case.case_id}")
    print(
        f"  体素 {tuple(round(float(v), 4) for v in case.spacing_zyx)} mm  "
        f"中心线节点 {case.centerline_voxels}  入口节点 {case.entry_node} ({case.entry_mode})"
    )
    print(f"  候选数 {case.candidate_count}  加载耗时 {case.load_seconds:.1f}s")
    for item in case.candidates:
        info = item.to_dict()
        print(
            f"    [客户端候选 {item.candidate_id}] 连通域标签 {item.component_label}"
            f"  {item.voxel_count} 体素  等效直径 "
            f"{info['equivalent_diameter_mm']}mm"
        )

    plans: dict[int, dict] = {}
    print("\n规划中……")
    for item in case.candidates:
        try:
            best, allp = plan_candidate(case, item.candidate_id)
            plans[item.candidate_id] = best.to_dict()
            plans[item.candidate_id]["_all"] = len(allp)
            print(
                f"    候选 {item.candidate_id}: {best.profile_name}  "
                f"长度 {best.metrics['route_length_mm']:.4f}mm  "
                f"靶距 {best.metrics['target_distance_mm']:.4f}mm  "
                f"方案数 {len(allp)}"
            )
        except RuntimeError as error:
            print(f"    候选 {item.candidate_id}: 规划失败 - {error}")

    plan_dir = case_dir / "interactive_plans"
    saved_files = sorted(plan_dir.glob("nodule_*_plan.json")) if plan_dir.is_dir() else []
    if not saved_files:
        print("\n[跳过对照] 没有找到 GUI 保存的规划结果")
        return 0

    print("\n" + "=" * 72)
    print("与 GUI 保存结果逐项对照")
    print("=" * 72)

    failures = 0
    for saved_path in saved_files:
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        gui_id = int(saved_path.stem.split("_")[1])
        want = saved.get("metrics", {})
        want_len = want.get("route_length_mm")

        # 自动找匹配的客户端候选
        matched = None
        for cand_id, plan in plans.items():
            got = plan.get("metrics", {}).get("route_length_mm")
            if got is not None and want_len is not None and numeric_match(got, want_len):
                if plan.get("profile") == saved.get("profile"):
                    matched = cand_id
                    break

        print(f"\n{saved_path.name}（GUI 编号 nodule_{gui_id:02d}，profile={saved.get('profile')}）")
        if matched is None:
            print("    未找到匹配的候选 —— 结果不一致")
            failures += 1
            continue

        tag = "一致" if matched == gui_id else f"一致（但客户端编号为候选项 {matched}）"
        print(f"    匹配到客户端候选 {matched}：数值 {tag}")

        problems = diff(want, plans[matched].get("metrics", {}), METRIC_KEYS)
        problems += diff(saved, plans[matched], TOP_KEYS)
        if problems:
            print("    存在差异：")
            for line in problems:
                print(line)
            failures += 1
        else:
            print(
                f"    全部 {len(METRIC_KEYS)} 项指标 + {len(TOP_KEYS)} 项字段逐位吻合"
            )
            print(f"    分叉序列：{saved.get('topology_signature')}")

    print("\n" + "=" * 72)
    if failures:
        print(f"结论：{failures} 处不一致，需排查")
        return 1
    print("结论：无 GUI 规划内核与 V1 数值完全一致，可安全承载 Agent 工具层")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
