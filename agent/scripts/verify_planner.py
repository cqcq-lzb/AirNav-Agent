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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.core.case_loader import load_case  # noqa: E402
from agent.core.planner import plan_candidate  # noqa: E402
from agent.core.sysmem import read_memory as _sysmem_read_memory  # noqa: E402

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


# ---------------------------------------------------------------- 内存预算
#
# `compare_with_baseline` 会在**整卷** airway 掩膜上做 skeletonize + 形态学膨胀，
# 峰值时同时持有若干份 `shape_zyx` 大小的数组。
#
# ⚠️ 为什么要**提前探**而不是 `except MemoryError` 兜住真正的失败：
#   那会让「机器腾不出内存」与「规划实现漂了」走进同一个分支，于是一次偶发的
#   环境抖动被写成 ❌。2026-09-30 就是这么误判的 —— 门禁报 geometry / planner 两项 ❌，
#   实际两处都是 `_ArrayMemoryError`（882 MiB / 36.8 MiB），而那一刻的读数极具迷惑性：
#   **可用物理还剩 12.5 GB，可用提交只剩 0.4 GB**。
#   `verify_geometry` 上一轮踩的是同一个病根，只是那次在物理口径上复发。
#   反过来把它写成 ✅ 更糟。所以：分配前先探一块，探不到就明确报 ⚪。

PLANNER_SLOTS = 4        # 峰值时同时在世的整卷数组份数（掩膜 / 膨胀 / 骨架 / 坐标）
PLANNER_SAFETY = 1.5     # 安全余量
PROBE_PAGE = 4096        # 逐页触碰的步长

# `load_case` 的**保守**门槛 —— 它必须做在加载之前（拿不到 case 就算不出确切需求）。
# 实测 shape (147, 512, 512)：CT int16 整卷 ≈77 MB，若干份 bool 掩膜每份 36.8 MB，
# 再加 SimpleITK 与 numpy 之间的拷贝 —— 峰值估计 ≈0.6 GB，取 1 GB 作为「低于此值别开始」。
# ⚠️ 2026-09-30 实测教训：门槛设 256 MB 时**放行了但照样崩**（当时余量 0.47 GB），
# 因为 `load_case` 的真实峰值是 CT + 多份掩膜叠加，不是单个 36.8 MB 数组。
LOAD_HEADROOM_MIN = 1024 * 2**20


def _planner_memory_need(size: int) -> int:
    """整卷规划一次的峰值内存估算（字节）。

    宁可多报 ⚪ —— 把资源问题记成 ❌ 比多报一次「不足以判定」坏得多。
    """
    return int(PLANNER_SAFETY * PLANNER_SLOTS * size)


def _try_reserve(nbytes: int) -> bool:
    """真的申请一块、**逐页触碰**、再释放。取不到返回 False。

    ⚠️ **必须逐页触碰**：`np.empty` 只保留虚拟地址；不写一遍的话，
    「虚拟地址申请到了」会被误当成「提交得出来」—— 而这正是本次要防的那件事。
    """
    try:
        probe = np.empty(nbytes, dtype=np.uint8)
        probe[::PROBE_PAGE] = 0
        del probe
        return True
    except (MemoryError, ValueError):
        return False


def _default_case_dir() -> str:
    """挑一个可用的病例目录（优先 LIDC_0089，否则取第一个）。

    ⚠️ 原来这里硬编码 `<repo>/cases/LIDC_0089/stage4_package`。真实病例缺席的
    机器上（含全新 clone），那一项是**硬门禁**，却指着一个不存在的目录 ——
    于是它要么直接报错，要么被 `--profile ci` 整项滤掉、从报告里消失。
    跟着 `cases_root()` 走，夹具才能真正顶上来。
    """
    from agent.core.case_loader import cases_root, list_local_cases

    root = cases_root()
    preferred = root / "LIDC_0089" / "stage4_package"
    if preferred.is_dir():
        return str(preferred)
    found = list_local_cases(root)
    if not found:
        raise SystemExit(f"找不到任何病例：{root}")
    return str(found[0]["package_dir"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", default=None)
    args = parser.parse_args()
    if args.case_dir is None:
        args.case_dir = _default_case_dir()

    case_dir = Path(args.case_dir).resolve()

    # ⚠️ 这一道必须在 `load_case` **之前** —— 病例加载自己就会在整卷上建若干份数组
    # （CT int16 整卷 + 多份 bool 掩膜 + SimpleITK→numpy 拷贝），2026-09-30 实测
    # 它正是在 `airway_baseline = array > 0` 这一步抛 `_ArrayMemoryError`。
    # 那时还没拿到 case，算不出确切需求，所以用保守门槛。判据是**提交**不是**物理**：
    # 那一刻可用物理还剩 12.5 GB，可用提交只剩 0.4 GB。
    if not _try_reserve(LOAD_HEADROOM_MIN):
        print(f"⚪ 探不到 {LOAD_HEADROOM_MIN / 2**20:,.0f} MB 连续可提交内存 —— 本项**不足以判定**")
        print("   病例加载本身就要建起 CT 与多份整卷掩膜，这不是规划实现的问题，")
        print("   是这台机器此刻的**可用提交**不足（注意：可用物理可能还剩很多）。")
        print("   补齐：关掉占内存的进程（浏览器 / Docker / 其它 python）后单跑：")
        print("     python -m agent.scripts.verify_planner")
        return 2

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
    baseline = getattr(case, "airway_baseline_mask", None)
    if baseline is not None:
        need = _planner_memory_need(int(baseline.size))
        reading = _sysmem_read_memory()
        now = reading.headroom_bytes
        print(
            f"\n  内存预算：整卷规划峰值 ≈{need / 2**20:,.0f} MB"
            f"（shape {tuple(baseline.shape)}，{baseline.size:,} 体素 × {PLANNER_SLOTS} 份）"
            f"；当前可申请 {('%.0f MB' % (now / 2**20)) if now else '未知'}"
            f"（取「物理可用 / 提交可用」的较小者）"
        )
        if not _try_reserve(need):
            print(f"  ⚪ 探不到 {need / 2**20:,.0f} MB 连续可提交内存 —— 本项**不足以判定**")
            print("     这不是规划实现的问题，是这台机器此刻腾不出整卷规划的内存。")
            print("     补齐：关掉占内存的进程（浏览器 / 其它 python / Docker）后单跑：")
            print("       python -m agent.scripts.verify_planner")
            return 2
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
        # ⚠️ 这里**不能返回 0**。
        #
        # 这个脚本有两半：前半「把 4 个候选都规划一遍」（任何病例都能跑），
        # 后半「与 V1 GUI 保存的结果逐位对拍」（只有真实病例才有对照物）。
        # 合成夹具没有 interactive_plans/，于是后半就是**没做**。
        # 原来这里 `return 0` —— 结果「没做」和「做对了」在门禁里长得一模一样，
        # 一个硬门禁就这样静默退化成了空转（判断纪律：不测就等于没有）。
        #
        # 按三态纪律：缺数据 ≠ 通过。退出码 2 = **不足以判定**，
        # 门禁会把它记成 `none` 而不是 `✅`，并带上下面这条补齐命令。
        print("\n[跳过对照] 没有找到 GUI 保存的规划结果 —— 数值对拍**未执行**")
        print(f"  病例目录：{case_dir}")
        print("  上述 4 个候选的规划数字只说明「跑起来了」，**不能**证明与 V1 一致。")
        print("  补齐：在装有真实病例的机器上重跑本项（cases/LIDC_0089 里有")
        print("        interactive_plans/nodule_XX_plan.json 时才会做逐位对拍）。")
        return 2

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
