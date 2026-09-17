"""工具返回体自检：**不产出文件的工具，不许给模型任何像链接的东西**。

    python -m agent.scripts.verify_payloads

## 为什么单独开一个入口

这是 E01 那一类缺陷的**载荷版本**，值得有个固定的地方盯着：

模型的回答是从**工具返回体**里长出来的。返回体里有什么，模型就会说什么。
所以只要返回体里出现一个 `xxx-001`，回答里迟早会出现 `![](xxx-001)` ——
用户看到的是一个点不开的破图，或者更糟，一句
「三位视图文件已生成，ID 为 xxx-001」的**假话**（那时根本没出图）。

2026-09-17 实测到的就是这个：

    "artifact_id": "route-001",
    "artifact_note": "路径折线数据已存入 artifact，渲染三维视图时用 render_viewer 引用该 id"

两句都不成立 —— `render_viewer` 的入参是 case_id / candidate_id，**不吃 id**；
而 `plan_route` 本身**不产出任何文件**。已确认**没有任何工具**吃 artifact id 入参，
所以把它暴露给模型是纯噪声，且必定被误用。

改提示词没用（试过，和 E01 一样）：**提示词压不住模型手里已有的数据。**
要改的是数据本身 —— 给文件，不给句柄。

## 四层

    第 0 层 静态：源码里不许再出现 "artifact_id" / "artifact_note" 这类键
    第 1 层 实跑：会留痕的三个工具，返回体里不许有 xxx-NNN 裸 id
    第 2 层 语义：不产出文件的工具必须明说「不产出文件」并指向 render_viewer
    第 3 层 正例：render_viewer 必须给出真文件名，供模型/界面交代给用户
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.tools import NavSession, build_registry  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# 产物 id 的形状：`route-001` / `viewer-002` / `segmentation-003`
ID_PATTERN = re.compile(r"(?:route|viewer|segmentation)-\d+")

# 会往 ArtifactStore 里写东西的工具 —— 也就是**最可能**把 id 漏进返回体的那几个
LEAKY_TOOLS: tuple[tuple[str, dict], ...] = (
    ("plan_route", {"case_id": "LIDC_0089", "candidate_id": 3, "device_diameter_mm": 1.5}),
    ("explain_route", {"case_id": "LIDC_0089", "candidate_id": 3, "device_diameter_mm": 1.5}),
    ("render_viewer", {"case_id": "LIDC_0089", "candidate_id": 3, "device_diameter_mm": 1.5}),
)

# 明确**不产出可查看文件**的工具：返回体里必须把这件事说出来
NON_VIEWING_TOOLS = ("plan_route", "explain_route")

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((ok, label, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"（{detail}）" if detail else ""))
    return ok


def layer_0_source_scan() -> int:
    """源码层面直接扫 —— 确保没有任何地方重新引入这两个键。

    比实跑更强：实跑只覆盖被调用的路径，源码扫描覆盖所有工具文件，包括
    那些这次没跑的。
    """
    print("\n[0] 静态扫描：工具源码里不许再出现 artifact_id / artifact_note")
    failures = 0
    offenders: list[str] = []
    for path in sorted((REPO / "agent" / "tools").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # 注释里说明「以前长这样」是允许的
            if re.search(r'["\'](artifact_id|artifact_note)["\']\s*:', line):
                offenders.append(f"{path.name}:{lineno}")
    failures += not check(
        not offenders,
        "没有任何工具把 artifact_id / artifact_note 放进返回体",
        "、".join(offenders) if offenders else "干净",
    )

    # payload 里塞了 xx-NNN 字面量也一样危险
    id_literals: list[str] = []
    for path in sorted((REPO / "agent" / "tools").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            if re.search(r'["\'](?:route|viewer|segmentation)-\d+["\']', line):
                id_literals.append(f"{path.name}:{lineno}")
    failures += not check(
        not id_literals,
        "源码里没有硬编码的产物 id 字面量",
        "、".join(id_literals) if id_literals else "干净",
    )
    return failures


def layer_1_live_payloads() -> int:
    """实跑一遍：返回体里不许出现任何 `xxx-NNN`。"""
    print("\n[1] 实跑：会留痕的工具，返回体里不许有裸 id")
    failures = 0
    session = NavSession()
    registry = build_registry()
    for name, args in LEAKY_TOOLS:
        result = registry.execute(name, dict(args), context=session)
        text = json.dumps(result, ensure_ascii=False, default=str)
        hits = ID_PATTERN.findall(text)
        failures += not check(
            not hits,
            f"{name} 的返回体不含产物 id",
            f"发现 {hits}" if hits else "干净",
        )
    return failures


def layer_2_notes() -> int:
    """不产出文件的工具，必须把「不产出文件」这件事说出来。

    光删掉 id 不够 —— 模型看不见任何指引时，仍然会顺着「路径已规划」去许诺
    一张并不存在的图。所以要给一句**正确且可执行**的话。
    """
    print("\n[2] 语义：不产出文件的工具必须明说，并指向 render_viewer")
    failures = 0
    session = NavSession()
    registry = build_registry()
    for name in NON_VIEWING_TOOLS:
        args = {"case_id": "LIDC_0089", "candidate_id": 3, "device_diameter_mm": 1.5}
        result = registry.execute(name, args, context=session)
        note = str(result.get("note") or "")
        failures += not check(
            "不产出任何可查看的文件" in note,
            f"{name} 的 note 明说「不产出任何可查看的文件」",
            note[:46] + "…" if note else "缺 note",
        )
        failures += not check(
            "render_viewer" in note,
            f"{name} 的 note 指向 render_viewer",
        )
    return failures


def layer_3_viewer_gives_file() -> int:
    """正例：真的产出了文件的那个工具，必须给文件名。"""
    print("\n[3] 正例：render_viewer 给真文件，不给句柄")
    session = NavSession()
    registry = build_registry()
    result = registry.execute(
        "render_viewer",
        {"case_id": "LIDC_0089", "candidate_id": 3, "device_diameter_mm": 1.5},
        context=session,
    )
    failures = 0
    failures += not check(bool(result.get("viewer_path")), "返回 viewer_path（完整路径）")
    failures += not check(
        str(result.get("viewer_file", "")).endswith(".html"),
        "返回 viewer_file（文件名，模型好念给用户）",
        str(result.get("viewer_file")),
    )
    return failures


def main() -> int:
    print("=" * 68)
    print("AirNav-Agent 工具返回体自检（不产出文件 ≠ 可以给个 id 让模型自己猜）")
    print("=" * 68)

    failures = 0
    failures += layer_0_source_scan()
    failures += layer_1_live_payloads()
    failures += layer_2_notes()
    failures += layer_3_viewer_gives_file()

    total = len(RESULTS)
    passed = sum(1 for ok, _, _ in RESULTS if ok)
    print("\n" + "=" * 68)
    if failures:
        print(f"未通过：{passed}/{total} 项通过，{failures} 项失败")
        for ok, label, detail in RESULTS:
            if not ok:
                print(f"  FAIL  {label}（{detail}）")
        return 1
    print(f"自检通过：{passed}/{total} 项全部符合预期")
    print("覆盖：源码扫描 · 实跑返回体 · 不产出文件的说明 · 出图工具给真文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
