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
而当时的 `plan_route` 本身**不产出任何文件**。已确认**没有任何工具**吃 artifact id
入参，所以把它暴露给模型是纯噪声，且必定被误用。

改提示词没用（试过，和 E01 一样）：**提示词压不住模型手里已有的数据。**
要改的是数据本身 —— 给文件，不给句柄。

⚠️ **第二天补的一刀（用户回了一句「连接呢」）**：删掉 id 之后模型不再说假话了，
但用户要的东西还是没有 —— 问一条路径，拿回一段文字、没有一个能点开的图。
所以光「不给句柄」不够，还得**让那句「三维视图已生成」成真**：
`plan_route` 现在规划成功就顺带把视图渲染好，返回体里给的是磁盘上真实的
`viewer_path`。第 3 层把这条契约钉住（返回的路径必须 `is_file()`）。

## 六层

    第 0 层 静态：源码里不许再出现 "artifact_id" / "artifact_note" 这类键
    第 1 层 实跑：会留痕的三个工具，返回体里不许有 xxx-NNN 裸 id
    第 2 层 语义：不产出文件的工具必须明说「不产出文件」并指向 render_viewer
    第 3 层 正例：产出文件的工具（render_viewer / plan_route）必须给真文件
    第 4 层 落盘：自检的渲染不碰入库目录；同参数重复渲染不重写文件；
                  不同器械外径的渲染不互相覆盖
    第 5 层 契约：规划失败必须结构化归因，且「最大可通过外径」要能真的执行
                  （按它重试必须成功）—— 2026-09-22 加，见 `layer_5_failure_contract`

第 4 层与「模型看到什么」无关，但它守的是同一件事的两面：**别让机器产生的
噪声混进成果里**。自检渲染用的是 1.5 mm，而入库的演示产物是 2.0 mm —— 不隔离
的话，跑一次自检就把仓库里那份演示 HTML 换成另一个器械尺寸。而 `generatedAt`
每次都变，所以同一组参数重渲染会在 `git status` 里留一处「只差一个时间戳」的
改动，跑几次演示就攒几条。

最后那条（不同外径不互相覆盖）来自用户实测：同一结节用 1.5 mm 渲染一次、
再用 2.0 mm 渲染一次，旧口径的文件名只编码「病例 + 候选号」，后者把前者顶掉，
用户点开看到的器械尺寸不是他问的那个。
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
NON_VIEWING_TOOLS = ("explain_route",)

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
    """正例：真的产出了文件的工具，必须给文件名。

    `plan_route` 从 2026-09-17 起也在这条正例里 —— 规划成功就顺带把视图渲染好。
    这一层的意义就是把「承诺」钉在**磁盘上真实存在的文件**上：
    只要它还成立，模型写「三维视图已生成 + 路径」就不会再是假话。
    """
    print("\n[3] 正例：产出了文件的工具必须给真文件，不给句柄")
    session = NavSession()
    registry = build_registry()
    failures = 0

    args = {"case_id": "LIDC_0089", "candidate_id": 3, "device_diameter_mm": 1.5}
    for name in ("render_viewer", "plan_route"):
        result = registry.execute(name, dict(args), context=session)
        path = Path(str(result.get("viewer_path") or ""))
        failures += not check(
            path.is_file(), f"{name} 给出落盘成功的 viewer_path", str(path)
        )
        failures += not check(
            str(result.get("viewer_file", "")).endswith(".html"),
            f"{name} 返回 viewer_file（文件名，模型好念给用户）",
            str(result.get("viewer_file")),
        )
    return failures


def layer_4_render_hygiene() -> int:
    """落盘卫生：自检的渲染不落进入库目录；同参数重渲染不重写；
    不同器械外径的产物不互相覆盖。"""
    print("\n[4] 落盘卫生：自检隔离 + 同参数重渲染不重写 + 不同外径不互相覆盖")
    session = NavSession()
    registry = build_registry()
    args = {"case_id": "LIDC_0089", "candidate_id": 3, "device_diameter_mm": 1.5}

    first = registry.execute("render_viewer", args, context=session)
    path = Path(str(first.get("viewer_path", "")))
    if not path.is_file():
        return not check(False, "首次渲染落盘", str(path))

    # 先记下字节与 mtime，再原样渲染第二次
    before_bytes, before_mtime = path.read_bytes(), path.stat().st_mtime_ns
    second = registry.execute("render_viewer", args, context=session)

    failures = 0
    tracked = REPO / "outputs" / "viewers"
    failures += not check(
        tracked not in path.parents,
        "自检渲染落在临时目录，不覆盖入库的演示产物",
        str(path.parent),
    )
    failures += not check(
        path.read_bytes() == before_bytes and path.stat().st_mtime_ns == before_mtime,
        "同参数再渲染一次不动文件（只有 generatedAt 会变，不该产生 diff）",
    )
    failures += not check(
        second.get("viewer_reused") is True,
        "重复渲染如实回报 viewer_reused=True",
        repr(second.get("viewer_reused")),
    )

    # 换器械外径：必须换一个文件，且两份都还在
    other = registry.execute(
        "render_viewer",
        {**args, "device_diameter_mm": 2.0},
        context=session,
    )
    other_path = Path(str(other.get("viewer_path", "")))
    failures += not check(
        other_path != path and other_path.is_file() and path.is_file(),
        "换器械外径渲染到另一个文件，不覆盖上一份",
        f"{path.name} / {other_path.name}",
    )
    return failures


def layer_5_failure_contract() -> int:
    """失败返回体必须是**可执行的指令**，不能只是一句人话。

    2026-09-22 加的。起因是 E14：`plan_route` 失败时返回体只有
    `{"ok": false, "error": "..."}` 两个 key，而工具描述里那句
    「失败就改用更细的器械重试」是**无条件**的 —— 对「编号越界」这种失败，
    「换个参数重试」正好等于**静默换目标**：E14 的 trace 里模型 step 2 已经
    写对了「99 超出了有效范围…范围是 1 到 4」，最终回答却交付了 3 号的
    完整路径与三维图，通篇不提 99。**知道，但没说。**

    这一层守两件事：
      ① 失败原因**分得开**（编号越界 / 器械太粗 / 气道不通），各给一条下一步；
      ② 给的数字**必须真的能用** —— 「最粗只能过 5.45mm」这句话要被执行一遍
         （按该外径重试必须成功）。否则它只是一句好听的文案，
         与「artifact_id 已生成」是同一类假话。
    """
    print("\n[5] 契约：规划失败必须结构化归因，而且给出的上限要能被真的执行")
    failures = 0
    session = NavSession()
    registry = build_registry()

    def run(name: str, **kwargs):
        return registry.execute(name, dict(kwargs), context=session)

    # ---- ① 编号越界 ----------------------------------------------------
    out = run("plan_route", case_id="LIDC_0089", candidate_id=99,
              device_diameter_mm=2.0)
    failure = out.get("failure") or {}
    failures += not check(
        out.get("ok") is False and failure.get("reason") == "candidate_out_of_range",
        "越界规划失败被归因为 candidate_out_of_range",
        f"reason={failure.get('reason')!r}",
    )
    failures += not check(
        failure.get("valid_candidate_id_range") == [1, 4]
        and failure.get("available_candidate_ids") == [1, 2, 3, 4],
        "越界失败体给出可用编号范围（而不是只写进 error 文案）",
        f"range={failure.get('valid_candidate_id_range')!r}",
    )
    failures += not check(
        "不要" in str(failure.get("next_step") or ""),
        "越界失败的 next_step 明确禁止「自己换一个编号」",
        str(failure.get("next_step"))[:40],
    )
    failures += not check(
        "max_device_diameter_mm" not in failure,
        "越界失败体不掺器械上限（别把两种原因混在一起）",
    )

    # ---- ② 器械太粗：上限要能被执行 -------------------------------------
    thick = run("plan_route", case_id="LIDC_0089", candidate_id=3,
                device_diameter_mm=6.0)
    tfail = thick.get("failure") or {}
    bound = tfail.get("max_device_diameter_mm")
    failures += not check(
        thick.get("ok") is False and tfail.get("reason") == "device_too_thick",
        "器械过粗被归因为 device_too_thick",
        f"reason={tfail.get('reason')!r}",
    )
    failures += not check(
        isinstance(bound, (int, float)) and bound < 6.0,
        "器械过粗的失败体给出最大可通过外径（原来完全没有这个数）",
        f"max_device_diameter_mm={bound!r}",
    )

    # 关键：拿它当真，按这个上限重试一次 —— 必须成功。
    # 这条把「上限」从一个说法变成**可执行的指令**。
    if isinstance(bound, (int, float)):
        retry = run("plan_route", case_id="LIDC_0089", candidate_id=3,
                    device_diameter_mm=bound)
        failures += not check(
            retry.get("ok") is True,
            f"按失败体给出的上限 {bound}mm 重试同一个候选必须成功",
            f"ok={retry.get('ok')!r} err={retry.get('error')!r}",
        )
        # 而且是**同一个候选** —— 不许为了通过而偷偷换目标
        failures += not check(
            retry.get("candidate_id") == 3,
            "重试仍落在同一个候选上（上限不等于换目标）",
            f"candidate_id={retry.get('candidate_id')!r}",
        )

    # 与 scan_device_fit 的答案一致（分辨率以内）——
    # 同一件事给两个数字，医生会以为其中一个错了。
    scan = run("scan_device_fit", case_id="LIDC_0089", candidate_id=3)
    scan_max = scan.get("max_device_diameter_mm")
    if isinstance(bound, (int, float)) and isinstance(scan_max, (int, float)):
        failures += not check(
            abs(float(scan_max) - float(bound)) <= 0.1 + 1e-9,
            "失败体的上限与 scan_device_fit 的答案一致（≤1 个分辨率步长）",
            f"failure={bound} scan={scan_max}",
        )

    # ---- ⑤ 扫描上界只影响成本，不影响结论 ------------------------------
    # 旧实现「命中上界就直接返回 + 回显参数 + 让模型调大再测一次」，
    # 稳定复现的病征是**答案被参数回显**：模型自己传 max_diameter_mm=3
    # （3mm 细镜的临床先验，< 真值 5.46），旧码就把 3.0 当答案交出去，
    # 还让医生「调大上界重测」——答案错了题，评测却照样判通过。
    # ⚠️ 别拿「调用次数」当病征：改前 4 份 GPU 产物里 1 次/4 次都有，是噪声
    # （docs/评测区分度_同义改写实验.md §8.9 的未修邻项与实测表）。
    # 现在改成工具内部把上界抬到「路径最窄处直径」这个天然硬上界后继续收敛，
    # 于是这条不变量必须成立：**上界取多少都不能改变答案**（除分辨率以内）。
    tight = run("scan_device_fit", case_id="LIDC_0089", candidate_id=3,
                max_diameter_mm=2.0)
    base = scan.get("max_device_diameter_mm")
    narrow = tight.get("max_device_diameter_mm")
    if isinstance(base, (int, float)) and isinstance(narrow, (int, float)):
        failures += not check(
            abs(float(narrow) - float(base)) <= 0.1 + 1e-9,
            "扫描上界取 2.0mm 与默认值得到同一个答案（≤1 个分辨率步长）",
            f"上界 2.0 → {narrow}mm，默认 → {base}mm",
        )
        # 旧实现最直接的病征：把调用方传进来的上界原样回显成「答案」。
        # 只在真值确实高于该上界时才断言（否则 2.0 本来就可能是正解）。
        if float(base) > 2.0 + 0.1 + 1e-9:
            failures += not check(
                abs(float(narrow) - 2.0) > 1e-9,
                "过小的扫描上界被工具内部抬高，而不是回显成答案",
                f"max_device_diameter_mm={narrow!r}（不能等于传入的 2.0）",
            )

    # ---- ③ 气道不通（真实数据凑不出，用派生用例补）--------------------
    # `LIDC_0089` 的四个候选在 0.1mm 器械下都还有路，所以走不到这一支。
    # 按「真实数据凑不出的边界值用派生夹具补」的纪律，直接调纯函数：
    # `device_diameter_mm<=0` 这一支**不碰 ctx**，可以离线构造。
    from ..tools.imaging import _plan_failure  # noqa: PLC0415

    airway = _plan_failure(
        None, "LIDC_0089", 3, RuntimeError("候选 3 在当前器械约束下没有可行路径"),
        0.0, 0.2, 4, [1, 2, 3, 4],
    )
    afail = airway.get("failure") or {}
    failures += not check(
        afail.get("reason") == "route_infeasible_by_airway"
        and afail.get("max_device_diameter_mm") is None,
        "未施加器械约束仍失败 → 归因为 airway（换器械无用），不给上限",
        f"reason={afail.get('reason')!r}",
    )

    # ---- ④ 反向：成功路径不许带 failure -------------------------------
    good = run("plan_route", case_id="LIDC_0089", candidate_id=3,
               device_diameter_mm=2.0)
    failures += not check(
        good.get("ok") is True and "failure" not in good,
        "规划成功时不带 failure 字段（失败字段不污染正例）",
    )

    return failures


def main() -> int:
    # 第 1/3/4 层要**真的调** render_viewer（器械 1.5 mm），而演示产物是 2.0 mm 那份。
    # 不隔离的话，跑一次自检就把 outputs/viewers 里的入库产物换成 1.5 mm 版本，
    # `git status` 里多出一处和源码无关的改动。
    from ..render.viewer import use_scratch_output

    scratch = use_scratch_output("verify_payloads")

    print("=" * 68)
    print("AirNav-Agent 工具返回体自检（不产出文件 ≠ 可以给个 id 让模型自己猜）")
    print("=" * 68)
    print(f"渲染输出改到临时目录：{scratch}")

    failures = 0
    failures += layer_0_source_scan()
    failures += layer_1_live_payloads()
    failures += layer_2_notes()
    failures += layer_3_viewer_gives_file()
    failures += layer_4_render_hygiene()
    failures += layer_5_failure_contract()

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
    print("覆盖：源码扫描 · 实跑返回体 · 不产出文件的说明 · 出图与规划都给真文件 · 落盘卫生"
          " · 失败的归因与上限可执行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
