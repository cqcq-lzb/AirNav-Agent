"""面向医生的临床规划报告 —— 单文件 HTML，可打印、可转发、可离线打开。

与治理报告的分工
----------------
`report.py`（运行与合规报告）的读者是**评审方与运维**：谁用过、链是否完整、有没有越权、
PHI 结论、评测有没有退步。
本脚本的读者是**医生**：这个靶点够不够得着、这条路径能不能过器械、哪一段最紧、
分割有没有掺进后处理重建的气道、这份报告基于哪个模型版本。

两者共用同一套地基（单文件 HTML + 打印样式 + 三态结论），所以本脚本**直接复用**
`report.py` 的三态常量与总判定规则 —— 两处各写一份「什么算通过」，迟早会漂。

三条设计取舍
------------
1. **不重跑规划**。数值全部来自 `ViewerOutput.payload`（或磁盘上的 viewer sidecar），
   与三维视图**同一份数据**。「报告里的参数与页面所见一致」于是是结构上成立的，
   不是靠人去对齐数字 —— 这正是差距清单给这项定的验收标准。
2. **三维图分两种**。屏幕给内嵌的活视图（单文件 viewer 的 HTML 塞进 `srcdoc`），
   打印给 Python 算出来的正交投影 SVG（见 `agent/render/sketch.py` 顶部说明为什么不截图）。
   打印样式里前者隐藏、后者保留，不留空白框。
3. **缺数据不许说「通过」**。没指定器械外径 → 通过性判「未判定」而不是默认能过；
   路径依赖了分割修复段 → 判「未判定」并给出复核指引（数据不足以支持「这段气道真实存在」）。

用法::

    python -m agent.scripts.clinical_report LIDC_0089 --candidate 3 --device 2.0
    python -m agent.scripts.clinical_report --from-sidecar outputs/viewers/viewer_LIDC_0089_c3.json
    python -m agent.scripts.clinical_report LNDB_0196 --candidate 2 --device 1.5 --no-view

退出码：`0` = 已生成且无异常项；`1` = 已生成但有异常项（如靶点不可达）；
`2` = 无法生成（病例 / 路径数据缺失）。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.render import sketch, viewer  # noqa: E402

# 三态与总判定规则**只此一处定义**（在 report.py 里），这里复用而不是重写。
# check_clinical.py 会断言两边是同一个对象，防止哪天被复制成两份。
from agent.scripts.report import (  # noqa: E402
    BAD,
    NONE,
    OK,
    STATE_COLOR,
    overall,
)

#: 判定文案按读者改，**颜色不改** —— 颜色是与治理报告共用的视觉契约，
#: 自检会断言两边取值一致。临床上「器械通过性：符合预期」这种说法很别扭，
#: 所以标签用「通过 / 异常 / 未判定」，色值仍与 `report.STATE_COLOR` 逐项相同。
CLINICAL_TAG = {OK: "通过", BAD: "异常", NONE: "未判定"}

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "reports"

#: 报告内嵌三维视图的默认网格步长。
#: 报告要的是「形态看得出」，不是 step=1 的精度；数值与步长无关
#: （管腔半径来自距离变换，见 `render/payload.py` 的说明），所以粗网格不会让任何数字变化，
#: 只是让单文件小一半。报告里会把这个步长如实写出来。
DEFAULT_MESH_STEP = 4

#: 靶点可达分级 -> 三态。用的是 planner 自己的 grade，不另立阈值。
GRADE_STATE = {
    "adjacent": OK,     # 紧邻气道，路径可达性好
    "reachable": OK,    # 位于气道邻近区域，可规划路径
    "marginal": NONE,   # 覆盖可能不足 —— 数据不足以判定远端真实连通
    "unreachable": BAD, # 远离气道，不建议作为导航目标
    "unknown": NONE,
}

#: 打印/PDF 里必须留住的说明 —— 交互视图打不出来，得说清楚去哪儿看图。
PRINT_NOTICE = "打印版不含交互式三维视图（WebGL 画布无法打印）。路径的空间关系请看上面的路径示意图。"


def _esc(value: Any) -> str:
    """一切进入 HTML 的文本都要转义 —— 病例 ID、档位名、告警文案都算外部输入。"""
    return html.escape("" if value is None else str(value))


def _fmt(value: Any, digits: int = 2, unit: str = "") -> str:
    """数值格式化。**缺失一律显示 `—`，不显示 0** —— 0 会被读成「测出来是零」。"""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}{unit}"


def _signed(value: Any, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:+.{digits}f}{unit}"


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _safe_name(value: Any, fallback: str = "case") -> str:
    """产物文件名里的一个片段：只保留 `[A-Za-z0-9_.-]`。

    病例 ID 来自数据面。它出现在 `REPORT_DIR / name` 里，所以 `../` 或 `D:\\` 会
    让报告写到别处去 —— 与网页版 `/artifacts/<name>` 的目录穿越防护是同一类问题。
    **显示用原文（转义后），落盘用净化值**，两者不能混。
    """
    text = _SAFE_NAME_RE.sub("_", str(value or "")).strip("._") or fallback
    return text[:64]


# ------------------------------------------------------------------ 数据装配


@dataclass
class Source:
    """报告的数据来源。`payload` 是唯一真相，其余都只是为了写清出处。"""

    payload: dict[str, Any]
    viewer_html: str | None = None
    viewer_name: str | None = None
    sidecar: Path | None = None
    sidecar_sha256: str | None = None
    reused_viewer: bool = False
    notes: list[str] = field(default_factory=list)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_from_sidecar(path: Path) -> Source:
    """从一份已存在的 viewer sidecar（纯 JSON）装配报告，完全不碰 `cases/`。

    这条路径的价值不只是「快」：`outputs/viewers/` 里那几份是**入了库的演示产物**，
    干净 clone 也拿得到。于是自检与门禁能在没有病例数据的机器上验完整链路
    （`cases/` 有 1.9 G，不入库）。
    """
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict) or "meta" not in payload:
        raise ValueError(f"{path.name} 不像 viewer sidecar（缺 meta）")
    source = Source(payload=payload, sidecar=path, sidecar_sha256=_sha256_text(text))
    sibling = path.with_suffix(".html")
    if sibling.is_file():
        source.viewer_html = sibling.read_text(encoding="utf-8")
        source.viewer_name = sibling.name
        source.notes.append(
            f"三维视图直接复用磁盘上已有的 {sibling.name}（未重新渲染）—— "
            "报告内嵌的就是用户点开的那一份文件。"
        )
    else:
        source.notes.append(
            f"未找到同目录的 {sibling.name}，报告不含三维视图"
            "（路径示意图不受影响）。"
        )
    return source


def render_fresh(
    case_id: str,
    candidate_id: int,
    device_diameter_mm: float,
    profile_name: str = "balanced",
    mesh_step: int = DEFAULT_MESH_STEP,
) -> Source:
    """现场重新规划 + 渲染。需要 `cases/<病例>/stage4_package` 存在。"""
    case_dir = Path(_cases_root()) / case_id / "stage4_package"
    if not case_dir.is_dir():
        raise FileNotFoundError(f"找不到病例目录：{case_dir}")
    from agent.core import case_loader  # 延迟导入：--from-sidecar 路径不需要它

    case = case_loader.load_case(case_dir, case_id=case_id)
    from agent.core.planner import plan_candidate

    plan, _alternatives = plan_candidate(
        case,
        candidate_id,
        profile_name=profile_name,
        device_diameter_mm=device_diameter_mm,
    )
    output = viewer.render_case_viewer(
        case, plan, device_diameter_mm, mesh_step=mesh_step
    )
    try:
        text = Path(output.path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:  # noqa: BLE001 - 读不到就说读不到
        text = None
        note = f"三维视图文件读取失败（{type(exc).__name__}），报告不含内嵌视图。"
    else:
        note = (
            f"三维视图随本次规划现场渲染：{Path(output.path).name}"
            f"（{'命中去重，复用同参数旧文件' if output.reused else '本次新写入'}）。"
        )
    return Source(
        payload=output.payload,
        viewer_html=text,
        viewer_name=Path(output.path).name,
        sidecar=Path(output.sidecar),
        sidecar_sha256=_sha256_text(output.sidecar.read_text(encoding="utf-8")),
        reused_viewer=output.reused,
        notes=[note],
    )


def _cases_root() -> str:
    from agent.core.case_loader import cases_root

    return cases_root()


# ------------------------------------------------------------------ 结论


def verdicts(payload: dict, *, embedded_view: bool, mesh_step: int) -> list[dict]:
    """五项三态结论。每项都说清「依据」与「数据来源」，`none` 一律带补齐方式。"""
    meta = payload.get("meta") or {}
    route = payload.get("route") or {}
    metrics = payload.get("metrics") or {}
    items: list[dict] = []

    device = float(meta.get("deviceDiameterMm") or 0.0)
    margin = float(meta.get("deviceMarginMm") or 0.0)
    clearances = route.get("clearanceMm") or []
    diameters = route.get("lumenRadiiMm") or []

    # ① 器械通过性：这条路径最窄的地方，器械加双侧余量放得下吗
    if device <= 0:
        items.append({
            "name": "器械通过性", "state": NONE,
            "detail": "本次未指定器械外径，无法判定这条路径能否通过器械。"
                      "「没测」不等于「能过」。",
            "fix": "带 --device <外径 mm> 重新生成（如 --device 2.0）",
            "source": "meta.deviceDiameterMm",
        })
    elif not clearances:
        items.append({
            "name": "器械通过性", "state": NONE,
            "detail": "路径负载里没有逐点余量数据，无法判定通过性。",
            "fix": "确认该 sidecar 由 agent/render/payload.py 生成（route.clearanceMm）",
            "source": "route.clearanceMm",
        })
    else:
        tightest = min(float(value) for value in clearances)
        narrow = min(float(value) for value in diameters) * 2.0 if diameters else None
        required = device + 2.0 * margin
        state = OK if tightest >= 0 else BAD
        detail = (
            f"路径最窄处直径 {_fmt(narrow)} mm；器械外径 {_fmt(device)} mm + 双侧安全余量 "
            f"{_fmt(margin)} mm = 要求 {_fmt(required)} mm；"
            f"全程最小余量 {_signed(tightest)} mm。"
            + ("" if state == OK else "器械在本路径最窄处放不下，当前口径不建议通过。")
        )
        items.append({
            "name": "器械通过性", "state": state, "detail": detail,
            "fix": None if state == OK else "换更细的器械重规划，或改用其它候选灶 / 代价配置",
            "source": "route.clearanceMm, meta.deviceDiameterMm",
        })

    # ② 路径有效性
    valid = metrics.get("route_valid")
    waypoints = route.get("waypointCount")
    if valid is None:
        # **键不存在 ≠ 键的值为假**。少了它只能判「未判定」：
        # 判 bad 是假警报，判 ok 就是拿缺数据冒充结论。
        items.append({
            "name": "路径有效性", "state": NONE,
            "detail": "负载里没有路径有效性标记（metrics.route_valid），无法判定。",
            "fix": "用 agent.scripts.cli 的 plan_route 重新规划后再导出报告",
            "source": "metrics.route_valid",
        })
    else:
        state = OK if valid else BAD
        detail = (
            f"路径共 {_fmt(waypoints, 0)} 个路点，总长 {_fmt(metrics.get('route_length_mm'), 1)} mm。"
        )
        rejected = metrics.get("degenerate_routes_rejected")
        if rejected:
            detail += f"规划过程中拦截了 {rejected} 条退化路径（起点即终点），未参与比较。"
        items.append({
            "name": "路径有效性", "state": state, "detail": detail,
            "fix": None if state == OK else "看规划返回的失败原因，换器械口径或代价配置重试",
            "source": "metrics.route_valid, metrics.route_length_mm",
        })

    # ③ 靶点可达分级（用 planner 自己的 grade）
    reach = payload.get("reachability") or {}
    grade = str(reach.get("grade") or "unknown")
    state = GRADE_STATE.get(grade, NONE)
    distance = reach.get("distance_mm")
    detail = (
        f"靶点距气道中心线 {_fmt(distance)} mm —— {reach.get('label') or '无分级说明'}"
        f"（分级：{grade}）"
    )
    fix = {
        NONE: "人工结合 CT 判断远端细小气道是否真实连通；必要时细化该区域分割后重规划",
        BAD: "该靶点不适合作为支气管镜导航目标；如仍需接近，请评估经皮等其它入路",
    }.get(state)
    items.append({
        "name": "靶点可达性", "state": state, "detail": detail, "fix": fix,
        "source": "reachability（agent/core/planner.py:classify_reachability）",
    })

    # ④ 分割依赖复核：路径有没有走过「分割后处理造出来的气道」
    bridge_raw = metrics.get("bridge_voxels_on_route")
    recovered_raw = metrics.get("recovered_voxels_on_route")
    repaired = payload.get("repaired") or {}
    if bridge_raw is None or recovered_raw is None:
        # 这里踩过一次坑：写成 `int(raw or 0)`，键缺失就变成 0，
        # 于是报告会**在没有数据时宣称「本路径未经过任何分割修复段」**——
        # 一个凭空的肯定结论。缺键必须落到「未判定」。
        items.append({
            "name": "分割依赖复核", "state": NONE,
            "detail": "负载里没有「路径是否经过分割修复段」的计数，"
                      "无法判定这条路径有没有依赖后处理造出的气道。",
            "fix": "确认该 sidecar 由 agent/render/payload.py 生成"
                   "（metrics.bridge_voxels_on_route / recovered_voxels_on_route）",
            "source": "metrics.bridge_voxels_on_route, metrics.recovered_voxels_on_route",
        })
    elif bridge_raw or recovered_raw:
        bridge, recovered = int(bridge_raw), int(recovered_raw)
        items.append({
            "name": "分割依赖复核", "state": NONE,
            "detail": (
                f"路径有 {recovered} 个体素落在「形态学修复」区域、{bridge} 个体素落在"
                "「连通性补桥」区域。这些段气道来自分割后处理而非原始影像，"
                "现有数据不足以判定该段气道真实存在，需人工复核。"
            ),
            "fix": "在三维视图里复核该段支路；必要时用更严的分割参数重新分割后重规划",
            "source": "metrics.bridge_voxels_on_route, metrics.recovered_voxels_on_route",
        })
    else:
        detail = "本路径未经过任何分割修复段，路径所依赖的气道均来自原始影像分割结果。"
        if repaired:
            labels = "、".join(
                str(seg.get("label")) for seg in repaired.values() if seg.get("label")
            )
            detail += f"（本病例存在修复段：{labels}；但本路径不经过。）"
        items.append({
            "name": "分割依赖复核", "state": OK, "detail": detail, "fix": None,
            "source": "metrics.*_voxels_on_route, repaired",
        })

    # ⑤ 交付完整性：这份文件本身有没有给出它声称给出的东西
    triangles = (payload.get("airwayStats") or {}).get("triangleCount")
    if not embedded_view:
        items.append({
            "name": "交付完整性", "state": NONE,
            "detail": "本次以 --no-view 生成，报告不含内嵌三维视图（路径示意图仍在）。",
            "fix": "去掉 --no-view 重新生成即可内嵌三维视图",
            "source": "命令行参数",
        })
    else:
        items.append({
            "name": "交付完整性", "state": OK,
            "detail": (
                f"三维视图已内嵌（网格步长 {mesh_step}，气道 {_fmt(triangles, 0)} 个三角面）；"
                "路径示意图由同一份负载投影生成。"
            ),
            "fix": None, "source": "airwayStats.triangleCount",
        })

    return items


def key_numbers(payload: dict) -> list[dict]:
    """指标卡。每项都带 `src`（来自负载的哪个键）与 `raw`（原始值）。

    `raw` 同时写进卡片的 `data-raw` 属性：报告本来就要求「结构化 + 图片」，
    把原始值连同来源键一起嵌进去，报告就成了**可被程序逐项复核**的东西 ——
    自检正是靠它断言「报告里的数字 == 负载里的数字」，而不是去解析显示文本再猜。
    """
    meta = payload.get("meta") or {}
    route = payload.get("route") or {}
    metrics = payload.get("metrics") or {}
    clearances = route.get("clearanceMm") or []
    device = float(meta.get("deviceDiameterMm") or 0.0)
    tightest = min(float(value) for value in clearances) if clearances else None
    return [
        {"k": "最窄处直径", "v": _fmt(metrics.get("minimum_diameter_mm")), "u": "mm",
         "n": "路径全程气道管腔最小值",
         "src": "metrics.minimum_diameter_mm", "raw": metrics.get("minimum_diameter_mm")},
        {"k": "全程最小余量", "v": (None if tightest is None else _signed(tightest)),
         "u": "mm", "n": "直径减去器械与双侧余量后的最紧处",
         "src": "route.clearanceMm", "raw": tightest},
        {"k": "路径总长", "v": _fmt(metrics.get("route_length_mm"), 1), "u": "mm",
         "n": f"{_fmt(route.get('waypointCount'), 0)} 个路点累计",
         "src": "metrics.route_length_mm", "raw": metrics.get("route_length_mm")},
        {"k": "最大转角", "v": _fmt(metrics.get("maximum_turn_angle_deg"), 1), "u": "°",
         "n": "相邻路点转折的最大值",
         "src": "metrics.maximum_turn_angle_deg", "raw": metrics.get("maximum_turn_angle_deg")},
        {"k": "靶点距中心线", "v": _fmt(metrics.get("target_distance_mm")), "u": "mm",
         "n": "靶点到气道中心线的最近距离",
         "src": "metrics.target_distance_mm", "raw": metrics.get("target_distance_mm")},
        {"k": "器械外径", "v": (None if device <= 0 else _fmt(device)), "u": "mm",
         "n": "本次规划采用的器械",
         "src": "meta.deviceDiameterMm", "raw": (None if device <= 0 else device)},
    ]


def param_rows(payload: dict) -> list[tuple[str, str]]:
    """器械与模型版本 —— 报告要能回答「这是哪一版算出来的」。"""
    meta = payload.get("meta") or {}
    centerline = payload.get("centerline") or {}
    spacing = meta.get("spacingZyx") or []
    size = meta.get("imageSize") or []
    spacing_text = " × ".join(_fmt(value, 4) for value in spacing) if spacing else "—"
    size_text = " × ".join(str(int(value)) for value in size) if size else "—"
    return [
        ("病例", str(meta.get("caseId") or "—")),
        ("候选灶", f"c{meta.get('candidateId')}（上游编号 {_fmt(meta.get('serverCandidateId'), 0)}）"),
        ("代价配置", f"{meta.get('profile')} · {meta.get('profileTitle')}"),
        ("器械外径", _fmt(meta.get("deviceDiameterMm"), 2, " mm")),
        ("双侧安全余量", _fmt(meta.get("deviceMarginMm"), 2, " mm")),
        ("入口模式", str(meta.get("entryMode") or "—")),
        ("拓扑签名", str(meta.get("topologySignature") or "—")),
        ("气道中心线", f"{_fmt(centerline.get('nodeCount'), 0)} 节点 / "
                       f"{_fmt(centerline.get('edgeCount'), 0)} 边 / "
                       f"{_fmt(centerline.get('junctionCount'), 0)} 个分叉点"),
        ("影像尺寸（层 × 行 × 列）", size_text),
        ("体素间距 Z×Y×X", spacing_text + " mm" if spacing else "—"),
        ("规划内核", str(meta.get("engine") or "—")),
        ("视图装配时刻", str(meta.get("generatedAt") or "—")),
    ]


# ------------------------------------------------------------------ 渲染

CSS = """
:root { --ok:#0f766e; --okbg:#ccfbf1; --bad:#a32d2d; --badbg:#fee2e2;
  --none:#78786f; --nonebg:#f1f0ec; --ink:#23231f; --dim:#6b6b64;
  --line:#e6e6e1; --paper:#f7f7f5; }
* { box-sizing: border-box; }
body { margin:0; padding:34px 30px 64px; background:var(--paper); color:var(--ink);
  font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif; font-size:14px; line-height:1.7; }
.wrap { max-width:1060px; margin:0 auto; }
.head { display:flex; justify-content:space-between; align-items:flex-start; gap:26px;
  padding-bottom:16px; border-bottom:2px solid var(--ink); }
.eyebrow { font-size:11.5px; letter-spacing:.16em; color:var(--dim); margin:0 0 10px; }
h1 { font-size:24px; font-weight:600; margin:0 0 7px; letter-spacing:-.3px; }
.sub { color:var(--dim); font-size:13px; margin:0; }
.stamp { text-align:right; flex:0 0 auto; }
.stamp .k { font-size:11px; color:var(--dim); margin:0; letter-spacing:.08em; }
.stamp .v { font-size:12.5px; margin:0 0 7px; font-variant-numeric:tabular-nums; }
h2 { font-size:15.5px; font-weight:600; margin:38px 0 12px; padding-bottom:8px;
  border-bottom:1px solid var(--line); }
h3 { font-size:13px; font-weight:600; margin:0 0 8px; color:var(--dim); }
.banner { border-radius:12px; padding:16px 20px; margin:18px 0 6px; border:1px solid; }
.banner .t { font-size:15px; font-weight:600; margin:0 0 4px; }
.banner .d { font-size:13px; margin:0; }
.banner.ok { background:#f0fdfa; border-color:#a7d9d0; color:#0b5b53; }
.banner.bad { background:#fef2f2; border-color:#e8b4b4; color:#8f2626; }
.banner.none { background:#fbfaf7; border-color:#e0ddd3; color:#6b6656; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; font-weight:600; font-size:11.5px; color:var(--dim); padding:8px 10px;
  border-bottom:1px solid var(--line); white-space:nowrap; }
td { padding:10px; border-bottom:1px solid #f0f0ec; vertical-align:top; }
tr:last-child td { border-bottom:none; }
.src { display:block; font-size:11px; color:#9a9a91; margin-top:4px; }
.fix { display:block; margin-top:5px; font-size:12px; color:#8a6d1f; }
.tag { display:inline-block; padding:1px 9px; border-radius:20px; font-size:12px; font-weight:500;
  white-space:nowrap; }
code { font-family:Consolas,"Courier New",monospace; font-size:11.5px; background:#f2f2ee;
  padding:1px 5px; border-radius:4px; word-break:break-all; }
.mono { font-family:Consolas,"Courier New",monospace; font-size:11.5px; color:#5c5c55; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.metric { background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.metric .k { font-size:12px; color:var(--dim); margin:0 0 6px; }
.metric .v { font-size:23px; font-weight:600; margin:0; letter-spacing:-.5px;
  font-variant-numeric:tabular-nums; }
.metric .u { font-size:13px; font-weight:400; color:var(--dim); margin-left:3px; }
.metric .n { font-size:11.5px; color:#8a8a81; margin:5px 0 0; }
.fig { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px;
  margin-bottom:14px; }
.fig h3 { margin-bottom:10px; }
.row { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.viewer { width:100%; height:560px; border:1px solid var(--line); border-radius:12px;
  background:#fff; display:block; }
.hint { font-size:12px; color:var(--dim); margin:9px 0 0; }
ul, ol { margin:8px 0; padding-left:22px; } li { margin:6px 0; }
.note { background:#fffdf5; border:1px solid #ece3c9; border-radius:10px; padding:15px 18px;
  font-size:13px; color:#4a4230; }
.note b { font-weight:600; }
.print-only { display:none; }
.boundary { font-size:12.5px; color:var(--dim); }
.boundary li { margin:5px 0; }
@media print {
  body { background:#fff; padding:0; font-size:11px; }
  .wrap { max-width:none; }
  h1 { font-size:18px; } h2 { font-size:13px; margin:20px 0 8px; page-break-after:avoid; }
  .head { border-bottom:1px solid #000; padding-bottom:8px; }
  .interactive-only { display:none !important; }
  .print-only { display:block; }
  .fig, .banner, .note, .metric, tr { break-inside:avoid; }
  a { color:inherit; text-decoration:none; }
  @page { margin:14mm; }
}
"""


def _tag(state: str) -> str:
    color, background = STATE_COLOR[state]
    return (
        f'<span class="tag" style="color:{color};background:{background}">'
        f"{_esc(CLINICAL_TAG[state])}</span>"
    )


def _verdict_table(items: list[dict]) -> str:
    rows = []
    for item in items:
        fix = item.get("fix")
        fix_html = f'<span class="fix">补齐：{_esc(fix)}</span>' if fix else ""
        rows.append(
            f"<tr><td><b>{_esc(item['name'])}</b>"
            f'<span class="src">来源：{_esc(item.get("source"))}</span></td>'
            f"<td>{_tag(item['state'])}</td>"
            f"<td>{_esc(item['detail'])}{fix_html}</td></tr>"
        )
    return (
        "<table><thead><tr><th>判定项</th><th>判定</th><th>依据</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _figure(svg: str, title: str) -> str:
    return f'<div class="fig"><h3>{_esc(title)}</h3>{svg}</div>'


def _viewer_block(source: Source, *, embedded: bool) -> str:
    if not embedded:
        # 屏幕上也要说清楚 —— 只写 print-only 的话，页面上这一节会整块空白。
        return (
            '<div class="note"><b>本次未内嵌三维视图（--no-view）。</b>'
            "路径的空间关系见上面的路径示意图；如需交互视图，去掉 --no-view 重新生成。"
            "</div>"
        )
    if not source.viewer_html:
        return (
            f'<div class="note"><b>本报告不含三维视图。</b>{_esc(" ".join(source.notes))}</div>'
        )
    # 单文件 viewer 直接塞进 `srcdoc`：不落第二个文件、不依赖相对路径，
    # 报告转发出去照样能转。属性值整体转义，内容无法从 iframe 里跑出来。
    payload_attr = html.escape(source.viewer_html, quote=True)
    name = _esc(source.viewer_name or "")
    return (
        f'<div class="interactive-only">'
        f'<iframe class="viewer" title="三维路径视图" loading="eager" '
        f'srcdoc="{payload_attr}"></iframe>'
        f'<p class="hint">拖动旋转 / 右键平移 / 滚轮缩放；视角预设见视图内工具条。'
        f'本视图为自包含单文件（{name}），可单独打开、可离线使用。</p></div>'
        f'<div class="print-only"><div class="note">{_esc(PRINT_NOTICE)}</div></div>'
    )


def build_html(
    source: Source,
    *,
    items: list[dict],
    state: str,
    sketches: list[tuple[str, str, dict]],
    mesh_step: int,
    embedded: bool,
) -> str:
    payload = source.payload
    meta = payload.get("meta") or {}
    case_id = str(meta.get("caseId") or "未知病例")
    candidate = meta.get("candidateId")
    banner_text = {
        OK: ("结论：在可判定的范围内未发现异常", "五项结论全部有据可依。"),
        BAD: ("结论：存在异常项，需人工介入", "有判定项与预期不符。报告只呈现事实，不代替临床判断。"),
        NONE: ("结论：证据不足，无法给出完整判定", "有判定项缺数据或需人工复核 —— 这不等于「没有问题」。"),
    }[state]

    cards = []
    for number in key_numbers(payload):
        unit = f'<span class="u">{_esc(number["u"])}</span>' if number["u"] else ""
        raw = "" if number["raw"] is None else repr(number["raw"])
        cards.append(
            f'<div class="metric" data-src="{_esc(number["src"])}" data-raw="{_esc(raw)}">'
            f'<p class="k">{_esc(number["k"])}</p>'
            f'<p class="v">{_esc(number["v"])}{unit}</p>'
            f'<p class="n">{_esc(number["n"])}</p>'
            f'<span class="src">来源：{_esc(number["src"])}</span></div>'
        )

    rows = "".join(
        f"<tr><td>{_esc(key)}</td><td>{_esc(value)}</td></tr>"
        for key, value in param_rows(payload)
    )

    warnings = payload.get("warnings") or []
    notes_list = "".join(f"<li>{_esc(text)}</li>" for text in warnings)
    if not notes_list:
        notes_list = "<li>规划内核未给出告警项。</li>"

    out_of_view = [
        name for _view, _svg, info in sketches for name in (info.get("out_of_view") or [])
    ]
    figure_notes = []
    if out_of_view:
        figure_notes.append(
            "示意图中带「视野外」的标记表示该点落在取景范围外，已夹到边缘并画出指向箭头 ——"
            "箭头方向才是它的真实方位，不要按标记位置读。"
        )
    figure_notes.append(
        "取景框按气道中心线范围缩放，故远处靶点可能落在框外（避免把气道树压成一小团）。"
    )

    main_fig = sketches[0]
    side_figs = sketches[1:]
    figures_html = _figure(main_fig[1], main_fig[0])
    if side_figs:
        figures_html += '<div class="row">' + "".join(
            _figure(svg, title) for title, svg, _info in side_figs
        ) + "</div>"

    source_name = "—"
    if source.sidecar is not None:
        source_name = source.sidecar.name
    trace = [
        f"数值与几何来源：{source_name}"
        + (f"（sha256 {source.sidecar_sha256[:16]}…）" if source.sidecar_sha256 else ""),
        f"三维视图：{source.viewer_name or '未包含'}",
        f"报告内嵌视图网格步长：{mesh_step}（仅影响显示精度，不影响任何数值 —— "
        "管腔半径来自距离变换，与显示网格无关）",
        "同源文件可用来逐项复核上面的数字：报告与三维视图读的是同一份负载，不是两次计算。",
    ]
    trace.extend(source.notes)

    # Python 3.10 的 f-string 不允许在表达式里复用外层引号（3.12 才放开），
    # 所以所有值一律先接到变量再进模板 —— 与 report.py 同一范式。
    esc_case = _esc(case_id)
    esc_candidate = _esc(candidate)
    esc_profile = _esc(meta.get("profile"))
    esc_profile_title = _esc(meta.get("profileTitle"))
    esc_engine = _esc(meta.get("engine"))
    sub_text = (
        f"器械外径 {_fmt(meta.get('deviceDiameterMm'))} mm · "
        f"双侧安全余量 {_fmt(meta.get('deviceMarginMm'))} mm · "
        f"代价配置 {esc_profile}（{esc_profile_title}）"
    )
    stamp_time = _esc(f"{datetime.now():%Y-%m-%d %H:%M}")
    banner_title = _esc(banner_text[0])
    banner_detail = _esc(banner_text[1])
    verdict_html = _verdict_table(items)
    viewer_html = _viewer_block(source, embedded=embedded)
    hint_text = _esc(" ".join(figure_notes))
    cards_html = "".join(cards)
    notes_html = notes_list
    params_html = rows
    boundary_html = "".join(f"<li>{_esc(line)}</li>" for line in trace)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>支气管镜导航规划报告 · {esc_case} c{esc_candidate}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div>
      <p class="eyebrow">AIRNAV-AGENT · 支气管镜导航规划报告</p>
      <h1>{esc_case} · 候选灶 c{esc_candidate}</h1>
      <p class="sub">{sub_text}</p>
    </div>
    <div class="stamp">
      <p class="k">生成时间</p><p class="v">{stamp_time}</p>
      <p class="k">规划内核</p><p class="v">{esc_engine}</p>
    </div>
  </div>

  <div class="banner {state}">
    <p class="t">{banner_title}</p>
    <p class="d">{banner_detail}</p>
  </div>

  <h2 id="conclusion">一、结论</h2>
  {verdict_html}

  <h2 id="figures">二、路径示意图</h2>
  {figures_html}
  <p class="hint">{hint_text}</p>

  <h2 id="numbers">三、关键参数</h2>
  <div class="cards">{cards_html}</div>

  <h2 id="view3d">四、三维视图</h2>
  {viewer_html}

  <h2 id="notes">五、注意事项</h2>
  <div class="note"><ul>{notes_html}</ul></div>

  <h2 id="params">六、器械与模型版本</h2>
  <table><thead><tr><th>项</th><th>值</th></tr></thead><tbody>{params_html}</tbody></table>

  <h2 id="method">七、方法与边界</h2>
  <ul class="boundary">
    <li>路径由 AirNav-Agent 无界面规划内核计算（<code>agent/core/planner.py</code>），
        规划数值与原 Airway 导航系统 V1 逐位一致，回归门禁含几何对拍与规划一致性两项硬检查。</li>
    <li>报告内所有数值与三维视图读的是<b>同一份负载</b>，不是两次计算 ——
        「报告与页面不一致」在结构上不会发生。</li>
    <li>三维视图是自包含单文件 HTML，无外部依赖、无内网地址，可直接转发与离线打开。</li>
    <li>本报告不含患者可识别信息（PHI）：数据面为 NIfTI 影像，不含 DICOM 标签，
        并有常驻守卫 <code>agent/scripts/scan_phi.py</code> 定期扫描。</li>
    <li>病例来自公开数据集（LIDC-IDRI / LNDb），<b>未经本院临床数据验证</b>；
        本报告为工程研究演示结果，<b>不用于临床诊断与治疗决策</b>。</li>
  </ul>
  <h3>可追溯性</h3>
  <ul class="boundary">{boundary_html}</ul>
</div>
</body>
</html>
"""


def _sketch_set(payload: dict) -> list[tuple[str, str, dict]]:
    """三张图：斜位大图立体的看，前后位 / 侧位按阅片习惯看。

    图内标题关掉、改用报告自己的 `<h3>` —— 同一句话说两遍很难看，
    而 `<h3>` 用的是页面排版（字号、间距统一），也更适合打印。
    标题文案取自 `sketch.VIEW_TITLES`，说的仍是同一件事。
    """
    plan = (
        ("oblique", 1040, 470, 2.0),
        ("ap", 508, 430, 2.5),
        ("lateral", 508, 430, 2.5),
    )
    out = []
    for view, width, height, min_px in plan:
        svg, info = sketch.render_sketch(
            payload, view=view, width=width, height=height,
            min_segment_px=min_px, show_title=False,
        )
        out.append((sketch.VIEW_TITLES[view], svg, info))
    return out


# ------------------------------------------------------------------ 入口


def generate(
    *,
    case_id: str | None,
    candidate_id: int | None,
    device_diameter_mm: float,
    profile_name: str,
    mesh_step: int,
    sidecar: Path | None,
    embedded: bool,
) -> tuple[str, Source, list[dict], str, Path]:
    if sidecar is not None:
        source = load_from_sidecar(sidecar)
        if case_id is None:
            case_id = str((source.payload.get("meta") or {}).get("caseId") or "未知病例")
    else:
        if not case_id or candidate_id is None:
            raise ValueError("需要 <病例> 与 --candidate（或用 --from-sidecar 指定已有 sidecar）")
        source = render_fresh(
            case_id, candidate_id, device_diameter_mm, profile_name, mesh_step
        )

    meta = source.payload.get("meta") or {}
    if candidate_id is None:
        candidate_id = meta.get("candidateId")
    # 显示用的步长必须取**负载里真实的那一个**，不能取命令行那个：
    # 从 sidecar 装配时视图是复用的，它的步长写在 meta 里。照抄命令行参数
    # 会让报告声称一个它没使用过的精度 —— 报告可以少说，不能错说。
    actual_step = int(meta.get("meshStep") or mesh_step)
    use_view = bool(embedded) and source.viewer_html is not None
    items = verdicts(source.payload, embedded_view=embedded, mesh_step=actual_step)
    state = overall(items)
    sketches = _sketch_set(source.payload)
    page = build_html(
        source, items=items, state=state, sketches=sketches,
        mesh_step=actual_step, embedded=embedded,
    )
    name = (
        f"clinical_{_safe_name(meta.get('caseId'))}_c{_safe_name(meta.get('candidateId'), '0')}"
        f"{viewer.param_tag(meta.get('deviceDiameterMm'), meta.get('profile'), actual_step)}"
        f"{'' if use_view else '_noview'}.html"
    )
    return page, source, items, state, REPORT_DIR / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.scripts.clinical_report",
        description="生成面向医生的单文件临床规划报告（含路径示意图与三维视图）。",
    )
    parser.add_argument("case_id", nargs="?", help="病例 ID（用 --from-sidecar 时可省略）")
    parser.add_argument("--candidate", type=int, help="候选灶编号（candidate_id）")
    parser.add_argument("--device", type=float, default=0.0, help="器械外径 mm（默认 0 = 不指定）")
    parser.add_argument("--profile", default="balanced", help="代价配置（默认 balanced）")
    parser.add_argument("--mesh-step", type=int, default=DEFAULT_MESH_STEP,
                        help=f"内嵌视图网格步长（默认 {DEFAULT_MESH_STEP}，只影响显示精度）")
    parser.add_argument("--from-sidecar", type=Path,
                        help="直接从已有的 viewer sidecar(JSON) 生成，不重跑规划、不需要病例数据")
    parser.add_argument("--no-view", action="store_true", help="不内嵌三维视图（报告体积约小一半）")
    parser.add_argument("--out", type=Path, help="输出 HTML 路径")
    parser.add_argument("--quiet", action="store_true", help="不打印摘要")
    args = parser.parse_args(argv)

    try:
        page, source, items, state, default_path = generate(
            case_id=args.case_id,
            candidate_id=args.candidate,
            device_diameter_mm=args.device,
            profile_name=args.profile,
            mesh_step=args.mesh_step,
            sidecar=args.from_sidecar,
            embedded=not args.no_view,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"无法生成报告：{exc}", file=sys.stderr)
        return 2

    target = Path(args.out) if args.out else default_path
    target.parent.mkdir(parents=True, exist_ok=True)
    # Python 写 .html 是本机 DLP 矩阵里的明文档，不需要补丁脚本。
    target.write_text(page, encoding="utf-8")

    if not args.quiet:
        pending = [item for item in items if item["state"] == NONE]
        failed = [item for item in items if item["state"] == BAD]
        print(f"已写入报告：{target}（{len(page.encode('utf-8'))} 字节）")
        print(f"  判定 {state} · 异常 {len(failed)} 项 · 未判定 {len(pending)} 项")
        for item in failed + pending:
            print(f"    [{item['state']}] {item['name']}：{item['detail'][:70]}")
        if source.viewer_html:
            print(f"  三维视图：{source.viewer_name}（已内嵌）")
        for note in source.notes:
            print(f"  说明：{note}")
    return 1 if state == BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
