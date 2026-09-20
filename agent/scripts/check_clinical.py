"""临床报告自检：三态不许放水、数字必须与负载逐位一致、投影可独立重算、打印不留白框。

    python -m agent.scripts.check_clinical

秒级、不联网、不要模型、**不需要 `cases/`**。夹具只用 `outputs/viewers/` 里**已入库**的两份
sidecar（JSON），所以干净 clone、CI 里都能跑完整链路（`cases/` 有 1.9 G，不入库）。

## 夹具为什么是「真实 + 派生」两种

- **真实夹具**（`LIDC_0089` 的 c3 / c1）：负责只有真病例才有的分支 ——
  全项通过、靶点 68 mm 不可达、路径经过分割修复段。这些**编不出来**。
- **派生夹具**：把一个真实负载的器械外径改到使最紧处余量恰好 `+0.05 mm` / `-0.05 mm`，
  并**按 `payload.py` 的同一个公式**重算 `route.clearanceMm`，保证负载自洽。
  真实数据凑不出这个边界值，而「余量落在 0 的哪一侧」正是三态最容易写错的地方
  （`>= 0` 写成 `> 0`，就会把「刚好放得下」判成异常）。自检先验一遍派生公式本身成立。

> 早先这里用的是另一份 `viewer_LNDB_0196_c2_d1.5.json`（磁盘上有、**没入库**）。
> 那种依赖在干净 clone 里会直接红 —— 自检的夹具必须是**跟着仓库走的**，
> 否则它保护的只是本机这一份工作区。

## 这一层在防什么

验收标准是「一次规划导出一份带图报告，**参数与页面所见一致**」。
「一致」最容易假过的方式是：报告自己算一遍数字，看上去对，其实和三维视图那份不是同一次计算。
所以这里不比对「显示得好不好看」，而是**从产物 HTML 里把 `data-raw` 抠出来，
和 sidecar JSON 里的原始值逐位比**；再单独验一遍显示文本确实是这个原始值的正确舍入。

另两条是治理侧报告踩过的坑，这里照原样防：
- **缺数据不许说通过**：没指定器械外径时，「器械通过性」必须是「未判定」而不是默认能过；
- **结论类产物必须能被触发**：夹具里特意放了不可达（`unreachable`）与依赖分割修复段两种真实病例，
  否则这些分支永远只有「一切正常」这一种表现，等于没测。
"""
from __future__ import annotations

import json
import html as html_mod
import re
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.render import sketch  # noqa: E402
from agent.scripts import clinical_report as clinical  # noqa: E402
from agent.scripts import report as report_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VIEWERS = ROOT / "outputs" / "viewers"

#: 真实夹具：绿色（全项通过）、红色（靶点 68 mm 不可达 + 未指定器械）
FIXTURES = {
    "green": VIEWERS / "viewer_LIDC_0089_c3.json",
    "red": VIEWERS / "viewer_LIDC_0089_c1.json",
}

#: 派生夹具的边界余量（mm）：刚好放得下 / 刚好放不下
EDGE_CLEARANCE_MM = 0.05

RESULTS: list[tuple[bool, str, str]] = []

#: 指标卡的绑定表：`data-src` -> 从 payload 里取原始值的取法。
#: 这份映射是**独立写出来的**，不从 clinical_report 里 import ——
#: 否则「报告算错了」和「检查器跟着算错」会一起过。
METRIC_SOURCES = {
    "metrics.minimum_diameter_mm": lambda p: p["metrics"]["minimum_diameter_mm"],
    "route.clearanceMm": lambda p: min(float(v) for v in p["route"]["clearanceMm"]),
    "metrics.route_length_mm": lambda p: p["metrics"]["route_length_mm"],
    "metrics.maximum_turn_angle_deg": lambda p: p["metrics"]["maximum_turn_angle_deg"],
    "metrics.target_distance_mm": lambda p: p["metrics"]["target_distance_mm"],
    "meta.deviceDiameterMm": lambda p: (
        None if float(p["meta"].get("deviceDiameterMm") or 0.0) <= 0
        else p["meta"]["deviceDiameterMm"]
    ),
}

CARD_RE = re.compile(
    r'<div class="metric" data-src="([^"]*)" data-raw="([^"]*)"'
    r'><p class="k">([^<]*)</p><p class="v">([^<]*)'
)


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((ok, label, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"（{detail}）" if detail else ""))
    return ok


def _payload(kind: str) -> dict:
    return json.loads(FIXTURES[kind].read_text(encoding="utf-8"))


def with_edge_clearance(payload: dict, target_mm: float) -> dict:
    """派生夹具：改器械外径，使最紧处余量恰好为 `target_mm`，并重算逐点余量。

    用的是 `render/payload.py` 里同一个式子（`clearance = 管腔半径 − 外径/2 − 安全余量`），
    所以产物负载是自洽的，而不是手改一个数字留下矛盾。
    调用方会先断言 `min(clearance) ≈ target_mm` —— 派生公式本身也要能被验。
    """
    import copy

    out = copy.deepcopy(payload)
    radii = [float(r) for r in out["route"]["lumenRadiiMm"]]
    margin = float(out["meta"].get("deviceMarginMm") or 0.0)
    device = 2.0 * (min(radii) - margin - target_mm)
    out["meta"]["deviceDiameterMm"] = device
    out["route"]["deviceDrawRadiusMm"] = max(device / 2.0, 1.2)
    out["route"]["clearanceMm"] = [
        round(r - device / 2.0 - margin, 3) for r in radii
    ]
    return out


def _write_payload(payload: dict, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _run_sidecar(sidecar: Path, out_dir: Path, name: str,
                 extra: list[str] | None = None) -> tuple[int, Path, str]:
    target = out_dir / f"{name}.html"
    argv = ["--from-sidecar", str(sidecar), "--out", str(target), "--quiet"]
    code = clinical.main(argv + list(extra or []))
    text = target.read_text(encoding="utf-8") if target.is_file() else ""
    return code, target, text


def _make(kind: str, out_dir: Path, extra: list[str] | None = None) -> tuple[int, Path, str]:
    """走一次真实命令行入口，返回 (退出码, 产物路径, 页面文本)。"""
    return _run_sidecar(FIXTURES[kind], out_dir, kind, extra)


def _state_of(page: str, name: str) -> str:
    """从结论表里抠出某一项的判定（靠 tag 里的背景色反推三态，不依赖文案）。"""
    color_state = {c[1]: state for state, c in report_mod.STATE_COLOR.items()}
    block = re.search(
        rf"<td><b>{re.escape(name)}</b>.*?</td><td>(.*?)</td>", page, re.S
    )
    if not block:
        return "?"
    found = re.search(r"background:(#[0-9a-f]{6})", block.group(1))
    return color_state.get(found.group(1), "?") if found else "?"


# ------------------------------------------------------------------ 一、三态


def layer_1_tristate(out_dir: Path) -> None:
    print("\n[1] 三态语义（真实病例夹具 + 余量边界派生夹具）")
    code_green, _p, page_green = _make("green", out_dir)
    code_red, _p, page_red = _make("red", out_dir)

    check(code_green == 0, "全项通过的病例：退出码 0", f"rc={code_green}")
    check('class="banner ok"' in page_green, "全项通过的病例：横幅走 ok 分支")
    check(
        all(_state_of(page_green, name) == "ok"
            for name in ("器械通过性", "路径有效性", "靶点可达性", "分割依赖复核", "交付完整性")),
        "全项通过的病例：五项判定全为 ok",
        " ".join(f"{n}={_state_of(page_green, n)}"
                 for n in ("器械通过性", "路径有效性", "靶点可达性", "分割依赖复核", "交付完整性")),
    )

    check(code_red == 1, "靶点不可达的病例：退出码 1", f"rc={code_red}")
    check('class="banner bad"' in page_red, "靶点不可达的病例：横幅走 bad 分支")
    check(_state_of(page_red, "靶点可达性") == "bad",
          "68 mm 处的靶点被判为「异常」而不是通过")
    # 「没指定器械外径」必须落在 none。写成 ok 就是拿「没测」冒充「能过」。
    check(_state_of(page_red, "器械通过性") == "none",
          "未指定器械外径：通过性判「未判定」，不默认能过")
    check("没测" in page_red, "未指定器械的判定里写明「没测不等于能过」")
    check(_state_of(page_red, "分割依赖复核") == "none",
          "路径经过分割修复段：判「未判定」（数据不足以认定该段气道真实存在）")
    check("人工复核" in page_red, "依赖修复段的判定给出人工复核指引")
    check(_state_of(page_red, "路径有效性") == "ok",
          "同一份报告里 ok 与 bad/none 可以并存（逐项独立）")

    # 边界派生夹具：真实数据凑不出「最紧处余量恰好落在 0 两侧」，
    # 而这正是三态最容易写错的地方（`>= 0` 写成 `> 0` 就会把「刚好放得下」判成异常）。
    base = _payload("green")
    edge_ok = with_edge_clearance(base, EDGE_CLEARANCE_MM)
    edge_bad = with_edge_clearance(base, -EDGE_CLEARANCE_MM)
    ok_min = min(float(v) for v in edge_ok["route"]["clearanceMm"])
    bad_min = min(float(v) for v in edge_bad["route"]["clearanceMm"])
    check(abs(ok_min - EDGE_CLEARANCE_MM) < 0.002,
          "派生夹具自洽：最紧处余量 = +0.05 mm（按 payload.py 同一公式重算）",
          f"{ok_min:+.4f}")
    check(abs(bad_min + EDGE_CLEARANCE_MM) < 0.002,
          "派生夹具自洽：最紧处余量 = -0.05 mm", f"{bad_min:+.4f}")

    code_ok, _p, page_ok = _run_sidecar(
        _write_payload(edge_ok, out_dir, "edge_ok"), out_dir, "edge_ok")
    code_bad, _p, page_bad = _run_sidecar(
        _write_payload(edge_bad, out_dir, "edge_bad"), out_dir, "edge_bad")
    check(code_ok == 0 and _state_of(page_ok, "器械通过性") == "ok",
          "余量恰好 +0.05 mm：判「通过」（边界含等号，写成 > 0 就错了）", f"rc={code_ok}")
    check("+0.05" in page_ok, "极小正余量如实带正号显示（不抹成 0）")
    check(code_bad == 1 and _state_of(page_bad, "器械通过性") == "bad",
          "余量恰好 -0.05 mm：判「异常」且退出码 1", f"rc={code_bad}")
    check("-0.05" in page_bad, "负余量带负号显示")
    check("放不下" in page_bad and "换更细的器械" in page_bad,
          "器械放不下时说明原因并给出补齐方式")

    # 总判定规则必须与治理报告同源：同一套「bad > none > ok」，不是各自实现的
    check(clinical.OK is report_mod.OK and clinical.BAD is report_mod.BAD
          and clinical.NONE is report_mod.NONE,
          "三态常量与治理报告是同一对象（不是复制的一份）")
    check(clinical.overall is report_mod.overall, "总判定规则复用治理报告的实现")
    for state in ("ok", "bad", "none"):
        check(clinical.STATE_COLOR[state] == report_mod.STATE_COLOR[state],
              f"色值 {state} 与治理报告一致", str(clinical.STATE_COLOR[state]))
    check(clinical.CLINICAL_TAG[clinical.BAD] == "异常",
          "判定文案按临床读者改写（颜色不变、词变了）")


# ------------------------------------------------------------------ 二、参数一致


def layer_2_consistency(out_dir: Path) -> None:
    print("\n[2] 参数与页面所见一致（逐位比对，不解析显示文本猜）")
    base = _payload("green")
    cases: list[tuple[str, dict, Path, str]] = []
    for kind in ("green", "red"):
        _code, _path, page = _make(kind, out_dir)
        cases.append((kind, _payload(kind), FIXTURES[kind], page))
    for label, target_mm in (("edge_ok", EDGE_CLEARANCE_MM),
                             ("edge_bad", -EDGE_CLEARANCE_MM)):
        payload = with_edge_clearance(base, target_mm)
        sidecar = _write_payload(payload, out_dir, label)
        _code, _path, page = _run_sidecar(sidecar, out_dir, label)
        cases.append((label, payload, sidecar, page))

    for kind, payload, sidecar, page in cases:
        cards = CARD_RE.findall(page)
        found = {src: raw for src, raw, _k, _v in cards}

        check(set(found) == set(METRIC_SOURCES),
              f"{kind}：指标卡的来源键集合完全吻合",
              f"{len(found)} 项")

        for src, getter in METRIC_SOURCES.items():
            expected = getter(payload)
            raw_text = found.get(src, "<缺>")
            if expected is None:
                check(raw_text == "", f"{kind}：{src} 缺数据时 data-raw 为空串", raw_text)
                continue
            try:
                got = float(raw_text) if raw_text else None
            except ValueError:
                got = None
            check(got is not None and abs(got - float(expected)) < 1e-9,
                  f"{kind}：{src} 与负载逐位一致",
                  f"报告 {raw_text} vs 负载 {expected}")

        # 显示文本必须是同一个原始值的正确舍入 —— 防止「绑定对但显示错」
        strip = lambda s: re.sub(r"<span.*", "", s)
        shown = {k: strip(v) for _s, _r, k, v in cards}
        check(shown.get("最窄处直径") == f"{payload['metrics']['minimum_diameter_mm']:.2f}",
              f"{kind}：最窄处直径的显示是原始值的两位舍入",
              shown.get("最窄处直径"))
        check(shown.get("路径总长") == f"{payload['metrics']['route_length_mm']:.1f}",
              f"{kind}：路径总长的显示是原始值的一位舍入",
              shown.get("路径总长"))
        check(shown.get("靶点距中心线") == f"{payload['metrics']['target_distance_mm']:.2f}",
              f"{kind}：靶点距中心线的显示是原始值的两位舍入",
              shown.get("靶点距中心线"))

        # 版本表：报告要能回答「这是哪一版算出来的」
        meta = payload["meta"]
        check(f"c{meta['candidateId']}" in page and str(meta["caseId"]) in page,
              f"{kind}：候选编号与病例 ID 出现在报告里")
        # 拓扑签名含 `>`，进 HTML 一定被转义成 `&gt;` —— 比对时按转义后的形态找，
        # 否则测的是「有没有原样落成标签」，那反而是错的。
        signature = html_mod.escape(str(meta["topologySignature"]))
        check(signature in page,
              f"{kind}：拓扑签名转义后原样入表（可据此判断走的是哪条支路）")
        # 内嵌视图的网格步长必须报**负载真实的那一个**，不能照抄命令行默认值
        check(f"网格步长 {meta['meshStep']}" in page,
              f"{kind}：报告的网格步长取自负载 meta.meshStep，不是命令行默认值",
              f"meta={meta['meshStep']}")
        # 来源指纹：报告印出 sidecar 的哈希前缀，第三方可据此确认「看的是同一份」
        digest = clinical._sha256_text(sidecar.read_text(encoding="utf-8"))[:16]
        check(f"sha256 {digest}" in page,
              f"{kind}：报告印出了来源 sidecar 的 sha256 前缀（可对质）")


# ------------------------------------------------------------------ 三、转义与落盘


def layer_3_escape(out_dir: Path) -> None:
    print("\n[3] 转义与产物落盘（数据面的字符串当外部输入处理）")
    payload = _payload("green")
    evil = {
        "caseId": '</script><script>alert(1)</script>',
        "profileTitle": '"><img src=x onerror=alert(2)>',
    }
    mutated = json.loads(json.dumps(payload))
    mutated["meta"].update(evil)
    sidecar = out_dir / "evil.json"
    sidecar.write_text(json.dumps(mutated, ensure_ascii=False), encoding="utf-8")

    target = out_dir / "evil.html"
    code = clinical.main(["--from-sidecar", str(sidecar), "--out", str(target), "--quiet"])
    page = target.read_text(encoding="utf-8") if target.is_file() else ""

    check(code in (0, 1), "注入型病例仍能出报告", f"rc={code}")
    check("<script>alert(1)</script>" not in page, "病例 ID 里的 </script> 未落成可执行脚本")
    check("&lt;/script&gt;" in page, "病例 ID 以转义形式呈现")
    check("<img src=x onerror=alert(2)>" not in page, "档位名里的 img/onerror 未落成标签")
    check("&quot;&gt;&lt;img" in page, "档位名以属性转义形式呈现")

    # 落盘文件名要净化：病例 ID 带路径分隔符时不能穿越目录
    outside = out_dir / "evil_c3.html"
    check(target.is_file() and outside.parent == out_dir,
          "产物落在指定的 --out，不因病例 ID 里的分隔符而漂移")
    nested = list(out_dir.rglob("clinical_*"))
    check(all(p.parent == clinical.REPORT_DIR or p.parent == out_dir for p in nested),
          "产物文件名不含未净化的路径分隔符")


# ------------------------------------------------------------------ 四、投影可对质


def layer_4_sketch() -> None:
    print("\n[4] 投影示意：坐标可独立重算，越界不假装")
    payload = _payload("green")
    for view in ("oblique", "ap", "lateral"):
        svg, info = sketch.render_sketch(payload, view=view, width=520, height=460)
        trans = info["transform"]
        # 拿报告给出的变换，独立把「最窄处」投影一遍，必须落在报告记的像素位置上
        uv, _depth = sketch.project([payload["bottleneck"]["position"]], view)
        xy = sketch.screen_map(uv, trans["scale"], trans["offset"])[0]
        got = info["markers"]["bottleneck"]["screen"]
        drift = ((float(xy[0]) - got[0]) ** 2 + (float(xy[1]) - got[1]) ** 2) ** 0.5
        check(drift < 0.01, f"{view}：最窄处标记的屏幕坐标可独立重算（漂移 < 0.01 px）",
              f"{drift:.6f} px")
        w, h = info["canvas"]
        check(all(0 <= m["screen"][0] <= w and 0 <= m["screen"][1] <= h
                  for m in info["markers"].values()),
              f"{view}：所有标记都在画布内（打印不会切掉）")
        check(svg.startswith("<svg") and svg.rstrip().endswith("</svg>"),
              f"{view}：输出是一段完整的自包含 SVG")
        check("http" not in svg.replace("http://www.w3.org", ""),
              f"{view}：SVG 内无外部引用（除 xml 命名空间）")

    # 远近排序：画家算法要求远处的先画。深度不单调就等于叠错层次。
    _svg, info = sketch.render_sketch(payload, view="oblique")
    check(info["kept_edges"] > 0 and info["kept_edges"] < info["total_edges"],
          "较短的投影边被剪掉（画出来也看不见，白占体积）",
          f"保留 {info['kept_edges']}/{info['total_edges']}")

    # 越界标记：68 mm 外的靶点会被夹到边缘，但**必须明说**，否则读图的人以为它在那儿。
    # 用斜位测：取景框按中心线范围定，只有斜位能把这个偏出去的点投到框外
    # （前后位 / 侧位各压掉一个轴，它会落进框内 —— 那不算错，只是这一支测不到）。
    far = json.loads(json.dumps(_payload("red")))
    svg_far, info_far = sketch.render_sketch(far, view="oblique")
    check("nodule" in (info_far.get("out_of_view") or []),
          "远靶点被判为「视野外」", str(info_far.get("out_of_view")))
    check("视野外" in info_far["markers"]["nodule"]["label"],
          "视野外的标记在文案里说明了（并画出指向箭头）")
    check("视野外" in svg_far and "marker-end" in svg_far, "说明文字与箭头确实进了 SVG")

    # 图例阈值取自负载，不另立一套
    tweaked = json.loads(json.dumps(payload))
    tweaked["clearanceScale"] = {"safeMm": 3.5, "tightMm": 0.9}
    svg_tweaked, _info = sketch.render_sketch(tweaked, view="ap")
    check("余量 ≥ 3.5 mm" in svg_tweaked and "余量 0.9" in svg_tweaked.replace("余量 0 ~ 0.9", ""),
          "图例阈值跟着 payload.clearanceScale 变（与三维视图同一套判据）")

    # 没有几何就不许编一张空图
    empty = {"meta": {"caseId": "X"}, "clearanceScale": {}}
    svg_empty, info_empty = sketch.render_sketch(empty, view="ap")
    check(info_empty["has_geometry"] is False, "缺几何时如实标记 has_geometry=False")
    check("无法出示意图" in svg_empty, "缺几何时图上写明原因，而不是留一张空白图")


# ------------------------------------------------------------------ 五、打印与自包含


def layer_5_print(out_dir: Path) -> None:
    print("\n[5] 打印友好与文件自包含")
    _code, _path, page = _make("green", out_dir)

    check("@media print" in page, "含打印样式")
    check(".interactive-only { display:none !important; }" in page,
          "打印时隐藏交互视图（WebGL 画布打不出来，留着会是一块空白）")
    check(".print-only { display:block; }" in page,
          "打印时显示替代说明（告诉读的人图在哪）")
    check("打印版不含交互式三维视图" in page,
          "打印说明写明了为什么没有交互视图，而不是默默消失")

    check('<iframe class="viewer"' in page and "srcdoc=" in page,
          "屏幕版内嵌了三维视图")
    check(page.count("<iframe") == 1, "只有一个内嵌框架", f"{page.count('<iframe')} 个")
    # 判据锚在**引用形态**上，不锚「出现过 https://」：
    # three.js 里有一条废弃告警的文案带 github 链接，全文搜「https」会被它骗过。
    for pattern in ('src="http', 'href="http', "url(http", "@import", 'src="//'):
        check(page.count(pattern) == 0, f"无外部引用：{pattern}", f"{page.count(pattern)} 处")
    check("<script src=" not in page, "没有外链脚本")

    for anchor in ("id=\"conclusion\"", "id=\"figures\"", "id=\"numbers\"",
                   "id=\"view3d\"", "id=\"notes\"", "id=\"params\"", "id=\"method\""):
        check(anchor in page, f"小节锚点存在：{anchor}")

    check(page.count("</svg>") >= 3, "三张投影示意图都在页面里", f"{page.count('</svg>')} 张")
    check("不用于临床诊断与治疗决策" in page, "边界声明（不用于临床决策）在报告里")
    check("未经本院临床数据验证" in page, "如实声明数据来源未经本院临床验证")
    check("PHI" in page, "报告说明了 PHI 口径")

    # 缺视图时也不能留白框
    _c, _p, page_noview = _make("green", out_dir, ["--no-view"])
    check('<iframe class="viewer"' not in page_noview, "--no-view 下确实没有内嵌框架")
    check("本次未内嵌三维视图" in page_noview,
          "--no-view 时屏幕上给出可见说明（不是只写在 print-only 里）")
    check(_state_of(page_noview, "交付完整性") == "none",
          "--no-view 时「交付完整性」判未判定，不谎称完整")
    check(len(page_noview) < len(page) * 0.6,
          "--no-view 确实让报告显著变小",
          f"{len(page_noview) // 1024} KB vs {len(page) // 1024} KB")


# ------------------------------------------------------------------ 六、拒绝下结论


def layer_6_honesty(out_dir: Path) -> None:
    print("\n[6] 信息不足时拒绝产出")
    missing = out_dir / "nope.json"
    check(clinical.main(["--from-sidecar", str(missing), "--quiet"]) == 2,
          "sidecar 不存在：退出码 2（不是 0，也不是崩栈）")

    junk = out_dir / "junk.json"
    junk.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    check(clinical.main(["--from-sidecar", str(junk), "--quiet"]) == 2,
          "sidecar 结构不像 viewer 负载：退出码 2")

    check(clinical.main(["--quiet"]) == 2, "既没给病例也没给 sidecar：退出码 2")

    blank = out_dir / "blank.json"
    source = _payload("green")
    source["route"] = {}
    source["metrics"] = {}
    source["reachability"] = {}
    blank.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    target = out_dir / "blank.html"
    code = clinical.main(["--from-sidecar", str(blank), "--out", str(target), "--quiet"])
    page = target.read_text(encoding="utf-8") if target.is_file() else ""
    names = ("器械通过性", "路径有效性", "靶点可达性", "分割依赖复核", "交付完整性")
    states = {name: _state_of(page, name) for name in names}
    # 关键断言：**缺数据时一项临床判定都不许判通过**。
    # 写成 `int(metrics.get(k) or 0)` 就会让「分割依赖复核」在缺计数时判 ok，
    # 也就是用没有的数据宣布「本路径未经过任何分割修复段」—— 一个凭空的肯定结论。
    check(states["分割依赖复核"] == "none",
          "缺修复段计数：判未判定，不宣称「未经过修复段」", states["分割依赖复核"])
    check(states["路径有效性"] == "none",
          "缺 route_valid：判未判定，不因键缺失而报异常", states["路径有效性"])
    # 「交付完整性」说的是**这份文件**有没有给出它声称给出的东西，不是临床结论，
    # 所以它可以（也应该）是 ok —— 这里只要求四项临床判定全都不通过。
    clinical_items = names[:4]
    check(all(states[name] != "ok" for name in clinical_items),
          "缺路径数据时四项临床判定没有任何一项通过",
          str({name: states[name] for name in clinical_items}))
    check(states["交付完整性"] == "ok",
          "「交付完整性」独立于临床结论：文件本身齐备就判通过")
    check('class="banner none"' in page and code == 0,
          "总判定落「不足」而不是「通过」；退出码 0（未判定不是失败）", f"rc={code}")

    # 缺数据时报「—」而不是 0 —— 0 会被读成「测出来是零」
    check('data-raw=""' in page, "缺失指标的原始值绑定为空串")
    check(re.search(r'<p class="v">—', page) is not None, "缺失指标显示「—」而不是 0")


def main() -> int:
    for name, path in FIXTURES.items():
        if not path.is_file():
            print(f"夹具缺失：{path}（{name}）—— 这几份 sidecar 是入库的演示产物，"
                  "不应被 .gitignore 排除")
            return 2

    work = Path(tempfile.mkdtemp(prefix="airnav_clinical_check_"))
    try:
        layer_1_tristate(work)
        layer_2_consistency(work)
        layer_3_escape(work)
        layer_4_sketch()
        layer_5_print(work)
        layer_6_honesty(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    failed = [item for item in RESULTS if not item[0]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} 项通过")
    for _ok, label, detail in failed:
        print(f"  FAIL  {label}" + (f"（{detail}）" if detail else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
