"""把 viewer 负载里的几何投影成二维示意图（SVG）。

为什么不直接截三维视图的图
--------------------------
1. **打印 / PDF 需要静态图**。三维视图是 WebGL 画布，`@media print` 打不出来 ——
   一份「带三维截图」的报告打印出来留一块空白，比根本没图更糟。
2. **无头浏览器是新增依赖**，还要能在门禁里无头、秒级、可复现地跑。本项目约定不新增依赖。
3. **一致性靠数据源，不靠人去对齐**：投影与三维视图读的是**同一份 payload**
   ——「报告里的参数与页面所见一致」于是是结构上成立的，不是靠人肉比对数字。

所以这里产出的是**正交投影示意图**，不是渲染截图。报告里也如实这么标注。
屏幕阅读仍然给活的三维视图（报告内嵌），两者分工：**屏幕用交互、打印用这张图**。

投影约定
--------
世界坐标（见 `mesh.lps_to_world`）：X = 患者左为正，Y = 头为正（向上），Z = 前为正。
屏幕 u = 水平向右，v = 竖直**向上**（SVG 的 y 轴朝下，落笔时取反）。
深度 = 沿视线方向的分量，越大越靠近观察者。
"""
from __future__ import annotations

import html
import math

import numpy as np

# 观察方向（方位角绕头脚轴，俯仰角抬头）。同一套几何，换视角就是换这两个数。
VIEWS: dict[str, tuple[float, float]] = {
    "oblique": (-34.0, 18.0),
    "ap": (0.0, 0.0),
    "lateral": (90.0, 0.0),
}

VIEW_TITLES: dict[str, str] = {
    "oblique": "斜位（旋转视角，方位为近似）",
    "ap": "前后位 AP（患者左在右侧）",
    "lateral": "侧位 LAT（自患者右侧看，前方在右侧）",
}

# 与 viewer 的余量着色同源：阈值取自 payload 的 clearanceScale，
# 只有「器械本体放不下」这一档是这里新增的（clearance < 0）。
COLOR_BLOCKED = "#991b1b"
COLOR_TIGHT = "#b91c1c"
COLOR_NEAR = "#b45309"
COLOR_SAFE = "#0f766e"

TREE_NEAR = 0.80
TREE_FAR = 0.28
TREE_WIDTH_NEAR = 2.0
TREE_WIDTH_FAR = 0.8
ROUTE_WIDTH = 3.4

FONT = '"Microsoft YaHei", "Segoe UI", system-ui, sans-serif'


def project(points, view: str = "oblique") -> tuple[np.ndarray, np.ndarray]:
    """世界坐标 (N,3) -> (屏幕 uv (N,2，v 朝上), 深度 (N,))。

    方位角 0° / 俯仰角 0° 即前后位：u = X（患者左在屏幕右侧）、
    v = Y（头在上）、depth = Z（前方更靠近观察者），与影像科阅片习惯一致。
    """
    if view not in VIEWS:
        raise KeyError(f"未知视角：{view}（可选 {sorted(VIEWS)}）")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    azimuth, elevation = VIEWS[view]
    a = math.radians(azimuth)
    e = math.radians(elevation)
    cos_a, sin_a = math.cos(a), math.sin(a)
    cos_e, sin_e = math.cos(e), math.sin(e)

    u = cos_a * pts[:, 0] + sin_a * pts[:, 2]
    into_plane = -sin_a * pts[:, 0] + cos_a * pts[:, 2]
    v = cos_e * pts[:, 1] - sin_e * into_plane
    depth = sin_e * pts[:, 1] + cos_e * into_plane
    return np.stack([u, v], axis=1), depth


def fit_transform(
    uv: np.ndarray, width: int, height: int, pad: int
) -> tuple[float, np.ndarray]:
    """等比缩放 + 居中：算出 (scale, offset)。

    **两个轴必须同一个 scale**，否则气道树会被拉扁，比例尺也就成了假话。
    v 朝上而 SVG 的 y 朝下，所以 y 方向由 `screen_map` 取反。
    """
    low = uv.min(axis=0)
    high = uv.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    scale = float(min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1]))
    offset = np.asarray(
        [
            pad + (width - 2 * pad - span[0] * scale) / 2.0 - low[0] * scale,
            pad + (height - 2 * pad - span[1] * scale) / 2.0 + high[1] * scale,
        ]
    )
    return scale, offset


def screen_map(uv: np.ndarray, scale: float, offset: np.ndarray) -> np.ndarray:
    """uv（v 朝上）-> SVG 像素坐标（y 朝下）。

    x = u * scale + off_x；y = off_y − v * scale（这里就是那次取反）。
    """
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    x = pts[:, 0] * scale + offset[0]
    y = offset[1] - pts[:, 1] * scale
    return np.stack([x, y], axis=1)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


class _LabelPlacer:
    """贪心标签避让。

    标签叠在一起是这类图最常见的丑法。位置算得准但字压在一起，读的人还是得猜。
    这里只做一件小事：按优先级给每个标签试几个候选偏移，取第一个不与已有标签相交的。
    """

    #: (dx, dy, anchor)：右下、右上、左上、左下、正上、正下
    CANDIDATES = (
        (10.0, -9.0, "start"),
        (10.0, 16.0, "start"),
        (-10.0, -9.0, "end"),
        (-10.0, 16.0, "end"),
        (0.0, -20.0, "middle"),
        (0.0, 26.0, "middle"),
    )

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.boxes: list[tuple[float, float, float, float]] = []

    @staticmethod
    def _text_width(text: str) -> float:
        """粗略估宽：中日韩字符按字号计，其余按 0.56 字号计。"""
        total = 0.0
        for ch in text:
            total += 11.0 if ord(ch) > 0x2E80 else 6.2
        return total

    def place(
        self, x: float, y: float, text: str, *, dy_first: float = 0.0
    ) -> tuple[float, float, str]:
        w = self._text_width(text)
        h = 13.0
        best = None
        for index, (dx, dy, anchor) in enumerate(self.CANDIDATES):
            shift = dy_first if index == 0 else 0.0
            lx, ly = x + dx, y + dy + shift
            if anchor == "start":
                box = (lx, ly - h / 2, lx + w, ly + h / 2)
            elif anchor == "end":
                box = (lx - w, ly - h / 2, lx, ly + h / 2)
            else:
                box = (lx - w / 2, ly - h / 2, lx + w / 2, ly + h / 2)
            inside = (
                box[0] >= 4 and box[2] <= self.width - 4
                and box[1] >= 4 and box[3] <= self.height - 4
            )
            if not inside:
                continue
            if best is None:
                best = (lx, ly, anchor, box)
            if not any(self._overlap(box, other) for other in self.boxes):
                best = (lx, ly, anchor, box)
                break
        if best is None:
            best = (x + 10.0, y - 9.0, "start",
                    (x + 10.0, y - 15.5, x + 10.0 + w, y - 2.5))
        self.boxes.append(best[3])
        return best[0], best[1], best[2]

    @staticmethod
    def _overlap(a, b) -> bool:
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _nice_scale_mm(span_mm: float) -> float:
    """取一个「整」的长度做比例尺，约为画面宽度的 1/4。

    取**最接近**的整值而不是「第一个大于等于目标的」：后者在目标 50 mm 时会跳到
    100 mm，比例尺一下占掉画面三分之一，喧宾夺主。
    """
    target = span_mm / 4.0
    ladder = (5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 500.0)
    return min(ladder, key=lambda candidate: abs(candidate - target))


def clearance_bands(payload: dict) -> list[tuple[float, float, str, str]]:
    """余量分档 (下界, 上界, 颜色, 文案)。上界用 `inf` 表示开区间。

    阈值直接取自 payload 的 `clearanceScale` —— 与三维视图同一套判据，
    改一处两边都变，不会出现「图上是绿的、页面说贴壁」。
    """
    scale = payload.get("clearanceScale") or {}
    safe = float(scale.get("safeMm", 2.0))
    tight = float(scale.get("tightMm", 0.5))
    return [
        (float("-inf"), 0.0, COLOR_BLOCKED, "器械放不下（余量 < 0）"),
        (0.0, tight, COLOR_TIGHT, f"贴壁（余量 0 ~ {_fmt(tight, 1)} mm）"),
        (tight, safe, COLOR_NEAR, f"偏紧（余量 {_fmt(tight, 1)} ~ {_fmt(safe, 1)} mm）"),
        (safe, float("inf"), COLOR_SAFE, f"宽松（余量 ≥ {_fmt(safe, 1)} mm）"),
    ]


def _color_for(clearance: float, bands) -> str:
    for low, high, color, _label in bands:
        if low <= clearance < high:
            return color
    return COLOR_SAFE


def render_sketch(
    payload: dict,
    *,
    view: str = "oblique",
    width: int = 520,
    height: int = 460,
    min_segment_px: float = 2.0,
    show_legend: bool = True,
    show_scale_bar: bool = True,
    show_orientation: bool = True,
    show_title: bool = True,
) -> tuple[str, dict]:
    """把负载渲染成一张 SVG，并返回可对质的事实（供自检）。

    返回的第二项不是装饰：`transform` 与 `markers.*.screen` 让调用方能**独立重算**
    投影坐标再比对。判据必须可对质，不能只是「函数自称画对了」。
    """
    pad = 26
    bands = clearance_bands(payload)
    route = payload.get("route") or {}
    centerline = payload.get("centerline") or {}
    nodules = payload.get("nodules") or []

    route_pts = np.asarray(route.get("positions") or [], dtype=np.float64).reshape(-1, 3)
    line_pts = np.asarray(centerline.get("positions") or [], dtype=np.float64).reshape(-1, 3)
    edges = list(centerline.get("edges") or [])

    frame_source = line_pts if len(line_pts) >= 2 else route_pts
    if len(frame_source) < 2:
        # 没有几何可画。不编一张空图冒充 —— 报告会拿 meta 里的说明去写「图缺失」。
        note = "负载里没有中心线或路径几何，无法出示意图"
        svg = (
            f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family={FONT!r} style="width:100%;height:auto">'
            f'<rect width="{width}" height="{height}" fill="#fbfbf9"/>'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            f'font-size="13" fill="#8a8a81">{html.escape(note)}</text></svg>'
        )
        return svg, {
            "view": view, "has_geometry": False, "note": note,
            "canvas": [width, height], "kept_edges": 0, "total_edges": len(edges),
        }

    frame_uv, _ = project(frame_source, view)
    scale, offset = fit_transform(frame_uv, width, height, pad)

    # ---- 中心线：先算屏幕长度，太短的丢掉（画出来也看不见，白占体积）
    kept: list[tuple[float, float, float, float, float]] = []
    for edge in edges:
        try:
            i, j = int(edge[0]), int(edge[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= i < len(line_pts) and 0 <= j < len(line_pts)):
            continue
        seg_uv, seg_depth = project(line_pts[[i, j]], view)
        seg = screen_map(seg_uv, scale, offset)
        length = float(math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]))
        if length < min_segment_px:
            continue
        kept.append(
            (float(seg_depth.mean()), seg[0][0], seg[0][1], seg[1][0], seg[1][1])
        )

    kept.sort(key=lambda item: item[0])  # 远的先画，近的后画（画家算法）
    depth_lo = kept[0][0] if kept else 0.0
    depth_hi = kept[-1][0] if kept else 1.0
    depth_span = max(depth_hi - depth_lo, 1e-6)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family={FONT!r} style="width:100%;height:auto" '
        f'role="img" aria-label="{html.escape(VIEW_TITLES.get(view, view))}">',
        f'<rect width="{width}" height="{height}" fill="#fbfbf9"/>',
        '<defs>'
        f'<marker id="arrow-{view}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="#6b6b64"/></marker>'
        '</defs>',
    ]

    for depth, x1, y1, x2, y2 in kept:
        t = (depth - depth_lo) / depth_span
        opacity = TREE_FAR + (TREE_NEAR - TREE_FAR) * t
        stroke_w = TREE_WIDTH_FAR + (TREE_WIDTH_NEAR - TREE_WIDTH_FAR) * t
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#9a9a92" stroke-width="{stroke_w:.2f}" '
            f'stroke-opacity="{opacity:.2f}" stroke-linecap="round"/>'
        )

    # ---- 路径：白色描边垫底（从灰色树里跳出来的最省事办法），再按余量分段上色
    route_screen = None
    if len(route_pts) >= 2:
        route_uv, _ = project(route_pts, view)
        route_screen = screen_map(route_uv, scale, offset)
        halo = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in route_screen)
        parts.append(
            f'<polyline points="{halo}" fill="none" stroke="#ffffff" '
            f'stroke-width="{ROUTE_WIDTH + 2.6:.1f}" stroke-opacity="0.9" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        clearances = route.get("clearanceMm") or []
        for index in range(len(route_screen) - 1):
            try:
                # 一段的颜色按**这一段里最紧的那点**定：说「这段能过」必须按最紧处说。
                left = float(clearances[index])
                right = float(clearances[index + 1])
            except (IndexError, TypeError, ValueError):
                left = right = float("inf")
            color = _color_for(min(left, right), bands)
            a, b = route_screen[index], route_screen[index + 1]
            parts.append(
                f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                f'stroke="{color}" stroke-width="{ROUTE_WIDTH}" stroke-linecap="round"/>'
            )

    # ---- 标记与标签
    placer = _LabelPlacer(width, height)
    markers: dict[str, dict] = {}
    texts: list[str] = []
    shapes: list[str] = []
    arrows: list[str] = []
    clamped: list[str] = []

    def _place(point_xyz, label: str, *, marker_kind: str) -> tuple[float, float]:
        uv, _ = project(np.asarray([point_xyz], dtype=np.float64), view)
        xy = screen_map(uv, scale, offset)[0]
        x, y = float(xy[0]), float(xy[1])
        # 落在取景框外的标记不能就这么画到画布外（打印出来就没了）。
        # 夹到边上 + 画箭头 + 文案加「视野外」—— 只把点挪进来而不说，
        # 读图的人会以为它真在那儿，这比不画还坏。
        out = not (
            pad * 0.5 <= x <= width - pad * 0.5
            and pad * 0.5 <= y <= height - pad * 0.5
        )
        cx = min(max(x, pad * 0.5 + 2), width - pad * 0.5 - 2)
        cy = min(max(y, pad * 0.5 + 2), height - pad * 0.5 - 2)
        if out:
            clamped.append(marker_kind)
            label = f"{label}（视野外）"
            dx, dy = x - cx, y - cy
            norm = math.hypot(dx, dy) or 1.0
            ax, ay = cx + dx / norm * 15.0, cy + dy / norm * 15.0
            arrows.append(
                f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" '
                f'stroke="#6b6b64" stroke-width="1.3" stroke-dasharray="3 2" '
                f'marker-end="url(#arrow-{view})"/>'
            )
        lx, ly, anchor = placer.place(cx, cy, label)
        texts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'font-size="11.5" fill="#23231f" stroke="#fbfbf9" stroke-width="3" '
            f'paint-order="stroke">{html.escape(label)}</text>'
        )
        markers[marker_kind] = {
            "screen": [round(cx, 2), round(cy, 2)],
            "out_of_view": out,
            "label": label,
        }
        return cx, cy

    # 入口点
    entry = payload.get("entryPoint")
    if entry:
        cx, cy = _place(entry, "入口", marker_kind="entry")
        shapes.append(f'<rect x="{cx - 3.2:.1f}" y="{cy - 3.2:.1f}" width="6.4" height="6.4" '
                      f'fill="#fff" stroke="#4b5563" stroke-width="1.6"/>')

    # 其余候选结节：淡圈 + 淡标号。只画圈不写字的话，读图的人只会看到几个
    # 没有说明的小灰点。标号排在最后，让主要标记先占位置。
    others: list[tuple[float, float, str]] = []
    for nodule in nodules:
        if nodule.get("selected"):
            continue
        uv, _ = project(np.asarray([nodule["position"]], dtype=np.float64), view)
        xy = screen_map(uv, scale, offset)[0]
        ox, oy = float(xy[0]), float(xy[1])
        inside = -10 <= ox <= width + 10 and -10 <= oy <= height + 10
        shapes.append(
            f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="3.4" fill="none" '
            'stroke="#b9b9b1" stroke-width="1.2"/>'
        )
        if inside:
            others.append((ox, oy, f"c{nodule.get('clientId')}"))

    # 靶点（被规划的那个）
    selected = next((n for n in nodules if n.get("selected")), None)
    if selected is not None:
        label = f"靶点 c{selected.get('clientId')}"
        cx, cy = _place(selected["position"], label, marker_kind="nodule")
        shapes.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.2" fill="#7c3aed" '
                      'fill-opacity="0.22" stroke="#7c3aed" stroke-width="1.8"/>')
        shapes.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.9" fill="#7c3aed"/>')

    bottleneck = payload.get("bottleneck")
    if bottleneck and bottleneck.get("position"):
        text = (f"最窄 Ø{_fmt(bottleneck.get('diameterMm'), 2)} mm"
                f"　余量 {float(bottleneck.get('clearanceMm') or 0):+.2f} mm")
        cx, cy = _place(bottleneck["position"], text, marker_kind="bottleneck")
        shapes.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.4" fill="none" '
                      f'stroke="{COLOR_TIGHT}" stroke-width="1.8"/>')

    turn = payload.get("sharpestTurn")
    if turn and turn.get("position"):
        text = f"最大转角 {_fmt(turn.get('angleDeg'), 1)}°"
        cx, cy = _place(turn["position"], text, marker_kind="sharpest_turn")
        shapes.append(
            f'<path d="M{cx:.1f},{cy - 4.4:.1f} L{cx + 4.4:.1f},{cy:.1f} '
            f'L{cx:.1f},{cy + 4.4:.1f} L{cx - 4.4:.1f},{cy:.1f} z" fill="none" '
            f'stroke="{COLOR_NEAR}" stroke-width="1.6"/>'
        )

    # 次要标号排在主要标记**之后**占位：标签避让是先到先得，
    # 让「入口 / 靶点 / 最窄处 / 最大转角」先挑位置，剩下的空隙才给其它候选。
    for ox, oy, text in others:
        lx, ly, anchor = placer.place(ox, oy, text)
        texts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="10.5" '
            f'fill="#a3a39b" stroke="#fbfbf9" stroke-width="3" '
            f'paint-order="stroke">{html.escape(text)}</text>'
        )

    parts.extend(shapes)
    parts.extend(arrows)
    parts.extend(texts)

    # ---- 视角标题（报告侧有小标题时关掉，免得同一句话说两遍）
    if show_title:
        title = VIEW_TITLES.get(view, view)
        parts.append(
            f'<text x="12" y="20" font-size="12.5" fill="#5c5c55" '
            f'stroke="#fbfbf9" stroke-width="3" paint-order="stroke">{html.escape(title)}</text>'
        )

    # ---- 方向标记：钉在**画布边缘**，不画在几何极值点上。
    # 极值点常常落在肺门附近（画面正中），标在那儿只增加噪声。
    # 方位由投影定义唯一确定（AP 的 u = +X，即患者左在屏幕右侧），按边放就是对的。
    # 斜位不是轴对齐视角，「左/右」只是近似 —— 不确定的事就不标。
    if show_orientation and view in ("ap", "lateral"):
        mid_y = height / 2.0
        if view == "ap":
            marks = ((14.0, "start", "R"), (width - 14.0, "end", "L"))
        else:
            marks = ((14.0, "start", "后"), (width - 14.0, "end", "前"))
        for mx, anchor, tag in marks:
            parts.append(
                f'<text x="{mx:.1f}" y="{mid_y:.1f}" font-size="12" fill="#a3a39b" '
                f'text-anchor="{anchor}" stroke="#fbfbf9" stroke-width="3" '
                f'paint-order="stroke">{html.escape(tag)}</text>'
            )

    # ---- 比例尺
    scale_bar_mm = None
    if show_scale_bar:
        span_mm = float(np.ptp(frame_uv[:, 0]))
        scale_bar_mm = _nice_scale_mm(span_mm)
        bar_px = scale_bar_mm * scale
        y0 = height - 18.0
        x0 = 14.0
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0 + bar_px:.1f}" y2="{y0:.1f}" '
            'stroke="#6b6b64" stroke-width="1.6"/>'
        )
        for tick in (x0, x0 + bar_px):
            parts.append(
                f'<line x1="{tick:.1f}" y1="{y0 - 4:.1f}" x2="{tick:.1f}" y2="{y0 + 4:.1f}" '
                'stroke="#6b6b64" stroke-width="1.6"/>'
            )
        parts.append(
            f'<text x="{x0 + bar_px + 6:.1f}" y="{y0 + 4:.1f}" font-size="11" '
            f'fill="#6b6b64">{scale_bar_mm:g} mm</text>'
        )

    # ---- 图例
    if show_legend:
        rows = [(color, label) for _lo, _hi, color, label in bands]
        box_w = 168.0
        box_h = 14.0 * len(rows) + 12.0
        bx = width - box_w - 12.0
        by = 30.0
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" '
            'rx="7" fill="#ffffff" fill-opacity="0.92" stroke="#e6e6e1"/>'
        )
        for row_index, (color, label) in enumerate(rows):
            ry = by + 17.0 + row_index * 14.0
            parts.append(
                f'<line x1="{bx + 10:.1f}" y1="{ry - 4:.1f}" x2="{bx + 26:.1f}" '
                f'y2="{ry - 4:.1f}" stroke="{color}" stroke-width="3.2" stroke-linecap="round"/>'
            )
            parts.append(
                f'<text x="{bx + 32:.1f}" y="{ry:.1f}" font-size="10.5" '
                f'fill="#5c5c55">{html.escape(label)}</text>'
            )

    parts.append("</svg>")
    meta = {
        "view": view,
        "has_geometry": True,
        "canvas": [width, height],
        "pad": pad,
        "total_edges": len(edges),
        "kept_edges": len(kept),
        "min_segment_px": min_segment_px,
        "route_segments": max(len(route_pts) - 1, 0),
        "transform": {
            "scale": round(scale, 8),
            "offset": [round(float(offset[0]), 6), round(float(offset[1]), 6)],
        },
        "scale_bar_mm": scale_bar_mm,
        "markers": markers,
        "out_of_view": clamped,
    }
    return "".join(parts), meta


__all__ = [
    "VIEWS",
    "VIEW_TITLES",
    "clearance_bands",
    "fit_transform",
    "project",
    "render_sketch",
    "screen_map",
]
