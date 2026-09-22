"""把评测结果渲染成单文件 HTML 看板。

为什么要有它：JSON 报告适合机器读，markdown 适合在终端里扫一眼，
但要一眼看清「哪些用例失败了、失败在哪个维度、两个策略差在哪」，
还是得有个看板。产物是自包含 HTML，不依赖网络，可以直接转发。

用法：
    python -m agent.eval.dashboard --report outputs/eval/eval_report.json \
                                   --compare outputs/eval/eval_sloppy.json \
                                   --out outputs/eval/eval_dashboard.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from .report import GRADER_LABELS

CATEGORY_ORDER = ("工具选型", "分项归因", "知识检索", "边界拒答", "稳健性")


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""))


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _rate(summary: dict[str, Any]) -> float:
    return float(summary.get("pass_rate") or 0.0)


def _attempt_rate(summary: dict[str, Any]) -> float:
    """尝试级通过率（`--repeat N`）。缺省时与 `_rate` 同值。"""
    return float(summary.get("run_pass_rate") or summary.get("pass_rate") or 0.0)


def _repeat_of(summary: dict[str, Any]) -> int:
    return int(summary.get("repeat") or 1)


def _case_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in summary.get("cases", [])}


def render(
    primary: dict[str, Any],
    compare: dict[str, Any] | None = None,
    title: str = "AirNav Agent 评测看板",
) -> str:
    main_cases = _case_map(primary)
    cmp_cases = _case_map(compare) if compare else {}

    order = sorted(
        main_cases,
        key=lambda cid: (
            CATEGORY_ORDER.index(main_cases[cid].get("category", ""))
            if main_cases[cid].get("category") in CATEGORY_ORDER
            else 99,
            cid,
        ),
    )

    # ---- 分类汇总 ----
    def category_stats(summary: dict[str, Any]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for row in summary.get("cases", []):
            bucket = out.setdefault(row.get("category", "其他"), [0, 0])
            bucket[1] += 1
            if row.get("passed"):
                bucket[0] += 1
        return out

    main_cat = category_stats(primary)
    cmp_cat = category_stats(compare) if compare else {}
    categories = [c for c in CATEGORY_ORDER if c in main_cat or c in cmp_cat]
    categories += [c for c in main_cat if c not in categories]

    # ---- 打分器汇总 ----
    def grader_stats(summary: dict[str, Any]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for row in summary.get("cases", []):
            for grade in row.get("grades", []):
                bucket = out.setdefault(grade.get("grader", "?"), [0, 0])
                bucket[1] += 1
                if grade.get("passed"):
                    bucket[0] += 1
        return out

    main_grader = grader_stats(primary)
    cmp_grader = grader_stats(compare) if compare else {}
    graders = list(main_grader) + [g for g in cmp_grader if g not in main_grader]

    parts: list[str] = []
    parts.append(
        f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{
    --bg: #f6f7f9; --panel: #ffffff; --line: #e4e7ec; --text: #1f2937;
    --muted: #6b7280; --ok: #15803d; --okbg: #eaf7ef;
    --bad: #b91c1c; --badbg: #fdeeee; --accent: #2563eb; --accentbg: #eef4ff;
    --warn: #b45309; --warnbg: #fef3c7;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 14px; line-height: 1.6; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px; margin-bottom: 28px; }}
  .card {{ background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px 20px; }}
  .card .label {{ font-size: 12px; color: var(--muted); letter-spacing: .04em; }}
  .card .big {{ font-size: 34px; font-weight: 650; line-height: 1.2; margin: 6px 0 2px; }}
  .card .note {{ font-size: 12px; color: var(--muted); }}
  .card.ok .big {{ color: var(--ok); }}
  .card.bad .big {{ color: var(--bad); }}
  h2 {{ font-size: 15px; margin: 32px 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--line); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
    border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--line);
    vertical-align: top; font-size: 13px; }}
  th {{ background: #fafbfc; font-weight: 600; color: var(--muted);
    font-size: 12px; letter-spacing: .04em; white-space: nowrap; }}
  tr:last-child td {{ border-bottom: none; }}
  .pill {{ display: inline-block; padding: 1px 9px; border-radius: 999px;
    font-size: 12px; font-weight: 600; white-space: nowrap; }}
  .pill.ok {{ background: var(--okbg); color: var(--ok); }}
  .pill.bad {{ background: var(--badbg); color: var(--bad); }}
  .pill.warn {{ background: var(--warnbg); color: var(--warn); }}
  .mono {{ font-family: ui-monospace, Consolas, monospace; font-size: 12px;
    color: #374151; word-break: break-all; }}
  .bar {{ height: 8px; border-radius: 999px; background: #edf0f4; overflow: hidden;
    min-width: 90px; }}
  .bar > span {{ display: block; height: 100%; background: var(--ok); }}
  .bar.bad > span {{ background: var(--bad); }}
  details {{ margin-top: 6px; }}
  summary {{ cursor: pointer; color: var(--accent); font-size: 12px; }}
  .detail {{ margin: 8px 0 0; padding: 10px 12px; background: var(--accentbg);
    border-radius: 8px; font-size: 12px; }}
  .detail li {{ margin-bottom: 4px; }}
  .q {{ color: var(--muted); font-size: 12px; }}
  .nowrap {{ white-space: nowrap; }}
  details > summary {{ list-style: none; }}
  details > summary::-webkit-details-marker {{ display: none; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(title)}</h1>
  <div class="sub">逐用例隔离运行 · 硬判定打分（无 LLM 当裁判）· 报告生成于
    {_esc(primary.get('started_at', ''))}</div>

  <div class="cards">"""
    )

    parts.append(
        f"""<div class="card {'ok' if _rate(primary) == 1 else ''}">
      <div class="label">基线策略（规则驱动）</div>
      <div class="big">{primary.get('passed')} / {primary.get('total')}</div>
      <div class="note">{_pct(_rate(primary))} · {_esc(primary.get('backend', ''))}</div>
    </div>"""
    )
    if compare:
        parts.append(
            f"""<div class="card bad">
      <div class="label">劣化策略（对照，验证区分度）</div>
      <div class="big">{compare.get('passed')} / {compare.get('total')}</div>
      <div class="note">{_pct(_rate(compare))} · 故意跳过编号确认 / 硬算器械外径 /
        编造数字 / 给临床结论</div>
    </div>"""
        )
        parts.append(
            f"""<div class="card">
      <div class="label">区分度</div>
      <div class="big">{( _rate(primary) - _rate(compare)) * 100:.0f} pt</div>
      <div class="note">同一套用例下两个策略通过率之差</div>
    </div>"""
        )

    # ---- 重复运行（--repeat N）----
    # 用例级通过率（N 次全过）与尝试级通过率是两个不同的量：
    # 前者保守、用来卡基线；后者才是「能力」。只显示一个都会误导。
    repeat = _repeat_of(primary)
    flaky_rows = primary.get("flaky") or []
    if repeat > 1:
        parts.append(
            f"""<div class="card {'bad' if flaky_rows else ''}">
      <div class="label">重复运行（每条 {repeat} 次）</div>
      <div class="big">{primary.get('attempts_passed')} / {primary.get('attempts_total')}</div>
      <div class="note">尝试级通过率 {_pct(_attempt_rate(primary))} ·
        上面那张卡是用例级（N 次全过，保守）· 偶发 <b>{len(flaky_rows)}</b> 条</div>
    </div>"""
        )

    # ---- 分类 ----
    parts.append("</div><h2>分类通过率</h2><table><tr><th>类别</th><th>基线</th>")
    if compare:
        parts.append("<th>劣化策略</th>")
    parts.append("<th>基线通过率</th></tr>")
    for category in categories:
        got, total = main_cat.get(category, [0, 0])
        rate = got / total if total else 0.0
        parts.append(
            f"<tr><td>{_esc(category)}</td>"
            f"<td class='mono'>{got} / {total}</td>"
        )
        if compare:
            cgot, ctotal = cmp_cat.get(category, [0, 0])
            parts.append(f"<td class='mono'>{cgot} / {ctotal}</td>")
        parts.append(
            f"<td><div class='bar{' bad' if rate < 1 else ''}'>"
            f"<span style='width:{rate * 100:.0f}%'></span></div></td></tr>"
        )
    parts.append("</table>")

    # ---- 打分器 ----
    parts.append("<h2>各打分器命中情况</h2><table><tr><th>打分器</th><th>基线</th>")
    if compare:
        parts.append("<th>劣化策略</th>")
    parts.append("</tr>")
    for grader in graders:
        label = GRADER_LABELS.get(grader, grader)
        got, total = main_grader.get(grader, [0, 0])
        parts.append(f"<tr><td>{_esc(label)}</td><td class='mono'>{got} / {total}</td>")
        if compare:
            cgot, ctotal = cmp_grader.get(grader, [0, 0])
            parts.append(f"<td class='mono'>{cgot} / {ctotal}</td>")
        parts.append("</tr>")
    parts.append("</table>")

    # ---- 用例明细 ----
    parts.append(
        "<h2>用例明细</h2><table><tr><th>用例</th><th>类别</th><th>结果</th>"
        "<th>调用链</th>"
    )
    if compare:
        parts.append("<th>劣化策略</th>")
    parts.append("</tr>")

    for case_id in order:
        row = main_cases[case_id]
        failures = [
            g for g in row.get("grades", [])
            if not g.get("passed") and g.get("severity", "error") == "error"
        ]
        if row.get("flaky"):
            # 偶发既不是通过也不是失败：写成「通过」会掩盖它挂过，
            # 写成「失败」又看不出它过过 —— 而两者的处置完全相反。
            mark = "<span class='pill warn'>偶发</span>"
        elif row.get("passed"):
            mark = "<span class='pill ok'>通过</span>"
        else:
            mark = "<span class='pill bad'>失败</span>"
        if _repeat_of(primary) > 1 and row.get("attempts"):
            mark += (
                f"<div class='q nowrap'>{row.get('pass_count')}/{row.get('attempts')}</div>"
            )
        tools = " → ".join(row.get("tools_called") or []) or "—"
        cell = (
            f"<td><b>{_esc(case_id)}</b><div class='q'>{_esc(row.get('question', ''))}"
            f"</div>"
        )
        if failures:
            seq = ""
            if row.get("attempts_detail"):
                chain = " ".join(
                    f"{a.get('index')}:{'过' if a.get('passed') else '挂'}"
                    for a in row["attempts_detail"]
                )
                tail = "（偶发 = 先看采样，别急着改代码）" if row.get("flaky") else ""
                seq = (
                    f"<li>重复 {row.get('attempts')} 次：<code>{_esc(chain)}</code>"
                    f"{tail}</li>"
                )
            items = seq + "".join(
                f"<li><b>{_esc(GRADER_LABELS.get(g.get('grader'), g.get('grader')))}</b>："
                f"{_esc(g.get('detail'))}</li>"
                for g in failures
            )
            cell += f"<details><summary>{len(failures)} 项失败</summary><ul class='detail'>{items}</ul></details>"
        cell += "</td>"
        parts.append(
            f"<tr>{cell}<td class='nowrap'>{_esc(row.get('category'))}</td>"
            f"<td class='nowrap'>{mark}</td>"
            f"<td class='mono'>{_esc(tools)}</td>"
        )
        if compare:
            crow = cmp_cases.get(case_id)
            if crow is None:
                parts.append("<td class='mono'>—</td>")
            else:
                cmark = (
                    "<span class='pill ok'>通过</span>"
                    if crow.get("passed")
                    else "<span class='pill bad'>失败</span>"
                )
                cfail = [
                    g for g in crow.get("grades", [])
                    if not g.get("passed") and g.get("severity", "error") == "error"
                ]
                extra = (
                    f"<details><summary>{len(cfail)} 项失败</summary><ul class='detail'>"
                    + "".join(
                        f"<li><b>{_esc(GRADER_LABELS.get(g.get('grader'), g.get('grader')))}</b>："
                        f"{_esc(g.get('detail'))}</li>"
                        for g in cfail
                    )
                    + "</ul></details>"
                    if cfail
                    else ""
                )
                parts.append(f"<td>{cmark}{extra}</td>")
        parts.append("</tr>")

    parts.append(
        """</table>
  <div class="sub" style="margin-top:24px">
    全部判定均为硬指标：工具选型、参数正确性、数字可溯源、编号口径、引用有效性、边界拒答、稳健性。
    打分器本身另有自检（agent/eval/selftest.py），用合成轨迹验证它确实能抓住对应缺陷。
  </div>
</div>
</body>
</html>"""
    )
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="渲染评测看板")
    parser.add_argument("--report", required=True, help="主报告 JSON")
    parser.add_argument("--compare", default=None, help="对照报告 JSON（可选）")
    parser.add_argument("--out", required=True, help="输出 HTML 路径")
    parser.add_argument("--title", default="AirNav Agent 评测看板")
    args = parser.parse_args()

    primary = _load(args.report)
    compare = _load(args.compare) if args.compare else None
    markup = render(primary, compare, title=args.title)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markup, encoding="utf-8")
    print(f"看板已写入 {out}（{len(markup) / 1024:.1f} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
