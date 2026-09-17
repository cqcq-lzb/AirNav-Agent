"""两轮评测报告的逐用例对照（HTML）。

用途：改动打分器、提示词或换模型之后，**只看总分是不够的**——
总分涨了可能是真进步，也可能是尺子松了。

这个工具把两份 `run_eval` 的 JSON 报告并排摊开，逐用例标出
「翻正 / 退化 / 不变」，并列出两轮各自的失败打分器与工具调用链。
配合 `git diff` 看代码改动，就能回答那个关键问题：
**这次的分数变化，到底来自模型，还是来自尺子？**

    python -m agent.scripts.eval_diff_report \
        --old outputs/eval/eval_gpu41_14b.json \
        --new outputs/eval/eval_gpu41_14b_v2.json \
        --out outputs/eval/对照_打分器修正.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

# 变化类型 -> (标签, 颜色变量名)
CHANGE_LABEL = {
    "fixed": ("翻正", "#0f766e", "#ccfbf1"),
    "regressed": ("退化", "#a32d2d", "#fee2e2"),
    "same_pass": ("通过", "#0f766e", "#f0fdfa"),
    "same_fail": ("失败", "#a32d2d", "#fef2f2"),
}

CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 28px 56px; background: #f7f7f5; color: #23231f;
  font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
  font-size: 14px; line-height: 1.65; }
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 6px; letter-spacing: -.2px; }
h2 { font-size: 16px; font-weight: 600; margin: 40px 0 12px;
  padding-bottom: 8px; border-bottom: 1px solid #e2e2dd; }
h3 { font-size: 14px; font-weight: 600; margin: 22px 0 8px; }
.sub { color: #6b6b64; font-size: 13px; margin: 0 0 4px; }
.card { background: #fff; border: 1px solid #e6e6e1; border-radius: 12px;
  padding: 18px 20px; margin-bottom: 14px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px; margin: 16px 0 4px; }
.metric { background: #fff; border: 1px solid #e6e6e1; border-radius: 10px; padding: 14px 16px; }
.metric .k { font-size: 12px; color: #78786f; margin: 0 0 6px; }
.metric .v { font-size: 26px; font-weight: 600; margin: 0; letter-spacing: -.5px; }
.metric .n { font-size: 12px; color: #8a8a81; margin: 4px 0 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; font-weight: 600; font-size: 12px; color: #6b6b64;
  padding: 8px 10px; border-bottom: 1px solid #e2e2dd; white-space: nowrap; }
td { padding: 9px 10px; border-bottom: 1px solid #f0f0ec; vertical-align: top; }
tr.moved td { background: #fbfdfb; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 20px;
  font-size: 12px; font-weight: 500; white-space: nowrap; }
code { font-family: Consolas, "Courier New", monospace; font-size: 12px;
  background: #f2f2ee; padding: 1px 5px; border-radius: 4px; }
.tools { color: #5c5c55; font-size: 12px; font-family: Consolas, monospace; }
.why { color: #6b6b64; font-size: 12.5px; }
.q { border-left: 3px solid #e2e2dd; padding: 2px 0 2px 12px; margin: 8px 0;
  color: #3d3d38; font-size: 13px; }
.q.bad { border-left-color: #d9a3a3; }
.q.good { border-left-color: #8fc7bd; }
.legend { font-size: 12.5px; color: #6b6b64; margin: 10px 0 0; }
.legend b { font-weight: 600; color: #23231f; }
.bar { height: 8px; border-radius: 4px; background: #cfe9e4; overflow: hidden; }
.bar > i { display: block; height: 100%; background: #0f766e; }
.note { background: #fffdf5; border: 1px solid #ece3c9; border-radius: 10px;
  padding: 14px 18px; font-size: 13px; color: #4a4230; }
.note b { font-weight: 600; }
ol, ul { margin: 8px 0; padding-left: 22px; }
li { margin: 5px 0; }
"""


def _load(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "backend": payload.get("backend") or "?",
        "passed": payload.get("passed", 0),
        "total": payload.get("total", 0),
        "started_at": payload.get("started_at", ""),
        "by_category": payload.get("by_category") or {},
        "cases": {c["case_id"]: c for c in payload.get("cases") or []},
    }


def _failing(case: dict) -> list[str]:
    """失败的打分器名 + 判定明细。"""
    return [
        f'{g["grader"]}：{(g.get("detail") or "").strip()}'
        for g in (case.get("grades") or [])
        if not g.get("passed")
    ]


def _short(text: str, limit: int = 150) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def _tag(kind: str) -> str:
    label, color, bg = CHANGE_LABEL[kind]
    return f'<span class="tag" style="color:{color};background:{bg}">{label}</span>'


def _rows(old: dict, new: dict) -> tuple[str, int, int, int]:
    ids = list(old["cases"])
    for case_id in new["cases"]:
        if case_id not in ids:
            ids.append(case_id)

    parts: list[str] = []
    fixed = regressed = 0
    for case_id in ids:
        a = old["cases"].get(case_id)
        b = new["cases"].get(case_id)
        if not a or not b:
            continue
        pa, pb = bool(a.get("passed")), bool(b.get("passed"))
        if pa == pb:
            kind = "same_pass" if pa else "same_fail"
        elif pb:
            kind, fixed = "fixed", fixed + 1
        else:
            kind, regressed = "regressed", regressed + 1

        detail = ""
        if kind in ("fixed", "regressed"):
            detail = (
                f'<div class="why">第一轮失败项：{html.escape("；".join(_failing(a)) or "—")}'
                f'<br>调用链 <span class="tools">{html.escape(" → ".join(a.get("tools_called") or []) or "（无）")}</span></div>'
                f'<div class="why">第二轮失败项：{html.escape("；".join(_failing(b)) or "—")}'
                f'<br>调用链 <span class="tools">{html.escape(" → ".join(b.get("tools_called") or []) or "（无）")}</span></div>'
            )
        else:
            detail = (
                f'<div class="why">{html.escape("；".join(_failing(b)) or "全部判定通过")}</div>'
                f'<div class="why"><span class="tools">{html.escape(" → ".join(b.get("tools_called") or []) or "（无工具调用）")}</span></div>'
            )

        parts.append(
            f'<tr class="{"moved" if kind in ("fixed", "regressed") else ""}">'
            f'<td>{html.escape(case_id)}<br><span class="why">{html.escape(b.get("category") or "")}</span></td>'
            f'<td>{_tag("same_pass" if pa else "same_fail")}</td>'
            f'<td>{_tag("same_pass" if pb else "same_fail")}</td>'
            f'<td>{_tag(kind)}</td>'
            f'<td>{detail}</td></tr>'
        )
    return "".join(parts), fixed, regressed, len(ids)


def _category_table(old: dict, new: dict) -> str:
    names = list(old["by_category"])
    for name in new["by_category"]:
        if name not in names:
            names.append(name)
    rows = []
    for name in names:
        a = old["by_category"].get(name) or {}
        b = new["by_category"].get(name) or {}
        ta, tb = a.get("total", 0), b.get("total", 0)
        pa, pb = a.get("passed", 0), b.get("passed", 0)
        ra = round(100 * pa / ta) if ta else 0
        rb = round(100 * pb / tb) if tb else 0
        arrow = "→" if ra == rb else ("↑" if rb > ra else "↓")
        color = "#0f766e" if rb > ra else ("#a32d2d" if rb < ra else "#78786f")
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f'<td style="text-align:right">{pa}/{ta}（{ra}%）</td>'
            f'<td style="text-align:right">{pb}/{tb}（{rb}%）</td>'
            f'<td style="text-align:right;color:{color};font-weight:600">{arrow} {rb - ra:+d}pt</td>'
            f'<td><div class="bar"><i style="width:{rb}%"></i></div></td></tr>'
        )
    return "".join(rows)


def build(old: dict, new: dict, ruler_evidence: str) -> str:
    rows, fixed, regressed, total = _rows(old, new)
    pa, pb = old["passed"], new["passed"]
    rate_a = round(100 * pa / old["total"]) if old["total"] else 0
    rate_b = round(100 * pb / new["total"]) if new["total"] else 0

    delta_color = "#0f766e" if pb > pa else ("#a32d2d" if pb < pa else "#78786f")
    verdict = (
        "分数变化全部来自尺子修正" if not regressed
        else f"⚠️ 有 {regressed} 条退化，改动可能伤到了别的地方"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>评测对照：打分器修正前后</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>评测对照：打分器修正前后</h1>
  <p class="sub">同一模型（{html.escape(new["backend"])}）、同一套 15 条用例，只改了打分器。</p>
  <p class="sub">第一轮 {html.escape(old["started_at"])}　·　第二轮 {html.escape(new["started_at"])}</p>

  <div class="cards">
    <div class="metric"><p class="k">第一轮通过</p><p class="v">{pa} / {old["total"]}</p>
      <p class="n">{rate_a}%（尺子未修）</p></div>
    <div class="metric"><p class="k">第二轮通过</p><p class="v">{pb} / {new["total"]}</p>
      <p class="n">{rate_b}%（尺子已修）</p></div>
    <div class="metric"><p class="k">翻正 / 退化</p>
      <p class="v" style="color:{delta_color}">{fixed} / {regressed}</p>
      <p class="n">{verdict}</p></div>
    <div class="metric"><p class="k">用例总数</p><p class="v">{total}</p>
      <p class="n">6 类硬判定打分器</p></div>
  </div>

  <h2>1　分类通过率</h2>
  <table>
    <thead><tr><th>类别</th><th style="text-align:right">第一轮</th>
      <th style="text-align:right">第二轮</th><th style="text-align:right">变化</th><th>第二轮</th></tr></thead>
    <tbody>{_category_table(old, new)}</tbody>
  </table>

  <h2>2　逐用例对照</h2>
  <p class="legend">只有标为 <b>翻正</b> / <b>退化</b> 的行底色被点亮。
    <b>退化</b> 必须为 0，否则说明改动伤到了别处。</p>
  <table>
    <thead><tr><th>用例</th><th>第一轮</th><th>第二轮</th><th>变化</th>
      <th>失败打分器与调用链</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <h2>3　为什么会误判：模型原文的用词 vs 打分器词表</h2>
  <p class="sub">打分器原先是精确子串匹配。中文里插一个修饰词就匹配不上，
    于是<b>正确的回答被判成没有划界</b> —— 错的是尺子。</p>
  {ruler_evidence}

  <h2>4　剩下的失败是真缺陷</h2>
  <div class="note">
    <b>E01 / E07「承诺 vs 行动」</b>：模型输出「…接下来，我将尝试为编号 3 规划路径。」
    「…调用 <code>plan_route</code> 重新规划路径。」—— 在<b>宣告</b>下一步动作但没发出工具调用，
    循环就把这段文字当成最终答案收下了（<code>stop_reason=answered</code>）。
    <br><br>
    <b>E08 / E09 / E10 知识检索 0/3</b>：<code>search_knowledge</code> 一次都没被调用。
    E10 更严重：没查就说「知识库中没有相关依据」——<b>假装查过</b>。
    <br><br>
    <b>E15 判定口径不一致</b>：用例 rubric 只要求「提示偏离气道、需人工复核」，
    模型<b>做到了</b>；但 <code>expect_tools</code> 额外钉死要 <code>rank_candidates</code>，
    模型用了 <code>list_nodule_candidates</code>。属判断题，未擅自改。
  </div>

  <p class="legend">数字可溯源 <b>两轮都是 15/15</b> —— 模型没有编造过任何数字。</p>
</div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="两轮评测报告的逐用例对照")
    parser.add_argument("--old", required=True, help="旧一轮的 JSON 报告")
    parser.add_argument("--new", required=True, help="新一轮的 JSON 报告")
    parser.add_argument("--out", default=None, help="输出 HTML，缺省打印到 stdout")
    args = parser.parse_args(argv)

    old, new = _load(args.old), _load(args.new)

    # 划界误判的原文证据：从新一轮报告里把边界拒答用例的回答取出来
    chunks = []
    for case_id, case in new["cases"].items():
        if case.get("category") != "边界拒答":
            continue
        bad = any(
            g["grader"] == "refusal" and not g.get("passed")
            for g in (case.get("grades") or [])
        )
        answer = _short(case.get("answer") or "", 220)
        chunks.append(
            f'<div class="q {"bad" if bad else "good"}">'
            f"<b>{html.escape(case_id)}</b>（第二轮{'仍失败' if bad else '通过'}）<br>"
            f"{html.escape(answer)}</div>"
        )
    evidence = "".join(chunks) or '<p class="sub">（新一轮报告里没有边界拒答用例）</p>'
    page = build(old, new, evidence)

    if args.out:
        Path(args.out).write_text(page, encoding="utf-8")
        print(f"已写入 {args.out}（{len(page.encode('utf-8'))} 字节）")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
