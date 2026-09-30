"""把「推理后端降级横幅」打到 `agent/web/static/index.html`。

## 为什么走脚本而不是直接编辑

与 `_patch_web_auth.py` 同因：**编辑器直写 `.html` 会落成连 Python 都解不开的
裸密文**（DLP 按写入进程决定形态），而服务端正是用 Python 读 `index.html`
再发给浏览器的 —— 直写等于把网页写废。前端改动一律走这条脚本化路径。

## 这个横幅在防什么

后端不可达时，网页端原本的表现是：用户提问 → SSE 推一条「连接失败」→
那条错误**看起来就像一次正常回答的结论**。用户无从知道
「是模型挂了」还是「这个病例真的没路径」—— 这两件事的处理完全不同。

所以降级状态必须**先于回答**出现（SSE 的第一条事件），也必须在
页面加载时就可见（`/api/meta` 里的 `backend_probe`）。

⚠️ 判据全部来自服务端（`agent/llm/client.probe_backend`，与 CLI 共用一份）。
前端**不自己猜**「后端通不通」—— 猜出来的状态会和 CLI / 审计记录对不上。

## 用法

    python agent/scripts/_patch_web_degraded.py            # 就地打补丁
    python agent/scripts/_patch_web_degraded.py --check     # 只检查锚点，不改文件

锚点全部断言存在；任何一条对不上就整体放弃（**不写半截文件**）。
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[2]
    / "agent" / "web" / "static" / "index.html"
)

MARK = 'id="degraded"'

# ------------------------------------------------------------------ 补丁内容

CSS_ANCHOR = '''  .dead-ref{display:inline-block;margin:4px 2px;padding:3px 9px;border-radius:6px;
    background:var(--warn-bg);border:1px solid #f0d9b0;color:var(--warn);font-size:12px}'''

CSS_NEW = '''  .dead-ref{display:inline-block;margin:4px 2px;padding:3px 9px;border-radius:6px;
    background:var(--warn-bg);border:1px solid #f0d9b0;color:var(--warn);font-size:12px}

  /* 降级横幅。⚠️ 用 warn 不用 bad：「降级」不等于「服务坏了」—— 纯工具查询
     照常能用，标成红色会让人以为整站挂了。等级要配得上事实。 */
  #degraded{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap;
    padding:8px 18px;background:var(--warn-bg);border-bottom:1px solid #f0d9b0;
    color:var(--warn);font-size:12px}
  /* ⚠️ 必须显式写这条：上面设了 `display:flex`，会把 `hidden` 属性自带的
     `display:none` 覆盖掉 —— 结果就是「加了 hidden 却照样显示」。 */
  #degraded[hidden]{display:none}
  #degraded .why{color:var(--tx-3);font:11px/1.5 var(--mono);word-break:break-all}'''

HTML_ANCHOR = '''</header>'''

HTML_NEW = '''</header>

<!--
  降级横幅：推理后端不可用时显示。默认 hidden —— 只有 /api/meta 或 SSE 明确
  说 degraded 时才亮。前端不猜状态，只显示服务端给的结论。
-->
<div id="degraded" hidden>
  <b>⚠️ 降级运行：推理后端不可用</b>
  <span id="degraded-note"></span>
  <span class="why" id="degraded-why"></span>
</div>'''

JSFN_ANCHOR = '''  function scrollDown(){ log.scrollTop = log.scrollHeight; }'''

JSFN_NEW = '''  function scrollDown(){ log.scrollTop = log.scrollHeight; }

  // ---------- 降级提示 ----------
  // 判据全部来自服务端（/api/meta 的 backend_probe 或 SSE 的 degraded 事件）。
  // 前端**不自己探**后端通不通：猜出来的状态会和 CLI / 审计记录对不上，
  // 而「同一件事两处说法不同」比没有提示更坏。
  function showDegraded(probe){
    const el = $("degraded");
    if (!el) return;
    if (!probe || !probe.degraded){ el.hidden = true; return; }
    $("degraded-note").textContent = probe.note || "依赖模型判断的步骤会失败。";
    $("degraded-why").textContent = probe.detail || "";
    el.hidden = false;
  }'''

JSINIT_ANCHOR = '''    backendDefault = meta.default_backend;'''

JSINIT_NEW = '''    backendDefault = meta.default_backend;
    // 页面一加载就把降级状态显示出来 —— 不必等第一次提问失败才知道
    showDegraded(meta.backend_probe);'''

JSCASE_ANCHOR = '''        case "start":
          traceRow("a", "", "后端 " + d.backend);
          break;'''

JSCASE_NEW = '''        case "degraded":
          // 后端不可用时这是**第一条**事件：先声明降级，回答才不会被误读成
          // 「这个病例真的没路径」。顺序本身就是信息。
          showDegraded(d.probe);
          traceRow("f", "", "后端降级");
          break;

        case "start":
          traceRow("a", "", "后端 " + d.backend);
          break;'''

PATCHES = [
    ("CSS  降级横幅样式", CSS_ANCHOR, CSS_NEW),
    ("HTML 降级横幅节点", HTML_ANCHOR, HTML_NEW),
    ("JS   showDegraded()", JSFN_ANCHOR, JSFN_NEW),
    ("JS   初始化时读 meta.backend_probe", JSINIT_ANCHOR, JSINIT_NEW),
    ("JS   SSE degraded 事件", JSCASE_ANCHOR, JSCASE_NEW),
]


def main(argv: list[str]) -> int:
    check_only = "--check" in argv

    if not TARGET.is_file():
        print(f"FAIL 目标不存在：{TARGET}")
        return 2

    data = TARGET.read_bytes()
    if data[:16] == b"%TSD-Header-###%":
        print("FAIL 目标文件是裸密文（DLP 形态），先 `git checkout -- <file>` 拿回明文")
        return 2
    text = data.decode("utf-8")

    if MARK in text:
        print('SKIP 补丁已打过（找到 id="degraded"），不动文件')
        return 0

    missing = [title for title, anchor, _ in PATCHES if anchor not in text]
    if missing:
        print("FAIL 以下锚点没对上，整体放弃（不写半截文件）：")
        for title in missing:
            print("   -", title)
        return 3

    for title, anchor, new in PATCHES:
        text = text.replace(anchor, new, 1)
        print("  ok", title)

    if check_only:
        print(f"CHECK {len(PATCHES)} 处锚点全部命中，可以打补丁（--check 未写文件）")
        return 0

    TARGET.write_bytes(text.encode("utf-8"))
    print(f"WROTE {TARGET}  {len(text.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
