"""把「代为执行」的前端改动打到 `agent/web/static/index.html`。

## 为什么要打这个补丁

Agent 循环新增了规则 C 的处置路径（见 `agent/agent_loop.py` 模块 docstring）：
模型把工具调用写成了正文（Qwen 原生 `<tool_call>` 格式漏进 `content`）时，
循环**不再纠偏，而是代为执行**，并记 `run.recovered_toolcalls`。

既然记了数，界面就必须显示 —— 否则一次「替模型擦屁股才完成」的运行
在页面上看起来和「一次到位」完全一样，而这个区别正是要给人看的东西
（判断纪律：**结论要能分辨「一次就答对」和「催了才答对」**）。

## 为什么补丁要写成脚本，而不是直接编辑文件

本机 DLP 按写入进程决定落盘形态：**编辑器直写 `.html` 会落成连 Python
都解不开的裸密文**，而服务端正是用 Python 读 `index.html` 再发给浏览器的。
所以前端改动一律走这条脚本化路径（脚本用 Python 落盘）。
详见 `_patch_web_auth.py` 的表格与 `.workbuddy/memory` 当日记录。

## 用法

    python agent/scripts/_patch_web_recovered.py            # 就地打补丁
    python agent/scripts/_patch_web_recovered.py --check    # 只检查锚点，不写文件

锚点全部断言存在；任何一条对不上就整体放弃（**不写半截文件**）。
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "agent" / "web" / "static" / "index.html"

MARK = 'case "recovered"'

# ------------------------------------------------------------------ 补丁内容

# 1) 轨迹里给「代为执行」一个自己的颜色。
#    与纠偏同色系（都是 harness 的干预），但用独立类名 —— 两者语义不同，
#    将来要分开调色时不必满文件找。
CSS_ANCHOR = """  .tl .c{color:#5b3f96}"""

CSS_NEW = """  .tl .c{color:#5b3f96}
  .tl .r{color:#5b3f96}"""

# 2) 事件分支。插在纠偏与产物之间，与事件在 SSE 上的自然顺序一致。
JS_CASE_ANCHOR = """          traceRow("c", d.step, "纠偏 " + d.kind + (d.count ? " #" + d.count : ""));
          scrollDown();
          break;
        }

        case "artifacts":"""

JS_CASE_NEW = """          traceRow("c", d.step, "纠偏 " + d.kind + (d.count ? " #" + d.count : ""));
          scrollDown();
          break;
        }

        // 模型把工具调用写成了正文（Qwen 原生格式漏进 content，见 agent_loop 规则 C）。
        // 这一类**不纠偏**：循环认出这段 JSON 是完整可执行的调用，直接代为执行。
        // 界面上必须显出来 —— 不然「代执行才跑通」和「一次到位」看起来一模一样。
        case "recovered": {
          ensureStep(d.step);
          const names = d.names || [];
          const n = document.createElement("div");
          n.className = "note corr";
          n.textContent = d.note || ("把工具调用写成了正文，已代为执行：" + names.join(", "));
          box.appendChild(n);
          traceRow("r", d.step, "代为执行 " + names.join(","));
          scrollDown();
          break;
        }

        case "artifacts":"""

# 3) 收尾统计栏。
JS_STATS_ANCHOR = """            (d.intent_corrections ? " · 纠偏 " + d.intent_corrections : "");"""

JS_STATS_NEW = """            (d.intent_corrections ? " · 纠偏 " + d.intent_corrections : "") +
            (d.recovered_toolcalls ? " · 代为执行 " + d.recovered_toolcalls : "");"""

PATCHES = [
    ("CSS  轨迹「代为执行」配色", CSS_ANCHOR, CSS_NEW),
    ("JS   recovered 事件分支", JS_CASE_ANCHOR, JS_CASE_NEW),
    ("JS   统计栏加「代为执行」", JS_STATS_ANCHOR, JS_STATS_NEW),
]


def main(argv: list[str]) -> int:
    check_only = "--check" in argv

    data = TARGET.read_bytes()
    if data[:16] == b"%TSD-Header-###%":
        print("FAIL 目标文件是裸密文（DLP 形态），先 `git checkout -- <file>` 拿回明文")
        return 2
    text = data.decode("utf-8")

    if MARK in text:
        print(f'SKIP 补丁已打过（找到 {MARK}），不动文件')
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
