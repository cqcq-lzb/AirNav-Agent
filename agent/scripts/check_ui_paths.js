// 网页版界面逻辑自检：验证「把回答里贴的本地路径换成可点按钮」这段。
//
// ## 为什么要单独测这一段
//
// 模型会在回答里贴本地文件路径，**提示词挡不住** —— 因为 `render_viewer`
// 的返回体里本来就带 `path` 字段，模型只是把它抄出来（和评测里 E01 是同一条教训：
// 提示词压不住模型手里已有的数据）。所以真正的修法在前端的 `markArtifacts`。
// 这段逻辑一旦写错，用户看到的就是一堆点不开的 `D:\...`，属于「静默坏掉」，必须钉住。
//
// ## 做法
//
// 从真实的 `index.html` 里**原样抽出** `esc` / `markArtifacts` / `restoreArtifacts`
// / `md` 四个函数（不是抄一份副本 —— 副本会随源码漂移，测了个寂寞），
// 在 Node 里喂真实模型回答的文本。
//
// 用例文本取自 2026-09-17 对真模型（gpu41 / qwen2.5:14b）跑网页版的**实际回答**：
// 模型写的是 markdown **图片**语法 `![](D:\...\x.html)`，不是 `[文字](路径)`。
// 只匹配后者的话，开头的 `!` 会残留在正文里 —— 用例 [2] 就是为它加的。
//
// ## 用法
//
//     node agent/scripts/check_ui_paths.js
//
// 退出码 0 = 全过。不需要浏览器，也不需要起服务。
// `check_web.py` 的 [4] 层会在能找到 node 时自动调它。

"use strict";

const fs = require("fs");
const path = require("path");

// 本文件在 agent/scripts/ 下，上溯两层到仓库根
const REPO = path.resolve(__dirname, "..", "..");
const INDEX = path.join(REPO, "agent", "web", "static", "index.html");

const html = fs.readFileSync(INDEX, "utf8");
const start = html.indexOf("function esc(");
const end = html.indexOf("function addMsg(");
if (start < 0 || end < 0 || end <= start) {
  console.error("FAIL 无法在 index.html 里定位要抽出的函数块（esc ... addMsg）");
  console.error("     界面结构可能被改动过，请同步更新本脚本的抽取边界。");
  process.exit(1);
}
const block = html.slice(start, end);

// 产物登记表：文件名 -> URL。真实运行时由 renderArtifacts 填。
const artByName = {
  "viewer_LIDC_0089_c3.html": "/artifacts/viewer_LIDC_0089_c3.html",
  "viewer_LIDC_0089_c1.html": "/artifacts/viewer_LIDC_0089_c1.html",
};

// 把抽出来的块塞进一个函数体，用参数注入 artByName
const factory = new Function(
  "artByName",
  block + "\nreturn { esc, markArtifacts, restoreArtifacts, md };"
);
const { md, markArtifacts } = factory(artByName);

let failures = 0;
function check(ok, label, detail) {
  console.log((ok ? "  PASS  " : "  FAIL  ") + label + (detail ? "（" + detail + "）" : ""));
  if (!ok) failures += 1;
}
const countButtons = (s) => (s.match(/<button/g) || []).length;

console.log("=".repeat(68));
console.log("网页版界面自检：回答里的本地路径 → 可点按钮");
console.log("=".repeat(68));

// ---------------------------------------------------------------- 用例 1
// `[文字](路径)`，且文字本身也是一条路径 —— 必须合成一个按钮，不能出现两个
console.log("\n[1] `[路径](路径)` —— 链接文字就是路径");
const answer1 =
  "已生成 LIDC_0089 病例中 3 号结节候选的三维视图，您可以直接查看：\n\n" +
  "[D:\\AirNav-Agent\\outputs\\viewers\\viewer_LIDC_0089_c3.html](D:\\AirNav-Agent\\outputs\\viewers\\viewer_LIDC_0089_c3.html)\n\n" +
  "该视图包含气道表面、中心线、路径双管、最窄处标记、结节候选及分割修复段等信息，" +
  "支持多种交互方式，适合演示与分析。";
const out1 = md(answer1);
check(!out1.includes("D:\\"), "回答里不再残留 Windows 路径");
check(out1.includes('data-open="/artifacts/viewer_LIDC_0089_c3.html"'), "替换成指向产物 URL 的按钮");
check(out1.includes(">打开三维视图</button>"), "按钮文案正确");
check(countButtons(out1) === 1, "链接与裸路径合并成一个按钮（不重复）", countButtons(out1) + " 个按钮");
check(out1.includes("该视图包含气道表面"), "正文其余内容保持不变");
check(out1.includes("<br>"), "换行仍被正确转换");

// ---------------------------------------------------------------- 用例 2
// 真模型实测输出：markdown 图片语法，前面多一个 `!`
console.log("\n[2] `![](路径)` —— markdown 图片语法（真模型实测形态）");
const answer2 =
  "在使用 1.2mm 的器械时，候选 1 的路径仍然存在一些问题，但已经生成了三维视图供您查看。" +
  "请参考以下路径信息：\n\n" +
  "- **路径长度**：303.666 mm\n" +
  "- **最窄直径**：1.641 mm\n\n" +
  "三维视图路径如下：\n" +
  "![](D:\\AirNav-Agent\\outputs\\viewers\\viewer_LIDC_0089_c1.html)\n\n" +
  "请注意，路径最窄处直径为 1.64 mm，仅比要求值 1.60 mm 略宽。";
const out2 = md(answer2);
const line2 = out2.split("<br>").find((l) => l.includes("inline-art")) || "";
check(!out2.includes("D:\\"), "回答里不再残留 Windows 路径");
check(!line2.includes("!"), "图片语法的 `!` 不残留（旧版会剩一个孤零零的叹号）", JSON.stringify(line2.slice(0, 40)));
check(out2.includes('data-open="/artifacts/viewer_LIDC_0089_c1.html"'), "按钮指向正确的产物 URL");
check(countButtons(out2) === 1, "只产生一个按钮", countButtons(out2) + " 个按钮");
check(
  out2.includes("<li><strong>路径长度</strong>：303.666 mm</li>") &&
    out2.includes("<li><strong>最窄直径</strong>：1.641 mm</li>"),
  "列表与粗体仍正常渲染"
);

// ---------------------------------------------------------------- 用例 3
// 真模型另一次实测：图片语法**带 alt 文本** `![三维视图](路径)`
// （空 alt 的 `![](路径)` 上是同位体，但 alt 有内容时 `[^\]]*` 才真正被走到）
console.log("\n[3] `![alt](路径)` —— 图片语法带 alt 文本（真模型实测）");
const answer3 =
  "这是 LIDC_0089 病例中候选 3 的三维视图：\n\n" +
  "- **气道表面**：半透明，可剖切\n" +
  "- **中心线**：包含分叉点层级\n\n" +
  "请注意，本视图为工程研究演示结果，不用于临床诊断与治疗决策。\n" +
  "![三维视图](D:\\AirNav-Agent\\outputs\\viewers\\viewer_LIDC_0089_c3.html)";
const out3 = md(answer3);
const line3 = out3.split("<br>").find((l) => l.includes("inline-art")) || "";
check(!out3.includes("D:\\"), "回答里不再残留 Windows 路径");
check(
  line3.trim() ===
    '<button type="button" class="inline-art" data-open="/artifacts/viewer_LIDC_0089_c3.html"' +
      ' data-name="viewer_LIDC_0089_c3.html">打开三维视图</button>',
  "该行只剩按钮（`!` 与 alt 文本都没有残留）",
  JSON.stringify(line3.slice(0, 60))
);
check(out3.includes('data-open="/artifacts/viewer_LIDC_0089_c3.html"'), "按钮指向正确的产物 URL");
check(countButtons(out3) === 1, "只产生一个按钮", countButtons(out3) + " 个按钮");
check(out3.includes("<li><strong>气道表面</strong>：半透明，可剖切</li>"), "列表与粗体仍正常渲染");

// ---------------------------------------------------------------- 用例 4
// 反向：产物不在登记表里时不能乱改，也不能留下占位符
console.log("\n[4] 反向用例 —— 未登记的路径");
const unknown = "见 [x](D:\\somewhere\\other_view.html) 和 ![](D:\\somewhere\\missing.html)";
const kept = markArtifacts(unknown);
check(kept.includes("other_view.html") && kept.includes("missing.html"), "未登记的路径保持原样");
check(!kept.includes("\u0001"), "不产生占位符残留");

console.log("\n" + "=".repeat(68));
console.log(failures ? `未通过：${failures} 项` : "全部通过");
for (const [i, out] of [out1, out2, out3].entries()) {
  console.log(`\n[${i + 1}] 替换后的片段：`);
  console.log(out.split("<br>").filter((l) => l.includes("button")).join("\n"));
}
process.exit(failures ? 1 : 0);
