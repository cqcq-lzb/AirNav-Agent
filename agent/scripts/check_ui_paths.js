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
  "viewer_LIDC_0089_c3_d1.5.html": "/artifacts/viewer_LIDC_0089_c3_d1.5.html",
  "viewer_LIDC_0089_c1.html": "/artifacts/viewer_LIDC_0089_c1.html",
};

// 产物 id -> {url, filename, kind}。**包括没有 url 的那些**（route 是内存对象）。
const artById = {
  "viewer-002": {
    url: "/artifacts/viewer_LIDC_0089_c3.html",
    filename: "viewer_LIDC_0089_c3.html",
    kind: "viewer",
  },
  "route-001": { url: "", filename: "", kind: "route" },
};

// ART_ID_RE 也从源码里取，不抄一份（抄的会随源码漂移）
const idReSource = html.match(/const ART_ID_RE = (\/[^\n]+\/);/);
if (!idReSource) {
  console.error("FAIL 在 index.html 里找不到 ART_ID_RE 的定义");
  process.exit(1);
}
const ART_ID_RE = eval(idReSource[1]);

// 把抽出来的块塞进一个函数体，用参数注入外部依赖
const factory = new Function(
  "artByName",
  "artById",
  "ART_ID_RE",
  block + "\nreturn { esc, markArtifacts, restoreArtifacts, md };"
);
const { md, markArtifacts } = factory(artByName, artById, ART_ID_RE);

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
// 用户实测报上来的现场：模型**没出图**，却宣称「三维路径可视化文件已生成」，
// 并把工具返回体里那个 `artifact_id` 当链接写成了 `![](route-001)`。
// route 是内存对象，没有文件 —— 界面必须把这件事说清楚，而不是渲染出破图。
console.log("\n[4] `![](route-001)` —— 把产物 id 当链接（用户实测现场）");
const answer4 =
  "对于病例 LIDC_0089 的 3 号结节候选，规划出的路径如下：\n\n" +
  "- **路径长度**：216.24 mm\n" +
  "- **最窄直径**：5.92 mm\n\n" +
  "三维路径可视化文件已生成，路径折线数据已存入 artifact，ID 为 route-001。\n\n" +
  "如需查看路径的三维可视化，请点击下方链接：\n" +
  "![](route-001)";
const out4 = md(answer4);
check(!out4.includes("](route-001)"), "不再留下 markdown 死链");
check(!out4.includes("<img"), "不会渲染成破图");
check(out4.includes('class="dead-ref"'), "换成一条能看懂的说明");
check(out4.includes("没有可查看的文件"), "说明里点出「不是文件」");
check(out4.includes("需要三维视图请让我出图"), "给出可执行的下一步");
check(countButtons(out4) === 0, "不该凭空造出一个按钮", countButtons(out4) + " 个按钮");
check(out4.includes("<li><strong>路径长度</strong>：216.24 mm</li>"), "正文其余内容不被搅乱");

// ---------------------------------------------------------------- 用例 5
// 反向的好情况：模型引用的是**真有文件**的产物 id，应当出按钮
console.log("\n[5] `![](viewer-002)` —— 引用可查看的产物 id");
const out5 = md("三维视图已生成：\n![](viewer-002)");
check(out5.includes('data-open="/artifacts/viewer_LIDC_0089_c3.html"'), "换成指向真实文件的按钮");
check(countButtons(out5) === 1, "一个按钮", countButtons(out5) + " 个按钮");
check(!out5.includes("dead-ref"), "不该误判成死引用");

// ---------------------------------------------------------------- 用例 6
// 形状像产物 id、但这次会话里根本没有
console.log("\n[6] `![](route-999)` —— 不存在的产物 id");
const out6 = md("见 ![x](route-999)");
check(out6.includes("引用了不存在的产物"), "明确说是「不存在的产物」");
check(countButtons(out6) === 0, "不造按钮");

// ---------------------------------------------------------------- 用例 7
// 模型现在被告知「给文件名」（viewer_file），于是它会在正文里裸提文件名 ——
// 这是**好行为**，网页里得能点。但只认登记表里有的名字，不能误伤随便一个 xxx.html。
console.log("\n[7] 裸文件名 `viewer_LIDC_0089_c3.html`（模型被引导后的好行为）");
const out7 = md(
  "三维视图已生成。请查看以下文件：\n\n- **三维视图文件**：viewer_LIDC_0089_c3.html"
);
check(out7.includes('data-open="/artifacts/viewer_LIDC_0089_c3.html"'), "裸文件名换成可点按钮");
check(countButtons(out7) === 1, "一个按钮", countButtons(out7) + " 个按钮");
check(out7.includes("<li><strong>三维视图文件</strong>："), "列表与粗体不被搅乱");
// 反向：没登记的名字不许动
const out7b = md("随便提一个 notes.html 和 report.json，它们不是产物");
check(!out7b.includes("<button"), "未登记的 .html / .json 名字保持纯文本");

// ---------------------------------------------------------------- 用例 8
// 反向：产物不在登记表里时不能乱改，也不能留下占位符
console.log("\n[8] 反向用例 —— 未登记的路径");
const unknown = "见 [x](D:\\somewhere\\other_view.html) 和 ![](D:\\somewhere\\missing.html)";
const kept = markArtifacts(unknown);
check(kept.includes("other_view.html") && kept.includes("missing.html"), "未登记的路径保持原样");
check(!kept.includes("\u0001"), "不产生占位符残留");

// ---------------------------------------------------------------- 用例 9
// 2026-09-17 新口径的实测形态：`plan_route` 顺带出图之后，空模型（真实 qwen2.5:14b）
// 写的是 `[三维视图](D:\...\viewer_LIDC_0089_c3_d1.5.html)` ——
// 链接文字**不是**路径，路径还带上了器械外径后缀。两者都不能让替换失效。
console.log("\n[9] `[三维视图](带外径后缀的路径)` —— 规划顺带出图后的实测形态");
const out9 = md(
  "**三维视图**：已生成，路径详情请查看 " +
    "[三维视图](D:\\AirNav-Agent\\outputs\\viewers\\viewer_LIDC_0089_c3_d1.5.html)。"
);
check(
  out9.includes('data-open="/artifacts/viewer_LIDC_0089_c3_d1.5.html"'),
  "带参数后缀的产物也能认出来"
);
check(countButtons(out9) === 1, "一个按钮", countButtons(out9) + " 个按钮");
check(!out9.includes("D:\\"), "路径不残留");

console.log("\n" + "=".repeat(68));
console.log(failures ? `未通过：${failures} 项` : "全部通过");
for (const [i, out] of [out1, out2, out3, out4, out5, out6, out7, out9].entries()) {
  console.log(`\n[${i + 1}] 替换后的片段：`);
  const lines = out
    .split("<br>")
    .filter((l) => l.includes("button") || l.includes("dead-ref"));
  console.log(lines.join("\n") || "（无）");
}
process.exit(failures ? 1 : 0);
