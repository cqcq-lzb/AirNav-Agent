// 端到端界面验证：把**真服务返回的真回答**喂给 index.html 里那段真函数。
//
// check_ui_paths.js 用的是 fixture（自己编的回答文本）；这里用的是
// 「真模型 → 真 web 服务 → 真 SSE 事件」产出的回答 + 产物表。
// 回答里那句路径能不能变成按钮，就是用户那句「连接呢」的直接答案，
// 所以值得端到端钉一次 —— 而不是只在 fixture 上成立。
//
// ## 用法（两步）
//
//     python -m agent.scripts.probe_web --dump D:\tmp\web_run.json
//     node agent/scripts/check_web_e2e.js D:\tmp\web_run.json
//
// 需要服务已启动 + 真模型可达，所以**不进常驻自检**。
// 退出码 0 = 全过。

"use strict";

const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "..", "..");
const INDEX = path.join(REPO, "agent", "web", "static", "index.html");
const runPath = process.argv[2];
if (!runPath || !fs.existsSync(runPath)) {
  console.error(`FAIL 找不到 ${runPath || "(未给路径)"}`);
  console.error("      先跑：python -m agent.scripts.probe_web --dump <路径>");
  process.exit(1);
}
const run = JSON.parse(fs.readFileSync(runPath, "utf8"));

// 从真实的 index.html 里**原样抽出**函数块（不抄副本 —— 副本会随源码漂移）
const html = fs.readFileSync(INDEX, "utf8");
const start = html.indexOf("function esc(");
const end = html.indexOf("function addMsg(");
if (start < 0 || end < 0 || end <= start) {
  console.error("FAIL 无法在 index.html 里定位要抽出的函数块（esc ... addMsg）");
  process.exit(1);
}
const block = html.slice(start, end);

// 产物登记表：按文件名与按 id 两张，都来自这次**真实运行**的 artifacts 事件
const artByName = {};
const artById = {};
for (const item of run.artifacts || []) {
  const name = item.filename || (item.path ? path.basename(item.path) : "");
  if (name && item.url) artByName[name] = item.url;
  if (item.id) {
    artById[item.id] = { url: item.url || "", filename: name, kind: item.kind };
  }
}
const idReSource = html.match(/const ART_ID_RE = (\/[^\n]+\/);/);
if (!idReSource) {
  console.error("FAIL 在 index.html 里找不到 ART_ID_RE 的定义");
  process.exit(1);
}
const ART_ID_RE = eval(idReSource[1]);
const factory = new Function(
  "artByName",
  "artById",
  "ART_ID_RE",
  block + "\nreturn { esc, markArtifacts, restoreArtifacts, md };"
);
const { md } = factory(artByName, artById, ART_ID_RE);

const out = md(run.answer || "");
const buttons = (out.match(/<button/g) || []).length;
const viewerUrl = ((run.artifacts || []).find((i) => i.kind === "viewer") || {}).url || "";

let failures = 0;
const check = (ok, label, detail) => {
  console.log((ok ? "  PASS  " : "  FAIL  ") + label + (detail ? "（" + detail + "）" : ""));
  if (!ok) failures += 1;
};

console.log("=".repeat(68));
console.log("端到端：真服务回答 → index.html 的真函数 → 按钮");
console.log("=".repeat(68));
console.log("产物表：" + JSON.stringify(artByName));
check(viewerUrl !== "", "本次运行产出了 viewer", viewerUrl || "（没有）");
check(!out.includes("D:\\"), "回答里不再残留 Windows 路径");
check(
  out.includes(`data-open="${viewerUrl}"`),
  "路径被换成指向该产物的按钮",
  `data-open="${viewerUrl}"`
);
check(buttons === 1, "恰好一个按钮", buttons + " 个按钮");
check(!out.includes("\u0001"), "无占位符残留");

console.log("\n替换后的相关片段：");
const lines = out.split("<br>").filter((l) => l.includes("button") || l.includes("dead-ref"));
console.log(lines.length ? lines.map((l) => "  " + l).join("\n") : "  （无）");
console.log("\n" + (failures ? `未通过：${failures} 项` : "全部通过"));
process.exit(failures ? 1 : 0);
