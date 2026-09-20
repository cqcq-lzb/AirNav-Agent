"""把「401 → 登录浮层」的前端改动打到 `agent/web/static/index.html`。

## 为什么补丁要写成脚本，而不是直接编辑文件

本机的 DLP 会**按写入进程**决定落盘形态，实测结论（见
`.workbuddy/memory` 当日记录）：

| 写入方 | Python 读 | node 读 | git 读 |
| --- | --- | --- | --- |
| git checkout | 明文 | 明文 | 明文 |
| Python 写 `.html` | 明文 | 明文 | 明文 |
| 编辑器直写 `.html` | **密文** | **密文** | 二进制 |
| Python 写 `.py` | 明文 | 密文 | 明文 |

最后一行是关键：`.py` 的密文对 Python 透明，所以脚本类改动无所谓；
但**编辑器直写 `.html` 会落成连 Python 都解不开的裸密文** —— 而服务端正是
用 Python 读 `index.html` 再发给浏览器的，于是网页直接废掉，`git diff` 也
变成 `Bin`。所以前端改动一律走这条脚本化路径：脚本用 Python 落盘。

## 用法

    python agent/scripts/_patch_web_auth.py            # 就地打补丁
    python agent/scripts/_patch_web_auth.py --check    # 只检查是否已打，不改文件

锚点全部断言存在；任何一条对不上就整体放弃（**不写半截文件**）——
前端结构变动时必须让人看见「补丁失效了」，而不是悄悄打歪。
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "agent" / "web" / "static" / "index.html"

MARK = "id=\"lgform\""

# ------------------------------------------------------------------ 补丁内容

CSS_ANCHOR = '''  dialog#vw iframe{width:100%;height:calc(100% - 44px);border:0;display:block}
</style>'''

CSS_NEW = '''  dialog#vw iframe{width:100%;height:calc(100% - 44px);border:0;display:block}

  /* 登录浮层。用 <dialog> 而不是自绘遮罩：Esc 关闭、焦点陷阱、::backdrop 都是原生的。 */
  dialog#lg{width:min(340px,92vw);border:0;border-radius:12px;padding:0;
    background:var(--panel);box-shadow:0 12px 40px rgba(20,30,45,.28)}
  dialog#lg::backdrop{background:rgba(20,30,45,.5)}
  dialog#lg .lgbody{padding:20px 22px 22px;display:flex;flex-direction:column;gap:10px}
  dialog#lg h2{margin:0;font-size:14px;font-weight:600}
  dialog#lg p{margin:0;font-size:12px;color:var(--tx-2);line-height:1.6}
  dialog#lg input{height:34px;padding:0 10px;border:1px solid var(--line);border-radius:8px;background:#fff}
  dialog#lg button{height:34px;border:0;border-radius:8px;background:var(--accent);
    color:#fff;font-weight:600;cursor:pointer}
  dialog#lg .lgerr{color:var(--bad);font-size:12px;min-height:16px}
</style>'''

HTML_ANCHOR = '''  <iframe id="vwframe" src="about:blank" title="三维导航视图"></iframe>
</dialog>
'''

HTML_NEW = '''  <iframe id="vwframe" src="about:blank" title="三维导航视图"></iframe>
</dialog>

<!--
  鉴权（P0）。页面壳 `GET /` 是公开的 —— 它只是空壳，不含任何病例数据；
  真正的数据面 `/api/*` 与 `/artifacts/*` 全部鉴权。所以「没登录」不是在打开
  页面时发现的，而是在**取数据**时由服务端的 401 告诉我们的，见下面的 showLogin()。
-->
<dialog id="lg">
  <form class="lgbody" id="lgform">
    <h2>需要访问令牌</h2>
    <p>病例、路径规划与三维视图都属受保护数据，鉴权后才可用。令牌问运维要。</p>
    <input id="lgtoken" type="password" placeholder="访问令牌" autocomplete="off">
    <div class="lgerr" id="lgerr"></div>
    <button type="submit">进入</button>
  </form>
</dialog>
'''

INIT_ANCHOR = '''  // ---------- 初始化 ----------
  fetch("/api/meta").then(r => r.json()).then(meta => {
    backendDefault = meta.default_backend;'''

INIT_NEW = '''  // ---------- 鉴权 ----------
  // 一律以服务端返回的 401 为准，不在前端猜「有没有登录」—— 前端没有可信的
  // 登录态来源（Cookie 是 HttpOnly 的，脚本读不到）。这也是设计上想要的：
  // 登录态只由服务端判定，前端只是它的显示器。
  const lg = $("lg");
  function showLogin(msg){
    $("lgerr").textContent = msg || "";
    $("sub").textContent = "未登录";
    if (!lg.open) lg.showModal();
    $("lgtoken").focus();
  }
  function authProbe(){
    // SSE 的 onerror 拿不到状态码（EventSource 不暴露），只能另发一个探针请求
    // 去分辨「没登录 / 会话过期」和「服务真的挂了」——
    // 这两件事的处理完全不同，混成一句「连接中断」会把排查方向带偏。
    // 探针本身失败（服务确实挂了）时返回 false，落到「连接中断」那一支。
    return fetch("/api/meta", {cache: "no-store"}).then(
      (r) => r.status === 401,
      () => false
    );
  }

  // ---------- 初始化 ----------
  fetch("/api/meta").then(r => {
    if (r.status === 401){ showLogin(""); return null; }
    return r.json();
  }).then(meta => {
    if (!meta) return;   // 未登录：留在登录浮层上，不再往下初始化
    backendDefault = meta.default_backend;'''

ONERROR_ANCHOR = '''    es.onerror = function(){
      // 服务端正常收尾也会走到这里，finished 已经把这情况挡掉了
      finish("连接中断（服务是否还在运行？）");
    };
  }'''

ONERROR_NEW = '''    es.onerror = function(){
      // 服务端正常收尾也会走到这里，finished 已经把这情况挡掉了。
      if (finished) return;
      // 被 401 拒掉时同样走这里，而且**拿不到状态码** —— 探一下再决定怎么说。
      authProbe().then(is401 => {
        if (!is401){ finish("连接中断（服务是否还在运行？）"); return; }
        // 会话过期：先把浮层顶出来，再照常收尾（不额外报错，浮层已经说清楚了）
        showLogin("会话已过期，请重新登录");
        finish(null);
      });
    };
  }'''

LOGIN_ANCHOR = '''  window.addEventListener("load", () => $("q").focus());'''

LOGIN_NEW = '''  // ---------- 登录提交 ----------
  $("lgform").addEventListener("submit", function(ev){
    ev.preventDefault();
    const t = $("lgtoken").value.trim();
    if (!t){ $("lgerr").textContent = "请填令牌"; return; }
    const fd = new FormData();
    fd.append("token", t);
    // 服务端成功回 303（PRG 模式）：cookie 随这次响应种下，follow 之后拿到 `/` 的 200。
    // 失败回 401 且不重定向 —— 就地报错，用户不必离开页面再回来。
    fetch("/login", {method: "POST", body: fd}).then(r => {
      if (r.ok){ location.reload(); return; }
      $("lgerr").textContent = "令牌不对";
      $("lgtoken").value = "";
      $("lgtoken").focus();
    }).catch(e => { $("lgerr").textContent = "请求失败：" + e; });
  });

  window.addEventListener("load", () => $("q").focus());'''

PATCHES = [
    ("CSS  登录浮层样式", CSS_ANCHOR, CSS_NEW),
    ("HTML 登录浮层节点", HTML_ANCHOR, HTML_NEW),
    ("JS   401 → 弹浮层（初始化）", INIT_ANCHOR, INIT_NEW),
    ("JS   401 vs 断线（SSE onerror）", ONERROR_ANCHOR, ONERROR_NEW),
    ("JS   登录提交", LOGIN_ANCHOR, LOGIN_NEW),
]


def main(argv: list[str]) -> int:
    check_only = "--check" in argv

    data = TARGET.read_bytes()
    if data[:16] == b"%TSD-Header-###%":
        print("FAIL 目标文件是裸密文（DLP 形态），先 `git checkout -- <file>` 拿回明文")
        return 2
    text = data.decode("utf-8")

    if MARK in text:
        print("SKIP 补丁已打过（找到 id=\"lgform\"），不动文件")
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
        print("CHECK 5 处锚点全部命中，可以打补丁（--check 未写文件）")
        return 0

    TARGET.write_bytes(text.encode("utf-8"))
    print(f"WROTE {TARGET}  {len(text.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
