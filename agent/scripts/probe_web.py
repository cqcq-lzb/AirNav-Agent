"""打一次网页版 Agent 的 SSE 接口，看它**实际**给不给链接。

    python -m agent.scripts.probe_web [--port 8777] [--backend gpu41] [--dump 路径]

## 为什么需要它

**源码修好了 ≠ 正在跑的东西修好了。** 2026-09-17 就栽在这上面：

    8777 上早就有一个进程（12:43 启动的后台服务），它 import 的是**改动前**的代码。
    源码里 `plan_route` 已经会顺带出图，但那个进程仍然只回旧字段 ——
    产物只有 `route`、回答里一个链接都没有。

所以判断「跑着的这一版到底什么行为」，唯一可靠的判据是**行为探针**：
真发一个请求，看返回体里有没有你要的东西。
不要比文件 mtime，更不要凭「我刚改过」。

## 用法一：判断哪一版代码在跑

    python -m agent.scripts.probe_web
    # 看「产物」那几行有没有 viewer、回答里有没有 viewer_path

## 用法二：端到端验证界面（回答里的路径 → 按钮）

    python -m agent.scripts.probe_web --dump D:\\tmp\\web_run.json
    node agent/scripts/check_web_e2e.js D:\\tmp\\web_run.json

把这一轮的**真回答 + 真产物表**喂给 `index.html` 里那段真函数，
验证用户点得到，而不是只在 fixture 上成立。

## 说明

- 需要**服务已启动**（`python -m agent.web.server`）与**真模型可达**，所以它不进
  `check_web` 那套秒级常驻自检，是手动入口。
- 本机的坑：`nohup … &` 起的服务会随那次 shell 调用结束被回收 ——
  同一次调用里 `/health` 返回 200，下一次调用就 `ConnectionRefused`。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_QUESTION = "LIDC_0089 的 3 号结节候选，用 1.5mm 器械规划一条路径，结果给我"


def main() -> int:
    parser = argparse.ArgumentParser(description="网页版 SSE 行为探针")
    parser.add_argument("--port", default="8777")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend", default="gpu41")
    parser.add_argument("--device-diameter", default="1.5")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--dump", default=None, help="把回答与产物表写成 JSON")
    args = parser.parse_args()

    url = (
        f"http://{args.host}:{args.port}/api/chat?"
        + urllib.parse.urlencode(
            {
                "q": args.question,
                "backend": args.backend,
                "device_diameter": args.device_diameter,
            }
        )
    )
    # 本机 http_proxy 会劫持 127.0.0.1，必须绕开
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=300) as resp:
            raw_lines = resp.readlines()
    except Exception as exc:  # noqa: BLE001
        print(f"打不通 http://{args.host}:{args.port} —— 服务起了吗？（{exc}）")
        return 1

    events: list[dict] = []
    for raw in raw_lines:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        try:
            events.append(json.loads(line[5:].strip()))
        except json.JSONDecodeError:
            pass

    print(f"事件数 {len(events)}")
    print("类型序：", [e.get("type") for e in events])

    items: list[dict] = []
    for event in events:
        if event.get("type") == "artifacts":
            items = event.get("items", [])
            for item in items:
                print(
                    f"  产物 {item.get('kind')}  "
                    f"{item.get('filename') or item.get('path')}  url={item.get('url')}"
                )
    if not any(i.get("kind") == "viewer" for i in items):
        print("  ⚠️ 没有 viewer 产物 —— 这一版大概率是 `plan_route` 顺带出图之前的代码")

    final = next((e for e in events if e.get("type") == "final"), None)
    answer = ""
    if final:
        answer = final.get("answer") or final.get("content") or ""
        print(
            f"\n统计：{final.get('stop')} / {final.get('steps')} 步 / "
            f"工具 {final.get('tool_calls')}"
        )
        print("\n--- 回答 ---")
        print(answer)
    if "viewer" in answer and ".html" not in answer:
        print("\n⚠️ 回答里提了 viewer 却没有文件路径 —— 界面做不出按钮")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(
                {"answer": answer, "artifacts": items}, fh, ensure_ascii=False, indent=1
            )
        print(f"\n已落盘：{args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
