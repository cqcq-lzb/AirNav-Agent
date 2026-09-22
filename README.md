# AirNav-Agent

把一套虚拟支气管镜导航工作站（原为 PySide6 + VTK 的本地 GUI）重构成 **Agent 形态**：
一个手写的 ReAct 循环 + 11 个领域工具 —— 规划结论可**溯源**、可**评测**、可**审计**。

> **这不是一个 demo。** 每项能力都配了自检，全部挂在同一道**回归门禁**上
> （13 项 / 约 400 条断言 / 约 3 分钟）。门禁的结论是**三态**的：
> ✅ 通过 · ❌ 未通过 · ⚪ **不足以判定** —— 缺数据 ≠ 通过。

规模（实测于 2026-09-22）：**24 次提交 / 6 天** · `agent/` **24 164 行 / 74 文件** ·
**11 个领域工具** · **30 条评测用例** · **6 类硬判定打分器** · 真实病例库 **12 例 / 1.9 GB**。

---

## 快速开始

```bash
PY=/path/to/.venv-mcp/Scripts/python.exe     # 本机为 D:/AirNav-Agent/.venv-mcp/Scripts/python.exe

# 全部回归（13 项，约 3 分钟）
"$PY" -m agent.scripts.gate

# 看清单 / 只跑某几项
"$PY" -m agent.scripts.gate --list
"$PY" -m agent.scripts.gate --only geometry,planner

# 网页版（每一步工具调用 SSE 实时可见）
"$PY" -m agent.web.server          # → http://127.0.0.1:8777
```

Windows 上也可以直接双击 `跑回归门禁.bat` / `启动网页Agent.bat`。

### 没有真实病例数据也能跑

真实病例（`cases/`，1.9 GB）按**隐私 + 体积**不入库。仓库因此自带一份
**合成夹具**（`fixtures/cases`，24 KB/例，不含任何患者信息）：

```bash
# 干净 clone 上：12 ✅ + 1 ⚪ + 0 ❌
AIRNAV_CASES_DIR="D:/AirNav-Agent/fixtures/cases" \
  "$PY" -m agent.scripts.gate --profile ci
```

> 那唯一一个 ⚪ 是 `planner`（与 V1 规划数值逐位对拍）：夹具上没有 V1 的对照结果，
> 所以对拍**没做** —— 门禁如实标「不足以判定」，而不是默认通过。
> 详见 **[`docs/合成夹具.md`](docs/合成夹具.md)**。

---

## 五层结构

| 层 | 是什么 | 钉着它的自检 |
|---|---|---|
| **底座** | Airway 导航 V1/V3 规划内核（**只读**，不修改） | 几何对拍（cKDTree vs scipy EDT 逐位一致）· 规划数值与 V1 逐位一致 |
| **Agent 面** | 手写 ReAct 循环 · 11 个领域工具 · ArtifactStore 上下文隔离 · 规划成功顺带渲染自包含三维 viewer | 控制流 `12/12 + 21/21 + 9/9 + 9/9` · 工具返回体契约 `26/26` |
| **评测面** | 30 条用例 · 6 类硬判定打分器 · **打分器自检**（植入缺陷必须被抓住） | 打分器自检 · 评测基线（不得低于历史） |
| **治理面** | 审计哈希链 · PHI 守卫 · 运行合规报告 · 临床规划报告 | `35/35` · `9/9` · `66/66` · `143/143` |
| **交付面** | CLI · 网页版（SSE + 鉴权）· MCP Server | 网页端 `63/63` · MCP 协议四层自检 |

领域工具：`list_cases` `inspect_case` `list_nodule_candidates` `plan_route`
`compare_profiles` `scan_device_fit` `explain_route` `rank_candidates` `render_viewer`
`search_knowledge` `explain_route_choice`

---

## 评测：区分度是这里最值钱的一件事

规则基线（手写 if-else）在原始用例上打满了下界，那说明**这份评测证明不了模型有用**：

| 用例集 | 模型（qwen2.5:14b） | 规则基线（手写 if-else） |
|---|---|---|
| 原始 15 条 | 15/15 | **15/15** ← 被下界打满，没有区分度 |
| **同义改写 15 条**（判定标准**一字不改**） | **15/15** | **3/15** |

同一批意图换一种说法，规则基线从 15/15 掉到 3/15，模型仍 15/15 —— **这个差距才是含金量**。

改写集用 `dataclasses.replace(base, ...)` 从原始用例**派生**，
所以「尺子没变」是由**结构**保证的，不靠人检查。
真机全量 30 条：**29/30**。实验记录见 [`docs/评测区分度_同义改写实验.md`](docs/评测区分度_同义改写实验.md)。

---

## 回归门禁的三种结论

```
✅ 通过        exit 0
❌ 未通过      exit 1   —— 真回归，必须查
⚪ 不足以判定   exit 2   —— 这一项**没跑成**（缺数据 / 内存不够），不是通过
```

`passed = not failed and not undecided and baseline_ok`；
**退出码只由 ❌ 决定** —— 否则一台没有真实病例的机器上，CI 会永远红着，门禁就没人看了。
但 ⚪ 同样**不**等于 ✅。

这套三态不是装饰：它修掉过两个**静默空转**的真 bug ——
`verify_planner` 在找不到 V1 对照物时曾 `return 0`（硬门禁静默退化成空转），
`scan_phi` 曾扫一个不存在的目录却报「0 命中 → 通过」（没扫到 ≠ 干净）。

---

## 目录

```
agent/
  agent_loop.py     手写 ReAct 循环（控制流自检 12/12 + 21/21 + 9/9 + 9/9）
  core/             病例加载 · 几何 · 规划 · 上下文隔离 · 系统内存读数
  tools/            11 个领域工具（返回体契约 26/26）
  eval/             30 条用例 · 6 类打分器 · 打分器自检 · 基线
  render/           自包含三维 viewer 产物
  web/              网页版（SSE + 鉴权）
  scripts/          gate（门禁）· 各类 check_*/verify_* · make_fixture_case
fixtures/           合成夹具（随仓库入库，24 KB/例）
docs/               设计文档与实验记录
使用指南.md          日常使用与运维
```

---

## 已知边界（诚实清单）

- **招牌项目叠在 V1 之上** —— 对外发布前必须先划清「哪些是自己的、哪些是 V1 的」
- `cases/` 不入库 → 合成夹具能顶大部分，但 `planner` 的 V1 对拍仍只能在有真实病例的机器上做
- **P1 服务化未做**：任务队列与状态机（远程计算目前同步跑在请求内，断线即失败）、
  结构化日志与 trace_id、降级自动切换 —— 一句话：**智力够用，缺可运营性**。
  见 [`docs/企业级差距清单.md`](docs/企业级差距清单.md)
- 仓库尚未配置 git remote（CI 工作流已就绪但未真正触发）
