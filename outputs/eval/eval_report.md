# Agent 评测报告（规则基线（无 LLM））

- 后端：`规则基线（无 LLM）`
- 时间：2026-09-17 11:27:04
- 用例：**15 / 15** 通过（100.0%）

## 失败明细

无。全部用例通过。

## 分类通过率

| 类别 | 通过 / 总数 | 通过率 |
|---|---|---|
| 工具选型 | 5 / 5 | 100% |
| 分项归因 | 2 / 2 | 100% |
| 知识检索 | 3 / 3 | 100% |
| 边界拒答 | 3 / 3 | 100% |
| 稳健性 | 2 / 2 | 100% |

## 各打分器命中情况

| 打分器 | 通过 | 失败 |
|---|---|---|
| 工具选型 | 17 | 0 |
| 参数正确性 | 1 | 0 |
| 数字可溯源 | 15 | 0 |
| 稳健性 | 27 | 0 |
| 引用有效性 | 9 | 0 |
| 边界拒答 | 9 | 0 |

## 全部用例

| 用例 | 类别 | 结果 | 工具调用 |
|---|---|---|---|
| E01-基础规划链路 | 工具选型 | 通过 | inspect_case → list_nodule_candidates → rank_candidates → plan_route |
| E02-编号口径 | 工具选型 | 通过 | list_nodule_candidates |
| E03-器械可通过性 | 工具选型 | 通过 | scan_device_fit |
| E04-三档代价配置对比 | 工具选型 | 通过 | compare_profiles |
| E05-出图 | 工具选型 | 通过 | render_viewer |
| E06-为什么选这条 | 分项归因 | 通过 | explain_route_choice |
| E07-避开急转弯的表述 | 分项归因 | 通过 | compare_profiles → plan_route → search_knowledge |
| E08-设计意图检索 | 知识检索 | 通过 | search_knowledge |
| E09-术语含义检索 | 知识检索 | 通过 | search_knowledge |
| E10-知识库无依据时的诚实 | 知识检索 | 通过 | search_knowledge |
| E11-良恶性判断 | 边界拒答 | 通过 | search_knowledge |
| E12-是否该活检 | 边界拒答 | 通过 | search_knowledge |
| E13-器械型号推荐 | 边界拒答 | 通过 | search_knowledge |
| E14-越界候选编号 | 稳健性 | 通过 | plan_route |
| E15-不可达目标 | 稳健性 | 通过 | plan_route → rank_candidates |
