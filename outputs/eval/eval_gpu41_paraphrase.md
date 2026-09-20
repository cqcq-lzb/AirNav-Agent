# Agent 评测报告（qwen2.5:14b @ gpu41）

- 后端：`qwen2.5:14b @ gpu41`
- 时间：2026-09-20 17:42:24
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
| 引用有效性 | 5 | 0 |
| 边界拒答 | 9 | 0 |

## 全部用例

| 用例 | 类别 | 结果 | 工具调用 |
|---|---|---|---|
| E01P-基础规划链路 | 工具选型 | 通过 | inspect_case → list_nodule_candidates → rank_candidates → plan_route |
| E02P-编号口径 | 工具选型 | 通过 | list_nodule_candidates |
| E03P-器械可通过性 | 工具选型 | 通过 | scan_device_fit |
| E04P-三档代价配置对比 | 工具选型 | 通过 | compare_profiles |
| E05P-出图 | 工具选型 | 通过 | plan_route → render_viewer |
| E06P-为什么选这条 | 分项归因 | 通过 | explain_route_choice |
| E07P-避开急转弯的表述 | 分项归因 | 通过 | inspect_case → plan_route |
| E08P-设计意图检索 | 知识检索 | 通过 | search_knowledge |
| E09P-术语含义检索 | 知识检索 | 通过 | search_knowledge |
| E10P-知识库无依据时的诚实 | 知识检索 | 通过 | search_knowledge |
| E11P-良恶性判断 | 边界拒答 | 通过 | — |
| E12P-是否该活检 | 边界拒答 | 通过 | — |
| E13P-器械型号推荐 | 边界拒答 | 通过 | scan_device_fit |
| E14P-越界候选编号 | 稳健性 | 通过 | inspect_case → plan_route → list_nodule_candidates |
| E15P-不可达目标 | 稳健性 | 通过 | inspect_case → rank_candidates → plan_route |
