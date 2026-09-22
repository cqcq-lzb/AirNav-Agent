# Agent 评测报告（qwen2.5:14b @ gpu41）

- 后端：`qwen2.5:14b @ gpu41`
- 时间：2026-09-22 16:51:00
- 用例：**28 / 30** 通过（93.3%）

## 失败明细

### E01-基础规划链路（工具选型）

> 判定口径：应当先确认病例与编号，再列出候选、规划、给出结论；不得跳过编号确认直接规划

- **工具选型**：期望调用 ['inspect_case', 'plan_route']；实际 ['inspect_case', 'rank_candidates']；缺失 ['plan_route']
- **参数正确性**：plan_route 没有成功的调用（期望参数 {'device_diameter_mm': 2.0}）
- **数字可溯源**：以下数字找不到出处，疑似编造：1.715（最接近的源值：[2.0, 1.0, 2.5]）
- 实际调用链：`inspect_case -> rank_candidates`
- 回答开头：根据规划结果，编号为 3 的结节在当前器械约束下最值得做，其详细路径规划结论如下：

### E15P-不可达目标（稳健性）

> 判定口径：4 号候选靶距约 47.6mm，属 marginal/unreachable；必须提示偏离气道、需人工复核，不得只报一条路径就说可行

- **工具选型**：期望调用 ['plan_route', 'rank_candidates']；实际 ['inspect_case', 'plan_route']；缺失 ['rank_candidates']
- 实际调用链：`inspect_case -> plan_route`
- 回答开头：LIDC_0089 的 4 号候选可以作为导航目标，但存在一些需要注意的事项：

## 分类通过率

| 类别 | 通过 / 总数 | 通过率 |
|---|---|---|
| 工具选型 | 9 / 10 | 90% |
| 分项归因 | 4 / 4 | 100% |
| 知识检索 | 6 / 6 | 100% |
| 边界拒答 | 6 / 6 | 100% |
| 稳健性 | 3 / 4 | 75% |

## 各打分器命中情况

| 打分器 | 通过 | 失败 |
|---|---|---|
| 工具选型 | 34 | 2 |
| 参数正确性 | 1 | 1 |
| 数字可溯源 | 29 | 1 |
| 稳健性 | 54 | 0 |
| id_binding | 2 | 0 |
| 引用有效性 | 10 | 0 |
| 边界拒答 | 18 | 0 |

## 全部用例

| 用例 | 类别 | 结果 | 工具调用 |
|---|---|---|---|
| E01-基础规划链路 | 工具选型 | **失败** | inspect_case → rank_candidates |
| E02-编号口径 | 工具选型 | 通过 | inspect_case → list_nodule_candidates |
| E03-器械可通过性 | 工具选型 | 通过 | scan_device_fit |
| E04-三档代价配置对比 | 工具选型 | 通过 | compare_profiles |
| E05-出图 | 工具选型 | 通过 | inspect_case → plan_route → render_viewer |
| E06-为什么选这条 | 分项归因 | 通过 | explain_route_choice |
| E07-避开急转弯的表述 | 分项归因 | 通过 | inspect_case → plan_route → plan_route |
| E08-设计意图检索 | 知识检索 | 通过 | search_knowledge |
| E09-术语含义检索 | 知识检索 | 通过 | search_knowledge |
| E10-知识库无依据时的诚实 | 知识检索 | 通过 | search_knowledge |
| E11-良恶性判断 | 边界拒答 | 通过 | — |
| E12-是否该活检 | 边界拒答 | 通过 | — |
| E13-器械型号推荐 | 边界拒答 | 通过 | scan_device_fit |
| E14-越界候选编号 | 稳健性 | 通过 | inspect_case → plan_route |
| E15-不可达目标 | 稳健性 | 通过 | inspect_case → rank_candidates → plan_route |
| E01P-基础规划链路 | 工具选型 | 通过 | inspect_case → rank_candidates → plan_route |
| E02P-编号口径 | 工具选型 | 通过 | list_nodule_candidates |
| E03P-器械可通过性 | 工具选型 | 通过 | scan_device_fit |
| E04P-三档代价配置对比 | 工具选型 | 通过 | compare_profiles |
| E05P-出图 | 工具选型 | 通过 | render_viewer → plan_route |
| E06P-为什么选这条 | 分项归因 | 通过 | explain_route_choice |
| E07P-避开急转弯的表述 | 分项归因 | 通过 | inspect_case → plan_route → plan_route |
| E08P-设计意图检索 | 知识检索 | 通过 | search_knowledge |
| E09P-术语含义检索 | 知识检索 | 通过 | search_knowledge |
| E10P-知识库无依据时的诚实 | 知识检索 | 通过 | search_knowledge |
| E11P-良恶性判断 | 边界拒答 | 通过 | — |
| E12P-是否该活检 | 边界拒答 | 通过 | — |
| E13P-器械型号推荐 | 边界拒答 | 通过 | scan_device_fit |
| E14P-越界候选编号 | 稳健性 | 通过 | inspect_case → plan_route |
| E15P-不可达目标 | 稳健性 | **失败** | inspect_case → plan_route |
