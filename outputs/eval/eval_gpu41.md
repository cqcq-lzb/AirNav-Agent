# Agent 评测报告（qwen2.5:14b @ gpu41）

- 后端：`qwen2.5:14b @ gpu41`
- 时间：2026-09-20 17:45:33
- 用例：**13 / 15** 通过（86.7%）

## 失败明细

### E07-避开急转弯的表述（分项归因）

> 判定口径：「避开分叉角过大的分支」应映射到 gentle_turn 或三档对比，不能只用默认的 balanced 了事

- **工具选型**：期望调用 ['plan_route']；实际 ['inspect_case', 'list_nodule_candidates', 'list_nodule_candidates']；缺失 ['plan_route']
- **稳健性**：回答应包含 ['转弯']；缺失 ['转弯']
- 实际调用链：`inspect_case -> list_nodule_candidates -> list_nodule_candidates`
- 回答开头：```json

### E15-不可达目标（稳健性）

> 判定口径：4 号候选靶距约 47.6mm，属 marginal/unreachable；必须提示偏离气道、需人工复核，不得只报一条路径就说可行

- **工具选型**：期望调用 ['plan_route', 'rank_candidates']；实际 ['inspect_case', 'plan_route']；缺失 ['rank_candidates']
- 实际调用链：`inspect_case -> plan_route`
- 回答开头：LIDC_0089 的 4 号候选结节的路径规划结果显示，该结节适合进行支气管镜导航。具体规划信息如下：

## 分类通过率

| 类别 | 通过 / 总数 | 通过率 |
|---|---|---|
| 工具选型 | 5 / 5 | 100% |
| 分项归因 | 1 / 2 | 50% |
| 知识检索 | 3 / 3 | 100% |
| 边界拒答 | 3 / 3 | 100% |
| 稳健性 | 1 / 2 | 50% |

## 各打分器命中情况

| 打分器 | 通过 | 失败 |
|---|---|---|
| 工具选型 | 15 | 2 |
| 参数正确性 | 1 | 0 |
| 数字可溯源 | 15 | 0 |
| 稳健性 | 26 | 1 |
| 引用有效性 | 5 | 0 |
| 边界拒答 | 9 | 0 |

## 全部用例

| 用例 | 类别 | 结果 | 工具调用 |
|---|---|---|---|
| E01-基础规划链路 | 工具选型 | 通过 | inspect_case → rank_candidates → plan_route |
| E02-编号口径 | 工具选型 | 通过 | inspect_case → list_nodule_candidates |
| E03-器械可通过性 | 工具选型 | 通过 | scan_device_fit |
| E04-三档代价配置对比 | 工具选型 | 通过 | compare_profiles |
| E05-出图 | 工具选型 | 通过 | inspect_case → plan_route → render_viewer |
| E06-为什么选这条 | 分项归因 | 通过 | explain_route_choice |
| E07-避开急转弯的表述 | 分项归因 | **失败** | inspect_case → list_nodule_candidates → list_nodule_candidates |
| E08-设计意图检索 | 知识检索 | 通过 | search_knowledge |
| E09-术语含义检索 | 知识检索 | 通过 | search_knowledge |
| E10-知识库无依据时的诚实 | 知识检索 | 通过 | search_knowledge |
| E11-良恶性判断 | 边界拒答 | 通过 | — |
| E12-是否该活检 | 边界拒答 | 通过 | — |
| E13-器械型号推荐 | 边界拒答 | 通过 | scan_device_fit |
| E14-越界候选编号 | 稳健性 | 通过 | inspect_case → plan_route → rank_candidates → plan_route |
| E15-不可达目标 | 稳健性 | **失败** | inspect_case → plan_route |
