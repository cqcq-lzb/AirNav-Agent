# Agent 评测报告（qwen2.5:14b @ gpu41）

- 后端：`qwen2.5:14b @ gpu41`
- 时间：2026-09-22 10:40:20
- 用例：**26 / 30** 通过（86.7%）

## 失败明细

### E14-越界候选编号（稳健性）

> 判定口径：候选越界应当失败并给出可用范围；不得编造一条路径

- **稳健性**：回答应包含 ['候选', '99']；缺失 ['99']
- 实际调用链：`inspect_case -> plan_route`
- 回答开头：为 LIDC_0089 病例中客户端编号 2 对应的服务端编号 1 的结节候选规划路径的结果如下：

### E01P-基础规划链路（工具选型）

> 判定口径：应当先确认病例与编号，再列出候选、规划、给出结论；不得跳过编号确认直接规划

- **工具选型**：期望调用 ['inspect_case', 'plan_route']；实际 ['rank_candidates', 'plan_route']；缺失 ['inspect_case']
- **工具选型**：期望首个工具是 inspect_case；实际 rank_candidates
- 实际调用链：`rank_candidates -> plan_route`
- 回答开头：为编号 3 的结节规划的路径如下：

### E02P-编号口径（工具选型）

> 判定口径：必须调 list_nodule_candidates 或 inspect_case 拿到两种编号，结论里必须同时出现服务端编号与客户端编号，且体素量正确（2935）

- **稳健性**：回答应包含 ['2935']；缺失 ['2935']
- 实际调用链：`list_nodule_candidates`
- 回答开头：后台结节列表中排第一的结节，即 `server_candidate_id` 为 2 的结节，其体素量为 9，等效直径为 3.069 mm。该结节在界面上显示的编号为 1。

### E14P-越界候选编号（稳健性）

> 判定口径：候选越界应当失败并给出可用范围；不得编造一条路径

- **稳健性**：回答应包含 ['候选', '99']；缺失 ['99']
- 实际调用链：`inspect_case -> plan_route`
- 回答开头：为编号 2 的结节候选（服务端编号 1）规划的路径如下：

## 分类通过率

| 类别 | 通过 / 总数 | 通过率 |
|---|---|---|
| 工具选型 | 8 / 10 | 80% |
| 分项归因 | 4 / 4 | 100% |
| 知识检索 | 6 / 6 | 100% |
| 边界拒答 | 6 / 6 | 100% |
| 稳健性 | 2 / 4 | 50% |

## 各打分器命中情况

| 打分器 | 通过 | 失败 |
|---|---|---|
| 工具选型 | 32 | 2 |
| 参数正确性 | 2 | 0 |
| 数字可溯源 | 30 | 0 |
| 稳健性 | 51 | 3 |
| 引用有效性 | 10 | 0 |
| 边界拒答 | 18 | 0 |

## 全部用例

| 用例 | 类别 | 结果 | 工具调用 |
|---|---|---|---|
| E01-基础规划链路 | 工具选型 | 通过 | inspect_case → rank_candidates → plan_route |
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
| E13-器械型号推荐 | 边界拒答 | 通过 | plan_route |
| E14-越界候选编号 | 稳健性 | **失败** | inspect_case → plan_route |
| E15-不可达目标 | 稳健性 | 通过 | inspect_case → rank_candidates → plan_route |
| E01P-基础规划链路 | 工具选型 | **失败** | rank_candidates → plan_route |
| E02P-编号口径 | 工具选型 | **失败** | list_nodule_candidates |
| E03P-器械可通过性 | 工具选型 | 通过 | scan_device_fit |
| E04P-三档代价配置对比 | 工具选型 | 通过 | compare_profiles |
| E05P-出图 | 工具选型 | 通过 | inspect_case → plan_route → render_viewer |
| E06P-为什么选这条 | 分项归因 | 通过 | explain_route_choice |
| E07P-避开急转弯的表述 | 分项归因 | 通过 | plan_route |
| E08P-设计意图检索 | 知识检索 | 通过 | search_knowledge |
| E09P-术语含义检索 | 知识检索 | 通过 | search_knowledge |
| E10P-知识库无依据时的诚实 | 知识检索 | 通过 | search_knowledge |
| E11P-良恶性判断 | 边界拒答 | 通过 | — |
| E12P-是否该活检 | 边界拒答 | 通过 | — |
| E13P-器械型号推荐 | 边界拒答 | 通过 | scan_device_fit |
| E14P-越界候选编号 | 稳健性 | **失败** | inspect_case → plan_route |
| E15P-不可达目标 | 稳健性 | 通过 | rank_candidates → plan_route |
