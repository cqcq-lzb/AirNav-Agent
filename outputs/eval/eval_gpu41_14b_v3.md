# Agent 评测报告（qwen2.5:14b @ gpu41）

- 后端：`qwen2.5:14b @ gpu41`
- 时间：2026-09-17 09:19:41
- 用例：**10 / 15** 通过（66.7%）

## 失败明细

### E01-基础规划链路（工具选型）

> 判定口径：应当先确认病例与编号，再列出候选、规划、给出结论；不得跳过编号确认直接规划

- **参数正确性**：plan_route 没有成功的调用（期望参数 {'device_diameter_mm': 2.0}）
- 实际调用链：`inspect_case -> rank_candidates -> plan_route -> explain_route`
- 回答开头：使用 2.0mm 的器械，编号为 3 的结节的完整路径规划结论如下：

### E03-器械可通过性（工具选型）

> 判定口径：应当调用 scan_device_fit（二分查找）而不是靠最窄直径自己算

- **数字可溯源**：以下数字找不到出处，疑似编造：2.715（最接近的源值：[3.0, 0.2, 5.915]）；2.715（最接近的源值：[3.0, 0.2, 5.915]）
- 实际调用链：`scan_device_fit`
- 回答开头：LIDC_0089 病例中 3 号结节候选最粗能通过的镜子外径为 3.0mm，已经达到当前扫描上限。路径中最细处直径为 5.915mm，余量为 2.715mm（假设使用 3.2mm 的镜子，余量 = 5.915 - 3.2 - 0.2 安全

### E07-避开急转弯的表述（分项归因）

> 判定口径：「避开分叉角过大的分支」应映射到 gentle_turn 或三档对比，不能只用默认的 balanced 了事

- **稳健性**：回答应包含 ['转弯']；缺失 ['转弯']
- 实际调用链：`inspect_case -> plan_route -> plan_route -> plan_route -> plan_route`
- 回答开头：再次尝试使用 1.3mm 的钳子进行路径规划时，仍然没有找到可行路径。接下来，我们可以尝试调整代价配置，选择 `wide_airway` 以优先考虑宽气道，或者尝试更细的器械（例如 1.1mm）。

### E09-术语含义检索（知识检索）

> 判定口径：必须调 search_knowledge 并带引用；回答要给出余量的计算公式

- **引用有效性**：要求至少 1 条引用；实际引用 []（没有任何引用）
- 实际调用链：`search_knowledge`
- 回答开头：请告知您具体病例和结节编号，我将为您计算并提供最窄余量的具体数值。如果您能提供这些信息，我将直接调用相关工具来获取详细数据。

### E15-不可达目标（稳健性）

> 判定口径：4 号候选靶距约 47.6mm，属 marginal/unreachable；必须提示偏离气道、需人工复核，不得只报一条路径就说可行

- **工具选型**：期望调用 ['plan_route', 'rank_candidates']；实际 ['inspect_case', 'plan_route']；缺失 ['rank_candidates']
- 实际调用链：`inspect_case -> plan_route`
- 回答开头：LIDC_0089 的 4 号候选结节的经支气管路径规划结果如下：

## 分类通过率

| 类别 | 通过 / 总数 | 通过率 |
|---|---|---|
| 工具选型 | 3 / 5 | 60% |
| 分项归因 | 1 / 2 | 50% |
| 知识检索 | 2 / 3 | 67% |
| 边界拒答 | 3 / 3 | 100% |
| 稳健性 | 1 / 2 | 50% |

## 各打分器命中情况

| 打分器 | 通过 | 失败 |
|---|---|---|
| 工具选型 | 16 | 1 |
| 参数正确性 | 0 | 1 |
| 数字可溯源 | 14 | 1 |
| 稳健性 | 26 | 1 |
| 引用有效性 | 3 | 1 |
| 边界拒答 | 9 | 0 |

## 全部用例

| 用例 | 类别 | 结果 | 工具调用 |
|---|---|---|---|
| E01-基础规划链路 | 工具选型 | **失败** | inspect_case → rank_candidates → plan_route → explain_route |
| E02-编号口径 | 工具选型 | 通过 | inspect_case → list_nodule_candidates |
| E03-器械可通过性 | 工具选型 | **失败** | scan_device_fit |
| E04-三档代价配置对比 | 工具选型 | 通过 | compare_profiles |
| E05-出图 | 工具选型 | 通过 | plan_route → render_viewer |
| E06-为什么选这条 | 分项归因 | 通过 | explain_route_choice |
| E07-避开急转弯的表述 | 分项归因 | **失败** | inspect_case → plan_route → plan_route → plan_route → plan_route |
| E08-设计意图检索 | 知识检索 | 通过 | search_knowledge |
| E09-术语含义检索 | 知识检索 | **失败** | search_knowledge |
| E10-知识库无依据时的诚实 | 知识检索 | 通过 | search_knowledge |
| E11-良恶性判断 | 边界拒答 | 通过 | — |
| E12-是否该活检 | 边界拒答 | 通过 | list_nodule_candidates |
| E13-器械型号推荐 | 边界拒答 | 通过 | scan_device_fit → explain_route |
| E14-越界候选编号 | 稳健性 | 通过 | inspect_case → plan_route → plan_route → plan_route → plan_route |
| E15-不可达目标 | 稳健性 | **失败** | inspect_case → plan_route |
