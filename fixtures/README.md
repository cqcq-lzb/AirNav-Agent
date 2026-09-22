# fixtures —— 随仓库入库的合成测试数据

本目录放**体积小、无隐私、可随仓库走**的合成夹具，让干净 clone 出来的机器
也能真跑依赖数据的自检。

## `cases/`

合成病例，**26.1 KB**（真实病例 75 MB）。

`stage4_package/` 六个文件：

| 文件 | 作用 |
|---|---|
| `ct.nii.gz` | 提供 spacing / origin（规划器只做物理坐标换算，不读 HU） |
| `airway_mask.nii.gz` | 气道掩膜 —— `geometry` 半径场对拍要它 |
| `nodule_raw.nii.gz` | 全部结节连通域 |
| `nodule_selected.nii.gz` | 靶点掩膜（取体积最大的那颗）—— `geometry` 靶点距离对拍要它 |
| `entry_point.nii.gz` | 入口点（固定入口模式） |
| `stage4_manifest.json` | 服务端编号清单（按体积降序，**刻意与客户端顺序不同**） |

> `nodule_selected.nii.gz` 是**后来补的**。真实 `stage4_package` 里每一份都有它，
> 夹具原先漏了 —— 后果不是「少一个文件」，是 `verify_geometry` 的靶点距离对拍
> 在干净 clone 上只能报 ⚪，**一条硬门禁在 CI 里永远跑不起来**。
> 补上之后，几何对拍的三个子项在夹具上全部真跑、全绿（200000/200000 逐位相同）。

```bash
# 用它跑全量门禁（来历会显式标出来）
AIRNAV_CASES_DIR=D:/AirNav-Agent/fixtures/cases \
  D:/AirNav-Agent/.venv-mcp/Scripts/python.exe -m agent.scripts.gate
```

- 生成脚本：`agent/scripts/make_fixture_case.py`
- 详细说明：**[`docs/合成夹具.md`](../docs/合成夹具.md)** ← 设计约束、来历机制、
  以及「它**不能**证明什么」都写在那里
- 来历：`case_loader.cases_root_provenance()` / `is_fixture()`；
  工具返回体里带 `"fixture": true`

> ⚠️ 它**不是**真实病例的替代品：夹具上没有 `interactive_plans/`，
> 所以 `planner` 项（与 V1 对拍）在它上面会明确报「不足以判定」（⚪），
> 而不是通过。

## 为什么 `cases/` 这条 gitignore 规则要放行

`.gitignore` 里的 `cases/` **匹配任意层级**的 `cases` 目录，包括 `fixtures/cases`。
所以放行必须写两步（父目录被排除时子级例外不生效）：

```gitignore
cases/
!fixtures/cases/
!fixtures/cases/**
```
