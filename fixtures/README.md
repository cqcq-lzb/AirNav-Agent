# fixtures —— 随仓库入库的合成测试数据

本目录放**体积小、无隐私、可随仓库走**的合成夹具，让干净 clone 出来的机器
也能真跑依赖数据的自检。

## `cases/`

合成病例，**24.4 KB**（真实病例 75 MB）。

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
