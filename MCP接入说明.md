# AirNav MCP Server 接入说明

把经支气管导航路径规划能力暴露成标准 **Model Context Protocol** 服务，
Claude Desktop / Cursor 等 MCP 客户端可以**直接驱动**这套工具，不需要跑本项目的 LLM 客户端。

- 服务名：`airnav-agent`　版本：`0.1.0`　传输：**stdio**
- 暴露工具：**11 个**
- 服务端代码：`agent/mcp_server.py`

---

## 1. 快速接入

### 1.1 Claude Desktop

编辑 `%APPDATA%\Claude\claude_desktop_config.json`，把下面这段合并进去：

```json
{
  "mcpServers": {
    "airnav-agent": {
      "command": "D:\\AirNav-Agent\\.venv-mcp\\Scripts\\python.exe",
      "args": ["-m", "agent.mcp_server"],
      "cwd": "D:\\AirNav-Agent",
      "env": {
        "PYTHONPATH": "D:\\AirNav-Agent",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1"
      }
    }
  }
}
```

改完**完全退出并重启 Claude Desktop**（只是关窗口不算）。之后对话里应能看到
`airnav-agent` 的工具列表。

### 1.2 Cursor

写到项目根目录的 `.cursor/mcp.json`（内容同上），或在
`Cursor Settings → MCP` 里手工添加。Cursor 会自动读取项目级配置。

### 1.3 三个必须照抄的点

| 配置项 | 为什么不能省 |
|---|---|
| `cwd` = 项目根 | `cases/` 病例目录与 `agent/rag/knowledge/` 知识库都是相对它解析的 |
| `PYTHONIOENCODING=utf-8` | 工具返回大量中文，Windows 默认 GBK 会损坏协议报文里的中文 |
| `PYTHONNOUSERSITE=1` | 显式屏蔽全局用户目录，保证启动结果不受它偶然内容影响（见第 3 节） |

配置模板见项目根目录 `mcp_config.example.json`。

---

## 2. 工具清单（11 个）

| 工具 | 作用 |
|---|---|
| `list_cases` | 列出本机已下载的病例，规划前先用它确认病例编号 |
| `inspect_case` | 病例概况：体素尺寸、气道规模、中心线节点数、入口点模式、候选数量 |
| `list_nodule_candidates` | 全部结节候选：体积、等效直径、三维中心、**两种编号** |
| `plan_route` | 为指定候选规划路径，返回长度 / 最窄直径 / 最大转角 / 可达性分级 |
| `compare_profiles` | 三档代价配置（balanced / wide_airway / gentle_turn）各规划一遍并对比 |
| `scan_device_fit` | 二分查找该路径能通过的最大器械外径（mm） |
| `explain_route` | 路径完整可解释信息：分叉序列逐级展开、最窄处与最急转弯位置 |
| `rank_candidates` | 全部候选各规划一次，按「好到达程度」排序 |
| `render_viewer` | 渲染自包含三维交互视图（单文件 HTML，离线可打开） |
| `search_knowledge` | 检索项目知识库，返回带引用号（形如 `[KB-01#3]`）的片段 |
| `explain_route_choice` | **「为什么选这条路径」的定量归因**，给出代价四项占比 |

### 使用上最容易踩的两点

1. **编号口径**。本系统的「客户端编号」与服务器清单的「服务端编号」并不一致。
   涉及具体候选之前先调 `inspect_case` 或 `list_nodule_candidates`，
   两个编号会同时返回。
2. **回答要带引用**。解释设计意图时用 `search_knowledge`，并在回答里带上
   `[KB-xx#n]` 引用号；检索返回空列表时应当直说「知识库没有依据」，不要编。

---

## 3. 运行环境：为什么解释器指向 `.venv-mcp`

### 3.1 conda 环境装不进去

`D:\miniforge` 是**管理员**安装的，ACL 对普通用户只有 `(RX)`：

```
WISEKING\Domain Users:(OI)(CI)(RX)
BUILTIN\Users:(OI)(CI)(RX)
BUILTIN\Administrators:(F)          ← 写权限在这里
```

所以 `D:\miniforge\envs\dicom\Lib\site-packages` 普通用户**写不进去**，
安装依赖会报 `WinError 5 拒绝访问`。加沙箱豁免也没用——这是 ACL 问题，不是沙箱问题。

### 3.2 conda 环境里 mcp 其实是「借来的」

`dicom` 环境是 Python **3.10.20**，`mcp` 及其全部依赖实际只存在于
Python 3.10 的 **user-site**：

```
%APPDATA%\Python\Python310\site-packages
```

当时实测：闭包 27 个包里，只有 `exceptiongroup`、`typing-extensions` 在 conda 环境自身，
**另外 25 个全部来自 user-site**。

危险之处在于**这件事不会报错**：

```text
$ pip list | grep mcp
mcp    2.2.0        ← 看起来装好了（pip 把 user-site 一起列出来了）

$ pip install --dry-run mcp==2.2.0
install: []         ← pip 认为无需安装
```

但一旦屏蔽 user-site，立刻现原形：

```text
$ PYTHONNOUSERSITE=1 python -c "import mcp"
ModuleNotFoundError: No module named 'mcp'
```

典型的「平时能跑、一部署就炸」。

### 3.3 解法：项目内独立环境

```powershell
# 1) 建环境（可在项目内写，不需要管理员）
D:\miniforge\envs\dicom\python.exe -m venv --system-site-packages D:\AirNav-Agent\.venv-mcp

# 2) 分析依赖归属，生成锁定文件（用 --stdout 加重定向，见 3.5）
python -m agent.scripts.lock_mcp_deps --stdout all > requirements_mcp.txt
python -m agent.scripts.lock_mcp_deps --stdout env > requirements_mcp_env.txt

# 3) 按锁定版本补装（会自动清洗文件的 NUL 填充，走清华镜像）
D:\AirNav-Agent\.venv-mcp\Scripts\python.exe -m agent.scripts.lock_mcp_deps --install
```

最后一步**不要**直接写 `pip install -r requirements_mcp_env.txt`：
本机 DLP 会把文件补 NUL 到 4096 字节块对齐，pip 会直接
`ERROR: Invalid requirement: '\x00\x00...'`。`--install` 会先把 pin 行
抽出来、清掉填充、写成临时文件再交给 pip，并在装完后逐个复核这些包
是否真的落进 `.venv-mcp` 自己的站点目录。

`--system-site-packages` 让新环境能看见 conda 里的
numpy / scipy / scikit-image / SimpleITK / vtk / paramiko，
自身站点则负责 MCP 那一层依赖。`--no-deps` 是刻意的：版本已经锁死，
不需要 pip 再解析一遍，也就不会顺手升级既有包。

### 3.4 锁定文件

| 文件 | 内容 |
|---|---|
| `requirements_mcp.txt` | `mcp` 的**完整依赖闭包**（27 个包，含精确版本） |
| `requirements_mcp_env.txt` | 只需补装的那部分（25 个包），可直接喂给 pip |

生成脚本：`agent/scripts/lock_mcp_deps.py`。
它用 `importlib.metadata.distributions(path=...)` **直接枚举**解释器自身的站点目录，
而不是相信 `pip list`——后者会把 user-site 一起算进来。

> 版本已锁定在 **MCP 2.x**。`FastMCP` 在 2.x 里已改名为 `MCPServer`
> （`from mcp.server.mcpserver import MCPServer`），网上大量 1.x 教程的
> `from mcp.server.fastmcp import FastMCP` 在这里会直接 `ModuleNotFoundError`。

### 3.5 锁定文件的产出方式（本机 DLP 注意）

本机装了透明加密类 DLP，它会用**两种**方式改写写出的文件，**触发条件都不确定**
（同类小文件在本机多数不会被动，所以不要赌）：

**方式一：整体加密**

`lock_mcp_deps.py --write` 产出的两个锁定文件在磁盘上可能是密文：

```text
$ head -c 15 requirements_mcp.txt
%TSD-Header-###%          # 长度还对 8192 字节对齐
```

python 与 pip 能透明读回来（所以安装过程一直是好的），但 `bash`、`od`、编辑器与
版本控制看到的是密文 —— 文件变得**不可读、不可 diff**。

**方式二：尾部补 NUL 到 4096 字节块对齐**

这个更阴，因为它**看起来完全正常**（`head` 看到的是明文），正文也没被改：

```text
requirements_mcp.txt   4096 字节
  ├─ 0    .. 1290   1291 字节  正文（头是 "# MCP server 完整依赖闭包"）
  └─ 1291 .. 4095   2805 字节  全是 \x00
```

坏处是填充段里**没有换行符**，于是按行解析的消费方会多出一条 2805 字节的假行：

| 消费方 | 表现 |
|---|---|
| `read_text().splitlines()` | 多出一个 2805 字节的"包名" → 自检误报 `1/28 个包位置不可识别` |
| `pip install -r` | `ERROR: Invalid requirement: '\x00\x00...' (from line 43)` |

> **触发条件仍然没钉死。** 我做过一轮隔离实验，结果如下：
>
> | 变量 | 处理 | 结果 |
> |---|---|---|
> | 内容 | 把锁定文件的正文原样写到 `zz_samecontent.txt` | 干净 |
> | 文件名 | 写 `requirements_probe.txt` / `requirements_mcp_x.txt` / `foo_mcp.txt` | 干净 |
> | 大小 | 1210 / 1507 / 3000 / 4095 / 8191 字节 | 干净 |
> | 目录 | `D:\AirNav-Agent`、`D:\tmp`、桌面 | 都干净 |
> | 写法 | 重定向、`write_bytes`、覆盖已有文件、新建文件 | 都可能干净 |
> | **真实锁定文件** | 用 `--stdout >` 重新产出 | **连续 2 次被补到 4096** |
>
> 也就是说：临时探针文件测了十几次一次没中，而这两个锁定文件反复中招。
> 结论不是"某个条件触发"，而是**这两个文件会被盯上，规律不明**。
> 既然如此就不要赌 —— 按"一定会有填充"来设计。

因此本项目里两处消费点都已加固：

| 位置 | 处理 |
|---|---|
| `check_mcp._locked_closure()` | 按字节读、清 NUL、再用 `^[A-Za-z0-9][A-Za-z0-9._-]*$` 校验包名 |
| `lock_mcp_deps.read_pins()` | 同上，服务于 `--install` |

加固是有效的 —— **把 2805 字节 NUL 人为补回去，第 0 层依然 PASS**，
并会打印一行 `锁定文件：锁定文件含 2805 字节 NUL 填充（DLP 块对齐），已忽略`。

产出方式仍推荐走标准输出加重定向（`bash` 写出的目前都是明文）：

```bash
# 产出
python -m agent.scripts.lock_mcp_deps --stdout all > requirements_mcp.txt
python -m agent.scripts.lock_mcp_deps --stdout env > requirements_mcp_env.txt

# 复查 1：不该有任何输出（防加密）
head -c 15 requirements_mcp.txt | grep -q "%TSD-Header" && echo "被加密了"

# 复查 2：不该有任何输出（防 NUL 填充）
grep -qP '\x00' requirements_mcp.txt && echo "有 NUL 填充，需清洗"
```

`--stdout` 模式下诊断报告会改走 stderr，不会污染重定向结果。
真要清洗填充，一行就够：

```bash
tr -d '\000' < requirements_mcp.txt > /tmp/clean && mv /tmp/clean requirements_mcp.txt
```

---

## 4. 验证

一条命令，四层检查，从依赖归属一路验到协议层真实调用：

```powershell
D:\AirNav-Agent\.venv-mcp\Scripts\python.exe -m agent.scripts.check_mcp --full
```

| 层 | 查什么 |
|---|---|
| 0 依赖归属 | 依赖有没有靠 user-site 兜底；屏蔽 user-site 后还能不能导入 |
| 1 静态 | 11 个工具是否齐全、描述非空、参数说明与 `ge`/枚举约束是否透传 |
| 2 握手 | 真的以子进程拉起 server，走一遍 stdio 的 `initialize` 握手 |
| 3 调用 | 通过标准 MCP 协议真实调用工具（列病例 / 知识检索 / 参数校验；`--full` 追加一次规划归因） |

最近一次结果：**四层全部通过**，`list_cases` 返回 12 个病例，
`search_knowledge` 命中 `KB-01#5`，越界参数被拦下并返回 `ok=false`，
`explain_route_choice` 返回四项占比 `{length 81.4, radius 17.0, curvature 1.2, branch 0.4}`
且 `verified_against_planner = True`。

数学校验另有两条独立入口（与 MCP 无关，纯几何/规划层面）：

```powershell
# 零分配几何与 scipy EDT 的等价性对拍（合成穷举 + 真实病例抽样）
D:\AirNav-Agent\.venv-mcp\Scripts\python.exe -m agent.scripts.verify_geometry

# 规划数值与 V1 GUI 保存结果的逐项对照
D:\AirNav-Agent\.venv-mcp\Scripts\python.exe -m agent.scripts.verify_planner
```

第 0 层的输出形如：

```text
[0] 依赖归属层：mcp 依赖是否依赖 user-site
         本解释器站点：D:\AirNav-Agent\.venv-mcp\Lib\site-packages
         user-site：在 sys.path 上（更危险）
  PASS  mcp 位于可接受站点（...\.venv-mcp\Lib\site-packages\mcp）
  PASS  闭包 27 个包均非 user-site 来源
  PASS  PYTHONNOUSERSITE=1 下仍可导入 server（不依赖 user-site）
```

> 注意第二行：venv 里 `site.ENABLE_USER_SITE` 仍然是 `True`，
> user-site 还在 `sys.path` 上、只是被 venv 站点遮住。
> 这正是配置里显式加 `PYTHONNOUSERSITE=1` 的原因——
> 让启动行为不再受全局用户目录的偶然内容影响。

---

## 5. 零分配几何：882MB 内存问题的根治

### 5.1 问题

MCP 的 `explain_route_choice` 曾稳定抛错：

```text
MemoryError: Unable to allocate 882. MiB for an array with shape
(3, 147, 512, 512) and data type float64
```

来源是 scipy 的 `ndi.distance_transform_edt`：它内部**必须**先分配
`(ndim, *体素数)` 的 int32 特征数组（441MB），再派生出 float64 距离场（882MB），
一次调用峰值 1.3GB。本机页面文件固定 8GB，而 MCP 服务进程里已经装了
病例数据与规划缓存，挤不下。

而整条规划链路里这些 EDT 调用**都只用到少数几个查询点的取值**
（中心线节点、基线骨架节点），为几个点付整卷 1.3GB 完全不划算。

### 5.2 做法

`agent/core/geometry.py` 用 KD 树把问题降维，两种情形各有依据：

| 情形 | 依据 |
|---|---|
| 到掩膜的最近距离 | 直接对**掩膜内体素**建树，与 EDT 定义完全一致 |
| 掩膜内的半径场 | 关键观察：对掩膜内的点，最近的背景体素**必定紧邻某个掩膜体素**（否则朝该点方向再走一步距离更小，矛盾）。所以只需对**掩膜外侧紧邻的一层壳**建树 |

气道掩膜下这层壳只有 26,851 个体素，占整卷 0.0697%。

### 5.3 效果

| 指标 | 原 EDT | 现在 |
|---|---|---|
| 单次内存 | 882 MB（另有 441MB 中间量） | **0.61 MB**（1,435×） |
| 查询 20 万点耗时 | 11.0 s | 2.9 s |
| 病例冷加载 | 15.96 s | **3.0 s** |
| 数值 | 基准 | 20 万随机点**逐位相同** |

### 5.4 等价性怎么保证

`agent/scripts/verify_geometry.py` 做两类对拍：

- **合成体积穷举**：24×32×40 的全部 30,720 个体素逐点比较，含非各向同性
  spacing，外加三个退化场景（全 True / 全 False 掩膜、空目标掩膜）
- **真实病例抽样**：LIDC_0089 上各 20 万随机体素，掩膜内外都覆盖

结果：**最大差异 0.0（逐位相同）**。合成场景下偶见 1e-15 的差异，
来自 `sqrt(0.7²) ≠ 0.7` 这类双精度末位舍入，比原子核半径还小 10 个数量级。

> 一个实测结论：**scipy 的 EDT 不做边界填充**。用「掩膜贴边但内部有背景体素」
> 的构造验证过——`EDT(0,0,0)` 给出的是到内部背景的真实距离，不是到数组外的一步。
> 因此几何实现里不需要任何边界修正项。
>
> 反过来，掩膜占满整卷（一个背景体素都没有）时 scipy 给的是**未初始化伪值**，
> 不是定义好的距离；本项目不会出现这种情况，实现里明确返回 `inf`。

需要覆盖的三处调用点：

| 位置 | 说明 |
|---|---|
| `case_loader` 半径场 | 只取骨架节点处的值 |
| `planner.compare_with_baseline` | 原为两次整卷 EDT，每次规划都会触发 |
| `navbridge.choose_entry_node_no_edt` | Agent 调用的 13 个 V1 函数中，唯一内部使用整卷 EDT 的一个；无入口掩膜时仍回退到 V1 原实现 |

最终把关仍是 `verify_planner`：**9 项指标 + 4 项字段与 V1 GUI 逐位吻合**。

---

## 6. 排查对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'mcp'` | 解释器指向了没装 mcp 的环境 | `command` 改成 `.venv-mcp\Scripts\python.exe` |
| 开发机正常，换机器 / 加 `-s` 就挂 | 依赖其实是 user-site 兜底 | 跑 `check_mcp` 第 0 层，按第 3.3 节重建 |
| 装依赖报 `WinError 5 拒绝访问` | miniforge 目录 ACL 只给普通用户 `(RX)` | 用项目内 `.venv-mcp`；或管理员提权后装进 conda 环境 |
| 中文乱码 | 少了编码声明 | 补 `PYTHONIOENCODING=utf-8` |
| 客户端连不上 / 协议报错 | stdout 被调试输出污染 | 日志一律写 **stderr**，stdout 只走协议帧 |
| 工具列表为空 | 工作目录不对 | `cwd` 必须是项目根 `D:\AirNav-Agent` |
| 提示 `cannot import name 'fastmcp'` | 抄了 MCP 1.x 的教程 | 本机是 2.x，用 `mcp.server.mcpserver.MCPServer` |
| `MemoryError: Unable to allocate 882 MiB` | 有调用点绕过了零分配几何、回到整卷 EDT | 跑 `verify_geometry` 确认；检查是否新引入了 `ndi.distance_transform_edt` |
| 锁定文件在编辑器里是乱码 | 被 DLP 加密了（python 直接写出的文件） | 用 `--stdout` 加 shell 重定向重新产出（见 3.5） |
| 自检报 `N/28 个包位置不可识别：['\x00...']` | 锁定文件被补了 NUL 到 4096 字节块对齐 | 已加固，会打印"已忽略 N 字节 NUL 填充"；要清文件见 3.5 |
| `pip install -r` 报 `Invalid requirement: '\x00...'` | 同上，pip 不认填充 | 改用 `python -m agent.scripts.lock_mcp_deps --install` |

---

## 7. 能力边界

本服务**只做几何计算**，不提供诊断结论、治疗建议或器械型号推荐。
服务端 `INSTRUCTIONS` 里也写了同样的边界声明，客户端接入后会自动读到。

可达性分级（`≤15 相邻` / `≤30 可达` / `≤50 勉强` / `>50 不可达`）是**几何口径**，
不是临床阈值。任何对外表述都应保留这一区分。
