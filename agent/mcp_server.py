"""MCP server：把 Agent 的工具集暴露成标准 Model Context Protocol 服务。

这样 Claude Desktop、Cursor、以及其他 MCP 客户端可以**直接驱动**这套
支气管镜路径规划能力，不需要跑本项目的 LLM 客户端。

本文件的两个技术要点
--------------------
1. **本机装的是 MCP 2.x**，`FastMCP` 已改名为 `MCPServer`
   （`from mcp.server.mcpserver import MCPServer`）。网上大量教程还是 1.x 的
   `from mcp.server.fastmcp import FastMCP`，在这里会直接报 ModuleNotFoundError。

2. **工具参数模型的摊平**。项目里每个工具的参数是 pydantic 模型
   （单一真相源），而 MCPServer 是通过**函数签名**来推导 JSON Schema 的。
   所以要把模型的字段转成 `inspect.Parameter`，并且注解必须写成
   `Annotated[类型, FieldInfo(description=...)]` 才能把字段级说明透传出去；
   只写裸类型的话，客户端看到的 schema 会丢掉全部参数说明。
   约束（ge/le）也要从原字段的 metadata 里一并带过去。

暴露范围：11 个工具。传输方式默认 stdio。
"""
from __future__ import annotations

import inspect
import json
from typing import Annotated, Any

from pydantic.fields import FieldInfo

from .tools import NavSession, ToolRegistry, build_registry
from .tools.registry import ToolSpec

SERVER_NAME = "airnav-agent"
SERVER_VERSION = "0.1.0"

INSTRUCTIONS = """\
经支气管肺结节导航系统的路径规划服务。

能力范围：读取已分割好的胸部 CT 病例，重建气道三维结构，规划经支气管路径，
给出几何指标（长度、最窄直径、最窄余量、最大转角、可达性分级），
生成自包含的三维交互视图，并检索项目知识库解释设计依据。

使用建议：
- 涉及具体候选之前，先调 inspect_case 或 list_nodule_candidates 确认编号口径。
  本系统的「客户端编号」与服务器清单的「服务端编号」并不一致，
  每个候选的两个编号都会同时返回。
- plan_route 失败时读返回体的 failure 字段：`failure.reason` 是为什么失败，
  `failure.next_step` 就是该怎么做。器械过粗时上限已在
  `failure.max_device_diameter_mm` 里，不必自己再调 scan_device_fit 去搜；
  编号越界时 `failure.valid_candidate_id_range` 给出可用范围，
  **不要自己挑别的编号去规划**，应把越界事实交回医生确认。
  （E14：模型曾把用户要的 99 号静默换成 3 号并交付完整路径 —— 本系统最危险的错配。）
- 解释「为什么选这条路径」请用 explain_route_choice，它会给出代价的四项占比。
- 解释系统设计意图请用 search_knowledge，并在回答里带上 [KB-xx#n] 引用号。

边界：本服务只做几何计算，不提供诊断结论、治疗建议或器械型号推荐。\
"""


# ------------------------------------------------------------------ 摊平


def _field_annotation(field: Any) -> Any:
    """把 pydantic 字段转成带说明与约束的注解。"""
    metadata: list[Any] = []
    description = field.description or ""
    if description:
        metadata.append(FieldInfo(description=description))
    # ge / le / min_length 这类约束挂在 field.metadata 里，一并带上，
    # 这样客户端能在 schema 层就看到取值范围，而不是等到运行时报错
    metadata.extend(field.metadata or [])

    if not metadata:
        return field.annotation
    return Annotated[tuple([field.annotation, *metadata])]


def make_wrapper(spec: ToolSpec, registry: ToolRegistry, session: NavSession):
    """按工具的参数模型生成一个签名正确的函数，供 MCPServer 推导 schema。"""

    def wrapper(**kwargs: Any) -> str:
        result = registry.execute(spec.name, kwargs, context=session)
        return json.dumps(result, ensure_ascii=False, default=str)

    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for name, field in spec.model.model_fields.items():
        annotation = _field_annotation(field)
        default = (
            inspect.Parameter.empty if field.is_required() else field.default
        )
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
        annotations[name] = annotation

    wrapper.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    wrapper.__annotations__ = {**annotations, "return": str}
    wrapper.__name__ = spec.name
    wrapper.__doc__ = spec.description
    return wrapper


# ------------------------------------------------------------------ 组装


def build_server(
    registry: ToolRegistry | None = None,
    session: NavSession | None = None,
):
    """组装 MCP server。

    注册表与会话是**进程级共享**的：MCP 服务在一次连接里会被连续调用多次，
    病例加载（十几秒）和规划结果必须能跨调用复用，否则体验会非常差。
    """
    from mcp.server.mcpserver import MCPServer

    registry = registry or build_registry()
    session = session or NavSession()

    server = MCPServer(
        name=SERVER_NAME,
        title="AirNav — 经支气管导航路径规划",
        version=SERVER_VERSION,
        instructions=INSTRUCTIONS,
    )

    for spec in registry.specs():
        server.add_tool(
            make_wrapper(spec, registry, session),
            name=spec.name,
            description=spec.description,
        )

    return server


def main() -> None:
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
