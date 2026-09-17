"""工具注册表。

设计要点
--------
1. **单一真相源**：每个工具用一个 pydantic 模型声明参数，
   OpenAI function-calling 的 JSON Schema 与运行时参数校验都从它派生，
   不会出现「文档写了但代码没校验」的漂移。
2. **返回值一律是 JSON 可序列化的 dict**，并且必须简短。
   大数组（路径折线、网格）不进返回值，而是存进 ArtifactStore 后返回一个 id，
   避免把上下文窗口塞爆 —— 这是 Agent 工程里最常见的翻车点。
3. **失败不抛异常**：工具内部异常被捕获并转成 {"ok": false, "error": ...}，
   交给 LLM 决定重试还是换策略。这是 ReAct 循环能自愈的前提。
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


@dataclass
class ToolSpec:
    """一个可被 LLM 调用的工具。"""

    name: str
    description: str
    model: type[BaseModel]
    handler: Callable[..., dict[str, Any]]

    def json_schema(self) -> dict[str, Any]:
        """转成 OpenAI function-calling 的参数 schema。"""
        schema = self.model.model_json_schema()
        schema.pop("title", None)
        # 去掉 pydantic 的 $defs 与字段级 title，压缩提示词体积
        schema.pop("$defs", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def validate(self, raw: dict[str, Any]) -> BaseModel:
        return self.model(**raw)


@dataclass
class ToolCallRecord:
    """一次工具调用的留痕，用于 trace 与评测。"""

    step: int
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any]
    error: str | None = None
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "result_preview": _preview(self.result),
        }


def _preview(payload: Any, limit: int = 420) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(payload)
    return text if len(text) <= limit else text[:limit] + " …(截断)"


class ArtifactStore:
    """存放不适合塞进上下文的中间数据（路径折线、网格、截图路径等）。"""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}
        self._counter = 0

    def put(self, kind: str, payload: Any, meta: dict[str, Any] | None = None) -> str:
        self._counter += 1
        key = f"{kind}-{self._counter:03d}"
        self._items[key] = {"kind": kind, "meta": meta or {}, "payload": payload}
        return key

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(f"没有这个 artifact：{key}")
        return self._items[key]["payload"]

    def meta(self, key: str) -> dict[str, Any]:
        return self._items[key]["meta"] if key in self._items else {}

    def keys(self) -> list[str]:
        return list(self._items)

    def summary(self) -> list[dict[str, Any]]:
        return [
            {"id": key, "kind": item["kind"], **item["meta"]}
            for key, item in self._items.items()
        ]


class ToolRegistry:
    """工具集合，负责执行、校验与留痕。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.calls: list[ToolCallRecord] = []

    # ---- 注册 ----

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具名重复：{spec.name}")
        self._tools[spec.name] = spec

    def tool(
        self,
        name: str,
        description: str,
        model: type[BaseModel],
    ) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
        def decorator(fn: Callable[..., dict[str, Any]]):
            self.register(ToolSpec(name, description, model, fn))
            return fn

        return decorator

    # ---- 查询 ----

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def openai_tools(self) -> list[dict[str, Any]]:
        return [spec.to_openai_tool() for spec in self._tools.values()]

    def describe(self) -> str:
        """给提示词用的紧凑工具清单。"""
        lines: list[str] = []
        for spec in self._tools.values():
            params = spec.json_schema().get("properties", {})
            required = set(spec.json_schema().get("required", []))
            args = ", ".join(
                f"{key}{'*' if key in required else ''}" for key in params
            )
            lines.append(f"- {spec.name}({args}): {spec.description}")
        return "\n".join(lines)

    # ---- 执行 ----

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | str | None,
        context: Any = None,
        step: int = 0,
    ) -> dict[str, Any]:
        if name not in self._tools:
            record = ToolCallRecord(
                step=step,
                name=name,
                arguments={},
                ok=False,
                result={},
                error=f"未注册的工具：{name}，可用工具：{self.names()}",
            )
            self.calls.append(record)
            return {"ok": False, "error": record.error}

        spec = self._tools[name]

        # 兼容 LLM 把参数写成 JSON 字符串的情况（常见失败模式）
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as error:
                record = ToolCallRecord(
                    step=step,
                    name=name,
                    arguments={},
                    ok=False,
                    result={},
                    error=f"参数不是合法 JSON：{error}",
                )
                self.calls.append(record)
                return {"ok": False, "error": record.error}
        arguments = arguments or {}

        started = time.time()
        try:
            validated = spec.validate(arguments)
        except ValidationError as error:
            # 把校验错误压缩成 LLM 能看懂的一行，便于它自己改参数重试
            issues = [
                f"{'.'.join(str(p) for p in item['loc'])}: {item['msg']}"
                for item in error.errors()
            ]
            message = "参数校验失败 -> " + "; ".join(issues)
            record = ToolCallRecord(
                step=step,
                name=name,
                arguments=arguments,
                ok=False,
                result={},
                error=message,
                elapsed_ms=int((time.time() - started) * 1000),
            )
            self.calls.append(record)
            return {"ok": False, "error": message}

        payload = validated.model_dump()
        try:
            if _wants_context(spec.handler):
                result = spec.handler(context, **payload)
            else:
                result = spec.handler(**payload)
            if not isinstance(result, dict):
                result = {"ok": True, "value": result}
            result.setdefault("ok", True)
            record = ToolCallRecord(
                step=step,
                name=name,
                arguments=payload,
                ok=True,
                result=result,
                elapsed_ms=int((time.time() - started) * 1000),
            )
            self.calls.append(record)
            return result
        except Exception as error:  # noqa: BLE001 - 必须兜住，交给 LLM 决策
            message = f"{type(error).__name__}: {error}"
            record = ToolCallRecord(
                step=step,
                name=name,
                arguments=payload,
                ok=False,
                result={},
                error=message,
                elapsed_ms=int((time.time() - started) * 1000),
            )
            self.calls.append(record)
            return {"ok": False, "error": message}

    # ---- trace ----

    def reset_trace(self) -> None:
        self.calls.clear()

    def trace(self) -> list[dict[str, Any]]:
        return [call.to_dict() for call in self.calls]

    def stats(self) -> dict[str, Any]:
        total = len(self.calls)
        failed = sum(1 for c in self.calls if not c.ok)
        return {
            "tool_calls": total,
            "failed_calls": failed,
            "success_rate": round((total - failed) / total, 4) if total else None,
            "total_ms": sum(c.elapsed_ms for c in self.calls),
            "by_tool": _count_by(self.calls),
        }


def _count_by(calls: list[ToolCallRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for call in calls:
        out[call.name] = out.get(call.name, 0) + 1
    return out


def _wants_context(handler: Callable[..., Any]) -> bool:
    """判断工具实现是否需要一个 context 参数（用于访问会话状态）。"""
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    params = list(signature.parameters.values())
    return bool(params) and params[0].name in {"ctx", "session", "context"}


__all__ = [
    "ArtifactStore",
    "ToolCallRecord",
    "ToolRegistry",
    "ToolSpec",
]
