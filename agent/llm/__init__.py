"""LLM 客户端层：OpenAI 兼容客户端与回放后端。"""
from .client import (
    PRESETS,
    ChatReply,
    LLMError,
    OpenAICompatClient,
    ScriptedClient,
    ToolCallRequest,
)

__all__ = [
    "ChatReply",
    "LLMError",
    "OpenAICompatClient",
    "PRESETS",
    "ScriptedClient",
    "ToolCallRequest",
]
