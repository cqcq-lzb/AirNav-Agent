"""Agent 工具层：注册表、会话状态、影像规划工具、知识检索与路径归因。"""
from .imaging import register_tools
from .knowledge import register_knowledge_tools
from .registry import ArtifactStore, ToolCallRecord, ToolRegistry, ToolSpec
from .session import NavSession, normalize_profile

__all__ = [
    "ArtifactStore",
    "NavSession",
    "ToolCallRecord",
    "ToolRegistry",
    "ToolSpec",
    "build_registry",
    "normalize_profile",
    "register_knowledge_tools",
    "register_tools",
]


def build_registry() -> ToolRegistry:
    """组装完整的工具集。"""
    registry = ToolRegistry()
    register_tools(registry)
    register_knowledge_tools(registry)
    return registry
