"""LLM 客户端。

只依赖 requests（环境已有），直接实现 OpenAI 的 /chat/completions 协议，
不引 SDK —— 这样既能接 Ollama 本地模型，也能接 DeepSeek / 通义 / OpenAI，
同时把 function-calling 的报文格式摊开，便于排查失败模式。

包含一个 ScriptedClient：按预设脚本回放，用于无网络环境下
跑通整条链路、以及评测集的可重复回放。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import requests

DEFAULT_TIMEOUT = 180


@dataclass
class ToolCallRequest:
    """模型请求调用一个工具。"""

    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，故意不解析，方便观察模型的真实输出

    def parsed(self) -> dict[str, Any]:
        text = (self.arguments or "").strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass
class ChatReply:
    """一次模型回复。"""

    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    """LLM 调用失败。"""


# ------------------------------------------------------------------ 预设


PRESETS: dict[str, dict[str, Any]] = {
    "gpu41": {
        # 内网 GPU 服务器（8×H100）。**地址一律走环境变量**，默认值只是
        # 本机开发时的兜底，方便不设变量也能跑通：
        #   set AIRNAV_GPU41_URL=http://<你的GPU主机>:11434/v1
        # 换机器 / 换端口改环境变量即可，不必改代码。
        "base_url": "http://192.168.8.41:11434/v1",  # airnav-allow-real-ip 本机兜底
        "url_env": "AIRNAV_GPU41_URL",
        "model": "qwen2.5:14b",
        "model_env": "AIRNAV_GPU41_MODEL",
        "api_key": "ollama",
        "label": "gpu41 服务器 Ollama（H100，内网）",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        # ⚠️ 这个预设应当指「本机自己那份 ollama」。核实过的本机事实：
        #   - 本机 `%LOCALAPPDATA%\Programs\Ollama` 里**只有 CPU 后端**
        #     （一堆 ggml-cpu-*.dll，没有 ggml-cuda.dll / ggml-vulkan.dll），
        #     所以本机推理必然是 CPU，并且以前撞过提交内存上限（HTTP 500）
        #   - 本机 `E:\ollama\models` 里**只装了 qwen2.5:7b**（4.4 GB），没有 14b
        #   - 本机 ollama 服务**平时根本没在跑**（无 ollama.exe 进程，
        #     server.log 停在 2026-09-16）
        # 那为什么曾经「感觉能用」？因为 **Cursor / VS Code 的 Remote-SSH
        # 会把 localhost:11434 转发到 gpu41** —— 于是 `ollama list` 返回的是
        # gpu41 上的 qwen2.5:14b，`ollama ps` 还显示 `100% GPU`，看起来
        # 就像本机真的跑起来了。**这是转发造成的假象，不是本机能力。**
        # 判据：本机只有 7b，看到 14b 就说明走的是转发。
        # 所以默认模型按「本机实际安装的」写 7b；真要在服务器上跑请用
        # `--backend gpu41`（那才是显式、不含糊的写法）。
        # 诊断入口：`python -m agent.cli doctor --backend ollama`
        # 会打印端口占用者，直接告诉你连的是本机还是被转发了。
        "model": "qwen2.5:7b",
        "api_key": "ollama",
        "label": "本地 Ollama（免费、无需联网）",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "label": "DeepSeek 官方 API",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "label": "阿里云百炼 / 通义千问",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "label": "OpenAI",
    },
}


# ------------------------------------------------------------------ 客户端


class OpenAICompatClient:
    """最小 OpenAI 兼容客户端。

    注意：本机常配了 http_proxy / https_proxy 环境变量（开发机上的常见情况），
    requests 默认会把 http://localhost 的请求也丢给代理，导致连本地 Ollama 时报 502。
    因此对本地地址显式关闭 trust_env，不走任何代理。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = 0.0,
        top_p: float | None = None,
        extra_body: dict[str, Any] | None = None,
        label: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "not-needed"
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.extra_body = extra_body or {}
        self.label = label or f"{model} @ {base_url}"
        self.is_local = _is_local_url(self.base_url)

        self._session = requests.Session()
        if self.is_local:
            # 本地服务必须绕过代理，否则会打到代理端口上得到 502
            self._session.trust_env = False

    @classmethod
    def from_preset(
        cls,
        name: str,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> "OpenAICompatClient":
        if name not in PRESETS:
            raise KeyError(f"未知后端 {name}，可选：{list(PRESETS)}")
        preset = dict(PRESETS[name])

        # 地址/模型名允许用环境变量给默认值（便于换机器、换端口而不改代码）。
        # 显式传入的参数优先级仍然最高。
        default_url = preset["base_url"]
        if preset.get("url_env"):
            default_url = os.environ.get(preset["url_env"]) or default_url
        default_model = preset["model"]
        if preset.get("model_env"):
            default_model = os.environ.get(preset["model_env"]) or default_model

        key = api_key or (
            os.environ.get(preset["env_key"]) if preset.get("env_key") else None
        )
        if preset.get("env_key") and not key:
            raise LLMError(
                f"后端 {name} 需要 API Key，请设置环境变量 {preset['env_key']}"
                f"或用 --api-key 传入"
            )
        # label 一律带上**实际生效**的模型名。
        # 缓存踩过的坑：原来把模型名写死在预设的 label 里，于是
        #   --backend ollama --model qwen2.5:14b
        # 明明用的是 14b，打印出来却是「本地 Ollama（qwen2.5:7b…）」——
        # 排错时会被这个标签带偏，以为模型没生效。
        effective_model = model or default_model
        template = preset.get("label", name)
        label = (
            template.format(model=effective_model)
            if "{model}" in template
            else f"{template} · {effective_model}"
        )

        return cls(
            base_url=base_url or default_url,
            model=effective_model,
            api_key=key or preset.get("api_key"),
            label=label,
            **kwargs,
        )

    # ---- 请求 ----

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        temperature: float | None = None,
    ) -> ChatReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if tools:
            payload["tools"] = list(tools)
            if tool_choice:
                payload["tool_choice"] = tool_choice
        payload.update(self.extra_body)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            response = self._session.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as error:
            hint = ""
            if self.is_local:
                hint = "（本地服务未启动？请先运行 ollama serve 或启动 Ollama 应用）"
            raise LLMError(f"无法连接 {self.label}{hint}：{error}") from error

        if response.status_code != 200:
            body = response.text[:400]
            raise LLMError(
                f"{self.label} 返回 HTTP {response.status_code}：{body}"
            )

        try:
            data = response.json()
        except ValueError as error:
            raise LLMError(f"{self.label} 返回的不是 JSON：{response.text[:200]}") from error

        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> ChatReply:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"响应里没有 choices：{json.dumps(data)[:300]}")
        message = choices[0].get("message") or {}

        calls: list[ToolCallRequest] = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            calls.append(
                ToolCallRequest(
                    id=raw.get("id") or f"call_{index}",
                    name=name,
                    arguments=function.get("arguments") or "{}",
                )
            )

        return ChatReply(
            content=(message.get("content") or "").strip(),
            tool_calls=calls,
            usage=data.get("usage") or {},
            finish_reason=choices[0].get("finish_reason"),
            raw=data,
        )

    def health(self) -> dict[str, Any]:
        """探活：拉一次模型列表，用于判断本地服务是否已启动。"""
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=8,
            )
            if response.status_code != 200:
                return {"ok": False, "detail": f"HTTP {response.status_code}"}
            models = [m.get("id") for m in (response.json().get("data") or [])]
            return {"ok": True, "models": models, "has_target": self.model in models}
        except requests.RequestException as error:
            return {"ok": False, "detail": str(error)}


# ------------------------------------------------------------------ 回放后端


class ScriptedClient:
    """按脚本回放回复的后端。

    用途：
      - 无网络 / 未启动 Ollama 时，端到端验证工具链路
      - 评测集里复现某条失败轨迹，做确定性对比

    脚本元素可以是 ChatReply，也可以是可调用对象
    (messages) -> ChatReply，便于写依赖上下文的断言。
    """

    def __init__(
        self,
        replies: Iterable[ChatReply | Any],
        label: str = "scripted",
    ) -> None:
        self._replies = list(replies)
        self._index = 0
        self.label = label
        self.calls: list[list[dict[str, Any]]] = []

    @staticmethod
    def tool(name: str, **arguments: Any) -> ChatReply:
        return ChatReply(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id=f"call_{name}_{abs(hash((name, tuple(sorted(arguments.items(), key=str))))) % 100000}",
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            ],
        )

    @staticmethod
    def say(text: str) -> ChatReply:
        return ChatReply(content=text)

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        temperature: float | None = None,
    ) -> ChatReply:
        self.calls.append(list(messages))
        if self._index >= len(self._replies):
            return ChatReply(content="[脚本已用尽] 没有后续回复")
        item = self._replies[self._index]
        self._index += 1
        if callable(item):
            return item(messages)
        return item

    def health(self) -> dict[str, Any]:
        return {"ok": True, "models": [self.label], "has_target": True}


def _is_local_url(url: str) -> bool:
    """判断是不是本机地址 —— 本机地址必须绕过代理。"""
    lowered = url.lower()
    return any(
        token in lowered
        for token in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "://192.168.", "://10.")
    )


__all__ = [
    "ChatReply",
    "LLMError",
    "OpenAICompatClient",
    "PRESETS",
    "ScriptedClient",
    "ToolCallRequest",
]
