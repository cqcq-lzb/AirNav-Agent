"""可选的向量检索后端。

本机现状：Ollama 里只有生成模型（qwen2.5:7b），**没有 embedding 模型**，
所以默认情况下 `make_embedder()` 返回 None，检索走纯 BM25。

这个模块存在的意义是把这个缺口做成**可插拔的**，而不是写死成词法检索：

- 设了环境变量 `AIRNAV_EMBED_MODEL` 且该模型在 Ollama 里存在时，
  检索自动升级为「BM25 + 向量」的 RRF 混合检索。
- 模型不存在或服务不可达时返回 None，检索静默退回纯 BM25，功能不受影响。

装一个 embedding 模型（例如 `ollama pull nomic-embed-text`）就能升级，
不需要改任何检索代码 —— 这是刻意留的升级路径。
"""
from __future__ import annotations

import os
from typing import Callable

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
ENV_MODEL = "AIRNAV_EMBED_MODEL"
ENV_BASE_URL = "AIRNAV_OLLAMA_URL"


def _session(base_url: str) -> requests.Session:
    session = requests.Session()
    if any(host in base_url for host in ("localhost", "127.0.0.1", "0.0.0.0")):
        # 与 LLM 客户端一致：本地服务绕过代理，否则会 502
        session.trust_env = False
    return session


def available_models(base_url: str | None = None) -> list[str]:
    url = (base_url or os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL)).rstrip("/")
    try:
        response = _session(url).get(f"{url}/api/tags", timeout=4)
        response.raise_for_status()
        return [item["name"] for item in response.json().get("models", [])]
    except Exception:
        return []


def make_embedder(
    model: str | None = None,
    base_url: str | None = None,
) -> Callable[[list[str]], list[list[float]]] | None:
    """有可用的 embedding 模型就返回编码函数，否则返回 None。"""
    name = model or os.environ.get(ENV_MODEL)
    if not name:
        return None

    url = (base_url or os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL)).rstrip("/")
    installed = available_models(url)
    # Ollama 的模型名可能带 :latest 后缀，做一次宽松匹配
    stem = name.split(":")[0]
    if not any(item.split(":")[0] == stem for item in installed):
        return None

    session = _session(url)

    def embed(texts: list[str]) -> list[list[float]]:
        response = session.post(
            f"{url}/api/embed",
            json={"model": name, "input": list(texts)},
            timeout=60,
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings")
        if vectors:
            return vectors
        # 老版本 Ollama 只有 /api/embeddings，逐条退回
        out: list[list[float]] = []
        for text in texts:
            single = session.post(
                f"{url}/api/embeddings",
                json={"model": name, "prompt": text},
                timeout=60,
            )
            single.raise_for_status()
            out.append(single.json()["embedding"])
        return out

    return embed


def describe_backend() -> dict:
    """给自检脚本用：当前检索跑在什么后端上。"""
    name = os.environ.get(ENV_MODEL)
    models = available_models()
    return {
        "env_model": name,
        "ollama_reachable": bool(models),
        "installed_models": models,
        "embedding_enabled": bool(name) and make_embedder(name) is not None,
        "hint": (
            "未启用向量检索（纯 BM25）。装一个 embedding 模型即可升级："
            "ollama pull nomic-embed-text，再设 AIRNAV_EMBED_MODEL=nomic-embed-text"
        ),
    }


__all__ = ["available_models", "describe_backend", "make_embedder"]
