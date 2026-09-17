"""RAG 层：领域知识语料 + 零依赖 BM25 检索。

对外只暴露两件事：
  - `get_retriever()` 拿到全局检索器
  - `RetrievalHit` 是带回引用号的检索结果

语料放在 `knowledge/*.md`，每篇用 front matter 声明 id / title / tags / source，
正文按二级标题切成可引用的小节（kb_id 形如 `KB-01#3`）。
"""
from __future__ import annotations

from .retriever import (
    BM25Index,
    KnowledgeChunk,
    RetrievalHit,
    Retriever,
    get_retriever,
    load_knowledge,
)
from .tokenizer import text_coverage, tokenize

__all__ = [
    "BM25Index",
    "KnowledgeChunk",
    "RetrievalHit",
    "Retriever",
    "get_retriever",
    "load_knowledge",
    "text_coverage",
    "tokenize",
]
