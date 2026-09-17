"""零依赖 BM25 检索器。

设计取舍
--------
**为什么是 BM25 而不是向量检索**：本机没有可用的 embedding 模型
（Ollama 里只有生成模型，pip 又装不了句子向量库）。与其写一个跑不起来的
向量检索，不如把词法检索做扎实——加了领域同义词扩展、字段加权、
以及「检索不到就明确说检索不到」的阈值。

**字段加权（BM25F 简化版）**：标题和标签里的词比正文里的更能代表主题，
所以词的加权词频是 `3×标题 + 2×标签 + 1×正文`。
文档长度也用这个加权长度，避免短文档被系统性高估。

**阈值为什么重要**：如果无论问什么都返回 top-k 段落，模型就会拿不相关的
段落硬编一个理由。低于阈值的查询直接返回空列表，让上游知道「知识库里
没有依据」，从而选择说「不确定」而不是编造。这是反幻觉的第一道闸门。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .tokenizer import tokenize

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# BM25 超参（文献常用取值）
K1 = 1.5
B = 0.75

# 字段权重
W_TITLE = 3.0
W_TAGS = 2.0
W_BODY = 1.0

# 低于这个分就不返回 —— 宁可空手而归，也不要给模型递不相关的材料
DEFAULT_MIN_SCORE = 1.0

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


@dataclass
class KnowledgeChunk:
    """一个可被引用的知识片段（对应语料里的一个二级标题小节）。"""

    kb_id: str
    doc_id: str
    title: str
    section: str
    tags: tuple[str, ...]
    source: str
    text: str
    cite: str = ""

    def to_dict(self) -> dict:
        return {
            "kb_id": self.kb_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "tags": list(self.tags),
            "source": self.source,
            "text": self.text,
        }


@dataclass
class RetrievalHit:
    chunk: KnowledgeChunk
    score: float

    def to_dict(self) -> dict:
        return {
            "kb_id": self.chunk.kb_id,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "tags": list(self.chunk.tags),
            "source": self.chunk.source,
            "score": round(self.score, 4),
            "text": self.chunk.text,
            "cite": f"[{self.chunk.kb_id}]",
        }


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """解析极简 front matter（`key: value` 与 `[a, b]` 列表），返回 (meta, body)。"""
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, raw[match.end() :]


def _parse_tags(value: str) -> tuple[str, ...]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def split_sections(body: str) -> list[tuple[str, str]]:
    """按二级标题切小节；标题前的内容归入「概述」。"""
    matches = list(_SECTION_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        return [("概述", stripped)] if stripped else []

    sections: list[tuple[str, str]] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(("概述", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[match.end() : end].strip()
        if text:
            sections.append((match.group(1).strip(), text))
    return sections


def load_knowledge(directory: Path | str | None = None) -> list[KnowledgeChunk]:
    """读取 knowledge/ 下全部 .md，切成带引用号的片段。"""
    root = Path(directory) if directory else KNOWLEDGE_DIR
    chunks: list[KnowledgeChunk] = []

    for path in sorted(root.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        doc_id = meta.get("id") or path.stem
        title = meta.get("title") or path.stem
        tags = _parse_tags(meta.get("tags", ""))
        source = meta.get("source", "")

        for order, (section, text) in enumerate(split_sections(body), start=1):
            kb_id = f"{doc_id}#{order}"
            chunks.append(
                KnowledgeChunk(
                    kb_id=kb_id,
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    tags=tags,
                    source=source,
                    text=text,
                    cite=f"[{kb_id}]",
                )
            )
    return chunks


class BM25Index:
    """带字段加权的 BM25。"""

    def __init__(self, chunks: Sequence[KnowledgeChunk]) -> None:
        self.chunks = list(chunks)
        self.tf: list[Counter[str]] = []
        self.length: list[float] = []
        self.df: Counter[str] = Counter()

        for chunk in self.chunks:
            title_tokens = tokenize(chunk.title)
            tag_tokens = tokenize(" ".join(chunk.tags))
            section_tokens = tokenize(chunk.section)
            body_tokens = tokenize(chunk.text)

            weighted: Counter[str] = Counter()
            for token, count in Counter(body_tokens).items():
                weighted[token] += W_BODY * count
            for token, count in Counter(section_tokens).items():
                weighted[token] += W_TITLE * count
            for token, count in Counter(title_tokens).items():
                weighted[token] += W_TITLE * count
            for token, count in Counter(tag_tokens).items():
                weighted[token] += W_TAGS * count

            self.tf.append(weighted)
            self.length.append(float(sum(weighted.values())) or 1.0)
            for token in weighted:
                self.df[token] += 1

        total = max(len(self.chunks), 1)
        self.avgdl = sum(self.length) / total
        # 用 log(1 + ...) 形式，保证 idf 恒为正 —— 词出现在全部文档里时也不会变负
        self.idf: dict[str, float] = {
            token: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for token, freq in self.df.items()
        }

    def score(self, query_tokens: Iterable[str]) -> list[float]:
        scores = [0.0] * len(self.chunks)
        for token in set(query_tokens):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, tf in enumerate(self.tf):
                freq = tf.get(token)
                if not freq:
                    continue
                norm = K1 * (1.0 - B + B * self.length[index] / self.avgdl)
                scores[index] += idf * (freq * (K1 + 1.0)) / (freq + norm)
        return scores


class Retriever:
    """知识检索器。可选接入 embedding 后端做混合检索（RRF 融合）。"""

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk] | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self.chunks = list(chunks) if chunks is not None else load_knowledge()
        self.index = BM25Index(self.chunks)
        self.min_score = min_score
        self.embed_fn = embed_fn
        self._doc_vectors: list[list[float]] | None = None

    # ---------------------------------------------------------------- 检索

    def search(
        self,
        query: str,
        top_k: int = 4,
        tags: Sequence[str] | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalHit]:
        threshold = self.min_score if min_score is None else min_score
        allowed = self._filter(tags)
        if not allowed:
            return []

        lexical = self.index.score(tokenize(query))
        ranked: list[RetrievalHit] = []

        if self.embed_fn is None:
            for index in allowed:
                if lexical[index] >= threshold:
                    ranked.append(RetrievalHit(self.chunks[index], lexical[index]))
        else:
            ranked = self._hybrid(query, allowed, lexical, threshold, top_k)

        ranked.sort(key=lambda hit: (-hit.score, hit.chunk.kb_id))
        return ranked[: max(top_k, 1)]

    # ---------------------------------------------------------------- 内部

    def _filter(self, tags: Sequence[str] | None) -> list[int]:
        if not tags:
            return list(range(len(self.chunks)))
        wanted = {tag.strip().lower() for tag in tags if tag.strip()}
        if not wanted:
            return list(range(len(self.chunks)))
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if wanted & set(chunk.tags)
        ]

    def _hybrid(
        self,
        query: str,
        allowed: list[int],
        lexical: list[float],
        threshold: float,
        top_k: int,
    ) -> list[RetrievalHit]:
        """词法 + 向量，用 RRF 融合。向量召回不足时退回纯词法。"""
        try:
            if self._doc_vectors is None:
                self._doc_vectors = self.embed_fn(  # type: ignore[misc]
                    [f"{c.title} {c.section} {c.text}" for c in self.chunks]
                )
            query_vector = self.embed_fn([query])[0]  # type: ignore[misc]
        except Exception:
            # embedding 后端不可用 -> 静默退回词法，绝不让检索整体失败
            return [
                RetrievalHit(self.chunks[i], lexical[i])
                for i in allowed
                if lexical[i] >= threshold
            ]

        dense = {
            index: _cosine(query_vector, self._doc_vectors[index]) for index in allowed
        }
        lexical_rank = sorted(allowed, key=lambda i: -lexical[i])
        dense_rank = sorted(allowed, key=lambda i: -dense[i])
        fused: dict[int, float] = {}
        for rank, index in enumerate(lexical_rank):
            fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank + 1)
        for rank, index in enumerate(dense_rank):
            fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank + 1)

        return [
            RetrievalHit(
                self.chunks[index],
                # 融合分很小，等比放大到与 BM25 同量级，便于统一阈值
                fused[index] * 100.0,
            )
            for index in fused
            if fused[index] * 100.0 >= threshold or lexical[index] >= threshold
        ]

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "documents": len({chunk.doc_id for chunk in self.chunks}),
            "vocabulary": len(self.index.idf),
            "avg_weighted_length": round(self.index.avgdl, 1),
            "hybrid": self.embed_fn is not None,
        }


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


_CACHED: Retriever | None = None


def get_retriever() -> Retriever:
    """进程内单例 —— 语料很小，索引一次即可。"""
    global _CACHED
    if _CACHED is None:
        embed_fn = None
        try:
            from .embedding import make_embedder

            embed_fn = make_embedder()
        except Exception:
            embed_fn = None
        _CACHED = Retriever(embed_fn=embed_fn)
    return _CACHED


__all__ = [
    "BM25Index",
    "KnowledgeChunk",
    "RetrievalHit",
    "Retriever",
    "get_retriever",
    "load_knowledge",
    "parse_front_matter",
    "split_sections",
]
