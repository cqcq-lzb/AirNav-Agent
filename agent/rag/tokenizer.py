"""检索用的分词器。

为什么不用现成的分词器
----------------------
本机没有 jieba，也没有任何 embedding 模型（Ollama 里只有 qwen2.5:14b 一个生成模型），
且 pip 直连 PyPI 不通、装不了新依赖。所以检索层必须零依赖自建。

中文 BM25 的关键是切词粒度。逐字切会丢掉词序信息，而中文的词边界没有空格，
按标点切又太粗。实践上最稳的零依赖方案是 **二元组（bigram）**：
「为什么选这条」→ 为什/什么/么选/选这/这条。这组 bigram 里天然包含
「选这」这类有区分度的片段，而常用虚词组成的 bigram 会被 IDF 自动压下去。

在此之上加一层**领域同义词扩展**：用户说「镜子多粗能过」和
「器械外径能多大」应当命中同一篇文档。做法是维护若干同义词组，
分词时若命中了组内任一说法，就补一个规范词（canonical token）到词袋里。
这是没有向量检索时的廉价替代方案，也是查全率的主要来源。
"""
from __future__ import annotations

import re
import unicodedata

# ASCII 词（含数字与下划线，保留 balanced / wide_airway / B6 这类记号）
_ASCII_RE = re.compile(r"[a-z0-9_]+")
# 连续的中日韩字符
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")

# 单字停用词：只用于丢弃「单字」token，不会拆散 bigram
_STOPWORDS = set(
    "的 了 是 在 和 与 或 有 不 我 你 他 这 那 它 们 吗 呢 吧 啊 把 被 给 对 从 到 而 就 也 都 还 又 "
    "很 更 最 会 能 要 想 让 请 帮 做 用 过 个 之 其 为 以 及 等 该 些 上 下 里 中 外 前 后 一 二 三 "
    "什么 怎么 为什么 哪些 哪个 是否 可以 需要 应该 如果 但是 因为 所以 然后 现在 一个 一下 时候".split()
)

# 领域同义词组：任一说法命中 -> 补一个规范词
# 这不是通用同义词表，只覆盖本系统里真实出现的说法，
# 目的是让「镜子 / 器械 / scope / 外径」这类不同叫法回到同一批文档。
_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dev", ("器械", "镜子", "镜体", "内镜", "支气管镜", "scope", "device", "导管", "探头", "外径")),
    ("airway", ("气道", "支气管", "airway", "管腔", "管道", "气道树", "支气管树")),
    ("nodule", ("结节", "病灶", "靶点", "靶区", "nodule", "lesion", "目标")),
    ("turn", ("转弯", "转角", "拐弯", "拐角", "角度", "turn", "curvature", "弯")),
    ("width", ("直径", "半径", "粗细", "宽窄", "宽度", "diameter", "radius")),
    ("branch", ("分叉", "分支", "岔口", "岔路", "junction", "branch")),
    ("length", ("长度", "长短", "距离", "路程", "length")),
    ("reach", ("可达", "可达性", "够得着", "reliability", "reachability", "adjacent")),
    ("clear", ("余量", "裕度", "间隙", "clearance", "余裕")),
    ("score", ("代价", "评分", "得分", "权重", "score", "cost", "排序")),
    ("profile", ("代价配置", "配置", "档位", "profile")),
    ("num", ("编号", "序号", "口径", "编号口径", "manifest")),
    ("repair", ("修复", "桥接", "断开", "断裂", "bridge", "recover", "残缺")),
    ("skin", ("骨架", "中心线", "骨架化", "skeleton", "centerline", "中线")),
    ("safe", ("安全", "风险", "并发症", "免责", "边界", "合规")),
    ("why", ("为什么", "原因", "理由", "依据", "解释", "why", "rationale")),
    ("b6", ("背段", "b6", "下叶上段")),
)

# 反查表：说法 -> 规范词（全部小写）
_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _forms in _ALIAS_GROUPS:
    for _form in _forms:
        _ALIAS_TO_CANON[_form.lower()] = _canon


def normalize(text: str) -> str:
    """全角转半角、大小写归一、压缩空白。"""
    folded = unicodedata.normalize("NFKC", text or "")
    return folded.lower()


def _cjk_tokens(run: str) -> list[str]:
    """对一段连续中文切 bigram；单字则保留该字。"""
    if len(run) == 1:
        return [] if run in _STOPWORDS else [run]
    bigrams = [run[i : i + 2] for i in range(len(run) - 1)]
    # 丢弃纯虚词的 bigram：整块是停用词，或两个成分都是停用单字
    return [
        gram
        for gram in bigrams
        if gram not in _STOPWORDS
        and not (gram[0] in _STOPWORDS and gram[1] in _STOPWORDS)
    ]


def tokenize(text: str, *, expand_aliases: bool = True) -> list[str]:
    """把一段文本切成词袋。

    expand_aliases=True 时额外追加规范词，用于「不同说法命中同一文档」。
    建索引与查询两侧都开，保证对称。
    """
    lowered = normalize(text)
    tokens: list[str] = []

    for match in _ASCII_RE.finditer(lowered):
        word = match.group(0)
        if len(word) == 1 and word not in {"b"} and not word.isdigit():
            # 单个无意义字母（例如中文夹一个 s）丢掉；数字与 b6 里的 b 保留
            continue
        tokens.append(word)

    for match in _CJK_RE.finditer(lowered):
        tokens.extend(_cjk_tokens(match.group(0)))

    if expand_aliases:
        # 在同义词表里查「原文是否出现某个说法」。
        # 表很小（< 100 条），直接子串扫描比构造自动机更省事，性能也够。
        for form, canon in _ALIAS_TO_CANON.items():
            if form in lowered:
                tokens.append(f"@canon:{canon}")

    return tokens


def token_set(text: str, *, expand_aliases: bool = True) -> set[str]:
    return set(tokenize(text, expand_aliases=expand_aliases))


def text_coverage(query: str, passage: str) -> float:
    """查询词中有多大比例出现在段落里 —— 用于「答非所问」的粗筛。"""
    wanted = token_set(query)
    if not wanted:
        return 0.0
    have = token_set(passage)
    return len(wanted & have) / len(wanted)


__all__ = ["normalize", "text_coverage", "token_set", "tokenize"]
