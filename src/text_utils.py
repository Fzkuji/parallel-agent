"""Text processing utilities for question analysis."""
from __future__ import annotations

from typing import Optional, Set

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "in",
    "on",
    "for",
    "with",
    "that",
    "which",
    "what",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "is",
    "are",
    "does",
    "do",
    "did",
    "was",
    "were",
    "how",
    "why",
    "请",
    "以及",
    "哪些",
    "什么",
}

REFERENCE_KEYWORDS = {
    "上述",
    "前文",
    "前面",
    "前一个",
    "这",
    "那",
    "它",
    "他们",
    "他",
    "她",
    "其",
    "这些",
    "those",
    "them",
    "it",
    "that",
    "previous",
}

AGGREGATE_KEYWORDS = {
    "总共",
    "总计",
    "一共",
    "合计",
    "总数",
    "平均",
    "列表",
    "列出",
    "罗列",
    "排序",
    "排名",
    "比较",
    "对比",
    "整体",
    "全部",
    "汇总",
    "aggregate",
    "list",
    "compare",
    "total",
    "average",
}


def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text, filtering stopwords."""
    normalized = []
    for ch in text.lower():
        if ch.isalnum():
            normalized.append(ch)
        else:
            normalized.append(" ")
    tokens = {
        tok for tok in "".join(normalized).split() if tok and tok not in STOPWORDS
    }
    tokens.update({ch for ch in text if "\u4e00" <= ch <= "\u9fff"})
    return tokens


def detect_reference_question(text: str) -> bool:
    """Check if question contains reference keywords (e.g., 'it', 'previous')."""
    return any(keyword in text for keyword in REFERENCE_KEYWORDS)


def detect_aggregate_question(text: str, type_hint: Optional[str]) -> bool:
    """Check if question is an aggregate/summary type."""
    if type_hint and type_hint.lower() in {"aggregate", "list", "compare"}:
        return True
    return any(keyword in text for keyword in AGGREGATE_KEYWORDS)
