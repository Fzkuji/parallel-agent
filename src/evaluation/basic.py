"""Basic evaluation metrics for short-form factual QA.

These metrics are designed for datasets like SQuAD and HotpotQA where answers
are typically short phrases or entities.

Metrics:
- EM (Exact Match): Strict equality after normalization
- F1: Token-level F1 score
- Lenient (Contains): Bidirectional substring containment
"""
from __future__ import annotations

import re
from typing import List


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Normalization steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_em(prediction: str, references: List[str]) -> float:
    """Compute exact match score.

    Returns 1.0 if normalized prediction matches any reference exactly, 0.0 otherwise.

    Args:
        prediction: Model's predicted answer
        references: List of gold/reference answers

    Returns:
        1.0 if exact match found, 0.0 otherwise
    """
    pred_norm = normalize_answer(prediction)
    for ref in references:
        if normalize_answer(ref) == pred_norm:
            return 1.0
    return 0.0


def compute_f1(prediction: str, references: List[str]) -> float:
    """Compute token-level F1 score.

    Calculates precision and recall based on token overlap, then computes F1.
    Returns the best F1 score across all references.

    Args:
        prediction: Model's predicted answer
        references: List of gold/reference answers

    Returns:
        Best F1 score (0.0 to 1.0)
    """
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        if not ref_tokens:
            continue

        overlap = set(pred_tokens) & set(ref_tokens)
        if not overlap:
            continue

        overlap_count = sum(
            min(pred_tokens.count(tok), ref_tokens.count(tok))
            for tok in overlap
        )
        precision = overlap_count / len(pred_tokens)
        recall = overlap_count / len(ref_tokens)

        if precision + recall == 0:
            continue

        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)

    return best


def compute_contains(prediction: str, references: List[str]) -> float:
    """Compute lenient containment score.

    Returns 1.0 if either:
    - Any reference is contained in the prediction, OR
    - The prediction is contained in any reference

    This is useful for cases where the model provides additional context
    or the reference is more verbose.

    Args:
        prediction: Model's predicted answer
        references: List of gold/reference answers

    Returns:
        1.0 if containment found, 0.0 otherwise
    """
    pred_norm = normalize_answer(prediction)
    for ref in references:
        ref_norm = normalize_answer(ref)
        if not ref_norm:
            continue
        if ref_norm in pred_norm or pred_norm in ref_norm:
            return 1.0
    return 0.0
