"""Generation metrics for long-form QA evaluation.

These metrics are designed for datasets like CMB where answers are
longer medical explanations rather than short factual phrases.

Metrics:
- BLEU-4: 4-gram precision with brevity penalty
- ROUGE-1: Unigram overlap F1
- ROUGE-2: Bigram overlap F1
- ROUGE-L: Longest common subsequence F1

For Chinese text, uses rouge-chinese with jieba word segmentation when available.
Falls back to character-level tokenization if not installed.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List

# Optional NLTK import for BLEU
try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Optional rouge-chinese with jieba for Chinese text
try:
    from rouge_chinese import Rouge as RougeChinese
    import jieba
    jieba.setLogLevel(jieba.logging.WARNING)  # Suppress jieba logs
    ROUGE_CHINESE_AVAILABLE = True
except ImportError:
    ROUGE_CHINESE_AVAILABLE = False



def _is_chinese_text(text: str) -> bool:
    """Check if text contains significant Chinese characters."""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    return chinese_chars > len(text) * 0.3  # More than 30% Chinese


def _tokenize(text: str) -> List[str]:
    """Tokenize text, using character-level for Chinese.

    For Chinese text: splits into individual characters (excluding whitespace/punctuation).
    For other text: splits on whitespace.
    """
    text = text.lower().strip()
    if _is_chinese_text(text):
        # Character-level tokenization for Chinese
        # Keep Chinese characters, digits, and letters
        tokens = list(re.findall(r'[\u4e00-\u9fff\w]', text))
        return tokens
    else:
        return text.split()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute longest common subsequence length."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # Use 1D DP for space efficiency
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


# -----------------------------------------------------------------------------
# BLEU-4 Metric
# -----------------------------------------------------------------------------

def _compute_bleu4_fallback(prediction: str, references: List[str]) -> float:
    """Fallback BLEU-4 implementation without nltk."""
    import math

    # Character-level tokenization (like LlamaFactory)
    pred_tokens = list(prediction)
    if len(pred_tokens) < 4:
        return 0.0

    best_score = 0.0
    for ref in references:
        ref_tokens = list(ref)
        if len(ref_tokens) < 4:
            continue

        # Compute n-gram precisions for n=1,2,3,4
        precisions = []
        for n in range(1, 5):
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                precisions.append(0.0)
                continue

            clipped_count = sum(
                min(pred_ngrams[ng], ref_ngrams.get(ng, 0))
                for ng in pred_ngrams
            )
            precisions.append(clipped_count / sum(pred_ngrams.values()))

        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            score = 0.0
        else:
            log_sum = sum(math.log(p) for p in precisions)
            score = math.exp(log_sum / 4)

        # Brevity penalty
        bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(
            1 - len(ref_tokens) / len(pred_tokens)
        )
        score *= bp

        best_score = max(best_score, score)

    return best_score


def compute_bleu4(prediction: str, references: List[str]) -> float:
    """Compute BLEU-4 score (best match against references).

    BLEU-4 measures n-gram precision (n=1,2,3,4) with a brevity penalty.
    Useful for evaluating long-form text generation quality.

    Args:
        prediction: Model's predicted answer
        references: List of gold/reference answers

    Returns:
        Best BLEU-4 score (0.0 to 1.0)
    """
    if not prediction.strip() or not references:
        return 0.0

    if NLTK_AVAILABLE:
        # Character-level tokenization (like LlamaFactory)
        pred_tokens = list(prediction)
        if len(pred_tokens) < 4:
            return 0.0

        best_score = 0.0
        smoothing = SmoothingFunction().method3  # NIST geometric sequence smoothing
        for ref in references:
            ref_tokens = list(ref)
            if len(ref_tokens) < 4:
                continue
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothing
                )
                best_score = max(best_score, score)
            except Exception:
                continue
        return best_score
    else:
        return _compute_bleu4_fallback(prediction, references)


# -----------------------------------------------------------------------------
# ROUGE Metrics
# -----------------------------------------------------------------------------

def _compute_rouge_fallback(
    prediction: str,
    references: List[str],
    rouge_type: str
) -> float:
    """Fallback ROUGE implementation without rouge_score package."""
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for ref in references:
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            continue

        if rouge_type == "rouge1":
            pred_ngrams = Counter(pred_tokens)
            ref_ngrams = Counter(ref_tokens)
            overlap = sum((pred_ngrams & ref_ngrams).values())
            precision = overlap / len(pred_tokens) if pred_tokens else 0
            recall = overlap / len(ref_tokens) if ref_tokens else 0
        elif rouge_type == "rouge2":
            pred_bigrams = _get_ngrams(pred_tokens, 2)
            ref_bigrams = _get_ngrams(ref_tokens, 2)
            if not pred_bigrams or not ref_bigrams:
                continue
            overlap = sum((pred_bigrams & ref_bigrams).values())
            precision = overlap / sum(pred_bigrams.values())
            recall = overlap / sum(ref_bigrams.values())
        elif rouge_type == "rougeL":
            lcs = _lcs_length(pred_tokens, ref_tokens)
            precision = lcs / len(pred_tokens) if pred_tokens else 0
            recall = lcs / len(ref_tokens) if ref_tokens else 0
        else:
            continue

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)

    return best_f1


# Cached rouge-chinese scorer instance
_rouge_chinese_instance: "RougeChinese | None" = None


def _get_rouge_chinese() -> "RougeChinese":
    """Get or create cached rouge-chinese scorer."""
    global _rouge_chinese_instance
    if _rouge_chinese_instance is None:
        _rouge_chinese_instance = RougeChinese()
    return _rouge_chinese_instance


def _compute_rouge_chinese(
    prediction: str,
    references: List[str],
    rouge_type: str,
) -> float:
    """Compute ROUGE using rouge-chinese with jieba segmentation."""
    # Segment with jieba
    pred_segmented = " ".join(jieba.cut(prediction))

    scorer = _get_rouge_chinese()
    best_score = 0.0

    for ref in references:
        if not ref.strip():
            continue
        ref_segmented = " ".join(jieba.cut(ref))
        try:
            scores = scorer.get_scores(pred_segmented, ref_segmented)[0]
            # rouge_type is "rouge1", "rouge2", or "rougeL" -> map to "rouge-1", "rouge-2", "rouge-l"
            key = rouge_type.replace("rouge", "rouge-").lower()
            best_score = max(best_score, scores[key]["f"])
        except Exception:
            continue

    return best_score


def compute_rouge1(prediction: str, references: List[str]) -> float:
    """Compute ROUGE-1 F1 score (best match against references).

    ROUGE-1 measures unigram overlap between prediction and reference.

    Args:
        prediction: Model's predicted answer
        references: List of gold/reference answers

    Returns:
        Best ROUGE-1 F1 score (0.0 to 1.0)
    """
    if not prediction.strip() or not references:
        return 0.0

    if ROUGE_CHINESE_AVAILABLE:
        return _compute_rouge_chinese(prediction, references, "rouge1")
    return _compute_rouge_fallback(prediction, references, "rouge1")


def compute_rouge2(prediction: str, references: List[str]) -> float:
    """Compute ROUGE-2 F1 score (best match against references).

    ROUGE-2 measures bigram overlap between prediction and reference.

    Args:
        prediction: Model's predicted answer
        references: List of gold/reference answers

    Returns:
        Best ROUGE-2 F1 score (0.0 to 1.0)
    """
    if not prediction.strip() or not references:
        return 0.0

    if ROUGE_CHINESE_AVAILABLE:
        return _compute_rouge_chinese(prediction, references, "rouge2")
    return _compute_rouge_fallback(prediction, references, "rouge2")


def compute_rouge_l(prediction: str, references: List[str]) -> float:
    """Compute ROUGE-L F1 score (best match against references).

    ROUGE-L measures longest common subsequence overlap.

    Args:
        prediction: Model's predicted answer
        references: List of gold/reference answers

    Returns:
        Best ROUGE-L F1 score (0.0 to 1.0)
    """
    if not prediction.strip() or not references:
        return 0.0

    if ROUGE_CHINESE_AVAILABLE:
        return _compute_rouge_chinese(prediction, references, "rougeL")
    return _compute_rouge_fallback(prediction, references, "rougeL")
