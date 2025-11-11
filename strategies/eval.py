import re
from typing import Dict, List, Tuple

from python import Question


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_contains(prediction: str, references: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for ref in references:
        ref_norm = normalize_answer(ref)
        if not ref_norm:
            continue
        if ref_norm in pred_norm or pred_norm in ref_norm:
            return 1.0
    return 0.0


def compute_em(prediction: str, references: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for ref in references:
        if normalize_answer(ref) == pred_norm:
            return 1.0
    return 0.0


def compute_f1(prediction: str, references: List[str]) -> float:
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
        overlap_count = sum(min(pred_tokens.count(tok), ref_tokens.count(tok)) for tok in overlap)
        precision = overlap_count / len(pred_tokens)
        recall = overlap_count / len(ref_tokens)
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def evaluate_predictions(
    predictions: Dict[str, Tuple[str, bool]],
    lookup: Dict[str, Question],
) -> Dict[str, float]:
    total = len(predictions)
    if total == 0:
        return {"strict_acc": 0.0, "lenient_acc": 0.0, "f1": 0.0}

    strict = lenient = f1_sum = 0.0
    for qid, (prediction, strict_valid) in predictions.items():
        refs = lookup[qid].references
        if strict_valid:
            strict += compute_em(prediction, refs)
        lenient += compute_contains(prediction, refs)
        f1_sum += compute_f1(prediction, refs)

    return {
        "strict_acc": strict / total,
        "lenient_acc": lenient / total,
        "f1": f1_sum / total,
    }
