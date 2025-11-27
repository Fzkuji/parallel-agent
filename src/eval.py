"""Evaluation metrics for question answering tasks.

This module re-exports from src/evaluation/ for backward compatibility.
For new code, prefer importing directly from src.evaluation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .evaluation.basic import (
    compute_contains,
    compute_em,
    compute_f1,
    normalize_answer,
)
from .evaluation.generation import (
    compute_bleu4,
    compute_rouge1,
    compute_rouge2,
    compute_rouge_l,
)
from .evaluation.config import (
    DATASET_METRICS,
    evaluate_for_dataset,
    get_dataset_metrics,
    get_metric_names,
)
from .models import Question


def evaluate_predictions(
    predictions: Dict[str, Tuple[str, bool]],
    lookup: Dict[str, Question],
    include_generation_metrics: bool = False,
    dataset: str = None,
) -> Dict[str, float]:
    """
    Evaluate predictions against references.

    Args:
        predictions: Dict of {qid: (prediction, strict_valid)}
        lookup: Dict of {qid: Question} with references
        include_generation_metrics: If True, include BLEU-4 and ROUGE metrics
            (useful for long-form generation tasks like CMB)
        dataset: If provided, use dataset-specific metrics

    Returns:
        Dict with metric scores (averaged over all predictions)
    """
    total = len(predictions)
    if total == 0:
        result = {"strict_acc": 0.0, "lenient_acc": 0.0, "f1": 0.0}
        if include_generation_metrics:
            result.update({
                "bleu4": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
            })
        return result

    # If dataset is specified, use dataset-specific metrics
    if dataset:
        refs_dict = {qid: lookup[qid].references for qid in predictions}
        return evaluate_for_dataset(dataset, predictions, refs_dict)

    strict = lenient = f1_sum = 0.0
    bleu4_sum = rouge1_sum = rouge2_sum = rougeL_sum = 0.0

    for qid, (prediction, strict_valid) in predictions.items():
        refs = lookup[qid].references
        if strict_valid:
            strict += compute_em(prediction, refs)
        lenient += compute_contains(prediction, refs)
        f1_sum += compute_f1(prediction, refs)

        if include_generation_metrics:
            bleu4_sum += compute_bleu4(prediction, refs)
            rouge1_sum += compute_rouge1(prediction, refs)
            rouge2_sum += compute_rouge2(prediction, refs)
            rougeL_sum += compute_rouge_l(prediction, refs)

    result = {
        "strict_acc": strict / total,
        "lenient_acc": lenient / total,
        "f1": f1_sum / total,
    }

    if include_generation_metrics:
        result.update({
            "bleu4": bleu4_sum / total,
            "rouge1": rouge1_sum / total,
            "rouge2": rouge2_sum / total,
            "rougeL": rougeL_sum / total,
        })

    return result


__all__ = [
    # Basic metrics
    "normalize_answer",
    "compute_em",
    "compute_f1",
    "compute_contains",
    # Generation metrics
    "compute_bleu4",
    "compute_rouge1",
    "compute_rouge2",
    "compute_rouge_l",
    # Evaluation functions
    "evaluate_predictions",
    "evaluate_for_dataset",
    # Config
    "DATASET_METRICS",
    "get_dataset_metrics",
    "get_metric_names",
]
