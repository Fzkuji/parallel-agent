"""Dataset-specific evaluation configuration.

This module defines which metrics are used for each dataset type:
- squad: Short-form factual QA → EM, F1, Lenient
- hotpot: Short-form factual QA → EM, F1, Lenient
- quac: Conversational QA → EM, F1, Lenient
- cmb: Long-form medical QA → BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from .basic import compute_choice_accuracy, compute_contains, compute_em, compute_f1
from .generation import (
    compute_bleu4,
    compute_rouge1,
    compute_rouge2,
    compute_rouge_l,
)

if TYPE_CHECKING:
    from ..models import Question

# Type alias for metric functions
MetricFunc = Callable[[str, List[str]], float]


# Dataset-specific metric configurations
# Each entry maps dataset name to a dict of {metric_name: metric_function}
DATASET_METRICS: Dict[str, Dict[str, MetricFunc]] = {
    # SQuAD: Short-form factual QA
    "squad": {
        "strict_acc": compute_em,
        "f1": compute_f1,
        "lenient_acc": compute_contains,
    },
    # HotpotQA: Short-form factual QA (multi-hop)
    "hotpot": {
        "strict_acc": compute_em,
        "f1": compute_f1,
        "lenient_acc": compute_contains,
    },
    # QuAC: Conversational QA
    "quac": {
        "strict_acc": compute_em,
        "f1": compute_f1,
        "lenient_acc": compute_contains,
    },
    # CMB: Long-form medical QA
    "cmb": {
        "bleu4": compute_bleu4,
        "rouge1": compute_rouge1,
        "rouge2": compute_rouge2,
        "rougeL": compute_rouge_l,
    },
    # QuALITY: Long-context multiple-choice reading comprehension
    "quality": {
        "acc": compute_choice_accuracy,
    },
    # DROP: Discrete reasoning over paragraphs (arithmetic, counting, sorting)
    "drop": {
        "strict_acc": compute_em,
        "f1": compute_f1,
        "lenient_acc": compute_contains,
    },
    # TriviaQA: Open-domain short-form factual QA
    "triviaqa": {
        "strict_acc": compute_em,
        "f1": compute_f1,
        "lenient_acc": compute_contains,
    },
    # CMB-Exam: Multiple-choice medical exam questions (all variants use accuracy)
    "cmb_exam": {
        "acc": compute_choice_accuracy,
    },
}


def get_dataset_metrics(dataset: str) -> Dict[str, MetricFunc]:
    """Get the metric functions for a specific dataset.

    Args:
        dataset: Dataset name (squad, hotpot, cmb, cmb_clin, cmb_exam_context, etc.)

    Returns:
        Dict mapping metric names to their functions

    Raises:
        ValueError: If dataset is not recognized
    """
    # Map CMB variants to their base metric configs
    if dataset == "cmb_clin":
        dataset = "cmb"  # CMB-Clin uses BLEU/ROUGE metrics
    elif dataset in ("cmb_exam_context", "cmb_exam_subdomain", "cmb_exam_random"):
        dataset = "cmb_exam"  # CMB-Exam variants use accuracy

    if dataset not in DATASET_METRICS:
        available = list(DATASET_METRICS.keys())
        raise ValueError(f"Unknown dataset: {dataset}. Available: {available}")
    return DATASET_METRICS[dataset]


def evaluate_for_dataset(
    dataset: str,
    predictions: Dict[str, Tuple[str, bool]],
    references: Dict[str, List[str]],
) -> Dict[str, float]:
    """Evaluate predictions using dataset-specific metrics.

    Args:
        dataset: Dataset name (squad, hotpot, cmb)
        predictions: Dict of {qid: (prediction, strict_valid)}
        references: Dict of {qid: list_of_reference_answers}

    Returns:
        Dict with averaged metric scores for the dataset
    """
    metrics = get_dataset_metrics(dataset)
    total = len(predictions)

    if total == 0:
        return {name: 0.0 for name in metrics}

    # Initialize accumulators
    sums: Dict[str, float] = {name: 0.0 for name in metrics}

    # Compute metrics for each prediction
    for qid, (prediction, strict_valid) in predictions.items():
        refs = references.get(qid, [])
        if not refs:
            continue

        for metric_name, metric_func in metrics.items():
            # For strict_acc (EM), only count if strict_valid is True
            if metric_name == "strict_acc" and not strict_valid:
                continue
            sums[metric_name] += metric_func(prediction, refs)

    # Compute averages
    return {name: sums[name] / total for name in metrics}


def get_metric_names(dataset: str) -> List[str]:
    """Get the list of metric names for a dataset.

    Args:
        dataset: Dataset name

    Returns:
        List of metric names used for this dataset
    """
    return list(get_dataset_metrics(dataset).keys())


def evaluate_predictions(
    predictions: Dict[str, Tuple[str, bool]],
    lookup: Dict[str, "Question"],
    include_generation_metrics: bool = False,
    dataset: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate predictions against references.

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
