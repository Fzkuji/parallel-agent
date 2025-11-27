"""Dataset-specific evaluation configuration.

This module defines which metrics are used for each dataset type:
- squad: Short-form factual QA → EM, F1, Lenient
- hotpot: Short-form factual QA → EM, F1, Lenient
- cmb: Long-form medical QA → BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .basic import compute_contains, compute_em, compute_f1
from .generation import (
    compute_bleu4,
    compute_rouge1,
    compute_rouge2,
    compute_rouge_l,
)

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
    # CMB: Long-form medical QA
    "cmb": {
        "bleu4": compute_bleu4,
        "rouge1": compute_rouge1,
        "rouge2": compute_rouge2,
        "rougeL": compute_rouge_l,
    },
}


def get_dataset_metrics(dataset: str) -> Dict[str, MetricFunc]:
    """Get the metric functions for a specific dataset.

    Args:
        dataset: Dataset name (squad, hotpot, cmb)

    Returns:
        Dict mapping metric names to their functions

    Raises:
        ValueError: If dataset is not recognized
    """
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
