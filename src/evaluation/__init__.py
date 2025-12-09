"""Evaluation metrics for question answering tasks.

This module provides different evaluation methods for different QA datasets:
- Basic metrics (EM, F1, Lenient): For short-form factual QA (SQuAD, HotpotQA)
- Generation metrics (BLEU-4, ROUGE): For long-form generation (CMB)
- LLM-based evaluation: For domain-specific tasks requiring expert judgment
"""

from .basic import (
    compute_choice_accuracy,
    compute_contains,
    compute_em,
    compute_f1,
    normalize_answer,
)

from .generation import (
    compute_bleu4,
    compute_rouge1,
    compute_rouge2,
    compute_rouge_l,
)

from .llm import (
    LLMEvalResult,
    OpenRouterEvaluator,
    compute_llm_metrics,
)

from .config import (
    DATASET_METRICS,
    get_dataset_metrics,
    get_metric_names,
    evaluate_for_dataset,
    evaluate_predictions,
)

__all__ = [
    # Basic metrics (short-form QA)
    "normalize_answer",
    "compute_em",
    "compute_f1",
    "compute_contains",
    "compute_choice_accuracy",
    # Generation metrics (long-form)
    "compute_bleu4",
    "compute_rouge1",
    "compute_rouge2",
    "compute_rouge_l",
    # LLM-based evaluation
    "LLMEvalResult",
    "OpenRouterEvaluator",
    "compute_llm_metrics",
    # Dataset-specific evaluation
    "DATASET_METRICS",
    "get_dataset_metrics",
    "get_metric_names",
    "evaluate_for_dataset",
    "evaluate_predictions",
]
