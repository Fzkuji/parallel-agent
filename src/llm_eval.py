"""LLM-based evaluation using OpenRouter API.

This module re-exports from src/evaluation/llm.py for backward compatibility.
For new code, prefer importing directly from src.evaluation.
"""
from .evaluation.llm import (
    CMB_EVAL_PROMPT_TEMPLATE,
    LLMEvalResult,
    OpenRouterEvaluator,
    compute_llm_metrics,
)

__all__ = [
    "LLMEvalResult",
    "OpenRouterEvaluator",
    "compute_llm_metrics",
    "CMB_EVAL_PROMPT_TEMPLATE",
]
