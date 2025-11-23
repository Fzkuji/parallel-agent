"""Lightweight strategy facade that re-exports core implementations under src.strategies."""

from src.strategies.executors import (
    run_all_in_one_strategy,
    run_all_in_one_multi_strategy,
    run_dependency_batch_strategy,
    run_full_batch_strategy,
    run_sequential_strategy,
    run_batch_multi_strategy,
    run_sequential_multi_strategy,
    StrategyResult,
)
from src.report import print_answer_table, summarize_results

__all__ = [
    "StrategyResult",
    "run_all_in_one_strategy",
    "run_all_in_one_multi_strategy",
    "run_dependency_batch_strategy",
    "run_full_batch_strategy",
    "run_sequential_strategy",
    "run_batch_multi_strategy",
    "run_sequential_multi_strategy",
    "print_answer_table",
    "summarize_results",
]
