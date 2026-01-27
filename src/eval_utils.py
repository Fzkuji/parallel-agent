"""Shared evaluation utilities for QA experiments.

This module provides common functionality used across evaluation scripts:
- GPU detection
- Dataset loading
- Result aggregation and saving
- Caching utilities
- Summary printing
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_available_gpus(min_free_memory_gb: float = 10.0) -> List[int]:
    """Get list of GPUs with sufficient free memory.

    Args:
        min_free_memory_gb: Minimum free GPU memory in GB.

    Returns:
        List of GPU indices with sufficient memory.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            available = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                gpu_idx = int(parts[0].strip())
                free_mb = float(parts[1].strip())
                free_gb = free_mb / 1024
                if free_gb >= min_free_memory_gb:
                    available.append(gpu_idx)
            if available:
                return available
    except Exception:
        pass

    try:
        import torch
        count = torch.cuda.device_count()
        return list(range(count))
    except Exception:
        return list(range(8))


def context_to_items(context_payload: dict) -> List[dict]:
    """Convert context format to items format.

    Args:
        context_payload: Dictionary containing either 'items' or 'context'+'questions'.

    Returns:
        List of item dictionaries with 'qid', 'question', 'context', 'references', 'answer_tokens'.
    """
    if "items" in context_payload:
        return context_payload["items"]

    context = context_payload["context"]
    items = []
    for q in context_payload["questions"]:
        items.append({
            "qid": q["qid"],
            "question": q["text"],
            "context": context,
            "references": q["references"],
            "answer_tokens": q.get("answer_tokens", 12),
        })
    return items


def load_dataset_groups(
    dataset: str,
    split: str,
    max_contexts: int,
    min_questions: int,
    max_questions: int,
    seed: int,
    fixed_question_count: Optional[int] = None,
) -> List[Dict]:
    """Unified dataset loading function.

    Args:
        dataset: Dataset name (squad, hotpot, quac, drop, triviaqa, quality, cmb, coqa).
        split: Dataset split (train, validation, test).
        max_contexts: Maximum number of contexts to load.
        min_questions: Minimum questions per context.
        max_questions: Maximum questions per context.
        seed: Random seed for shuffling.
        fixed_question_count: If set, take exactly this many questions per context.

    Returns:
        List of context dictionaries.
    """
    if dataset == "hotpot":
        from .datasets.hotpot import load_hotpot_groups
        return load_hotpot_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "squad":
        from .datasets.squad import load_squad_groups
        return load_squad_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed,
            fixed_question_count=fixed_question_count
        )
    elif dataset == "quac":
        from .datasets.quac import load_quac_groups
        return load_quac_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "drop":
        from .datasets.drop import load_drop_groups
        return load_drop_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "triviaqa":
        from .datasets.triviaqa import load_triviaqa_groups
        return load_triviaqa_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "quality":
        from .datasets.quality import load_quality_groups
        return load_quality_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "cmb":
        from .datasets.cmb import load_cmb_exam_context_groups
        return load_cmb_exam_context_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "coqa":
        from .datasets.coqa import load_coqa_groups
        return load_coqa_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def aggregate_context_results(contexts: List[Dict]) -> Dict[str, Any]:
    """Aggregate per-context results into summary metrics.

    Args:
        contexts: List of per-context result dictionaries.

    Returns:
        Dictionary with aggregated metrics.
    """
    if not contexts:
        return {}

    total_questions = sum(ctx["num_questions"] for ctx in contexts)
    total_contexts = len(contexts)
    total_latency = sum(ctx["latency"] for ctx in contexts)
    avg_questions_per_context = total_questions / total_contexts if total_contexts else 0

    # Deduplicated tokens (context counted once)
    total_prompt_tokens = sum(ctx["prompt_tokens"] for ctx in contexts)
    total_generated_tokens = sum(ctx["generated_tokens"] for ctx in contexts)

    # API tokens (actual tokens sent/received)
    total_prompt_tokens_api = sum(ctx.get("prompt_tokens_api", ctx["prompt_tokens"]) for ctx in contexts)
    total_generated_tokens_api = sum(ctx.get("generated_tokens_api", ctx["generated_tokens"]) for ctx in contexts)

    # Dependency generation cost (for collab_llm/cross_batch)
    total_dep_prompt_tokens = sum(ctx.get("dep_prompt_tokens", 0) for ctx in contexts)
    total_dep_generated_tokens = sum(ctx.get("dep_generated_tokens", 0) for ctx in contexts)
    total_dep_latency = sum(ctx.get("dep_latency", 0) for ctx in contexts)

    # Weighted averages for metrics
    total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in contexts)
    total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in contexts)
    total_lenient = sum(ctx["metrics"].get("lenient_acc", 0) * ctx["num_questions"] for ctx in contexts)

    return {
        "strict_acc": total_em / total_questions if total_questions > 0 else 0,
        "f1": total_f1 / total_questions if total_questions > 0 else 0,
        "lenient_acc": total_lenient / total_questions if total_questions > 0 else 0,
        "avg_latency": total_latency / total_contexts if total_contexts else 0,
        "total_latency": total_latency,
        # Deduplicated tokens
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "avg_prompt_tokens": total_prompt_tokens / total_contexts if total_contexts else 0,
        "avg_generated_tokens": total_generated_tokens / total_contexts if total_contexts else 0,
        # API tokens
        "total_prompt_tokens_api": total_prompt_tokens_api,
        "total_generated_tokens_api": total_generated_tokens_api,
        "avg_prompt_tokens_api": total_prompt_tokens_api / total_contexts if total_contexts else 0,
        "avg_generated_tokens_api": total_generated_tokens_api / total_contexts if total_contexts else 0,
        # Dependency tokens
        "total_dep_prompt_tokens": total_dep_prompt_tokens,
        "total_dep_generated_tokens": total_dep_generated_tokens,
        "total_dep_latency": total_dep_latency,
        "avg_dep_prompt_tokens": total_dep_prompt_tokens / total_contexts if total_contexts else 0,
        "avg_dep_generated_tokens": total_dep_generated_tokens / total_contexts if total_contexts else 0,
        "avg_dep_latency": total_dep_latency / total_contexts if total_contexts else 0,
        # Counts
        "num_contexts": total_contexts,
        "num_questions": total_questions,
        "avg_questions_per_context": avg_questions_per_context,
    }


def save_evaluation_results(
    output_dir: Path,
    results: Dict[str, Any],
    config: Dict[str, Any],
    filename: str = "results.json",
) -> Path:
    """Save evaluation results with standard structure.

    Args:
        output_dir: Output directory path.
        results: Dictionary with strategy results.
        config: Configuration dictionary.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": config,
        "strategies": results,
    }

    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    return output_path


def print_results_summary(
    results: Dict[str, Any],
    strategies: List[str],
    dataset: str = "squad",
) -> None:
    """Print formatted results summary table.

    Args:
        results: Dictionary mapping strategy names to result dictionaries.
        strategies: List of strategy names to display.
        dataset: Dataset name (for display).
    """
    logger.info("\n" + "=" * 160)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 160)

    # Get sample counts from first strategy
    first_strategy = next((s for s in strategies if s in results), None)
    if first_strategy:
        first_metrics = results[first_strategy].get("aggregate_metrics", results[first_strategy])
        avg_q = first_metrics.get('avg_questions_per_context', 0)
        logger.info(f"Contexts: {first_metrics.get('num_contexts', 0)}, "
                   f"Questions: {first_metrics.get('num_questions', 0)}, "
                   f"Avg Q/Context: {avg_q:.2f}")

    # Combined comparison table
    header = (f"{'Strategy':<15} | {'EM':>6} | {'F1':>6} | {'Lenient':>7} | "
              f"{'Q/Ctx':>5} | {'PromptTok':>10} | {'GenTok':>8} | "
              f"{'PromptTok_API':>13} | {'GenTok_API':>10} | {'DepTok':>8} | {'Latency':>8}")
    separator = "-" * len(header)
    logger.info("\n" + header)
    logger.info(separator)

    for strategy in strategies:
        if strategy not in results:
            continue
        metrics = results[strategy].get("aggregate_metrics", results[strategy])
        dep_tok = metrics.get('avg_dep_prompt_tokens', 0) + metrics.get('avg_dep_generated_tokens', 0)
        logger.info(
            f"{strategy:<15} | "
            f"{metrics.get('strict_acc', 0):>6.3f} | "
            f"{metrics.get('f1', 0):>6.3f} | "
            f"{metrics.get('lenient_acc', 0):>7.3f} | "
            f"{metrics.get('avg_questions_per_context', 0):>5.1f} | "
            f"{metrics.get('avg_prompt_tokens', 0):>10.1f} | "
            f"{metrics.get('avg_generated_tokens', 0):>8.1f} | "
            f"{metrics.get('avg_prompt_tokens_api', 0):>13.1f} | "
            f"{metrics.get('avg_generated_tokens_api', 0):>10.1f} | "
            f"{dep_tok:>8.1f} | "
            f"{metrics.get('avg_latency', 0):>6.2f}s"
        )

    logger.info("=" * 160)


# Cache utilities

def get_cache_key(config: Dict[str, Any]) -> str:
    """Generate a cache key based on evaluation configuration.

    Args:
        config: Configuration dictionary with model, dataset, etc.

    Returns:
        Cache key string.
    """
    key_parts = [
        config.get("model", "unknown").replace("/", "_"),
        config.get("dataset", "unknown"),
        f"n{config.get('eval_samples', config.get('max_contexts', 0))}",
        f"q{config.get('min_questions', 1)}-{config.get('max_questions', 10)}",
        f"tok{config.get('max_new_tokens', 96)}",
    ]

    # Add optional parameters if present
    if config.get("group_size"):
        key_parts.append(f"gs{config['group_size']}")
    if config.get("strategy"):
        key_parts.append(config["strategy"])

    return "_".join(key_parts)


def load_cached_results(cache_dir: Path, cache_key: str, strategy: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load cached results if available.

    Args:
        cache_dir: Directory containing cache files.
        cache_key: Cache key string.
        strategy: Optional strategy name for strategy-specific cache.

    Returns:
        Cached results dictionary or None if not found.
    """
    cache_dir = Path(cache_dir)
    if strategy:
        cache_file = cache_dir / f"cache_{cache_key}_{strategy}.json"
    else:
        cache_file = cache_dir / f"cache_{cache_key}.json"

    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded cached results from {cache_file}")
            return data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache: {e}")
    return None


def save_cached_results(
    cache_dir: Path,
    cache_key: str,
    results: Dict[str, Any],
    config: Dict[str, Any],
    strategy: Optional[str] = None,
) -> None:
    """Save results to cache.

    Args:
        cache_dir: Directory for cache files.
        cache_key: Cache key string.
        results: Results dictionary to cache.
        config: Configuration dictionary.
        strategy: Optional strategy name for strategy-specific cache.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if strategy:
        cache_file = cache_dir / f"cache_{cache_key}_{strategy}.json"
    else:
        cache_file = cache_dir / f"cache_{cache_key}.json"

    data_with_meta = {
        "config": config,
        **results,
    }

    with open(cache_file, 'w') as f:
        json.dump(data_with_meta, f, indent=2)
    logger.info(f"Saved cache to {cache_file}")


# System prompt for QA tasks (shared across strategies)
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
You MUST wrap your answer in <answer></answer> tags. Be concise.

Example:
Question: What color is the sky?
<answer>blue</answer>"""


# All-in-one system prompt
ALL_IN_ONE_SYSTEM_PROMPT = """You are a helpful assistant that answers multiple questions from a single background.
Answer each question using exactly this format: QID: <answer>text</answer>

Example:
Q1: <answer>Paris</answer>
Q2: <answer>42</answer>

Rules:
- Use the exact question ID (e.g., Q1, Q2)
- Put answer inside <answer></answer> tags
- Extract answers directly from the background passage
- One answer per line, no extra text"""
