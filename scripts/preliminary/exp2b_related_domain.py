#!/usr/bin/env python3
"""Experiment 2b: Related Domain (Semantic Similarity - Weak Association)

Dataset: MATH (7 mathematical domains with labels)

Research Question: Do questions from the same mathematical domain benefit from
being processed together? This tests "weak" semantic association (same topic)
vs. exp2a's "strong" association (shared context).

Design:
- 7 domains in MATH dataset
- Each domain samples 49 questions (7 × 49 = 343 total)
- Each group processes 7 questions sequentially
- Same Domain: all 7 questions from one domain
- Cross Domain: 7 questions from 7 different domains (one per domain)

Conditions:
- Independent: Each question answered separately (baseline)
- Seq. Same Domain: 7 questions all from the same domain, with history
- Seq. Cross Domain: 7 questions from 7 different domains, with history

Key comparison: Same Domain vs Cross Domain at same sequence lengths (7 questions).
If Same Domain > Cross Domain, semantic similarity helps even without shared context.

Usage:
    # Single GPU (auto-detect)
    python scripts/preliminary/exp2b_related_domain.py \
        --models Qwen/Qwen3-8B \
        --use-local --use-vllm \
        --n-per-domain 49

    # Multi-model with auto GPU detection
    python scripts/preliminary/exp2b_related_domain.py \
        --models "Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B" \
        --use-local --use-vllm \
        --n-per-domain 49
"""

from __future__ import annotations

import os
# Suppress vLLM verbose logging
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from utils import (
    ExperimentConfig,
    ExperimentResult,
    LLMClient,
    compute_exact_match,
    print_summary,
    save_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MATH dataset domains
MATH_DOMAINS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# System prompt
SYSTEM_PROMPT = """You are a helpful math tutor. Solve the problem step by step and give your final answer in \\boxed{}.
Be concise but show key steps."""


def load_math_by_domain(
    n_per_domain: int = -1,
    seed: int = 42,
    split: str = "test",
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Load MATH dataset grouped by domain.

    Args:
        n_per_domain: Number of questions to sample per domain (-1 for all)
        seed: Random seed
        split: Dataset split to use

    Returns:
        Tuple of (all_questions, domain_to_indices)
    """
    logger.info(f"Loading MATH dataset (split={split})...")

    # Load each domain separately
    domain_questions = defaultdict(list)
    for domain in MATH_DOMAINS:
        try:
            dataset = load_dataset(
                "EleutherAI/hendrycks_math",
                domain,
                split=split,
                trust_remote_code=True
            )
            for item in dataset:
                domain_questions[domain].append({
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "level": item.get("level", "unknown"),
                    "domain": domain,
                })
            logger.info(f"  Loaded {len(domain_questions[domain])} from {domain}")
        except Exception as e:
            logger.warning(f"Failed to load {domain}: {e}")

    # Sample from each domain
    random.seed(seed)
    all_questions = []
    domain_to_indices = defaultdict(list)

    for domain in MATH_DOMAINS:
        questions = domain_questions.get(domain, [])
        if n_per_domain > 0 and len(questions) > n_per_domain:
            questions = random.sample(questions, n_per_domain)

        start_idx = len(all_questions)
        for i, q in enumerate(questions):
            all_questions.append(q)
            domain_to_indices[domain].append(start_idx + i)

    logger.info(f"Loaded {len(all_questions)} questions from {len(domain_to_indices)} domains")
    for domain, indices in domain_to_indices.items():
        logger.info(f"  {domain}: {len(indices)} questions")

    return all_questions, dict(domain_to_indices)


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...} format."""
    # Find all \boxed{...} patterns (handle nested braces)
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()  # Return last match (usually the final answer)

    # Fallback: look for "answer is" patterns
    match = re.search(r'(?:answer|result)\s*(?:is|=)\s*[:\s]*([^\n.,]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Last fallback: return last line
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else ""


def normalize_answer(answer: str) -> str:
    """Normalize math answer for comparison."""
    # Remove common LaTeX formatting
    answer = answer.strip()
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\$', '', answer)
    answer = re.sub(r'\\,', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer.lower()


def compare_math_answers(pred: str, gold: str) -> bool:
    """Compare predicted and gold math answers."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Exact match after normalization
    if pred_norm == gold_norm:
        return True

    # Try numeric comparison
    try:
        # Extract numbers
        pred_nums = re.findall(r'-?\d+\.?\d*', pred_norm)
        gold_nums = re.findall(r'-?\d+\.?\d*', gold_norm)

        if pred_nums and gold_nums:
            pred_val = float(pred_nums[-1])
            gold_val = float(gold_nums[-1])
            if abs(pred_val - gold_val) < 1e-6:
                return True
            # Also check relative error for larger numbers
            if gold_val != 0 and abs((pred_val - gold_val) / gold_val) < 1e-4:
                return True
    except (ValueError, IndexError):
        pass

    # Check if gold is contained in pred (for symbolic answers)
    if gold_norm in pred_norm:
        return True

    return False


def run_independent(
    questions: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Independent: Each question answered separately."""
    logger.info("Running Independent condition...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for q in tqdm(questions, desc="Independent"):
        problem = q["problem"]
        gold_solution = q["solution"]
        gold_answer = extract_boxed_answer(gold_solution)

        prompt = f"""Problem:
{problem}

Solve step by step and give your final answer in \\boxed{{}}."""

        pred_raw, response = client.generate(prompt, max_tokens=512, system_prompt=SYSTEM_PROMPT)
        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        pred_answer = extract_boxed_answer(pred_raw)
        is_correct = compare_math_answers(pred_answer, gold_answer)
        total_correct += int(is_correct)
        total_questions += 1

        details.append({
            "domain": q["domain"],
            "level": q["level"],
            "problem": problem[:100] + "...",
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "correct": is_correct,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="independent",
        dataset="math",
        n_samples=len(questions),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"accuracy": accuracy, "em": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        details=details,
    )


def run_sequential_same_domain(
    questions: List[Dict[str, Any]],
    domain_to_indices: Dict[str, List[int]],
    client: LLMClient,
    include_context: bool = True,
    seed: int = 42,
) -> ExperimentResult:
    """Sequential with 7 questions all from the same domain.

    Design: Each group has 7 questions from ONE domain.
    With 7 domains and 49 questions each, we get 7 groups per domain = 49 total groups.

    Args:
        questions: All questions
        domain_to_indices: Mapping from domain to question indices
        client: LLM client
        include_context: If True, include previous Q&A in context
        seed: Random seed
    """
    n_questions = 7  # Fixed: 7 questions per group (one per domain slot)
    condition = f"seq_same_domain{'_full' if include_context else ''}"
    logger.info(f"Running {condition} condition (n_questions={n_questions}, all same domain)...")

    random.seed(seed)

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Process each domain
    for domain, indices in tqdm(domain_to_indices.items(), desc=f"Same Domain"):
        # Shuffle indices within domain
        indices = indices.copy()
        random.shuffle(indices)

        # Group into batches of 7
        for group_start in range(0, len(indices), n_questions):
            group_indices = indices[group_start:group_start + n_questions]
            if len(group_indices) < n_questions:
                continue  # Skip incomplete groups

            conversation_history = []

            for turn_idx, q_idx in enumerate(group_indices):
                q = questions[q_idx]
                problem = q["problem"]
                gold_solution = q["solution"]
                gold_answer = extract_boxed_answer(gold_solution)

                # Build prompt
                if include_context and conversation_history:
                    # Include previous Q&A
                    history_text = "\n\n".join([
                        f"Previous Problem {i+1}:\n{h['problem']}\nAnswer: {h['answer']}"
                        for i, h in enumerate(conversation_history)
                    ])
                    prompt = f"""{history_text}

Current Problem:
{problem}

Solve step by step and give your final answer in \\boxed{{}}."""
                else:
                    prompt = f"""Problem:
{problem}

Solve step by step and give your final answer in \\boxed{{}}."""

                pred_raw, response = client.generate(prompt, max_tokens=512, system_prompt=SYSTEM_PROMPT)
                total_latency += response.latency
                total_prompt_tokens += response.prompt_tokens
                total_completion_tokens += response.completion_tokens

                pred_answer = extract_boxed_answer(pred_raw)
                is_correct = compare_math_answers(pred_answer, gold_answer)
                total_correct += int(is_correct)
                total_questions += 1

                # Add to history
                conversation_history.append({
                    "problem": problem,
                    "answer": pred_answer,
                })

                details.append({
                    "domain": domain,
                    "level": q["level"],
                    "turn": turn_idx,
                    "problem": problem[:100] + "...",
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "correct": is_correct,
                })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition=condition,
        dataset="math",
        n_samples=len(questions),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"accuracy": accuracy, "em": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        details=details,
    )


def run_sequential_cross_domain(
    questions: List[Dict[str, Any]],
    domain_to_indices: Dict[str, List[int]],
    client: LLMClient,
    include_context: bool = True,
    seed: int = 42,
) -> ExperimentResult:
    """Sequential with 7 questions from 7 DIFFERENT domains (one per domain).

    Design: Each group has exactly 7 questions, one from each domain.
    This ensures maximum domain diversity within each group.

    Args:
        questions: All questions
        domain_to_indices: Mapping from domain to question indices
        client: LLM client
        include_context: If True, include previous Q&A in context
        seed: Random seed
    """
    n_questions = 7  # Fixed: 7 questions per group (one per domain)
    condition = f"seq_cross_domain{'_full' if include_context else ''}"
    logger.info(f"Running {condition} condition (n_questions={n_questions}, all different domains)...")

    random.seed(seed)
    domains = list(domain_to_indices.keys())

    # Shuffle indices within each domain
    shuffled_indices = {}
    for domain in domains:
        indices = domain_to_indices[domain].copy()
        random.shuffle(indices)
        shuffled_indices[domain] = indices

    # Find minimum number of questions per domain (should be 49)
    min_per_domain = min(len(indices) for indices in shuffled_indices.values())
    n_groups = min_per_domain  # 49 groups

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Process groups: each group has one question from each domain
    for group_idx in tqdm(range(n_groups), desc="Cross Domain"):
        # Collect one question from each domain for this group
        group_question_indices = []
        for domain in domains:
            q_idx = shuffled_indices[domain][group_idx]
            group_question_indices.append(q_idx)

        # Shuffle the order within the group (so domains appear in random order)
        random.shuffle(group_question_indices)

        conversation_history = []

        for turn_idx, q_idx in enumerate(group_question_indices):
            q = questions[q_idx]
            problem = q["problem"]
            gold_solution = q["solution"]
            gold_answer = extract_boxed_answer(gold_solution)

            # Build prompt
            if include_context and conversation_history:
                history_text = "\n\n".join([
                    f"Previous Problem {i+1}:\n{h['problem']}\nAnswer: {h['answer']}"
                    for i, h in enumerate(conversation_history)
                ])
                prompt = f"""{history_text}

Current Problem:
{problem}

Solve step by step and give your final answer in \\boxed{{}}."""
            else:
                prompt = f"""Problem:
{problem}

Solve step by step and give your final answer in \\boxed{{}}."""

            pred_raw, response = client.generate(prompt, max_tokens=512, system_prompt=SYSTEM_PROMPT)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            pred_answer = extract_boxed_answer(pred_raw)
            is_correct = compare_math_answers(pred_answer, gold_answer)
            total_correct += int(is_correct)
            total_questions += 1

            conversation_history.append({
                "problem": problem,
                "answer": pred_answer,
            })

            details.append({
                "domain": q["domain"],
                "level": q["level"],
                "turn": turn_idx,
                "group_idx": group_idx,
                "problem": problem[:100] + "...",
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "correct": is_correct,
            })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition=condition,
        dataset="math",
        n_samples=len(questions),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"accuracy": accuracy, "em": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        details=details,
    )


def _merge_results(results: List[ExperimentResult]) -> ExperimentResult:
    """Merge results from multiple ranks into one."""
    if not results:
        raise ValueError("No results to merge")
    if len(results) == 1:
        return results[0]

    total_correct = 0
    total_questions = 0
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_unique_prompt_tokens = 0
    total_sum_completion_tokens = 0
    all_details = []

    for r in results:
        total_questions += r.n_questions
        total_correct += int(r.accuracy * r.n_questions)
        total_latency += r.latency
        total_prompt_tokens += r.prompt_tokens
        total_completion_tokens += r.completion_tokens
        total_unique_prompt_tokens += r.unique_prompt_tokens
        total_sum_completion_tokens += r.total_completion_tokens
        all_details.extend(r.details)

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition=results[0].condition,
        dataset=results[0].dataset,
        n_samples=sum(r.n_samples for r in results),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"accuracy": accuracy, "em": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=total_unique_prompt_tokens,
        total_completion_tokens=total_sum_completion_tokens,
        details=all_details,
    )


def worker_process(
    rank: int,
    world_size: int,
    gpu_id: int,
    model: str,
    use_vllm: bool,
    questions: List[Dict[str, Any]],
    domain_to_indices: Dict[str, List[int]],
    conditions: List[str],
    seed: int,
    output_dir: str,
    enable_thinking: bool = False,
):
    """Worker process that runs on a single GPU."""
    import json

    # IMPORTANT: Set environment variables BEFORE importing anything CUDA-related
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_DISABLE_FRONTEND_MULTIPROCESSING"] = "1"
    os.environ["VLLM_NO_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ["TQDM_DISABLE"] = "1"

    logger.info(f"[Worker {rank}] GPU {gpu_id}: Starting, {len(questions)} questions")

    # Initialize LLM client
    client = LLMClient(
        model=model,
        use_local=True,
        use_vllm=use_vllm,
        tensor_parallel_size=1,
        enable_thinking=enable_thinking,
    )

    logger.info(f"[Worker {rank}] Model loaded, running conditions: {conditions}")

    # Run conditions
    results = []

    if "independent" in conditions:
        logger.info(f"[Worker {rank}] Running independent...")
        results.append(run_independent(questions, client))

    if "seq_same_domain_full" in conditions:
        logger.info(f"[Worker {rank}] Running seq_same_domain_full...")
        results.append(run_sequential_same_domain(
            questions, domain_to_indices, client,
            include_context=True, seed=seed + rank
        ))

    if "seq_cross_domain_full" in conditions:
        logger.info(f"[Worker {rank}] Running seq_cross_domain_full...")
        results.append(run_sequential_cross_domain(
            questions, domain_to_indices, client,
            include_context=True, seed=seed + rank
        ))

    # Save results to temp file
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f"temp_rank{rank}.json")

    # Convert results to serializable format
    results_data = []
    for r in results:
        results_data.append({
            "condition": r.condition,
            "n_samples": r.n_samples,
            "n_questions": r.n_questions,
            "accuracy": r.accuracy,
            "metrics": r.metrics,
            "latency": r.latency,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "unique_prompt_tokens": r.unique_prompt_tokens,
            "total_completion_tokens": r.total_completion_tokens,
            "details": r.details,
        })

    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f)

    logger.info(f"[Worker {rank}] Done, saved to {temp_file}")


def run_experiment_for_model(
    model: str,
    questions: List[Dict[str, Any]],
    domain_to_indices: Dict[str, List[int]],
    conditions: List[str],
    args,
    num_gpus: int,
):
    """Run experiment for a single model."""
    import multiprocessing as mp
    import json

    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment for model: {model}")
    logger.info(f"{'='*60}\n")

    if args.use_local and num_gpus > 1:
        # Multi-GPU parallel mode with multiprocessing
        gpus = list(range(num_gpus))
        world_size = num_gpus

        logger.info(f"Parallel mode with {world_size} GPUs: {gpus}")

        # Set spawn method (required for CUDA)
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Clean up old temp files
        for rank in range(world_size):
            temp_file = os.path.join(args.output_dir, f"temp_rank{rank}.json")
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # Shard data by domain
        domain_list = list(domain_to_indices.keys())
        shards = [defaultdict(list) for _ in range(world_size)]
        shard_questions = [[] for _ in range(world_size)]

        for i, domain in enumerate(domain_list):
            rank = i % world_size
            indices = domain_to_indices[domain]
            # Map indices to local question list
            for idx in indices:
                new_idx = len(shard_questions[rank])
                shard_questions[rank].append(questions[idx])
                shards[rank][domain].append(new_idx)

        # Start all workers
        processes = []
        for rank, gpu_id in enumerate(gpus):
            p = mp.Process(
                target=worker_process,
                args=(rank, world_size, gpu_id, model, args.use_vllm,
                      shard_questions[rank], dict(shards[rank]), conditions,
                      args.seed, args.output_dir,
                      args.enable_thinking)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker {rank} on GPU {gpu_id} (PID: {p.pid})")

        # Wait for all workers
        for p in processes:
            p.join()

        logger.info("All workers finished, merging results...")

        # Merge results from all workers
        all_results_by_condition = {}
        for rank in range(world_size):
            temp_file = os.path.join(args.output_dir, f"temp_rank{rank}.json")
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                for r in results_data:
                    cond = r["condition"]
                    if cond not in all_results_by_condition:
                        all_results_by_condition[cond] = []
                    all_results_by_condition[cond].append(ExperimentResult(
                        condition=r["condition"],
                        dataset="math",
                        n_samples=r["n_samples"],
                        n_questions=r["n_questions"],
                        accuracy=r["accuracy"],
                        metrics=r["metrics"],
                        latency=r["latency"],
                        prompt_tokens=r["prompt_tokens"],
                        completion_tokens=r["completion_tokens"],
                        unique_prompt_tokens=r.get("unique_prompt_tokens", 0),
                        total_completion_tokens=r.get("total_completion_tokens", 0),
                        details=r["details"],
                    ))
                os.remove(temp_file)

        # Merge results
        final_results = []
        for cond, results_list in all_results_by_condition.items():
            merged = _merge_results(results_list)
            final_results.append(merged)

    else:
        # Single process mode (API or single GPU)
        if args.use_local and num_gpus == 1:
            logger.info("Single GPU mode: using GPU 0")

        client = LLMClient(
            model=model,
            use_local=args.use_local,
            use_vllm=args.use_vllm,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_thinking=args.enable_thinking,
        )

        final_results = []

        if "independent" in conditions:
            final_results.append(run_independent(questions, client))

        if "seq_same_domain_full" in conditions:
            final_results.append(run_sequential_same_domain(
                questions, domain_to_indices, client,
                include_context=True, seed=args.seed
            ))

        if "seq_cross_domain_full" in conditions:
            final_results.append(run_sequential_cross_domain(
                questions, domain_to_indices, client,
                include_context=True, seed=args.seed
            ))

    # Print and save results for this model
    config = ExperimentConfig(
        exp_name="exp2b_related_domain",
        dataset="math",
        model=model,
        n_samples=args.n_per_domain,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print_summary(final_results)
    save_results(final_results, config)

    return final_results


def summarize_from_files(output_dir: str, pattern: str = "exp2b_related_domain_*.json"):
    """Read saved result files and print combined summary in markdown format."""
    import glob
    import re as regex

    # Valid exp2b conditions
    VALID_CONDITIONS = ["independent", "seq_same_domain_full", "seq_cross_domain_full"]

    # Find all matching files
    file_pattern = os.path.join(output_dir, pattern)
    files = glob.glob(file_pattern)

    if not files:
        print(f"No files found matching: {file_pattern}")
        return

    # Load data, keep only the file with largest n_samples per model
    model_files = {}
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model = data.get("config", {}).get("model", "unknown")
        n_samples = data.get("config", {}).get("n_samples", 0)
        timestamp = data.get("timestamp", "")

        # Check if this file has valid conditions
        result_conditions = [r.get("condition", "") for r in data.get("results", [])]
        has_valid = any(c in VALID_CONDITIONS for c in result_conditions)
        if not has_valid:
            continue

        # Keep the file with largest n_samples (or latest timestamp if same)
        if model not in model_files or n_samples > model_files[model]["n_samples"]:
            model_files[model] = {"data": data, "n_samples": n_samples, "timestamp": timestamp}

    if not model_files:
        print("No valid results found.")
        return

    # Sort models by size
    def model_size(name):
        match = regex.search(r'(\d+(?:\.\d+)?)[Bb]', name)
        return float(match.group(1)) if match else 0

    sorted_models = sorted(model_files.keys(), key=model_size)

    # Get n_samples and detect model series
    first_data = model_files[sorted_models[0]]["data"]
    n_samples = first_data.get("config", {}).get("n_samples", "all")

    # Detect model series
    first_model = sorted_models[0]
    if "Qwen3" in first_model:
        model_series = "Qwen3"
    elif "Qwen2.5" in first_model:
        model_series = "Qwen2.5-Instruct"
    else:
        model_series = "Unknown"

    # Extract model sizes for display
    def get_model_size(name):
        name_short = name.split("/")[-1] if "/" in name else name
        size = name_short.replace("Qwen2.5-", "").replace("Qwen3-", "").replace("-Instruct", "")
        return size

    model_sizes = [get_model_size(m) for m in sorted_models]

    # Print header
    print("# Experiment 2b: Related Domain Study - Full Results")
    print()
    print("## Dataset")
    print("- **MATH**: Mathematical problems grouped by domain")
    print(f"- **Domains**: {len(MATH_DOMAINS)} ({', '.join(MATH_DOMAINS)})")
    print(f"- **Questions per domain**: {n_samples}")
    print(f"- **Models**: {model_series} series ({', '.join(model_sizes)})")
    print()
    print("## Experimental Conditions")
    print()
    print("| Condition | Description |")
    print("|-----------|-------------|")
    print("| independent | Each question answered separately |")
    print("| seq_same_domain_full | 7 questions all from same domain, with history |")
    print("| seq_cross_domain_full | 7 questions from 7 different domains, with history |")
    print()
    print("---")
    print()

    # Print each model's results
    all_model_results = {}
    for model in sorted_models:
        data = model_files[model]["data"]
        results = data.get("results", [])

        # Filter to valid conditions only
        results = [r for r in results if r.get("condition", "") in VALID_CONDITIONS]

        model_short = model.split("/")[-1] if "/" in model else model
        all_model_results[model_short] = {}

        print(f"## {model}")
        print()
        print("**EXPERIMENT SUMMARY**")
        print()

        headers = ["Condition", "Accuracy", "Samples", "Avg SeqLen", "Avg Latency (s)"]
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")

        for r in results:
            condition = r.get("condition", "")
            accuracy = r.get("accuracy", 0)
            r_n_samples = r.get("n_samples", 0)
            latency = r.get("latency", 0)
            n_questions = r.get("n_questions", 1)

            avg_latency = latency / n_questions if n_questions > 0 else 0

            all_model_results[model_short][condition] = {"accuracy": accuracy}

            # Calculate sequence length
            unique_prompt = r.get("unique_prompt_tokens", 0)
            total_compl = r.get("total_completion_tokens", 0)
            if unique_prompt > 0:
                seq_length = unique_prompt + total_compl
                avg_seq_len = seq_length / (n_questions // 7) if n_questions > 0 else 0  # per group (7 questions)
            else:
                avg_seq_len = 0

            row = [
                condition,
                f"{accuracy:.4f}",
                str(n_questions),
                f"{avg_seq_len:.1f}",
                f"{avg_latency:.2f}",
            ]
            print("| " + " | ".join(row) + " |")

        print()
        print("---")
        print()

    # Print final summary table
    print("## Summary Table (Accuracy)")
    print()

    headers = ["Model"] + VALID_CONDITIONS
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["-------"] * len(headers)) + "|")

    for model in sorted_models:
        model_short = model.split("/")[-1] if "/" in model else model
        size = get_model_size(model)
        results = all_model_results.get(model_short, {})
        row = [size]
        for cond in VALID_CONDITIONS:
            acc = results.get(cond, {}).get("accuracy", 0)
            row.append(f"{acc:.4f}")
        print("| " + " | ".join(row) + " |")

    print()

    # Key findings section
    print("## Key Findings")
    print()
    print("1. **Same domain vs Cross domain**: Compare seq_same_domain_full vs seq_cross_domain_full")
    print("2. **If same > cross**: Semantic similarity helps even without shared context")
    print("3. **Both conditions**: Same sequence length (7 questions per group)")
    print()


def check_existing_results(
    model: str,
    n_per_domain: int,
    conditions: List[str],
    output_dir: str,
) -> Optional[str]:
    """Check if results already exist for given model and conditions."""
    import glob

    # Build expected filename pattern
    model_name = model.replace("/", "_").replace("\\", "_")
    n_samples_str = "all" if n_per_domain == -1 else str(n_per_domain)
    filename = f"exp2b_related_domain_{model_name}_n{n_samples_str}.json"
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        return None

    # Check if the file has all requested conditions
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        existing_conditions = set(r.get("condition", "") for r in data.get("results", []))
        requested_conditions = set(conditions)

        if requested_conditions.issubset(existing_conditions):
            return filepath
        else:
            missing = requested_conditions - existing_conditions
            logger.info(f"Existing results missing conditions: {missing}")
            return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Error reading existing results: {e}")
        return None


def load_results_from_file(filepath: str) -> List[ExperimentResult]:
    """Load experiment results from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for r in data.get("results", []):
        results.append(ExperimentResult(
            condition=r["condition"],
            dataset=r["dataset"],
            n_samples=r["n_samples"],
            n_questions=r["n_questions"],
            accuracy=r["accuracy"],
            metrics=r["metrics"],
            latency=r["latency"],
            prompt_tokens=r["prompt_tokens"],
            completion_tokens=r["completion_tokens"],
            unique_prompt_tokens=r.get("unique_prompt_tokens", 0),
            total_completion_tokens=r.get("total_completion_tokens", 0),
            details=r.get("details", []),
        ))
    return results


def main():
    import torch

    parser = argparse.ArgumentParser(
        description="Exp 2b: Related Domain - MATH"
    )
    parser.add_argument(
        "--models", type=str, default="Qwen/Qwen3-8B",
        help="Comma-separated list of models (e.g., 'Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B')"
    )
    parser.add_argument(
        "--use-local", action="store_true",
        help="Use local model instead of API"
    )
    parser.add_argument(
        "--use-vllm", action="store_true",
        help="Use vLLM for faster inference (requires --use-local)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (vLLM only, for single-GPU mode)"
    )
    parser.add_argument(
        "--n-per-domain", type=int, default=49,
        help="Questions per domain (default 49 for 7 groups of 7)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--conditions", type=str,
        default="independent,seq_same_domain_full,seq_cross_domain_full",
        help="Comma-separated conditions to run"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory"
    )
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Enable thinking mode for Qwen3 models"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run experiments even if results already exist"
    )
    args = parser.parse_args()

    # Parse models and conditions
    models = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    # Detect number of GPUs
    if args.use_local and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s)")
    else:
        num_gpus = 0
        logger.info("Using CPU or API mode")

    # Check which models need to be run
    models_to_run = []
    models_with_results = []

    for model in models:
        if args.force:
            models_to_run.append(model)
        else:
            existing_file = check_existing_results(
                model=model,
                n_per_domain=args.n_per_domain,
                conditions=conditions,
                output_dir=args.output_dir,
            )
            if existing_file:
                logger.info(f"Found existing results for {model}: {existing_file}")
                models_with_results.append((model, existing_file))
            else:
                models_to_run.append(model)

    # Load data only if we need to run experiments
    questions = None
    domain_to_indices = None
    if models_to_run:
        questions, domain_to_indices = load_math_by_domain(
            n_per_domain=args.n_per_domain,
            seed=args.seed,
        )
        logger.info(f"Loaded {len(questions)} questions")

    # Run experiments for models that need it
    all_model_results = {}

    # First, load existing results
    for model, filepath in models_with_results:
        logger.info(f"Using cached results for {model}: {filepath}")
        results = load_results_from_file(filepath)
        results = [r for r in results if r.condition in conditions]
        all_model_results[model] = results

    # Then, run new experiments
    for model in models_to_run:
        results = run_experiment_for_model(
            model=model,
            questions=questions,
            domain_to_indices=domain_to_indices,
            conditions=conditions,
            args=args,
            num_gpus=num_gpus,
        )
        all_model_results[model] = results

    # Print final combined summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL MODELS")
    print("=" * 80 + "\n")
    summarize_from_files(args.output_dir)


if __name__ == "__main__":
    main()
