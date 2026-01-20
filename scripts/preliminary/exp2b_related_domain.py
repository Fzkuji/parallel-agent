#!/usr/bin/env python3
"""Experiment 2b: Related Domain (Semantic Similarity - Weak Association)

Dataset: MATH (7 mathematical domains with labels)

Research Question: Do questions from the same mathematical domain benefit from
being processed together? This tests "weak" semantic association (same topic)
vs. exp2a's "strong" association (shared context).

Conditions:
- Same Domain (Sequential): Questions grouped by domain, processed sequentially
- Random Domain (Sequential): Questions randomly mixed across domains
- Independent: Each question answered separately (baseline)

Key comparison: Same Domain vs Random Domain at similar sequence lengths.
If Same Domain > Random Domain, semantic similarity helps even without shared context.

Usage:
    # Single GPU
    python scripts/preliminary/exp2b_related_domain.py \
        --model Qwen/Qwen3-8B \
        --n-per-domain 20 \
        --n-questions 5

    # Multi-GPU (data parallel)
    python scripts/preliminary/exp2b_related_domain.py \
        --model Qwen/Qwen3-8B \
        --n-per-domain 50 \
        --n-gpus 4
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

# System prompts
SYSTEM_PROMPT = """You are a helpful math tutor. Solve the problem step by step and give your final answer in \\boxed{}.
Be concise but show key steps."""

SYSTEM_PROMPT_MULTI = """You are a helpful math tutor. Solve each problem step by step.
Format your response as:
Problem 1: [solution] \\boxed{answer1}
Problem 2: [solution] \\boxed{answer2}
..."""


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
    n_questions: int = 5,
    include_context: bool = True,
) -> ExperimentResult:
    """Sequential with questions grouped by domain.

    Args:
        questions: All questions
        domain_to_indices: Mapping from domain to question indices
        client: LLM client
        n_questions: Number of questions per group
        include_context: If True, include previous Q&A in context (sequential with history)
    """
    condition = f"seq_same_domain{'_full' if include_context else ''}"
    logger.info(f"Running {condition} condition (n_questions={n_questions})...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Process each domain
    for domain, indices in tqdm(domain_to_indices.items(), desc=f"Same Domain"):
        # Split into groups of n_questions
        random.shuffle(indices)

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


def run_sequential_random_domain(
    questions: List[Dict[str, Any]],
    client: LLMClient,
    n_questions: int = 5,
    include_context: bool = True,
    seed: int = 42,
) -> ExperimentResult:
    """Sequential with questions randomly mixed across domains.

    Args:
        questions: All questions
        client: LLM client
        n_questions: Number of questions per group
        include_context: If True, include previous Q&A in context
        seed: Random seed
    """
    condition = f"seq_random_domain{'_full' if include_context else ''}"
    logger.info(f"Running {condition} condition (n_questions={n_questions})...")

    # Shuffle all questions
    random.seed(seed)
    indices = list(range(len(questions)))
    random.shuffle(indices)

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Process in groups
    for group_start in tqdm(range(0, len(indices), n_questions), desc="Random Domain"):
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


def run_all_in_one_same_domain(
    questions: List[Dict[str, Any]],
    domain_to_indices: Dict[str, List[int]],
    client: LLMClient,
    n_questions: int = 5,
) -> ExperimentResult:
    """All-in-One: All questions from same domain in one prompt."""
    logger.info(f"Running all_in_one_same_domain condition (n_questions={n_questions})...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for domain, indices in tqdm(domain_to_indices.items(), desc="All-in-One Same Domain"):
        random.shuffle(indices)

        for group_start in range(0, len(indices), n_questions):
            group_indices = indices[group_start:group_start + n_questions]
            if len(group_indices) < n_questions:
                continue

            # Build multi-problem prompt
            problems_text = "\n\n".join([
                f"Problem {i+1}:\n{questions[idx]['problem']}"
                for i, idx in enumerate(group_indices)
            ])

            prompt = f"""{problems_text}

Solve each problem step by step. Give each final answer in \\boxed{{}}.
Format: Problem 1: ... \\boxed{{answer1}}
Problem 2: ... \\boxed{{answer2}}
..."""

            pred_raw, response = client.generate(prompt, max_tokens=2048, system_prompt=SYSTEM_PROMPT_MULTI)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            # Extract answers for each problem
            # Try to parse "Problem N:" format
            pred_answers = []
            for i in range(len(group_indices)):
                pattern = rf'Problem\s*{i+1}[:\s].*?\\boxed\{{([^}}]+)\}}'
                match = re.search(pattern, pred_raw, re.IGNORECASE | re.DOTALL)
                if match:
                    pred_answers.append(match.group(1).strip())
                else:
                    # Fallback: find all boxed answers
                    all_boxed = re.findall(r'\\boxed\{([^}]+)\}', pred_raw)
                    if i < len(all_boxed):
                        pred_answers.append(all_boxed[i])
                    else:
                        pred_answers.append("")

            # Evaluate each question
            for i, q_idx in enumerate(group_indices):
                q = questions[q_idx]
                gold_answer = extract_boxed_answer(q["solution"])
                pred_answer = pred_answers[i] if i < len(pred_answers) else ""

                is_correct = compare_math_answers(pred_answer, gold_answer)
                total_correct += int(is_correct)
                total_questions += 1

                details.append({
                    "domain": domain,
                    "level": q["level"],
                    "problem": q["problem"][:100] + "...",
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "correct": is_correct,
                })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="all_in_one_same_domain",
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


def worker_process(rank: int, world_size: int, args, questions: List[Dict], domain_to_indices: Dict):
    """Worker process for multi-GPU parallel inference."""
    import torch

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, device 0 is our assigned GPU

    logger.info(f"Worker {rank}/{world_size} starting on GPU {rank}")

    # Shard data
    all_indices = list(range(len(questions)))
    shard_size = len(all_indices) // world_size
    start_idx = rank * shard_size
    end_idx = start_idx + shard_size if rank < world_size - 1 else len(all_indices)
    local_indices = all_indices[start_idx:end_idx]

    local_questions = [questions[i] for i in local_indices]

    # Build local domain_to_indices
    local_domain_to_indices = defaultdict(list)
    for new_idx, old_idx in enumerate(local_indices):
        domain = questions[old_idx]["domain"]
        local_domain_to_indices[domain].append(new_idx)

    logger.info(f"Worker {rank}: processing {len(local_questions)} questions")

    # Initialize client
    client = LLMClient(
        model=args.model,
        use_local=True,
        use_vllm=True,
        tensor_parallel_size=1,
        enable_thinking=args.enable_thinking,
    )

    # Run conditions
    conditions = [c.strip() for c in args.conditions.split(",")]
    results = []

    if "independent" in conditions:
        results.append(run_independent(local_questions, client))

    if "seq_same_domain" in conditions:
        results.append(run_sequential_same_domain(
            local_questions, dict(local_domain_to_indices), client,
            n_questions=args.n_questions, include_context=False
        ))

    if "seq_same_domain_full" in conditions:
        results.append(run_sequential_same_domain(
            local_questions, dict(local_domain_to_indices), client,
            n_questions=args.n_questions, include_context=True
        ))

    if "seq_random_domain" in conditions:
        results.append(run_sequential_random_domain(
            local_questions, client,
            n_questions=args.n_questions, include_context=False, seed=args.seed + rank
        ))

    if "seq_random_domain_full" in conditions:
        results.append(run_sequential_random_domain(
            local_questions, client,
            n_questions=args.n_questions, include_context=True, seed=args.seed + rank
        ))

    if "all_in_one_same_domain" in conditions:
        results.append(run_all_in_one_same_domain(
            local_questions, dict(local_domain_to_indices), client,
            n_questions=args.n_questions
        ))

    return results


def merge_results(all_results: List[List[ExperimentResult]]) -> List[ExperimentResult]:
    """Merge results from multiple workers."""
    if not all_results:
        return []

    # Group by condition
    condition_results = defaultdict(list)
    for worker_results in all_results:
        for result in worker_results:
            condition_results[result.condition].append(result)

    # Merge each condition
    merged = []
    for condition, results in condition_results.items():
        total_correct = sum(r.accuracy * r.n_questions for r in results)
        total_questions = sum(r.n_questions for r in results)
        total_latency = sum(r.latency for r in results)
        total_prompt = sum(r.prompt_tokens for r in results)
        total_completion = sum(r.completion_tokens for r in results)
        all_details = []
        for r in results:
            all_details.extend(r.details)

        accuracy = total_correct / total_questions if total_questions > 0 else 0

        merged.append(ExperimentResult(
            condition=condition,
            dataset="math",
            n_samples=sum(r.n_samples for r in results),
            n_questions=total_questions,
            accuracy=accuracy,
            metrics={"accuracy": accuracy, "em": accuracy},
            latency=total_latency,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            unique_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            details=all_details,
        ))

    return merged


def main():
    parser = argparse.ArgumentParser(description="Exp 2b: Related Domain - MATH")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model name (default: Qwen/Qwen3-8B)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking mode for Qwen3 models")

    # Data
    parser.add_argument("--n-per-domain", type=int, default=50,
                        help="Questions per domain (-1 for all)")
    parser.add_argument("--n-questions", type=int, default=5,
                        help="Questions per group in sequential conditions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Conditions
    parser.add_argument("--conditions", type=str,
                        default="independent,seq_same_domain_full,seq_random_domain_full",
                        help="Comma-separated conditions to run")

    # Parallel
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Number of GPUs for data parallel")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/preliminary",
                        help="Output directory")

    args = parser.parse_args()

    # Load data
    questions, domain_to_indices = load_math_by_domain(
        n_per_domain=args.n_per_domain,
        seed=args.seed,
    )

    if args.n_gpus > 1:
        # Multi-GPU: use multiprocessing
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        with mp.Pool(args.n_gpus) as pool:
            all_results = pool.starmap(
                worker_process,
                [(rank, args.n_gpus, args, questions, domain_to_indices) for rank in range(args.n_gpus)]
            )

        results = merge_results(all_results)
    else:
        # Single GPU
        client = LLMClient(
            model=args.model,
            use_local=True,
            use_vllm=True,
            tensor_parallel_size=1,
            enable_thinking=args.enable_thinking,
        )

        conditions = [c.strip() for c in args.conditions.split(",")]
        results = []

        if "independent" in conditions:
            results.append(run_independent(questions, client))

        if "seq_same_domain" in conditions:
            results.append(run_sequential_same_domain(
                questions, domain_to_indices, client,
                n_questions=args.n_questions, include_context=False
            ))

        if "seq_same_domain_full" in conditions:
            results.append(run_sequential_same_domain(
                questions, domain_to_indices, client,
                n_questions=args.n_questions, include_context=True
            ))

        if "seq_random_domain" in conditions:
            results.append(run_sequential_random_domain(
                questions, client,
                n_questions=args.n_questions, include_context=False, seed=args.seed
            ))

        if "seq_random_domain_full" in conditions:
            results.append(run_sequential_random_domain(
                questions, client,
                n_questions=args.n_questions, include_context=True, seed=args.seed
            ))

        if "all_in_one_same_domain" in conditions:
            results.append(run_all_in_one_same_domain(
                questions, domain_to_indices, client,
                n_questions=args.n_questions
            ))

    # Print and save results
    config = ExperimentConfig(
        exp_name="exp2b_related_domain",
        dataset="math",
        model=args.model,
        n_samples=len(questions),
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print_summary(results)
    save_results(results, config)


if __name__ == "__main__":
    main()
