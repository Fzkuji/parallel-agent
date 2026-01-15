#!/usr/bin/env python3
"""Experiment 2b: Related Domain (Semantic - Weak)

Dataset: MATH (7 mathematical domains with labels)

Research Question: Do questions from the same domain benefit from being
processed together?

Conditions:
- Oracle: Group questions by mathematical domain
- Method: Use encoder embeddings to cluster similar questions
- Random: Mix questions from different domains randomly

Expected: Oracle >= Method > Random
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
    print_summary,
    save_results,
    setup_distributed,
    cleanup_distributed,
    shard_data,
    gather_results,
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


def load_math_by_domain(
    n_per_domain: int = -1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Load MATH dataset grouped by domain.

    Returns:
        Tuple of (all_questions, domain_to_indices)
    """
    logger.info("Loading MATH dataset...")

    # Load each domain separately
    domain_questions = defaultdict(list)
    for domain in MATH_DOMAINS:
        try:
            dataset = load_dataset("EleutherAI/hendrycks_math", domain, split="test", trust_remote_code=True)
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

    return all_questions, domain_to_indices


def extract_answer(solution: str) -> str:
    """Extract final answer from MATH solution (usually in \\boxed{})."""
    import re

    # Find \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", solution)
    if match:
        return match.group(1)

    # Fallback: last line or number
    lines = solution.strip().split("\n")
    return lines[-1] if lines else ""


def run_oracle(
    questions: List[Dict[str, Any]],
    domain_to_indices: Dict[str, List[int]],
    client: LLMClient,
    batch_size: int = 4,
) -> ExperimentResult:
    """Oracle condition: Group questions by domain."""
    logger.info("Running Oracle condition (domain grouping)...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for domain, indices in tqdm(domain_to_indices.items(), desc="Oracle (domains)"):
        # Process in batches within domain
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]

            for idx in batch_indices:
                q = questions[idx]
                problem = q["problem"]
                gold_answer = extract_answer(q["solution"])

                prompt = f"""Solve this math problem. Show your work and give the final answer.

Problem:
{problem}

Solution:"""

                pred, response = client.generate(prompt, max_tokens=512)
                total_latency += response.latency
                total_prompt_tokens += response.prompt_tokens
                total_completion_tokens += response.completion_tokens

                # Extract and evaluate answer
                pred_answer = extract_answer(pred) or pred.strip().split("\n")[-1]
                is_correct = _compare_math_answers(pred_answer, gold_answer)
                total_correct += int(is_correct)
                total_questions += 1

                details.append({
                    "domain": domain,
                    "problem": problem[:100] + "...",
                    "gold_answer": gold_answer,
                    "prediction": pred_answer,
                    "correct": is_correct,
                })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="oracle",
        dataset="math",
        n_samples=len(questions),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"accuracy": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_method(
    questions: List[Dict[str, Any]],
    client: LLMClient,
    batch_size: int = 4,
) -> ExperimentResult:
    """Method condition: Use encoder embeddings to cluster similar questions."""
    logger.info("Running Method condition (embedding clustering)...")

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
    except ImportError:
        logger.error("Please install sentence-transformers and scikit-learn")
        raise

    # Encode all questions
    logger.info("Encoding questions with sentence transformer...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [q["problem"] for q in questions]
    embeddings = encoder.encode(texts, show_progress_bar=True)

    # Cluster into groups
    n_clusters = len(questions) // batch_size
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Group by cluster
    cluster_to_indices = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        cluster_to_indices[label].append(idx)

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for cluster_id, indices in tqdm(cluster_to_indices.items(), desc="Method (clusters)"):
        for idx in indices:
            q = questions[idx]
            problem = q["problem"]
            gold_answer = extract_answer(q["solution"])

            prompt = f"""Solve this math problem. Show your work and give the final answer.

Problem:
{problem}

Solution:"""

            pred, response = client.generate(prompt, max_tokens=512)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            pred_answer = extract_answer(pred) or pred.strip().split("\n")[-1]
            is_correct = _compare_math_answers(pred_answer, gold_answer)
            total_correct += int(is_correct)
            total_questions += 1

            details.append({
                "cluster": int(cluster_id),
                "domain": q["domain"],
                "problem": problem[:100] + "...",
                "gold_answer": gold_answer,
                "prediction": pred_answer,
                "correct": is_correct,
            })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="method",
        dataset="math",
        n_samples=len(questions),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"accuracy": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_random(
    questions: List[Dict[str, Any]],
    client: LLMClient,
    batch_size: int = 4,
    seed: int = 42,
) -> ExperimentResult:
    """Random condition: Mix questions from different domains."""
    logger.info("Running Random condition...")

    random.seed(seed)
    indices = list(range(len(questions)))
    random.shuffle(indices)

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Process in random batches
    for i in tqdm(range(0, len(indices), batch_size), desc="Random"):
        batch_indices = indices[i:i + batch_size]

        for idx in batch_indices:
            q = questions[idx]
            problem = q["problem"]
            gold_answer = extract_answer(q["solution"])

            prompt = f"""Solve this math problem. Show your work and give the final answer.

Problem:
{problem}

Solution:"""

            pred, response = client.generate(prompt, max_tokens=512)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            pred_answer = extract_answer(pred) or pred.strip().split("\n")[-1]
            is_correct = _compare_math_answers(pred_answer, gold_answer)
            total_correct += int(is_correct)
            total_questions += 1

            details.append({
                "domain": q["domain"],
                "problem": problem[:100] + "...",
                "gold_answer": gold_answer,
                "prediction": pred_answer,
                "correct": is_correct,
            })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="random",
        dataset="math",
        n_samples=len(questions),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"accuracy": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def _compare_math_answers(pred: str, gold: str) -> bool:
    """Compare math answers (handles numeric and symbolic)."""
    import re

    # Normalize
    pred = pred.strip().lower()
    gold = gold.strip().lower()

    # Exact match
    if pred == gold:
        return True

    # Try numeric comparison
    try:
        pred_num = float(re.sub(r"[^\d.\-]", "", pred))
        gold_num = float(re.sub(r"[^\d.\-]", "", gold))
        if abs(pred_num - gold_num) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # Check containment
    if gold in pred:
        return True

    return False


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
    all_details = []

    for r in results:
        total_questions += r.n_questions
        total_correct += int(r.accuracy * r.n_questions)
        total_latency += r.latency
        total_prompt_tokens += r.prompt_tokens
        total_completion_tokens += r.completion_tokens
        all_details.extend(r.details)

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition=results[0].condition,
        dataset=results[0].dataset,
        n_samples=sum(r.n_samples for r in results),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"accuracy": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=all_details,
    )


def _shard_questions_by_domain(
    questions: List[Dict[str, Any]],
    domain_to_indices: Dict[str, List[int]],
    rank: int,
    world_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Shard questions while preserving domain structure."""
    # Shard questions list
    local_questions = shard_data(questions, rank, world_size)

    # Build new domain_to_indices for local shard
    local_domain_to_indices = defaultdict(list)
    for new_idx, q in enumerate(local_questions):
        domain = q["domain"]
        local_domain_to_indices[domain].append(new_idx)

    return local_questions, dict(local_domain_to_indices)


def main():
    parser = argparse.ArgumentParser(
        description="Exp 2b: Related Domain - MATH"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Model to use for inference (e.g., gpt-4o-mini, Qwen/Qwen2.5-7B-Instruct)"
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
        help="Number of GPUs for tensor parallelism (vLLM only)"
    )
    parser.add_argument(
        "--n-per-domain", type=int, default=-1,
        help="Number of questions per domain (-1 for all)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--conditions", type=str, default="oracle,method,random",
        help="Comma-separated list of conditions to run"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Setup distributed (each GPU loads one model)
    rank, world_size = setup_distributed()

    # Configuration
    config = ExperimentConfig(
        exp_name="exp2b_related_domain",
        dataset="math",
        model=args.model,
        n_samples=args.n_per_domain * len(MATH_DOMAINS),
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Load data (all ranks load the same data, then shard)
    all_questions, all_domain_to_indices = load_math_by_domain(
        n_per_domain=args.n_per_domain,
        seed=args.seed,
    )

    # Shard data across GPUs
    questions, domain_to_indices = _shard_questions_by_domain(
        all_questions, all_domain_to_indices, rank, world_size
    )
    logger.info(f"Rank {rank}/{world_size}: processing {len(questions)}/{len(all_questions)} questions")

    # Initialize LLM client (will use current GPU)
    client = LLMClient(
        model=args.model,
        use_local=args.use_local,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Run conditions on local shard
    conditions = [c.strip() for c in args.conditions.split(",")]
    local_results = []

    if "oracle" in conditions:
        local_results.append(run_oracle(questions, domain_to_indices, client, batch_size=args.batch_size))

    if "method" in conditions:
        local_results.append(run_method(questions, client, batch_size=args.batch_size))

    if "random" in conditions:
        local_results.append(run_random(questions, client, batch_size=args.batch_size, seed=args.seed))

    # Gather results from all GPUs
    all_results_by_condition = {}
    for result in local_results:
        gathered = gather_results([result], world_size)
        if rank == 0:
            merged = _merge_results(gathered)
            all_results_by_condition[result.condition] = merged

    # Print and save results (only rank 0)
    if rank == 0:
        results = list(all_results_by_condition.values())
        print_summary(results)
        save_results(results, config)

    cleanup_distributed()


if __name__ == "__main__":
    main()
