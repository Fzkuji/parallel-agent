#!/usr/bin/env python3
"""Experiment 2a: Shared Context (Multi-Query QA on SQuAD)

Dataset: SQuAD (multiple questions per paragraph)

Research Question: How do different multi-query processing strategies compare
when questions share the same context?

Conditions:
- Independent: Each question + context answered separately
- All-in-One: All questions from same context in one prompt
- Sequential (Random): Questions answered in random order, with Q&A history
- Sequential (LLM-based): LLM determines optimal question order, with Q&A history

Expected: Sequential strategies may benefit from information sharing between questions.
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

from datasets import load_dataset
from tqdm import tqdm

from utils import (
    ExperimentConfig,
    ExperimentResult,
    LLMClient,
    compute_exact_match,
    compute_f1,
    compute_contains,
    print_summary,
    save_results,
    setup_distributed,
    cleanup_distributed,
    shard_data,
    gather_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_squad_groups(
    n_groups: int = -1,
    min_questions: int = 3,
    max_questions: int = 6,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load SQuAD dataset grouped by paragraph.

    Returns list of groups, each containing:
    - context: The paragraph text
    - questions: List of {question, answer} dicts
    """
    logger.info("Loading SQuAD dataset...")

    dataset = load_dataset("rajpurkar/squad", split="validation")

    # Group questions by context
    context_groups = defaultdict(list)
    for item in dataset:
        context = item["context"]
        context_groups[context].append({
            "question": item["question"],
            "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
            "id": item["id"],
        })

    # Filter groups with enough questions
    valid_groups = []
    for context, questions in context_groups.items():
        if min_questions <= len(questions) <= max_questions:
            valid_groups.append({
                "context": context,
                "questions": questions,
                "n_questions": len(questions),
            })

    # Shuffle and sample
    random.seed(seed)
    random.shuffle(valid_groups)
    if n_groups > 0:
        groups = valid_groups[:n_groups]
    else:
        groups = valid_groups  # Use all groups

    logger.info(f"Loaded {len(groups)} groups with {min_questions}-{max_questions} questions each")

    return groups


def run_independent(
    groups: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Independent: Each question + context answered separately."""
    logger.info("Running Independent condition...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for group in tqdm(groups, desc="Independent"):
        context = group["context"]
        questions = group["questions"]

        for q_item in questions:
            question = q_item["question"]
            gold_answer = q_item["answer"]

            prompt = f"""Read the following passage and answer the question.

Passage:
{context}

Question: {question}

Answer (be concise):"""

            pred, response = client.generate(prompt, max_tokens=64)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            is_correct = compute_contains(pred, gold_answer) > 0
            total_correct += int(is_correct)
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "correct": is_correct,
            })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="independent",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"em": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_all_in_one(
    groups: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """All-in-One: All questions from same context in one prompt."""
    logger.info("Running All-in-One condition...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for group in tqdm(groups, desc="All-in-One"):
        context = group["context"]
        questions = group["questions"]

        # Build multi-question prompt
        q_list = "\n".join([f"Q{i+1}: {q['question']}" for i, q in enumerate(questions)])

        prompt = f"""Read the following passage and answer all questions.

Passage:
{context}

Questions:
{q_list}

Answer each question concisely. Format: Q1: [answer], Q2: [answer], ..."""

        pred, response = client.generate(prompt, max_tokens=256)
        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        # Parse answers
        answers = _parse_batch_answers(pred, len(questions))

        for i, q_item in enumerate(questions):
            gold_answer = q_item["answer"]
            pred_answer = answers.get(i, "")

            is_correct = compute_contains(pred_answer, gold_answer) > 0
            total_correct += int(is_correct)
            total_questions += 1

            details.append({
                "question": q_item["question"],
                "gold_answer": gold_answer,
                "prediction": pred_answer,
                "correct": is_correct,
            })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="all_in_one",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"em": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_sequential_random(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    seed: int = 42,
) -> ExperimentResult:
    """Sequential (Random): Questions in random order, each sees previous Q&A history."""
    logger.info("Running Sequential (Random) condition...")

    random.seed(seed)

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for group in tqdm(groups, desc="Sequential-Random"):
        context = group["context"]
        questions = group["questions"].copy()

        # Shuffle questions randomly
        random.shuffle(questions)

        qa_history = []

        for q_item in questions:
            question = q_item["question"]
            gold_answer = q_item["answer"]

            # Build prompt with Q&A history
            if qa_history:
                history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_history])
                prompt = f"""Read the following passage and answer the question.
You may use the previous Q&A pairs as reference.

Passage:
{context}

Previous Q&A:
{history_str}

New Question: {question}

Answer (be concise):"""
            else:
                prompt = f"""Read the following passage and answer the question.

Passage:
{context}

Question: {question}

Answer (be concise):"""

            pred, response = client.generate(prompt, max_tokens=64)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            # Add to history
            qa_history.append((question, pred.strip()))

            is_correct = compute_contains(pred, gold_answer) > 0
            total_correct += int(is_correct)
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "correct": is_correct,
                "history_length": len(qa_history) - 1,
            })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="sequential_random",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"em": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_sequential_llm(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    ordering_client: Optional[LLMClient] = None,
) -> ExperimentResult:
    """Sequential (LLM-based): LLM determines optimal question order, then answers sequentially."""
    logger.info("Running Sequential (LLM-based) condition...")

    # Use same client for ordering if not specified
    if ordering_client is None:
        ordering_client = client

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for group in tqdm(groups, desc="Sequential-LLM"):
        context = group["context"]
        questions = group["questions"]

        # Step 1: Ask LLM to determine optimal order
        q_list = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(questions)])

        ordering_prompt = f"""Given the following passage and questions, determine the optimal order to answer them.
Consider which questions might provide useful information for answering subsequent questions.

Passage:
{context}

Questions:
{q_list}

Return ONLY a comma-separated list of question numbers in the optimal order (e.g., "2,1,3,4").
Optimal order:"""

        order_response, order_meta = ordering_client.generate(ordering_prompt, max_tokens=32)
        total_latency += order_meta.latency
        total_prompt_tokens += order_meta.prompt_tokens
        total_completion_tokens += order_meta.completion_tokens

        # Parse order
        ordered_indices = _parse_order(order_response, len(questions))
        ordered_questions = [questions[i] for i in ordered_indices]

        # Step 2: Answer questions in determined order
        qa_history = []

        for q_item in ordered_questions:
            question = q_item["question"]
            gold_answer = q_item["answer"]

            # Build prompt with Q&A history
            if qa_history:
                history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_history])
                prompt = f"""Read the following passage and answer the question.
You may use the previous Q&A pairs as reference.

Passage:
{context}

Previous Q&A:
{history_str}

New Question: {question}

Answer (be concise):"""
            else:
                prompt = f"""Read the following passage and answer the question.

Passage:
{context}

Question: {question}

Answer (be concise):"""

            pred, response = client.generate(prompt, max_tokens=64)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            # Add to history
            qa_history.append((question, pred.strip()))

            is_correct = compute_contains(pred, gold_answer) > 0
            total_correct += int(is_correct)
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "correct": is_correct,
                "history_length": len(qa_history) - 1,
                "llm_order": ordered_indices,
            })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="sequential_llm",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"em": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def _parse_batch_answers(response: str, n_questions: int) -> Dict[int, str]:
    """Parse batch answers from response like 'Q1: answer1, Q2: answer2'."""
    answers = {}

    # Try pattern: Q1: answer
    for i in range(n_questions):
        pattern = rf"Q{i+1}[:\s]+([^Q\n]+)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answers[i] = match.group(1).strip().rstrip(",")

    # Fallback: split by comma or newline if no Q patterns found
    if not answers:
        parts = re.split(r"[,\n]", response)
        for i, part in enumerate(parts[:n_questions]):
            clean = re.sub(r"^Q\d+[:\s]*", "", part.strip(), flags=re.IGNORECASE)
            answers[i] = clean

    return answers


def _parse_order(response: str, n_questions: int) -> List[int]:
    """Parse question order from LLM response like '2,1,3,4'."""
    # Extract numbers from response
    numbers = re.findall(r'\d+', response)

    # Convert to 0-indexed and validate
    indices = []
    seen = set()
    for num in numbers:
        idx = int(num) - 1  # Convert to 0-indexed
        if 0 <= idx < n_questions and idx not in seen:
            indices.append(idx)
            seen.add(idx)

    # Add any missing indices at the end
    for i in range(n_questions):
        if i not in seen:
            indices.append(i)

    return indices


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
        metrics={"em": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=all_details,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Exp 2a: Shared Context - SQuAD"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to use for inference"
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
        "--n-groups", type=int, default=-1,
        help="Number of context groups to evaluate (-1 for all)"
    )
    parser.add_argument(
        "--min-questions", type=int, default=3,
        help="Minimum questions per group"
    )
    parser.add_argument(
        "--max-questions", type=int, default=6,
        help="Maximum questions per group"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--conditions", type=str, default="independent,all_in_one,sequential_random,sequential_llm",
        help="Comma-separated list of conditions: independent,all_in_one,sequential_random,sequential_llm"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Setup distributed
    rank, world_size = setup_distributed()

    # Configuration
    config = ExperimentConfig(
        exp_name="exp2a_shared_context",
        dataset="squad",
        model=args.model,
        n_samples=args.n_groups,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Load data
    all_groups = load_squad_groups(
        n_groups=args.n_groups,
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        seed=args.seed,
    )

    # Shard data across GPUs
    groups = shard_data(all_groups, rank, world_size)
    logger.info(f"Rank {rank}/{world_size}: processing {len(groups)}/{len(all_groups)} groups")

    # Initialize LLM client
    client = LLMClient(
        model=args.model,
        use_local=args.use_local,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Run conditions
    conditions = [c.strip() for c in args.conditions.split(",")]
    local_results = []

    if "independent" in conditions:
        local_results.append(run_independent(groups, client))

    if "all_in_one" in conditions:
        local_results.append(run_all_in_one(groups, client))

    if "sequential_random" in conditions:
        local_results.append(run_sequential_random(groups, client, seed=args.seed))

    if "sequential_llm" in conditions:
        local_results.append(run_sequential_llm(groups, client))

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
