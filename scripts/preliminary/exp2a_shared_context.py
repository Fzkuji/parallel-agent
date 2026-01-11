#!/usr/bin/env python3
"""Experiment 2a: Shared Context (Semantic - Medium)

Dataset: SQuAD (multiple questions per paragraph)

Research Question: Do questions sharing the same context benefit from being
processed together in the same batch?

Conditions:
- Oracle: Group questions by paragraph (same paragraph = same batch)
- Random: Mix questions from different paragraphs in each batch

Expected: Oracle > Random (shared context enables information reuse)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_squad_groups(
    n_groups: int = 100,
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

    dataset = load_dataset("squad", split="validation")

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
    groups = valid_groups[:n_groups]

    logger.info(f"Loaded {len(groups)} groups with {min_questions}-{max_questions} questions each")

    return groups


def run_oracle(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    batch_size: int = 4,
) -> ExperimentResult:
    """Oracle condition: Process questions with their shared context."""
    logger.info("Running Oracle condition...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for group in tqdm(groups, desc="Oracle"):
        context = group["context"]
        questions = group["questions"]

        # Process all questions with shared context
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

            # Evaluate
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
        condition="oracle",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "exact_match": accuracy,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_random(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    batch_size: int = 4,
    seed: int = 42,
) -> ExperimentResult:
    """Random condition: Mix questions from different paragraphs."""
    logger.info("Running Random condition...")

    random.seed(seed)

    # Flatten all questions with their contexts
    all_questions = []
    for group in groups:
        context = group["context"]
        for q_item in group["questions"]:
            all_questions.append({
                "context": context,
                "question": q_item["question"],
                "answer": q_item["answer"],
            })

    # Shuffle questions
    random.shuffle(all_questions)

    # Create random batches (questions from different contexts)
    batches = []
    for i in range(0, len(all_questions), batch_size):
        batch = all_questions[i:i + batch_size]
        batches.append(batch)

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for batch in tqdm(batches, desc="Random"):
        # Process batch - each question needs its own context
        for q_item in batch:
            context = q_item["context"]
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

            # Evaluate
            is_correct = compute_contains(pred, gold_answer) > 0
            total_correct += int(is_correct)
            total_questions += 1

            details.append({
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "correct": is_correct,
                "batch_size": len(batch),
            })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="random",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "exact_match": accuracy,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_oracle_batch(
    groups: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Oracle-Batch: Process multiple questions from same context in one prompt."""
    logger.info("Running Oracle-Batch condition...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for group in tqdm(groups, desc="Oracle-Batch"):
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

        # Evaluate each question
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
        condition="oracle_batch",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "exact_match": accuracy,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_random_batch(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    batch_size: int = 4,
    seed: int = 42,
) -> ExperimentResult:
    """Random-Batch: Process questions from different contexts in one prompt."""
    logger.info("Running Random-Batch condition...")

    random.seed(seed)

    # Flatten all questions with their contexts
    all_questions = []
    for group in groups:
        context = group["context"]
        for q_item in group["questions"]:
            all_questions.append({
                "context": context,
                "question": q_item["question"],
                "answer": q_item["answer"],
            })

    # Shuffle and create batches
    random.shuffle(all_questions)
    batches = [all_questions[i:i + batch_size] for i in range(0, len(all_questions), batch_size)]

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for batch in tqdm(batches, desc="Random-Batch"):
        # Build multi-context, multi-question prompt
        prompt_parts = []
        for i, q_item in enumerate(batch):
            prompt_parts.append(f"Passage {i+1}:\n{q_item['context']}\n\nQ{i+1}: {q_item['question']}")

        prompt = "\n\n---\n\n".join(prompt_parts)
        prompt += "\n\nAnswer each question concisely. Format: Q1: [answer], Q2: [answer], ..."

        pred, response = client.generate(prompt, max_tokens=256)
        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        # Parse answers
        answers = _parse_batch_answers(pred, len(batch))

        # Evaluate each question
        for i, q_item in enumerate(batch):
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
        condition="random_batch",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "exact_match": accuracy,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def _parse_batch_answers(response: str, n_questions: int) -> Dict[int, str]:
    """Parse batch answers from response like 'Q1: answer1, Q2: answer2'."""
    import re

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
            # Remove Q1:, Q2:, etc. prefix if present
            clean = re.sub(r"^Q\d+[:\s]*", "", part.strip(), flags=re.IGNORECASE)
            answers[i] = clean

    return answers


def main():
    parser = argparse.ArgumentParser(
        description="Exp 2a: Shared Context - SQuAD"
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
        "--n-groups", type=int, default=100,
        help="Number of context groups to evaluate"
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
        "--batch-size", type=int, default=4,
        help="Batch size for batched conditions"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--conditions", type=str, default="oracle,random",
        help="Comma-separated list of conditions: oracle,random,oracle_batch,random_batch"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory for results"
    )
    args = parser.parse_args()

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
    groups = load_squad_groups(
        n_groups=args.n_groups,
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        seed=args.seed,
    )

    # Initialize LLM client
    client = LLMClient(
        model=args.model,
        use_local=args.use_local,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Run conditions
    conditions = [c.strip() for c in args.conditions.split(",")]
    results = []

    if "oracle" in conditions:
        results.append(run_oracle(groups, client, batch_size=args.batch_size))

    if "random" in conditions:
        results.append(run_random(groups, client, batch_size=args.batch_size, seed=args.seed))

    if "oracle_batch" in conditions:
        results.append(run_oracle_batch(groups, client))

    if "random_batch" in conditions:
        results.append(run_random_batch(groups, client, batch_size=args.batch_size, seed=args.seed))

    # Print and save results
    print_summary(results)
    save_results(results, config)


if __name__ == "__main__":
    main()
