#!/usr/bin/env python3
"""Experiment 3: Format Similarity (Structural)

Dataset: ARC-Challenge (science multiple-choice questions)

Research Question: Do questions with the same format (multiple-choice) benefit
from being processed together?

Conditions:
- Oracle: Process all choice questions together with consistent format
- Method: Rule-based format detection and grouping
- Random: Mix with other question types or shuffle order

Metrics:
- Accuracy: Answer correctness
- Format Consistency: Output follows expected format (A/B/C/D)
- Answer Validity: Response contains a valid choice

Expected: Format Consistency: Oracle ≈ Method >> Random
"""

from __future__ import annotations

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
    print_summary,
    save_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_arc_challenge(
    n_samples: int = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load ARC-Challenge dataset.

    Each sample contains:
    - question: The question text
    - choices: List of choice texts
    - answerKey: The correct answer (A, B, C, D, or 1, 2, 3, 4)
    """
    logger.info("Loading ARC-Challenge dataset...")

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    samples = []
    for item in dataset:
        choices = item["choices"]
        choice_texts = choices["text"]
        choice_labels = choices["label"]

        # Build choice string
        choice_str = "\n".join([
            f"{label}. {text}"
            for label, text in zip(choice_labels, choice_texts)
        ])

        samples.append({
            "question": item["question"],
            "choices": choice_str,
            "choice_labels": choice_labels,
            "choice_texts": choice_texts,
            "answer_key": item["answerKey"],
            "id": item["id"],
        })

    # Shuffle and sample
    random.seed(seed)
    random.shuffle(samples)
    samples = samples[:n_samples]

    logger.info(f"Loaded {len(samples)} ARC-Challenge questions")

    return samples


def load_open_ended_questions(
    n_samples: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load open-ended questions (SQuAD) for mixing with choice questions."""
    logger.info("Loading open-ended questions for mixing...")

    dataset = load_dataset("squad", split="validation")

    samples = []
    for item in dataset:
        samples.append({
            "question": item["question"],
            "context": item["context"][:500],  # Truncate for efficiency
            "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
            "type": "open_ended",
        })

    random.seed(seed)
    random.shuffle(samples)

    return samples[:n_samples]


def _format_choice_prompt(question: str, choices: str) -> str:
    """Format a multiple-choice question prompt."""
    return f"""Answer the following multiple-choice question. Respond with ONLY the letter (A, B, C, or D).

Question: {question}

{choices}

Answer:"""


def _format_open_prompt(question: str, context: str) -> str:
    """Format an open-ended question prompt."""
    return f"""Answer the following question based on the context.

Context: {context}

Question: {question}

Answer:"""


def _extract_choice(response: str) -> Tuple[str, bool]:
    """Extract choice letter from response.

    Returns:
        Tuple of (extracted_choice, is_valid_format)
    """
    response = response.strip().upper()

    # Check for direct letter answer
    if response in ["A", "B", "C", "D"]:
        return response, True

    # Check for letter at start
    match = re.match(r"^([A-D])", response)
    if match:
        return match.group(1), True

    # Check for letter anywhere
    match = re.search(r"\b([A-D])\b", response)
    if match:
        return match.group(1), False  # Found but not clean format

    # Check for numeric (1, 2, 3, 4)
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
    for num, letter in num_to_letter.items():
        if num in response:
            return letter, False

    return "", False


def run_oracle(
    samples: List[Dict[str, Any]],
    client: LLMClient,
    batch_size: int = 4,
) -> ExperimentResult:
    """Oracle condition: All choice questions with consistent format."""
    logger.info("Running Oracle condition (consistent choice format)...")

    total_correct = 0
    total_valid_format = 0
    total_valid_answer = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for sample in tqdm(samples, desc="Oracle"):
        question = sample["question"]
        choices = sample["choices"]
        gold_answer = sample["answer_key"]

        prompt = _format_choice_prompt(question, choices)

        pred, response = client.generate(prompt, max_tokens=16)
        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        # Extract and evaluate
        extracted, is_valid_format = _extract_choice(pred)

        is_correct = extracted == gold_answer
        is_valid_answer = extracted in ["A", "B", "C", "D"]

        total_correct += int(is_correct)
        total_valid_format += int(is_valid_format)
        total_valid_answer += int(is_valid_answer)
        total_questions += 1

        details.append({
            "question": question[:100] + "...",
            "gold_answer": gold_answer,
            "raw_prediction": pred,
            "extracted": extracted,
            "correct": is_correct,
            "valid_format": is_valid_format,
            "valid_answer": is_valid_answer,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    format_consistency = total_valid_format / total_questions if total_questions > 0 else 0
    answer_validity = total_valid_answer / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="oracle",
        dataset="arc_challenge",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "accuracy": accuracy,
            "format_consistency": format_consistency,
            "answer_validity": answer_validity,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_method(
    samples: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Method condition: Rule-based format detection and grouping."""
    logger.info("Running Method condition (rule-based grouping)...")

    # In this case, Method ≈ Oracle since format detection for choice questions
    # is trivial (just check for A/B/C/D choices)
    # We simulate the "detection" step

    total_correct = 0
    total_valid_format = 0
    total_valid_answer = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Group by detected format (all are choice questions here)
    choice_questions = []
    for sample in samples:
        # Rule-based detection: has choices field
        if "choices" in sample and sample["choices"]:
            choice_questions.append(sample)

    logger.info(f"Detected {len(choice_questions)} choice questions")

    for sample in tqdm(choice_questions, desc="Method"):
        question = sample["question"]
        choices = sample["choices"]
        gold_answer = sample["answer_key"]

        prompt = _format_choice_prompt(question, choices)

        pred, response = client.generate(prompt, max_tokens=16)
        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        extracted, is_valid_format = _extract_choice(pred)

        is_correct = extracted == gold_answer
        is_valid_answer = extracted in ["A", "B", "C", "D"]

        total_correct += int(is_correct)
        total_valid_format += int(is_valid_format)
        total_valid_answer += int(is_valid_answer)
        total_questions += 1

        details.append({
            "question": question[:100] + "...",
            "gold_answer": gold_answer,
            "raw_prediction": pred,
            "extracted": extracted,
            "correct": is_correct,
            "valid_format": is_valid_format,
            "valid_answer": is_valid_answer,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    format_consistency = total_valid_format / total_questions if total_questions > 0 else 0
    answer_validity = total_valid_answer / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="method",
        dataset="arc_challenge",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "accuracy": accuracy,
            "format_consistency": format_consistency,
            "answer_validity": answer_validity,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_random(
    choice_samples: List[Dict[str, Any]],
    open_samples: List[Dict[str, Any]],
    client: LLMClient,
    seed: int = 42,
) -> ExperimentResult:
    """Random condition: Mix choice questions with open-ended questions."""
    logger.info("Running Random condition (mixed question types)...")

    random.seed(seed)

    # Interleave choice and open-ended questions
    all_samples = []
    for sample in choice_samples:
        sample["type"] = "choice"
        all_samples.append(sample)
    for sample in open_samples:
        sample["type"] = "open_ended"
        all_samples.append(sample)

    random.shuffle(all_samples)

    total_correct = 0
    total_valid_format = 0
    total_valid_answer = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for sample in tqdm(all_samples, desc="Random"):
        if sample["type"] == "choice":
            question = sample["question"]
            choices = sample["choices"]
            gold_answer = sample["answer_key"]

            # Use same prompt format but in mixed context
            prompt = _format_choice_prompt(question, choices)

            pred, response = client.generate(prompt, max_tokens=16)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            extracted, is_valid_format = _extract_choice(pred)

            is_correct = extracted == gold_answer
            is_valid_answer = extracted in ["A", "B", "C", "D"]

            total_correct += int(is_correct)
            total_valid_format += int(is_valid_format)
            total_valid_answer += int(is_valid_answer)
            total_questions += 1

            details.append({
                "type": "choice",
                "question": question[:100] + "...",
                "gold_answer": gold_answer,
                "raw_prediction": pred,
                "extracted": extracted,
                "correct": is_correct,
                "valid_format": is_valid_format,
                "valid_answer": is_valid_answer,
            })
        else:
            # Process open-ended question (not counted in choice metrics)
            question = sample["question"]
            context = sample["context"]

            prompt = _format_open_prompt(question, context)

            pred, response = client.generate(prompt, max_tokens=64)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            # Don't count open-ended in choice metrics
            details.append({
                "type": "open_ended",
                "question": question[:100] + "...",
                "prediction": pred,
            })

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    format_consistency = total_valid_format / total_questions if total_questions > 0 else 0
    answer_validity = total_valid_answer / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="random",
        dataset="arc_challenge",
        n_samples=len(choice_samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "accuracy": accuracy,
            "format_consistency": format_consistency,
            "answer_validity": answer_validity,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_random_order_only(
    samples: List[Dict[str, Any]],
    client: LLMClient,
    seed: int = 42,
) -> ExperimentResult:
    """Random condition (order only): Shuffle choice questions order."""
    logger.info("Running Random-Order condition (shuffled choice questions)...")

    random.seed(seed)
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)

    total_correct = 0
    total_valid_format = 0
    total_valid_answer = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for sample in tqdm(shuffled_samples, desc="Random-Order"):
        question = sample["question"]
        choices = sample["choices"]
        gold_answer = sample["answer_key"]

        prompt = _format_choice_prompt(question, choices)

        pred, response = client.generate(prompt, max_tokens=16)
        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        extracted, is_valid_format = _extract_choice(pred)

        is_correct = extracted == gold_answer
        is_valid_answer = extracted in ["A", "B", "C", "D"]

        total_correct += int(is_correct)
        total_valid_format += int(is_valid_format)
        total_valid_answer += int(is_valid_answer)
        total_questions += 1

        details.append({
            "question": question[:100] + "...",
            "gold_answer": gold_answer,
            "raw_prediction": pred,
            "extracted": extracted,
            "correct": is_correct,
            "valid_format": is_valid_format,
            "valid_answer": is_valid_answer,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    format_consistency = total_valid_format / total_questions if total_questions > 0 else 0
    answer_validity = total_valid_answer / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="random_order",
        dataset="arc_challenge",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "accuracy": accuracy,
            "format_consistency": format_consistency,
            "answer_validity": answer_validity,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Format Similarity - ARC-Challenge"
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
        "--n-samples", type=int, default=200,
        help="Number of choice samples to evaluate"
    )
    parser.add_argument(
        "--n-open-samples", type=int, default=100,
        help="Number of open-ended samples for mixing"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--conditions", type=str, default="oracle,method,random",
        help="Comma-separated list of conditions: oracle,method,random,random_order"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Configuration
    config = ExperimentConfig(
        exp_name="exp3_format_similarity",
        dataset="arc_challenge",
        model=args.model,
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Load data
    choice_samples = load_arc_challenge(n_samples=args.n_samples, seed=args.seed)

    # Initialize LLM client
    client = LLMClient(model=args.model, use_local=args.use_local)

    # Run conditions
    conditions = [c.strip() for c in args.conditions.split(",")]
    results = []

    if "oracle" in conditions:
        results.append(run_oracle(choice_samples, client))

    if "method" in conditions:
        results.append(run_method(choice_samples, client))

    if "random" in conditions:
        open_samples = load_open_ended_questions(n_samples=args.n_open_samples, seed=args.seed)
        results.append(run_random(choice_samples, open_samples, client, seed=args.seed))

    if "random_order" in conditions:
        results.append(run_random_order_only(choice_samples, client, seed=args.seed))

    # Print and save results
    print_summary(results)
    save_results(results, config)


if __name__ == "__main__":
    main()
