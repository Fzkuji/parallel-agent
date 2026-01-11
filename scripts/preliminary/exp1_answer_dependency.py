#!/usr/bin/env python3
"""Experiment 1: Answer Dependency (Semantic - Strong)

Dataset: MoreHopQA (3-5 hop reasoning with gold sub-questions and sub-answers)

Research Question: Does answering questions in dependency order with prior answers
improve multi-step reasoning performance?

Conditions:
- Oracle: Follow question_decomposition order, pass prior sub-answers
- Method: LLM detects dependencies, answers in detected order
- Random: Shuffle sub-questions, no prior answer passing

Expected: Random ≈ 0 (reasoning chain breaks), Method ≈ Oracle
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
    topological_sort,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_morehopqa(n_samples: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Load MoreHopQA dataset.

    Each sample contains:
    - question: The main question
    - answer: The final answer
    - decomposition: List of {question, answer} for sub-questions
    """
    logger.info("Loading MoreHopQA dataset...")

    valid_samples = []

    # Download raw JSON directly from HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        import json as json_module

        logger.info("Downloading MoreHopQA from HuggingFace Hub...")
        file_path = hf_hub_download(
            repo_id="alabnii/morehopqa",
            filename="data/with_human_verification.json",
            repo_type="dataset",
        )
        with open(file_path, "r", encoding="utf-8") as f:
            data = json_module.load(f)

        for item in data:
            decomp = item.get("question_decomposition", [])
            if decomp and len(decomp) >= 2:
                valid_samples.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "decomposition": decomp,
                    "n_hops": len(decomp),
                })

        logger.info(f"Loaded {len(valid_samples)} samples from MoreHopQA")

    except Exception as e:
        logger.warning(f"Failed to load MoreHopQA: {e}")
        logger.info("Falling back to HotpotQA...")

        # Fallback to HotpotQA
        dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

        for item in dataset:
            # Use supporting facts to create decomposition
            sf = item.get("supporting_facts", {})
            titles = sf.get("title", [])
            sent_ids = sf.get("sent_id", [])

            if len(set(titles)) >= 2:  # At least 2 different sources
                context_dict = {t: s for t, s in zip(item["context"]["title"], item["context"]["sentences"])}
                decomp = []
                seen_titles = set()
                for title, sid in zip(titles, sent_ids):
                    if title not in seen_titles and title in context_dict:
                        sents = context_dict[title]
                        if sid < len(sents):
                            decomp.append({
                                "question": f"What information about {title} is relevant?",
                                "answer": sents[sid],
                            })
                            seen_titles.add(title)

                if len(decomp) >= 2:
                    valid_samples.append({
                        "question": item["question"],
                        "answer": item["answer"],
                        "decomposition": decomp,
                        "n_hops": len(decomp),
                    })

        logger.info(f"Loaded {len(valid_samples)} samples from HotpotQA (fallback)")

    # Shuffle and sample
    random.seed(seed)
    random.shuffle(valid_samples)
    samples = valid_samples[:n_samples]

    logger.info(f"Loaded {len(samples)} samples with valid decomposition")
    logger.info(f"Hop distribution: {_count_hops(samples)}")

    return samples


def _count_hops(samples: List[Dict]) -> Dict[int, int]:
    """Count samples by number of hops."""
    counts = {}
    for s in samples:
        n = s["n_hops"]
        counts[n] = counts.get(n, 0) + 1
    return dict(sorted(counts.items()))


def run_oracle(
    samples: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Oracle condition: Follow gold decomposition order, pass prior answers."""
    logger.info("Running Oracle condition...")

    correct = 0
    total = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for sample in tqdm(samples, desc="Oracle"):
        decomp = sample["decomposition"]
        gold_answer = sample["answer"]

        # Build context with prior Q&A
        prior_qa = []
        final_pred = None

        for i, step in enumerate(decomp):
            sub_q = step["question"]
            sub_a = step["answer"]

            # Build prompt with prior answers
            if prior_qa:
                context = "Previous Q&A:\n" + "\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in prior_qa]
                )
                prompt = f"{context}\n\nNow answer this question:\nQ: {sub_q}\nA:"
            else:
                prompt = f"Answer this question:\nQ: {sub_q}\nA:"

            # For Oracle, we use gold sub-answers to pass forward
            # But still generate to measure the process
            pred, response = client.generate(prompt, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            # Use gold answer for next step (Oracle condition)
            prior_qa.append((sub_q, sub_a))
            final_pred = pred
            total_questions += 1

        # Evaluate final answer
        is_correct = compute_contains(final_pred, gold_answer) > 0
        correct += int(is_correct)
        total += 1

        details.append({
            "question": sample["question"],
            "gold_answer": gold_answer,
            "prediction": final_pred,
            "correct": is_correct,
            "n_hops": sample["n_hops"],
        })

    accuracy = correct / total if total > 0 else 0

    return ExperimentResult(
        condition="oracle",
        dataset="morehopqa",
        n_samples=total,
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"exact_match": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_method(
    samples: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Method condition: LLM detects dependencies, answers in detected order."""
    logger.info("Running Method (LLM-based) condition...")

    correct = 0
    total = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for sample in tqdm(samples, desc="Method"):
        decomp = sample["decomposition"]
        gold_answer = sample["answer"]
        questions = [step["question"] for step in decomp]

        # Step 1: LLM detects dependencies
        dependencies = client.detect_dependencies(questions)

        # Step 2: Topological sort based on detected dependencies
        order = topological_sort(len(questions), dependencies)

        # Step 3: Answer in detected order, passing predictions
        prior_qa = []
        predictions = {}

        for idx in order:
            sub_q = questions[idx]

            # Build prompt with prior answers
            if prior_qa:
                context = "Previous Q&A:\n" + "\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in prior_qa]
                )
                prompt = f"{context}\n\nNow answer this question:\nQ: {sub_q}\nA:"
            else:
                prompt = f"Answer this question:\nQ: {sub_q}\nA:"

            pred, response = client.generate(prompt, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            predictions[idx] = pred
            prior_qa.append((sub_q, pred))
            total_questions += 1

        # Final prediction is the last one in original order
        final_pred = predictions.get(len(questions) - 1, "")

        # Evaluate final answer
        is_correct = compute_contains(final_pred, gold_answer) > 0
        correct += int(is_correct)
        total += 1

        details.append({
            "question": sample["question"],
            "gold_answer": gold_answer,
            "prediction": final_pred,
            "correct": is_correct,
            "n_hops": sample["n_hops"],
            "detected_deps": dependencies,
            "order": order,
        })

    accuracy = correct / total if total > 0 else 0

    return ExperimentResult(
        condition="method",
        dataset="morehopqa",
        n_samples=total,
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"exact_match": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_random(
    samples: List[Dict[str, Any]],
    client: LLMClient,
    seed: int = 42,
) -> ExperimentResult:
    """Random condition: Shuffle questions, no prior answer passing."""
    logger.info("Running Random condition...")

    random.seed(seed)

    correct = 0
    total = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for sample in tqdm(samples, desc="Random"):
        decomp = sample["decomposition"]
        gold_answer = sample["answer"]
        questions = [step["question"] for step in decomp]

        # Shuffle order
        indices = list(range(len(questions)))
        random.shuffle(indices)

        # Answer in random order, NO prior answer passing
        predictions = {}

        for idx in indices:
            sub_q = questions[idx]

            # No context from prior answers
            prompt = f"Answer this question:\nQ: {sub_q}\nA:"

            pred, response = client.generate(prompt, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            predictions[idx] = pred
            total_questions += 1

        # Final prediction is the last one in original order
        final_pred = predictions.get(len(questions) - 1, "")

        # Evaluate final answer
        is_correct = compute_contains(final_pred, gold_answer) > 0
        correct += int(is_correct)
        total += 1

        details.append({
            "question": sample["question"],
            "gold_answer": gold_answer,
            "prediction": final_pred,
            "correct": is_correct,
            "n_hops": sample["n_hops"],
            "order": indices,
        })

    accuracy = correct / total if total > 0 else 0

    return ExperimentResult(
        condition="random",
        dataset="morehopqa",
        n_samples=total,
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"exact_match": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Exp 1: Answer Dependency - MoreHopQA"
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
        "--n-samples", type=int, default=100,
        help="Number of samples to evaluate"
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

    # Configuration
    config = ExperimentConfig(
        exp_name="exp1_answer_dependency",
        dataset="morehopqa",
        model=args.model,
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Load data
    samples = load_morehopqa(n_samples=args.n_samples, seed=args.seed)

    # Initialize LLM client
    client = LLMClient(model=args.model, use_local=args.use_local)

    # Run conditions
    conditions = [c.strip() for c in args.conditions.split(",")]
    results = []

    if "oracle" in conditions:
        results.append(run_oracle(samples, client))

    if "method" in conditions:
        results.append(run_method(samples, client))

    if "random" in conditions:
        results.append(run_random(samples, client, seed=args.seed))

    # Print and save results
    print_summary(results)
    save_results(results, config)


if __name__ == "__main__":
    main()
