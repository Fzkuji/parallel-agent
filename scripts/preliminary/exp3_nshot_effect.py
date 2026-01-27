#!/usr/bin/env python3
"""
Experiment 3: N-Shot Effect Analysis

Tests how the number of consecutive questions (N-shot) affects model performance
across different question types.

Research Question:
  Does answering multiple questions sequentially improve performance through
  context accumulation and in-context learning?

Setup:
  For each dataset, test N = 0, 1, 2, 3, 4, 5 shot configurations:
    - 0-shot: Independent answering (baseline, batch mode)
    - N-shot: Answer N questions sequentially with context accumulation

Datasets and their characteristics:
  - SQuAD: Extractive QA, short phrase answers, shared context
  - DROP: Numeric reasoning, number answers, shared context
  - TriviaQA: Open domain QA, entity answers, no shared context
  - MMLU: Multiple choice, letter answers (A/B/C/D), grouped by subject
  - GSM8K: Math reasoning, numeric answers, independent problems

Metrics:
  - Overall Accuracy by N-shot
  - Per-position accuracy within N-shot sessions
  - N-shot benefit = N-shot_acc - 0-shot_acc

Usage:
  # Single GPU:
  python exp3_nshot_effect.py --model Qwen/Qwen2.5-7B-Instruct --datasets squad drop

  # Multi-GPU with DDP (e.g., 4 GPUs):
  torchrun --nproc_per_node=4 exp3_nshot_effect.py --model Qwen/Qwen2.5-7B-Instruct --datasets squad drop
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import argparse
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.preliminary.utils import (
    setup_distributed, cleanup_distributed, shard_data, gather_results
)
from src.datasets.squad import load_squad_groups
from src.datasets.drop import load_drop_groups
from src.datasets.triviaqa import load_triviaqa_groups
from src.datasets.mmlu import load_mmlu
from src.datasets.gsm8k import load_gsm8k
from src.models import Question
from src.strategies import run_sequential_strategy
from src.prompts import build_single_prompt, MULTIPLE_CHOICE_DATASETS
from src.inference import extract_answer
from src.evaluation import evaluate_predictions


def load_dataset_for_nshot(
    dataset_name: str,
    split: str,
    max_contexts: int,
    n_shot: int,
    seed: int = 42,
) -> Tuple[List[Dict], str]:
    """
    Load dataset with exactly n_shot questions per context.

    For n_shot=0 (batch mode), we load single questions per context
    to measure baseline performance.

    Args:
        dataset_name: Name of the dataset
        split: Dataset split ('validation', 'test', etc.)
        max_contexts: Maximum number of contexts to load
        n_shot: Number of questions per context (0 = single question batch)
        seed: Random seed

    Returns:
        Tuple of (data, dataset_key)
    """
    # For 0-shot, we still need at least 1 question per context
    effective_n = max(1, n_shot)

    if dataset_name == "squad":
        data = load_squad_groups(
            split=split,
            max_contexts=max_contexts,
            min_questions=effective_n,
            max_questions=effective_n,
            seed=seed
        )
        return data, "squad"

    elif dataset_name == "drop":
        data = load_drop_groups(
            split=split,
            max_contexts=max_contexts,
            min_questions=effective_n,
            max_questions=effective_n,
            seed=seed
        )
        return data, "drop"

    elif dataset_name == "triviaqa":
        # TriviaQA: open-domain, group random questions together
        data = load_triviaqa_groups(
            split=split,
            max_groups=max_contexts,
            min_questions=effective_n,
            max_questions=effective_n,
            seed=seed
        )
        return data, "triviaqa"

    elif dataset_name == "mmlu":
        data = load_mmlu(
            split=split,
            max_contexts=max_contexts,
            min_questions=effective_n,
            max_questions=effective_n,
            seed=seed
        )
        return data, "mmlu"

    elif dataset_name == "gsm8k":
        # GSM8K uses 'test' split for validation
        gsm_split = "test" if split == "validation" else split
        data = load_gsm8k(
            split=gsm_split,
            max_contexts=max_contexts,
            min_questions=effective_n,
            max_questions=effective_n,
            seed=seed
        )
        return data, "gsm8k"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_batched_0shot(
    data: List[Dict],
    tokenizer,
    model,
    max_new_tokens: int,
    dataset: str,
    batch_size: int = 8,
    is_main: bool = True,
) -> Dict[str, Any]:
    """
    Run 0-shot with batched parallel inference.

    Collects all questions across contexts and processes them in batches.
    """
    # Collect all questions with their contexts
    all_items = []
    question_lookup = {}

    for ctx_idx, context_data in enumerate(data):
        background = context_data["context"]
        for q_data in context_data["questions"]:
            qid = f"ctx{ctx_idx}_{q_data['qid']}"
            question = Question(
                qid=qid,
                text=q_data.get("text", q_data.get("question", "")),
                priority=1.0,
                answer_tokens=q_data.get("answer_tokens", 32),
                type_hint=None,
                references=q_data.get("references", [])
            )
            question_lookup[qid] = question
            all_items.append({
                "qid": qid,
                "question": question,
                "background": background,
            })

    if not all_items:
        return {
            "n_shot": 0,
            "accuracy": 0.0,
            "total_questions": 0,
            "total_correct": 0,
            "position_metrics": {},
        }

    # Build all prompts
    all_prompts = []
    for item in all_items:
        system_prompt, user_prompt = build_single_prompt(
            item["background"], item["question"], dataset
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)

    # Batch inference
    answer_records = {}
    total_batches = (len(all_prompts) + batch_size - 1) // batch_size

    pbar = tqdm(total=total_batches, desc="0-shot batch", disable=not is_main)

    for batch_start in range(0, len(all_prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_items = all_items[batch_start:batch_end]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and extract answers
        for i, item in enumerate(batch_items):
            input_len = inputs["input_ids"][i].shape[0]
            generated = outputs[i][input_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True)

            answer, valid = extract_answer(response, dataset)
            answer_records[item["qid"]] = (answer, valid)

        pbar.update(1)

    pbar.close()

    # Evaluate
    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    total_questions = len(answer_records)
    total_correct = int(round(metrics.get("strict_acc", 0) * total_questions))

    return {
        "n_shot": 0,
        "accuracy": metrics.get("strict_acc", 0),
        "total_questions": total_questions,
        "total_correct": total_correct,
        "position_metrics": {},
    }


def run_nshot_condition(
    data: List[Dict],
    n_shot: int,
    tokenizer,
    model,
    max_new_tokens: int,
    dataset: str,
    batch_size: int = 8,
    is_main: bool = True,
) -> Dict[str, Any]:
    """
    Run N-shot experiment condition.

    Args:
        data: List of contexts, each with n_shot questions
        n_shot: Number of questions per session (0 = batch mode)
        tokenizer: Model tokenizer
        model: Language model
        max_new_tokens: Max tokens to generate
        dataset: Dataset name for evaluation
        batch_size: Batch size for 0-shot inference
        is_main: Whether this is the main process (for progress bar)

    Returns:
        Result dictionary with aggregated metrics
    """
    # 0-shot: Use batched parallel inference
    if n_shot == 0:
        return run_batched_0shot(
            data, tokenizer, model, max_new_tokens, dataset, batch_size, is_main
        )

    # N-shot: Sequential processing with context accumulation
    position_correct = {}
    position_total = {}
    all_correct = 0
    all_total = 0

    pbar = tqdm(data, desc=f"{n_shot}-shot", disable=not is_main)

    for context_data in pbar:
        background = context_data["context"]
        questions = [
            Question(
                qid=q["qid"],
                text=q.get("text", q.get("question", "")),
                priority=1.0,
                answer_tokens=q.get("answer_tokens", 32),
                type_hint=None,
                references=q.get("references", [])
            )
            for q in context_data["questions"]
        ]

        result = run_sequential_strategy(
            background=background,
            questions=questions,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            dataset=dataset,
        )

        # Aggregate metrics
        n_questions = len(questions)
        n_correct = int(round(result.metrics.get("strict_acc", 0) * n_questions))
        all_correct += n_correct
        all_total += n_questions

        # Track per-position metrics
        if "turns" in result.details:
            for i, turn in enumerate(result.details["turns"]):
                pos = i + 1
                if pos not in position_correct:
                    position_correct[pos] = 0
                    position_total[pos] = 0
                position_total[pos] += 1
                if turn.get("strict_valid", False):
                    position_correct[pos] += 1

    # Compute position-wise accuracy
    position_metrics = {}
    for pos in sorted(position_correct.keys()):
        if position_total[pos] > 0:
            position_metrics[pos] = {
                "accuracy": position_correct[pos] / position_total[pos],
                "correct": position_correct[pos],
                "total": position_total[pos],
            }

    accuracy = all_correct / all_total if all_total > 0 else 0.0

    return {
        "n_shot": n_shot,
        "accuracy": accuracy,
        "total_questions": all_total,
        "total_correct": all_correct,
        "position_metrics": position_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: N-Shot Effect Analysis")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model to use")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["squad", "drop", "triviaqa", "mmlu", "gsm8k"],
                       help="Datasets to test")
    parser.add_argument("--shots", type=int, nargs="+",
                       default=[0, 1, 2, 3, 4, 5],
                       help="N-shot values to test (0 = batch mode)")
    parser.add_argument("--max-contexts", type=int, default=100,
                       help="Maximum contexts per dataset per shot")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for 0-shot inference")
    parser.add_argument("--output-dir", type=str,
                       default="outputs/preliminary/exp3",
                       help="Output directory")

    args = parser.parse_args()

    # Setup distributed (DDP)
    rank, world_size = setup_distributed()
    is_main = (rank == 0)

    # Create output directory (only on rank 0)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device for this rank
    if torch.cuda.is_available():
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        device = "cpu"

    # Load model
    if is_main:
        print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    if is_main:
        print(f"Model loaded on {device} (rank {rank}/{world_size})")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure shots are sorted
    shots = sorted(args.shots)
    max_shot = max(shots)

    # Run experiments for each dataset
    all_results = {}

    for dataset_name in args.datasets:
        if is_main:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}\n")

        dataset_results = {}

        for n_shot in shots:
            if is_main:
                print(f"\n--- Testing {n_shot}-shot ---")

            # Load dataset with appropriate question count
            # For fair comparison, load same number of contexts for each shot
            if is_main:
                print(f"Loading {dataset_name} data for {n_shot}-shot...")

            data, dataset_key = load_dataset_for_nshot(
                dataset_name,
                split="validation",
                max_contexts=args.max_contexts,
                n_shot=n_shot,
                seed=42
            )

            total_contexts = len(data)
            if is_main:
                print(f"Loaded {total_contexts} contexts")

            # Shard data across ranks
            local_data = shard_data(data, rank, world_size)
            if is_main:
                print(f"Rank {rank} processing {len(local_data)} contexts")

            # Run condition
            local_result = run_nshot_condition(
                data=local_data,
                n_shot=n_shot,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
                dataset=dataset_key,
                batch_size=args.batch_size,
                is_main=is_main,
            )

            # Gather results from all ranks
            local_metrics = {
                "total_correct": local_result["total_correct"],
                "total_questions": local_result["total_questions"],
                "position_metrics": local_result["position_metrics"],
            }
            all_metrics = gather_results([local_metrics], world_size)

            # Aggregate across all ranks
            total_correct = sum(m["total_correct"] for m in all_metrics)
            total_questions = sum(m["total_questions"] for m in all_metrics)
            accuracy = total_correct / total_questions if total_questions > 0 else 0.0

            # Aggregate position metrics
            combined_position = {}
            for m in all_metrics:
                for pos, pm in m["position_metrics"].items():
                    if pos not in combined_position:
                        combined_position[pos] = {"correct": 0, "total": 0}
                    combined_position[pos]["correct"] += pm["correct"]
                    combined_position[pos]["total"] += pm["total"]

            for pos in combined_position:
                combined_position[pos]["accuracy"] = (
                    combined_position[pos]["correct"] / combined_position[pos]["total"]
                    if combined_position[pos]["total"] > 0 else 0.0
                )

            result = {
                "n_shot": n_shot,
                "accuracy": accuracy,
                "total_questions": total_questions,
                "total_correct": total_correct,
                "position_metrics": combined_position,
            }

            dataset_results[str(n_shot)] = result

            if is_main:
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  Correct: {result['total_correct']}/{result['total_questions']}")
                if combined_position:
                    print(f"  Position accuracy:")
                    for pos in sorted(combined_position.keys()):
                        pm = combined_position[pos]
                        print(f"    Position {pos}: {pm['accuracy']:.4f} ({pm['correct']}/{pm['total']})")

        # Compute N-shot benefits
        baseline_acc = dataset_results["0"]["accuracy"]
        for n_shot_str, result in dataset_results.items():
            result["benefit"] = result["accuracy"] - baseline_acc

        all_results[dataset_name] = dataset_results

        if is_main:
            print(f"\n{'='*40}")
            print(f"Summary for {dataset_name}:")
            print(f"{'='*40}")
            print(f"{'N-shot':<10} {'Accuracy':>10} {'Benefit':>10}")
            print(f"{'-'*40}")
            for n_shot in shots:
                r = dataset_results[str(n_shot)]
                print(f"{n_shot:<10} {r['accuracy']:>9.2%} {r['benefit']:>+9.2%}")
            print(f"{'='*40}\n")

    # Save results (only on rank 0)
    if is_main:
        output_file = output_dir / "exp3_nshot_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "model": args.model,
                "datasets": args.datasets,
                "shots": shots,
                "max_contexts": args.max_contexts,
                "results": all_results,
            }, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Print final summary table
        print(f"\n{'='*70}")
        print("FINAL SUMMARY TABLE")
        print(f"{'='*70}")

        # Header
        header = f"{'Dataset':<12}"
        for n_shot in shots:
            header += f" {n_shot}-shot".rjust(10)
        print(header)
        print(f"{'-'*70}")

        # Data rows
        for dataset_name in args.datasets:
            row = f"{dataset_name:<12}"
            for n_shot in shots:
                acc = all_results[dataset_name][str(n_shot)]["accuracy"]
                row += f" {acc:>9.2%}"
            print(row)

        print(f"{'='*70}")

        # Benefit summary
        print(f"\nN-shot Benefit (vs 0-shot):")
        print(f"{'-'*70}")
        header = f"{'Dataset':<12}"
        for n_shot in shots[1:]:  # Skip 0-shot
            header += f" {n_shot}-shot".rjust(10)
        print(header)
        print(f"{'-'*70}")

        for dataset_name in args.datasets:
            row = f"{dataset_name:<12}"
            for n_shot in shots[1:]:
                benefit = all_results[dataset_name][str(n_shot)]["benefit"]
                row += f" {benefit:>+9.2%}"
            print(row)

        print(f"{'='*70}\n")

    # Cleanup distributed
    cleanup_distributed()


if __name__ == "__main__":
    main()
