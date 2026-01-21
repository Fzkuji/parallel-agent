#!/usr/bin/env python3
"""
Experiment 3: Task-Specific Context Dependency

Tests how different question types (answer formats) depend on shared context.

Research Question:
  Do different reasoning patterns exhibit different levels of context dependency?

Hypothesis:
  - Extractive QA (SQuAD): High context dependency
  - Numeric reasoning (DROP): Medium-high dependency
  - Open domain (TriviaQA): Medium dependency
  - Multiple choice (MMLU): Low-medium dependency
  - Math reasoning (GSM8K): Low dependency

Setup (Simplified):
  For each dataset, test 2 conditions:
    1. Batch: Independent answering (baseline)
    2. ICL (In-Context Learning): 5 questions sharing context, answered sequentially

Metrics:
  - Accuracy (Batch vs. ICL)
  - ICL Benefit = ICL - Batch

Usage:
  python exp3_task_dependency_5datasets.py --model Qwen/Qwen2.5-7B-Instruct --datasets squad drop triviaqa mmlu gsm8k
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import argparse
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.datasets.squad import load_squad_groups
from src.datasets.drop import load_drop_groups
from src.datasets.triviaqa import load_triviaqa
from src.datasets.mmlu import load_mmlu
from src.datasets.gsm8k import load_gsm8k
from src.models import Question
from src.strategies.sequential import run_sequential_strategy
from src.strategies.independent import run_independent_strategy


def load_dataset_by_name(dataset_name: str, split: str = "validation", max_contexts: int = 100):
    """Load dataset and format for experiments."""

    if dataset_name == "squad":
        data = load_squad_groups(
            split=split,
            max_contexts=max_contexts,
            min_questions=5,
            max_questions=5,
            seed=42
        )
        return data, "squad"

    elif dataset_name == "drop":
        data = load_drop_groups(
            split=split,
            max_contexts=max_contexts,
            min_questions=5,
            max_questions=5,
            seed=42
        )
        return data, "drop"

    elif dataset_name == "triviaqa":
        # TriviaQA: 1 question per context (open-domain)
        # Need more contexts since only 1 question each
        data = load_triviaqa(
            split=split,
            max_contexts=max_contexts * 5,
            min_questions=1,
            max_questions=1,
            seed=42
        )
        return data, "triviaqa"

    elif dataset_name == "mmlu":
        data = load_mmlu(
            split=split,
            max_contexts=max_contexts,
            min_questions=5,
            max_questions=5,
            seed=42
        )
        return data, "mmlu"

    elif dataset_name == "gsm8k":
        # Use 'test' split for GSM8K
        gsm_split = "test" if split == "validation" else split
        data = load_gsm8k(
            split=gsm_split,
            max_contexts=max_contexts,
            min_questions=5,
            max_questions=5,
            seed=42
        )
        return data, "gsm8k"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_condition(
    data: List[Dict],
    condition: str,
    tokenizer,
    model,
    max_new_tokens: int,
    dataset: str,
) -> Dict[str, Any]:
    """
    Run one experimental condition.

    Args:
        data: List of contexts with questions
        condition: "batch" or "icl"
        tokenizer: Model tokenizer
        model: Language model
        max_new_tokens: Max tokens to generate
        dataset: Dataset name for evaluation

    Returns:
        Result dictionary with metrics
    """

    if condition == "batch":
        # Independent answering - each question answered separately
        all_results = []
        for context_data in data:
            background = context_data["context"]
            questions = [
                Question(
                    qid=q["qid"],
                    text=q["question"],
                    priority=1.0,
                    answer_tokens=q.get("answer_tokens", 32),
                    type_hint=None,
                    references=q.get("references", [])
                )
                for q in context_data["questions"]
            ]

            result = run_independent_strategy(
                background=background,
                questions=questions,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens,
                strategy_name="batch",
                dataset=dataset,
            )
            all_results.append(result)

        # Aggregate metrics
        total_correct = sum(r.metrics.get("correct", 0) for r in all_results)
        total_questions = sum(r.metrics.get("total", 0) for r in all_results)
        accuracy = total_correct / total_questions if total_questions > 0 else 0.0

        return {
            "condition": condition,
            "accuracy": accuracy,
            "total_questions": total_questions,
            "total_correct": total_correct,
        }

    elif condition == "icl":
        # In-Context Learning: Sequential answering with shared context
        all_results = []
        for context_data in data:
            background = context_data["context"]
            questions = [
                Question(
                    qid=q["qid"],
                    text=q["question"],
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
                strategy_name="icl",
                dataset=dataset,
            )
            all_results.append(result)

        # Aggregate metrics
        total_correct = sum(r.metrics.get("correct", 0) for r in all_results)
        total_questions = sum(r.metrics.get("total", 0) for r in all_results)
        accuracy = total_correct / total_questions if total_questions > 0 else 0.0

        return {
            "condition": condition,
            "accuracy": accuracy,
            "total_questions": total_questions,
            "total_correct": total_correct,
        }

    else:
        raise ValueError(f"Unknown condition: {condition}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Task-Specific Context Dependency")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model to use")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["squad", "drop", "triviaqa", "mmlu", "gsm8k"],
                       help="Datasets to test (squad, drop, triviaqa, mmlu, gsm8k)")
    parser.add_argument("--max-contexts", type=int, default=100,
                       help="Maximum contexts per dataset")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                       help="Maximum tokens to generate")
    parser.add_argument("--output-dir", type=str,
                       default="outputs/preliminary/exp3",
                       help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run experiments for each dataset
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}\n")

        # Load dataset
        print(f"Loading {dataset_name} data...")
        data, dataset_key = load_dataset_by_name(
            dataset_name,
            split="validation",
            max_contexts=args.max_contexts
        )
        print(f"Loaded {len(data)} contexts\n")

        # Test each condition
        dataset_results = {}
        conditions = ["batch", "icl"]

        for condition in conditions:
            print(f"\nRunning condition: {condition}")
            result = run_condition(
                data=data,
                condition=condition,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
                dataset=dataset_key,
            )

            dataset_results[condition] = result
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Correct: {result['total_correct']}/{result['total_questions']}")

        # Compute derived metrics
        batch_acc = dataset_results["batch"]["accuracy"]
        icl_acc = dataset_results["icl"]["accuracy"]

        icl_benefit = icl_acc - batch_acc

        dataset_results["metrics"] = {
            "icl_benefit": icl_benefit,
        }

        print(f"\n{'='*40}")
        print(f"Summary for {dataset_name}:")
        print(f"  Batch: {batch_acc:.4f}")
        print(f"  ICL:   {icl_acc:.4f}")
        print(f"  ICL Benefit: {icl_benefit:+.4f}")
        print(f"{'='*40}\n")

        all_results[dataset_name] = dataset_results

    # Save results
    output_file = output_dir / "exp3_results_5datasets.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": args.model,
            "datasets": args.datasets,
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Batch':>10} {'ICL':>10} {'Benefit':>10}")
    print(f"{'-'*60}")

    for dataset_name in args.datasets:
        r = all_results[dataset_name]
        print(f"{dataset_name:<20} "
              f"{r['batch']['accuracy']:>9.2%} "
              f"{r['icl']['accuracy']:>9.2%} "
              f"{r['metrics']['icl_benefit']:>+9.2%}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
