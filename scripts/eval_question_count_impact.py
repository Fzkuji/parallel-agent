#!/usr/bin/env python3
"""
Evaluate the impact of question count per context on model performance.

Tests different question counts (1, 5, 10, 20) to understand how performance
scales with the number of questions per context.

For each question count:
1. Evaluate pretrained baseline (all strategies)
2. Train SFT-LoRA models (batch and sequential)
3. Evaluate SFT-LoRA models

Usage:
    python scripts/eval_question_count_impact.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset squad \
        --eval-samples 100 \
        --train-samples 1000 \
        --num-gpus 8
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate question count impact")

    # Model and dataset
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="squad",
                       choices=["squad", "hotpot", "quac", "drop", "triviaqa", "quality", "cmb"])
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=1000)

    # Question counts to test
    parser.add_argument("--question-counts", type=str, default="1,5,10,20",
                       help="Comma-separated list of question counts to test")

    # Hardware
    parser.add_argument("--num-gpus", type=int, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/question_count_study")
    parser.add_argument("--skip-pretrained", action="store_true", help="Skip pretrained evaluation")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT training")

    return parser.parse_args()


def run_pretrained_eval(args, question_count: int, output_dir: Path) -> Dict:
    """Run baseline_pretrained.py for a specific question count."""
    logger.info(f"\n{'='*80}")
    logger.info(f"PRETRAINED EVALUATION: {question_count} questions/context")
    logger.info(f"{'='*80}")

    result_file = output_dir / f"pretrained_q{question_count}.json"

    # Check if already exists
    if result_file.exists():
        logger.info(f"Loading existing results from {result_file}")
        with open(result_file, 'r') as f:
            return json.load(f)

    cmd = [
        sys.executable, "scripts/baseline_pretrained.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--eval-samples", str(args.eval_samples),
        "--min-questions", str(question_count),
        "--max-questions", str(question_count),
        "--strategies", "all_in_one,sequential,batch,collab_llm",
        "--seed", str(args.seed),
    ]

    if args.num_gpus:
        cmd.extend(["--num-gpus", str(args.num_gpus)])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Pretrained evaluation failed for {question_count} questions")
        return {}

    # Load results from the script's output directory
    baseline_results_file = Path("outputs/eval_baselines") / f"results_{args.dataset}.json"
    if baseline_results_file.exists():
        with open(baseline_results_file, 'r') as f:
            results = json.load(f)
        # Save a copy with question count in name
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {result_file}")
        return results
    else:
        logger.warning(f"Results file not found: {baseline_results_file}")

    return {}


def run_sft_training(args, question_count: int, output_dir: Path) -> Dict:
    """Run baseline_sft.py for a specific question count."""
    logger.info(f"\n{'='*80}")
    logger.info(f"SFT TRAINING & EVALUATION: {question_count} questions/context")
    logger.info(f"{'='*80}")

    result_file = output_dir / f"sft_q{question_count}.json"

    # Check if already exists
    if result_file.exists():
        logger.info(f"Loading existing results from {result_file}")
        with open(result_file, 'r') as f:
            return json.load(f)

    cmd = [
        sys.executable, "scripts/baseline_sft.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--train-samples", str(args.train_samples),
        "--eval-samples", str(args.eval_samples),
        "--min-questions", str(question_count),
        "--max-questions", str(question_count),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--train-format", "all",  # Train both formats
    ]

    if args.num_gpus:
        cmd.extend(["--num-gpus", str(args.num_gpus)])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"SFT training failed for {question_count} questions")
        return {}

    # Try to load results from the script's output directory
    sft_results_file = Path("outputs/sft_lora") / f"sft_lora_results_{args.dataset}.json"
    if sft_results_file.exists():
        with open(sft_results_file, 'r') as f:
            results = json.load(f)
        # Save a copy with question count in name
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {result_file}")
        return results

    return {}


def generate_summary_table(args, results_by_count: Dict[int, Dict], output_dir: Path):
    """Generate summary table comparing all question counts."""
    logger.info(f"\n{'='*120}")
    logger.info("SUMMARY: Performance vs Question Count")
    logger.info(f"{'='*120}")

    summary_file = output_dir / "summary.txt"

    with open(summary_file, 'w') as f:
        f.write(f"# Question Count Impact Study\n")
        f.write(f"# Dataset: {args.dataset}\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Eval Samples: {args.eval_samples}\n")
        f.write(f"# Train Samples: {args.train_samples}\n")
        f.write(f"# Date: 2026-01-24\n\n")

        # Pretrained Results - EM
        f.write("## Pretrained Baseline Results - EM (Exact Match)\n\n")
        f.write(f"{'Q/Ctx':<6} | {'all_in_one':>11} | {'sequential':>11} | {'batch':>11} | {'collab_llm':>11}\n")
        f.write("-" * 70 + "\n")

        for q_count in sorted(results_by_count.keys()):
            data = results_by_count[q_count]
            if "pretrained" not in data:
                continue

            pretrained = data["pretrained"]
            strategies = pretrained.get("strategies", {})

            f.write(f"{q_count:<6} | ")
            f.write(f"{strategies.get('all_in_one', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{strategies.get('sequential', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{strategies.get('batch', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{strategies.get('collab_llm', {}).get('strict_acc', 0):>11.3f}\n")

        # Pretrained Results - F1
        f.write("\n## Pretrained Baseline Results - F1\n\n")
        f.write(f"{'Q/Ctx':<6} | {'all_in_one':>11} | {'sequential':>11} | {'batch':>11} | {'collab_llm':>11}\n")
        f.write("-" * 70 + "\n")

        for q_count in sorted(results_by_count.keys()):
            data = results_by_count[q_count]
            if "pretrained" not in data:
                continue

            pretrained = data["pretrained"]
            strategies = pretrained.get("strategies", {})

            f.write(f"{q_count:<6} | ")
            f.write(f"{strategies.get('all_in_one', {}).get('f1', 0):>11.3f} | ")
            f.write(f"{strategies.get('sequential', {}).get('f1', 0):>11.3f} | ")
            f.write(f"{strategies.get('batch', {}).get('f1', 0):>11.3f} | ")
            f.write(f"{strategies.get('collab_llm', {}).get('f1', 0):>11.3f}\n")

        # SFT Results - EM
        f.write("\n## SFT-LoRA Results - EM (Exact Match)\n\n")
        f.write(f"{'Q/Ctx':<6} | {'batch baseline':>14} | {'batch SFT':>10} | {'Δ':>7} | {'seq baseline':>13} | {'seq SFT':>9} | {'Δ':>7}\n")
        f.write("-" * 100 + "\n")

        for q_count in sorted(results_by_count.keys()):
            data = results_by_count[q_count]
            if "sft" not in data:
                continue

            sft = data["sft"]
            batch_data = sft.get("batch", {})
            seq_data = sft.get("sequential", {})

            batch_baseline = batch_data.get("baseline", {}).get("strict_acc", 0)
            batch_sft = batch_data.get("sft_lora", {}).get("strict_acc", 0)
            batch_delta = batch_sft - batch_baseline

            seq_baseline = seq_data.get("baseline", {}).get("strict_acc", 0)
            seq_sft = seq_data.get("sft_lora", {}).get("strict_acc", 0)
            seq_delta = seq_sft - seq_baseline

            f.write(f"{q_count:<6} | ")
            f.write(f"{batch_baseline:>14.3f} | ")
            f.write(f"{batch_sft:>10.3f} | ")
            f.write(f"{batch_delta:>+7.3f} | ")
            f.write(f"{seq_baseline:>13.3f} | ")
            f.write(f"{seq_sft:>9.3f} | ")
            f.write(f"{seq_delta:>+7.3f}\n")

        # SFT Results - F1
        f.write("\n## SFT-LoRA Results - F1\n\n")
        f.write(f"{'Q/Ctx':<6} | {'batch baseline':>14} | {'batch SFT':>10} | {'Δ':>7} | {'seq baseline':>13} | {'seq SFT':>9} | {'Δ':>7}\n")
        f.write("-" * 100 + "\n")

        for q_count in sorted(results_by_count.keys()):
            data = results_by_count[q_count]
            if "sft" not in data:
                continue

            sft = data["sft"]
            batch_data = sft.get("batch", {})
            seq_data = sft.get("sequential", {})

            batch_baseline_f1 = batch_data.get("baseline", {}).get("f1", 0)
            batch_sft_f1 = batch_data.get("sft_lora", {}).get("f1", 0)
            batch_delta_f1 = batch_sft_f1 - batch_baseline_f1

            seq_baseline_f1 = seq_data.get("baseline", {}).get("f1", 0)
            seq_sft_f1 = seq_data.get("sft_lora", {}).get("f1", 0)
            seq_delta_f1 = seq_sft_f1 - seq_baseline_f1

            f.write(f"{q_count:<6} | ")
            f.write(f"{batch_baseline_f1:>14.3f} | ")
            f.write(f"{batch_sft_f1:>10.3f} | ")
            f.write(f"{batch_delta_f1:>+7.3f} | ")
            f.write(f"{seq_baseline_f1:>13.3f} | ")
            f.write(f"{seq_sft_f1:>9.3f} | ")
            f.write(f"{seq_delta_f1:>+7.3f}\n")

        # Token efficiency
        f.write("\n## Token Efficiency (Pretrained)\n\n")
        f.write(f"{'Q/Ctx':<6} | {'sequential PromptTok':>20} | {'batch PromptTok':>16} | {'collab_llm PromptTok':>20}\n")
        f.write("-" * 80 + "\n")

        for q_count in sorted(results_by_count.keys()):
            data = results_by_count[q_count]
            if "pretrained" not in data:
                continue

            pretrained = data["pretrained"]
            strategies = pretrained.get("strategies", {})

            f.write(f"{q_count:<6} | ")
            f.write(f"{strategies.get('sequential', {}).get('avg_prompt_tokens', 0):>20.1f} | ")
            f.write(f"{strategies.get('batch', {}).get('avg_prompt_tokens', 0):>16.1f} | ")
            f.write(f"{strategies.get('collab_llm', {}).get('avg_prompt_tokens', 0):>20.1f}\n")

    logger.info(f"\nSummary table saved to {summary_file}")

    # Also print to console
    with open(summary_file, 'r') as f:
        print(f.read())


def main():
    args = parse_args()

    # Parse question counts
    question_counts = [int(x.strip()) for x in args.question_counts.split(',')]
    logger.info(f"Testing question counts: {question_counts}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store all results
    results_by_count = {}

    for q_count in question_counts:
        logger.info(f"\n\n{'#'*120}")
        logger.info(f"# Testing with {q_count} questions per context")
        logger.info(f"{'#'*120}\n")

        results_by_count[q_count] = {}

        # Step 1: Pretrained evaluation
        if not args.skip_pretrained:
            pretrained_results = run_pretrained_eval(args, q_count, output_dir)
            if pretrained_results:
                results_by_count[q_count]["pretrained"] = pretrained_results

        # Step 2: SFT training and evaluation
        if not args.skip_sft:
            sft_results = run_sft_training(args, q_count, output_dir)
            if sft_results:
                results_by_count[q_count]["sft"] = sft_results

    # Generate summary table
    generate_summary_table(args, results_by_count, output_dir)

    # Save all results
    all_results_file = output_dir / "all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(results_by_count, f, indent=2)
    logger.info(f"\nAll results saved to {all_results_file}")

    logger.info("\n" + "="*80)
    logger.info("DONE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
