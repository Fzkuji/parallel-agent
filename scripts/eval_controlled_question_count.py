#!/usr/bin/env python3
"""
Controlled question count experiment.

Tests performance on the same set of contexts with different question counts.
Uses contexts that have ≥20 questions, then samples 1/5/10/20 questions from each.

This ensures fair comparison - all tests use the exact same contexts.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled question count experiment")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="squad")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-counts", type=str, default="1,5,10,20")
    parser.add_argument("--output-dir", type=str, default="outputs/controlled_q_count")

    return parser.parse_args()


def run_pretrained(args, q_count, output_dir):
    """Run baseline_pretrained.py with fixed question count."""
    logger.info(f"\n{'='*80}")
    logger.info(f"PRETRAINED: {q_count} questions from 20Q contexts")
    logger.info(f"{'='*80}\n")

    cmd = [
        sys.executable, "scripts/baseline_pretrained.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--eval-samples", "1000",
        "--min-questions", "20",
        "--max-questions", "20",
        "--strategies", "all_in_one,sequential,batch,collab_llm",
        "--seed", str(args.seed),
        "--num-gpus", str(args.num_gpus),
        "--force",
    ]

    # Add fixed-question-count if not testing full 20
    if q_count < 20:
        cmd.extend(["--fixed-question-count", str(q_count)])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Failed for {q_count} questions")
        return None

    # Read results
    results_file = Path("outputs/eval_baselines/results_squad.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
        # Save copy
        output_file = output_dir / f"pretrained_fixed_q{q_count}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to {output_file}")
        return data
    else:
        logger.error(f"Results file not found")
        return None


def generate_summary(results_by_count, output_dir):
    """Generate summary table."""
    logger.info(f"\n{'='*100}")
    logger.info("CONTROLLED EXPERIMENT SUMMARY")
    logger.info(f"{'='*100}\n")

    summary_file = output_dir / "summary.txt"

    with open(summary_file, 'w') as f:
        f.write("# Controlled Question Count Experiment\n")
        f.write("# Using the same set of contexts (≥20 questions) for all tests\n")
        f.write("# Questions are sampled in order (first N questions)\n\n")

        # Pretrained EM
        f.write("## Pretrained Results - EM\n\n")
        f.write(f"{'Q/Ctx':<6} | {'all_in_one':>11} | {'sequential':>11} | {'batch':>11} | {'collab_llm':>11} | {'Contexts':>10}\n")
        f.write("-" * 85 + "\n")

        for q in sorted(results_by_count.keys()):
            if results_by_count[q] is None:
                continue
            strategies = results_by_count[q].get("strategies", {})
            config = results_by_count[q].get("config", {})
            num_contexts = strategies.get("batch", {}).get("aggregate_metrics", {}).get("num_contexts", 0)

            f.write(f"{q:<6} | ")
            f.write(f"{strategies.get('all_in_one', {}).get('aggregate_metrics', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{strategies.get('sequential', {}).get('aggregate_metrics', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{strategies.get('batch', {}).get('aggregate_metrics', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{strategies.get('collab_llm', {}).get('aggregate_metrics', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{num_contexts:>10}\n")

        # Pretrained F1
        f.write("\n## Pretrained Results - F1\n\n")
        f.write(f"{'Q/Ctx':<6} | {'all_in_one':>11} | {'sequential':>11} | {'batch':>11} | {'collab_llm':>11}\n")
        f.write("-" * 70 + "\n")

        for q in sorted(results_by_count.keys()):
            if results_by_count[q] is None:
                continue
            strategies = results_by_count[q].get("strategies", {})

            f.write(f"{q:<6} | ")
            f.write(f"{strategies.get('all_in_one', {}).get('aggregate_metrics', {}).get('f1', 0):>11.3f} | ")
            f.write(f"{strategies.get('sequential', {}).get('aggregate_metrics', {}).get('f1', 0):>11.3f} | ")
            f.write(f"{strategies.get('batch', {}).get('aggregate_metrics', {}).get('f1', 0):>11.3f} | ")
            f.write(f"{strategies.get('collab_llm', {}).get('aggregate_metrics', {}).get('f1', 0):>11.3f}\n")

        # Token efficiency
        f.write("\n## Token Efficiency\n\n")
        f.write(f"{'Q/Ctx':<6} | {'seq PromptTok':>14} | {'batch PromptTok':>16} | {'collab PromptTok':>17}\n")
        f.write("-" * 70 + "\n")

        for q in sorted(results_by_count.keys()):
            if results_by_count[q] is None:
                continue
            strategies = results_by_count[q].get("strategies", {})

            f.write(f"{q:<6} | ")
            f.write(f"{strategies.get('sequential', {}).get('aggregate_metrics', {}).get('avg_prompt_tokens', 0):>14.1f} | ")
            f.write(f"{strategies.get('batch', {}).get('aggregate_metrics', {}).get('avg_prompt_tokens', 0):>16.1f} | ")
            f.write(f"{strategies.get('collab_llm', {}).get('aggregate_metrics', {}).get('avg_prompt_tokens', 0):>17.1f}\n")

    logger.info(f"Summary saved to {summary_file}\n")

    # Print summary
    with open(summary_file, 'r') as f:
        print(f.read())


def main():
    args = parse_args()

    # Parse question counts
    question_counts = [int(x.strip()) for x in args.question_counts.split(',')]
    logger.info(f"Testing question counts: {question_counts}")
    logger.info(f"All tests will use contexts with ≥20 questions\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pretrained evaluation for each question count
    results_by_count = {}

    for q_count in question_counts:
        results = run_pretrained(args, q_count, output_dir)
        results_by_count[q_count] = results

    # Generate summary
    generate_summary(results_by_count, output_dir)

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
