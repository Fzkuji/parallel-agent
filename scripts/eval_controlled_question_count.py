#!/usr/bin/env python3
"""
Controlled question count experiment.

Tests performance on the same set of 9 contexts with different question counts.
Uses contexts that have 20 questions, then samples 1/5/10/20 questions from each.

- 1 Q/ctx: 9 contexts × 1 question = 9 total questions
- 5 Q/ctx: 9 contexts × 5 questions = 45 total questions
- 10 Q/ctx: 9 contexts × 10 questions = 90 total questions
- 20 Q/ctx: 9 contexts × 20 questions = 180 total questions

All strategies test the same question set for each configuration.
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
    logger.info(f"QUESTION COUNT: {q_count} questions per context")
    logger.info(f"{'='*80}\n")

    cmd = [
        sys.executable, "scripts/baseline_pretrained.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--eval-samples", "1000",
        "--min-questions", "20",
        "--max-questions", "20",
        "--strategies", "all_in_one,sequential,batch,collab_llm",
        "--fixed-question-count", str(q_count),
        "--seed", str(args.seed),
        "--num-gpus", str(args.num_gpus),
        "--force",
    ]

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
        output_file = output_dir / f"q{q_count}.json"
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
    logger.info("QUESTION COUNT EXPERIMENT SUMMARY")
    logger.info(f"{'='*100}\n")

    summary_file = output_dir / "summary.txt"

    with open(summary_file, 'w') as f:
        f.write("# Controlled Question Count Experiment\n")
        f.write("# Dataset: Same 9 contexts for all tests\n")
        f.write("# Questions per context varies: 1, 5, 10, 20\n")
        f.write("# All strategies test the same questions for each configuration\n\n")

        # EM Results
        f.write("## Results - EM (Exact Match)\n\n")
        f.write(f"{'Q/Ctx':<8} | {'all_in_one':>11} | {'sequential':>11} | {'batch':>11} | {'collab_llm':>11} | {'Total Q':>10}\n")
        f.write("-" * 85 + "\n")

        for q in sorted(results_by_count.keys()):
            if results_by_count[q] is None:
                continue
            strategies = results_by_count[q].get("strategies", {})
            num_q = strategies.get("batch", {}).get("aggregate_metrics", {}).get("num_questions", 0)

            f.write(f"{q:<8} | ")
            f.write(f"{strategies.get('all_in_one', {}).get('aggregate_metrics', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{strategies.get('sequential', {}).get('aggregate_metrics', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{strategies.get('batch', {}).get('aggregate_metrics', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{strategies.get('collab_llm', {}).get('aggregate_metrics', {}).get('strict_acc', 0):>11.3f} | ")
            f.write(f"{num_q:>10}\n")

        # F1 Results
        f.write("\n## Results - F1\n\n")
        f.write(f"{'Q/Ctx':<8} | {'all_in_one':>11} | {'sequential':>11} | {'batch':>11} | {'collab_llm':>11}\n")
        f.write("-" * 70 + "\n")

        for q in sorted(results_by_count.keys()):
            if results_by_count[q] is None:
                continue
            strategies = results_by_count[q].get("strategies", {})

            f.write(f"{q:<8} | ")
            f.write(f"{strategies.get('all_in_one', {}).get('aggregate_metrics', {}).get('f1', 0):>11.3f} | ")
            f.write(f"{strategies.get('sequential', {}).get('aggregate_metrics', {}).get('f1', 0):>11.3f} | ")
            f.write(f"{strategies.get('batch', {}).get('aggregate_metrics', {}).get('f1', 0):>11.3f} | ")
            f.write(f"{strategies.get('collab_llm', {}).get('aggregate_metrics', {}).get('f1', 0):>11.3f}\n")

        # Latency
        f.write("\n## Latency Comparison (seconds)\n\n")
        f.write(f"{'Q/Ctx':<8} | {'all_in_one':>11} | {'sequential':>11} | {'batch':>11} | {'collab_llm':>11}\n")
        f.write("-" * 70 + "\n")

        for q in sorted(results_by_count.keys()):
            if results_by_count[q] is None:
                continue
            strategies = results_by_count[q].get("strategies", {})

            f.write(f"{q:<8} | ")
            f.write(f"{strategies.get('all_in_one', {}).get('aggregate_metrics', {}).get('total_latency', 0):>11.2f} | ")
            f.write(f"{strategies.get('sequential', {}).get('aggregate_metrics', {}).get('total_latency', 0):>11.2f} | ")
            f.write(f"{strategies.get('batch', {}).get('aggregate_metrics', {}).get('total_latency', 0):>11.2f} | ")
            f.write(f"{strategies.get('collab_llm', {}).get('aggregate_metrics', {}).get('total_latency', 0):>11.2f}\n")

    logger.info(f"Summary saved to {summary_file}\n")

    # Print summary
    with open(summary_file, 'r') as f:
        print(f.read())


def main():
    args = parse_args()

    # Parse question counts
    question_counts = [int(x.strip()) for x in args.question_counts.split(',')]
    logger.info(f"Testing question counts: {question_counts}")
    logger.info(f"All tests will use the same 9 contexts\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation for each question count
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
