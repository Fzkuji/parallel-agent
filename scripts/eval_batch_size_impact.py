#!/usr/bin/env python3
"""
Test the impact of different batch sizes on batch strategy performance.

All tests use the same 180 questions (9 contexts × 20 questions).
Only the batch size varies: 1, 5, 10, 20.
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
    parser = argparse.ArgumentParser(description="Batch size impact experiment")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="squad")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-sizes", type=str, default="1,5,10,20")
    parser.add_argument("--output-dir", type=str, default="outputs/batch_size_study")

    return parser.parse_args()


def run_evaluation(args, batch_size, output_dir):
    """Run baseline_pretrained.py with specific batch size."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH SIZE: {batch_size}")
    logger.info(f"{'='*80}\n")

    cmd = [
        sys.executable, "scripts/baseline_pretrained.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--eval-samples", "1000",
        "--min-questions", "20",
        "--max-questions", "20",
        "--strategies", "batch",
        "--batch-size", str(batch_size),
        "--seed", str(args.seed),
        "--num-gpus", str(args.num_gpus),
        "--force",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Failed for batch size {batch_size}")
        return None

    # Read results
    results_file = Path("outputs/eval_baselines/results_squad.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Save copy
        output_file = output_dir / f"batch_size_{batch_size}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to {output_file}")
        return data
    else:
        logger.error(f"Results file not found")
        return None


def generate_summary(results_by_batch_size, output_dir):
    """Generate summary table."""
    logger.info(f"\n{'='*80}")
    logger.info("BATCH SIZE IMPACT STUDY SUMMARY")
    logger.info(f"{'='*80}\n")

    summary_file = output_dir / "summary.txt"

    with open(summary_file, 'w') as f:
        f.write("# Batch Size Impact Study\n")
        f.write("# Dataset: 9 contexts, 20 questions each, 180 total questions\n")
        f.write("# All tests use the same 180 questions\n")
        f.write("# Only batch size varies\n\n")

        f.write("## Results - EM\n\n")
        f.write(f"{'Batch Size':<12} | {'EM':>8} | {'F1':>8} | {'Latency (s)':>12} | {'Questions':>10}\n")
        f.write("-" * 65 + "\n")

        for bs in sorted(results_by_batch_size.keys()):
            if results_by_batch_size[bs] is None:
                continue

            batch_data = results_by_batch_size[bs].get("strategies", {}).get("batch", {})
            metrics = batch_data.get("aggregate_metrics", {})

            em = metrics.get("strict_acc", 0)
            f1 = metrics.get("f1", 0)
            latency = metrics.get("total_latency", 0)
            num_q = metrics.get("num_questions", 0)

            f.write(f"{bs:<12} | {em:>8.3f} | {f1:>8.3f} | {latency:>12.2f} | {num_q:>10}\n")

        f.write("\n## Interpretation\n\n")
        f.write("If batch is truly independent:\n")
        f.write("- All batch sizes should produce identical EM/F1\n")
        f.write("- Only latency should differ\n\n")
        f.write("If results differ:\n")
        f.write("- Padding/attention may cause interference in larger batches\n")

    logger.info(f"Summary saved to {summary_file}\n")

    # Print summary
    with open(summary_file, 'r') as f:
        print(f.read())


def main():
    args = parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    logger.info(f"Testing batch sizes: {batch_sizes}")
    logger.info(f"All tests will use the same 180 questions (9 contexts × 20 questions)\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation for each batch size
    results_by_batch_size = {}

    for batch_size in batch_sizes:
        results = run_evaluation(args, batch_size, output_dir)
        results_by_batch_size[batch_size] = results

    # Generate summary
    generate_summary(results_by_batch_size, output_dir)

    # Save all results
    all_results_file = output_dir / "all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(results_by_batch_size, f, indent=2)
    logger.info(f"\nAll results saved to {all_results_file}")

    logger.info("\n" + "="*80)
    logger.info("DONE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
