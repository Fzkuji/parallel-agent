#!/usr/bin/env python3
"""Run all preliminary experiments.

Usage:
    python scripts/preliminary/run_all.py --model gpt-4o-mini --quick
    python scripts/preliminary/run_all.py --model gpt-4o --full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_experiment(script_name: str, args: list) -> int:
    """Run a single experiment script."""
    script_path = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path)] + args

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all preliminary experiments")
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Model to use (e.g., gpt-4o-mini, Qwen/Qwen2.5-7B-Instruct)"
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
        "--quick", action="store_true",
        help="Run quick version with fewer samples"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full version with more samples"
    )
    parser.add_argument(
        "--exp", type=str, default="all",
        help="Which experiments to run: all, exp1, exp2a, exp2b, exp3"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory"
    )
    args = parser.parse_args()

    # Determine sample sizes
    if args.quick:
        n_samples = 20
        n_per_domain = 10
        n_groups = 20
    elif args.full:
        n_samples = 200
        n_per_domain = 50
        n_groups = 100
    else:
        n_samples = 50
        n_per_domain = 20
        n_groups = 50

    base_args = [
        "--models", args.model,
        "--seed", str(args.seed),
        "--output-dir", args.output_dir,
    ]
    if args.use_local:
        base_args.append("--use-local")
    if args.use_vllm:
        base_args.append("--use-vllm")

    experiments = {
        "exp1": {
            "script": "exp1_answer_dependency.py",
            "args": base_args + ["--n-samples", str(n_samples)],
        },
        "exp2a": {
            "script": "exp2a_shared_context.py",
            "args": base_args + ["--n-groups", str(n_groups)],
        },
        "exp2b": {
            "script": "exp2b_related_domain.py",
            "args": base_args + ["--n-per-domain", str(n_per_domain)],
        },
        "exp3": {
            "script": "exp3_format_similarity.py",
            "args": base_args + ["--n-samples", str(n_samples * 2)],
        },
    }

    # Determine which experiments to run
    if args.exp == "all":
        to_run = list(experiments.keys())
    else:
        to_run = [args.exp]

    # Run experiments
    results = {}
    for exp_name in to_run:
        if exp_name not in experiments:
            print(f"Unknown experiment: {exp_name}")
            continue

        exp = experiments[exp_name]
        returncode = run_experiment(exp["script"], exp["args"])
        results[exp_name] = "SUCCESS" if returncode == 0 else "FAILED"

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for exp_name, status in results.items():
        print(f"  {exp_name}: {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
