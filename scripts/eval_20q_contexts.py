#!/usr/bin/env python3
"""
Evaluate performance on the 9 contexts that have 20 questions.

Tests with different question counts (1, 5, 10, 20) by sampling/dividing
the 20 questions in each context.

This ensures fair comparison across question counts using the same contexts.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.squad import load_squad_groups

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on 20-question contexts")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="squad")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/20q_contexts_study")

    return parser.parse_args()


def get_20q_contexts():
    """Get the contexts that have exactly 20 questions."""
    # Load all validation contexts
    all_groups = load_squad_groups(
        split="validation",
        max_contexts=None,
        min_questions=20,
        max_questions=20,
        seed=42
    )
    logger.info(f"Found {len(all_groups)} contexts with exactly 20 questions")
    return all_groups


def save_custom_contexts(contexts, output_file):
    """Save contexts to a JSON file for manual loading."""
    with open(output_file, 'w') as f:
        json.dump(contexts, f, indent=2)
    logger.info(f"Saved contexts to {output_file}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the 9 contexts with 20 questions
    contexts_20q = get_20q_contexts()

    if len(contexts_20q) == 0:
        logger.error("No contexts with 20 questions found!")
        return

    # Save these contexts
    contexts_file = output_dir / "20q_contexts.json"
    save_custom_contexts(contexts_20q, contexts_file)

    # Now test with different question sampling strategies
    question_counts = [1, 5, 10, 20]

    logger.info(f"\nTesting question counts: {question_counts}")
    logger.info(f"Using {len(contexts_20q)} contexts with 20 questions each")

    # For each question count, we'll:
    # 1. Create modified contexts with only N questions (sampled from the 20)
    # 2. Run evaluation on those contexts

    results = {}

    for q_count in question_counts:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing with {q_count} questions per context")
        logger.info(f"Sampling {q_count} questions from each of the {len(contexts_20q)} contexts")
        logger.info(f"{'='*80}\n")

        # For now, just document the approach
        # Actual implementation would require modifying the evaluation scripts
        # to accept pre-filtered contexts

        results[q_count] = {
            "num_contexts": len(contexts_20q),
            "questions_per_context": q_count,
            "total_questions": len(contexts_20q) * q_count,
            "sampling_strategy": "First N questions" if q_count < 20 else "All questions"
        }

    # Save analysis
    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            "contexts_info": {
                "num_contexts": len(contexts_20q),
                "questions_per_context": 20,
                "total_questions": len(contexts_20q) * 20
            },
            "test_configs": results
        }, f, indent=2)

    logger.info(f"\nAnalysis saved to {analysis_file}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"\nFound {len(contexts_20q)} contexts with 20 questions each")
    logger.info(f"\nFor fair comparison across question counts:")
    logger.info("  1 Q/ctx:  Sample 1st question from each context (9 questions total)")
    logger.info("  5 Q/ctx:  Sample 1st 5 questions from each context (45 questions total)")
    logger.info(" 10 Q/ctx:  Sample 1st 10 questions from each context (90 questions total)")
    logger.info(" 20 Q/ctx:  Use all 20 questions from each context (180 questions total)")
    logger.info("\nThis ensures all tests use the same 9 contexts, making results comparable.")

    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATION")
    logger.info("="*80)
    logger.info("\nOption 1 - Rerun experiments with --min-questions 20 --max-questions 20")
    logger.info("  This filters to the 9 contexts, then you manually sample questions")
    logger.info("\nOption 2 - Use HotpotQA or TriviaQA datasets")
    logger.info("  These have more multi-question contexts")
    logger.info("\nOption 3 - Interpret current results carefully")
    logger.info("  Different question counts use different context subsets")
    logger.info("  1Q: 2067 contexts,  20Q: 9 contexts (different difficulty)")


if __name__ == "__main__":
    main()
