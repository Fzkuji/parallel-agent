"""
Complete training and evaluation pipeline for cross-batch module.

This script:
1. Loads baseline (batch) strategy results (or runs if not cached)
2. Trains cross-batch module for multiple epochs
3. Evaluates after each epoch and compares with baseline
4. Saves checkpoints and results progressively
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.squad import load_squad_groups
from src.models import Question, StrategyResult
from src.strategies.sequential_batch import run_batch_multi_strategy
from src.strategies.cross_batch import run_cross_batch_multi_strategy
from src.evaluation import evaluate_predictions
from src.cross_batch import train_cross_batch_module


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate cross-batch module")

    # Model and dataset
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset", type=str, default="squad", help="Dataset name")
    parser.add_argument("--train-samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--eval-samples", type=int, default=100, help="Number of evaluation samples")
    parser.add_argument("--min-questions", type=int, default=3, help="Min questions per context")
    parser.add_argument("--max-questions", type=int, default=5, help="Max questions per context")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (contexts per batch)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--module-type", type=str, default="multi_layer", choices=["simple", "multi_layer", "attention", "mixer"], help="Cross-batch module type")
    parser.add_argument("--mix-layers", type=str, default=None, help="Comma-separated layer indices for multi_layer mode (None = all layers)")

    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens for generation")

    # Paths
    parser.add_argument("--output-dir", type=str, default="outputs/train_and_eval", help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Checkpoint directory")
    parser.add_argument("--cache-baseline", action="store_true", help="Cache baseline results to avoid recomputation")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    return parser.parse_args()


def squad_to_items(context_payload: dict) -> List[dict]:
    """Convert SQuAD context format to items format for batch strategy."""
    context = context_payload["context"]
    items = []
    for q in context_payload["questions"]:
        items.append({
            "qid": q["qid"],
            "question": q["text"],
            "context": context,
            "references": q["references"],
            "answer_tokens": q.get("answer_tokens", 12),
        })
    return items


def load_or_run_baseline(
    args,
    eval_contexts: List[Dict],
    tokenizer,
    model,
    cache_path: Path,
) -> Dict[str, Any]:
    """Load cached baseline results or run baseline strategy."""

    if cache_path.exists() and args.cache_baseline:
        logging.info(f"Loading cached baseline results from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    logging.info("Running baseline (batch) strategy...")
    baseline_results = {
        "contexts": [],
        "aggregate_metrics": {},
    }

    total_em = 0.0
    total_f1 = 0.0
    total_questions = 0
    total_latency = 0.0
    total_prompt_tokens = 0
    total_generated_tokens = 0

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = squad_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        logging.info(f"Baseline: Processing {idx}/{len(eval_contexts)}: {title}")

        # Run batch strategy
        result = run_batch_multi_strategy(
            items,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            strategy_name="batch",
            dataset=args.dataset,
        )

        # Extract questions for display
        questions = [
            Question(
                qid=item["qid"],
                text=item["question"],
                priority=1.0,
                answer_tokens=item.get("answer_tokens", 12),
                type_hint=None,
                references=item.get("references", []),
            )
            for item in items
        ]

        # Store results
        context_result = {
            "title": title,
            "questions": [{"qid": q.qid, "text": q.text, "references": q.references} for q in questions],
            "answers": result.answers,
            "metrics": result.metrics,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "latency": result.latency,
        }
        baseline_results["contexts"].append(context_result)

        # Accumulate metrics
        n_questions = len(questions)
        total_em += result.metrics.get("strict_acc", 0) * n_questions
        total_f1 += result.metrics.get("f1", 0) * n_questions
        total_questions += n_questions
        total_latency += result.latency
        total_prompt_tokens += result.prompt_tokens
        total_generated_tokens += result.generated_tokens

    # Compute aggregate metrics
    baseline_results["aggregate_metrics"] = {
        "strict_acc": total_em / total_questions if total_questions > 0 else 0,
        "f1": total_f1 / total_questions if total_questions > 0 else 0,
        "avg_latency": total_latency / len(eval_contexts),
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
    }

    # Save cache
    if args.cache_baseline:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        logging.info(f"Saved baseline results to {cache_path}")

    return baseline_results


def evaluate_checkpoint(
    args,
    checkpoint_path: str,
    eval_contexts: List[Dict],
    tokenizer,
    model,
    epoch: int,
) -> Dict[str, Any]:
    """Evaluate a trained checkpoint."""

    logging.info(f"Evaluating checkpoint from epoch {epoch}...")

    results = {
        "epoch": epoch,
        "contexts": [],
        "aggregate_metrics": {},
    }

    total_em = 0.0
    total_f1 = 0.0
    total_questions = 0
    total_latency = 0.0
    total_prompt_tokens = 0
    total_generated_tokens = 0

    # Parse mix_layers if provided
    mix_layers = None
    if args.mix_layers:
        mix_layers = [int(x.strip()) for x in args.mix_layers.split(',')]

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = squad_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        logging.info(f"Epoch {epoch}: Evaluating {idx}/{len(eval_contexts)}: {title}")

        # Run cross-batch strategy
        result = run_cross_batch_multi_strategy(
            items,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            strategy_name=f"collab_hidden_epoch{epoch}",
            dataset=args.dataset,
            mix_method=args.module_type,
            mix_layer=-1,
            mix_layers=mix_layers,
            checkpoint_path=checkpoint_path,
            enable_cross_batch=True,
        )

        # Extract questions
        questions = [
            Question(
                qid=item["qid"],
                text=item["question"],
                priority=1.0,
                answer_tokens=item.get("answer_tokens", 12),
                type_hint=None,
                references=item.get("references", []),
            )
            for item in items
        ]

        # Store results
        context_result = {
            "title": title,
            "questions": [{"qid": q.qid, "text": q.text, "references": q.references} for q in questions],
            "answers": result.answers,
            "metrics": result.metrics,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "latency": result.latency,
        }
        results["contexts"].append(context_result)

        # Accumulate metrics
        n_questions = len(questions)
        total_em += result.metrics.get("strict_acc", 0) * n_questions
        total_f1 += result.metrics.get("f1", 0) * n_questions
        total_questions += n_questions
        total_latency += result.latency
        total_prompt_tokens += result.prompt_tokens
        total_generated_tokens += result.generated_tokens

    # Compute aggregate metrics
    results["aggregate_metrics"] = {
        "strict_acc": total_em / total_questions if total_questions > 0 else 0,
        "f1": total_f1 / total_questions if total_questions > 0 else 0,
        "avg_latency": total_latency / len(eval_contexts),
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
    }

    return results


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model and tokenizer
    logging.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # Load training data
    logging.info(f"Loading training data: {args.train_samples} samples")
    train_groups = load_squad_groups(
        split="train",
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        max_contexts=args.train_samples,
        seed=args.seed,
    )

    # Load evaluation data
    logging.info(f"Loading evaluation data: {args.eval_samples} samples")
    eval_contexts = load_squad_groups(
        split="validation",
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        max_contexts=args.eval_samples,
        seed=args.seed,
    )

    # Get or compute baseline results
    baseline_cache_path = output_dir / "baseline_results.json"
    logging.info("=" * 60)
    logging.info("STEP 1: Baseline (Batch) Strategy")
    logging.info("=" * 60)
    baseline_results = load_or_run_baseline(
        args,
        eval_contexts,
        tokenizer,
        model,
        baseline_cache_path,
    )

    logging.info("\nBaseline Results:")
    logging.info(f"  EM:  {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
    logging.info(f"  F1:  {baseline_results['aggregate_metrics']['f1']:.4f}")
    logging.info(f"  Latency: {baseline_results['aggregate_metrics']['avg_latency']:.2f}s")

    # Train cross-batch module
    logging.info("\n" + "=" * 60)
    logging.info("STEP 2: Training Cross-Batch Module")
    logging.info("=" * 60)

    # Generate checkpoint path
    safe_model_name = args.model.replace('/', '_')
    checkpoint_base = Path(args.checkpoint_dir) / args.dataset / safe_model_name
    checkpoint_base.mkdir(parents=True, exist_ok=True)

    # Results tracking
    all_results = {
        "config": vars(args),
        "baseline": baseline_results["aggregate_metrics"],
        "epochs": [],
    }

    # Train for multiple epochs with evaluation after each epoch
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Epoch {epoch}/{args.epochs}")
        logging.info("=" * 60)

        # Train for one epoch
        logging.info(f"Training epoch {epoch}...")

        # Parse mix_layers
        mix_layers = None
        if args.mix_layers:
            mix_layers = [int(x.strip()) for x in args.mix_layers.split(',')]

        # Get checkpoint path for this epoch
        checkpoint_path = str(checkpoint_base / f"{args.module_type}_frozen_epoch{epoch}.pt")

        # Train (this will save checkpoint at the end)
        train_cross_batch_module(
            model=model,
            tokenizer=tokenizer,
            train_groups=train_groups,
            dataset_name=args.dataset,
            module_type=args.module_type,
            mix_layer=-1,
            mix_layers=mix_layers,
            num_epochs=1,  # Train for 1 epoch at a time
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_dir=str(checkpoint_base),
            checkpoint_name=f"{args.module_type}_frozen_epoch{epoch}.pt",
        )

        # Evaluate this epoch's checkpoint
        logging.info(f"\nEvaluating epoch {epoch}...")
        epoch_results = evaluate_checkpoint(
            args,
            checkpoint_path,
            eval_contexts,
            tokenizer,
            model,
            epoch,
        )

        # Display results
        logging.info(f"\nEpoch {epoch} Results:")
        logging.info(f"  EM:  {epoch_results['aggregate_metrics']['strict_acc']:.4f} "
                    f"(Δ {epoch_results['aggregate_metrics']['strict_acc'] - baseline_results['aggregate_metrics']['strict_acc']:+.4f})")
        logging.info(f"  F1:  {epoch_results['aggregate_metrics']['f1']:.4f} "
                    f"(Δ {epoch_results['aggregate_metrics']['f1'] - baseline_results['aggregate_metrics']['f1']:+.4f})")
        logging.info(f"  Latency: {epoch_results['aggregate_metrics']['avg_latency']:.2f}s")

        # Store epoch results
        all_results["epochs"].append(epoch_results["aggregate_metrics"])

        # Save progressive results
        results_path = output_dir / f"results_epoch{epoch}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Saved results to {results_path}")

    # Final summary
    logging.info("\n" + "=" * 60)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 60)

    logging.info("\nBaseline (Batch):")
    logging.info(f"  EM: {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
    logging.info(f"  F1: {baseline_results['aggregate_metrics']['f1']:.4f}")

    logging.info("\nCross-Batch Results by Epoch:")
    for epoch, epoch_metrics in enumerate(all_results["epochs"], start=1):
        logging.info(f"\nEpoch {epoch}:")
        logging.info(f"  EM: {epoch_metrics['strict_acc']:.4f} "
                    f"(Δ {epoch_metrics['strict_acc'] - baseline_results['aggregate_metrics']['strict_acc']:+.4f})")
        logging.info(f"  F1: {epoch_metrics['f1']:.4f} "
                    f"(Δ {epoch_metrics['f1'] - baseline_results['aggregate_metrics']['f1']:+.4f})")

    # Find best epoch
    best_epoch = max(range(len(all_results["epochs"])),
                     key=lambda i: all_results["epochs"][i]["f1"]) + 1
    logging.info(f"\nBest Epoch: {best_epoch} (F1: {all_results['epochs'][best_epoch-1]['f1']:.4f})")

    # Save final results
    final_results_path = output_dir / "final_results.json"
    with open(final_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nSaved final results to {final_results_path}")

    logging.info("\n" + "=" * 60)
    logging.info("DONE!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
