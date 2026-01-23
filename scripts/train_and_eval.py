#!/usr/bin/env python3
"""
Cross-batch training and evaluation script.

Usage:
    python scripts/train_and_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --epochs 3 \
        --cache-baseline
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate cross-batch module")

    # Model and dataset
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="squad")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--min-questions", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=5)

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--module-type", type=str, default="multi_layer",
                       choices=["simple", "multi_layer", "multi_layer_attention", "attention", "mixer"])
    parser.add_argument("--mix-layers", type=str, default=None)

    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=96)

    # Paths
    parser.add_argument("--output-dir", type=str, default="outputs/train_and_eval")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--cache-baseline", action="store_true")
    parser.add_argument("--force", action="store_true")

    # LoRA
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # vLLM
    parser.add_argument("--enable-thinking", action="store_true")

    # Other
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    import torch
    from src.datasets.squad import load_squad_groups
    from src.datasets.hotpot import load_hotpot_groups
    from src.baseline import run_vllm_baseline
    from src.training import CrossBatchPipeline
    from src.training.pipeline import TrainingConfig

    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation data
    logger.info(f"Loading evaluation data: {args.eval_samples} samples")
    if args.dataset == "hotpot":
        eval_contexts = load_hotpot_groups(
            split="validation",
            max_contexts=args.eval_samples,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            seed=args.seed + 1000,
        )
    else:
        eval_contexts = load_squad_groups(
            split="validation",
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.eval_samples,
            seed=args.seed + 1000,
        )
    logger.info(f"Loaded {len(eval_contexts)} evaluation contexts")

    # Step 1: Baseline (vLLM)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Baseline (vLLM)")
    logger.info("=" * 60)

    baseline_results = run_vllm_baseline(
        model_name=args.model,
        eval_contexts=eval_contexts,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        dataset=args.dataset,
        enable_thinking=args.enable_thinking,
        cache_baseline=args.cache_baseline,
        force=args.force,
    )

    logger.info("\nBaseline Results:")
    logger.info(f"  EM:  {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
    logger.info(f"  F1:  {baseline_results['aggregate_metrics']['f1']:.4f}")
    logger.info(f"  Latency: {baseline_results['aggregate_metrics']['avg_latency']:.2f}s")

    # Step 2: Training
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Training Cross-Batch Module")
    logger.info("=" * 60)

    # Load training data
    logger.info(f"Loading training data: {args.train_samples} samples")
    if args.dataset == "hotpot":
        train_groups = load_hotpot_groups(
            split="train",
            max_contexts=args.train_samples,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            seed=args.seed,
        )
    else:
        train_groups = load_squad_groups(
            split="train",
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.train_samples,
            seed=args.seed,
        )

    # Parse mix_layers
    mix_layers = None
    if args.mix_layers:
        mix_layers = [int(x.strip()) for x in args.mix_layers.split(',')]

    # Create config
    config = TrainingConfig(
        model_name=args.model,
        dataset=args.dataset,
        module_type=args.module_type,
        mix_layers=mix_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_new_tokens=args.max_new_tokens,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[m.strip() for m in args.lora_target_modules.split(',')],
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )

    # Run pipeline
    pipeline = CrossBatchPipeline(config)
    pipeline.run(train_groups, eval_contexts, baseline_results)

    logger.info("\n" + "=" * 60)
    logger.info("DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
