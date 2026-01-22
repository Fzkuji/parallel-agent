"""
Complete training and evaluation pipeline for cross-batch module with DDP support.

This script:
1. Loads baseline (batch) strategy results (or runs if not cached)
2. Trains cross-batch module for multiple epochs (single-GPU)
3. Evaluates after each epoch using multi-GPU (if available)
4. Saves checkpoints and results progressively

Usage:
  Single GPU:
    python scripts/train_and_eval.py --model Qwen/Qwen2.5-7B-Instruct --epochs 3 --cache-baseline

  Multi-GPU evaluation (8 GPUs):
    torchrun --nproc_per_node=8 scripts/train_and_eval.py --model Qwen/Qwen2.5-7B-Instruct --epochs 3 --cache-baseline
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.squad import load_squad_groups
from src.models import Question
from src.strategies.sequential_batch import run_batch_multi_strategy
from src.strategies.cross_batch import run_cross_batch_multi_strategy
from src.cross_batch import (
    CrossBatchTrainer,
    SQuADGroupedDataset,
    SimpleCrossBatchGate,
    MultiLayerCrossBatch,
    CrossBatchAttention,
    CrossBatchEmbeddingMixer,
)


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


def run_baseline_on_shard(
    args,
    eval_contexts: List[Dict],
    tokenizer,
    model,
) -> Dict[str, Any]:
    """Run baseline strategy on this rank's shard."""
    shard_results = {"contexts": []}

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = squad_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        logging.info(f"Baseline: Processing {idx}/{len(eval_contexts)}: {title}")

        result = run_batch_multi_strategy(
            items,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            strategy_name="batch",
            dataset=args.dataset,
        )

        context_result = {
            "title": title,
            "metrics": result.metrics,
            "latency": result.latency,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "num_questions": len(items),
        }
        shard_results["contexts"].append(context_result)

    return shard_results


def load_or_run_baseline(
    args,
    eval_contexts: List[Dict],
    tokenizer,
    model,
    cache_path: Path,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    """Load cached baseline or run on all ranks and gather."""
    # Rank 0 checks cache
    if rank == 0 and cache_path.exists() and args.cache_baseline:
        logging.info(f"Loading cached baseline from {cache_path}")
        with open(cache_path, 'r') as f:
            baseline_results = json.load(f)
        return baseline_results

    # All ranks run baseline on their shard
    logging.info("Running baseline on shard...")
    shard_results = run_baseline_on_shard(args, eval_contexts, tokenizer, model)

    # Gather results
    if world_size > 1 and dist.is_initialized():
        all_shards = [None for _ in range(world_size)]
        dist.all_gather_object(all_shards, shard_results)

        if rank == 0:
            # Merge
            all_contexts = []
            for shard in all_shards:
                if shard:
                    all_contexts.extend(shard["contexts"])

            # Aggregate
            total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in all_contexts)
            total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in all_contexts)
            total_questions = sum(ctx["num_questions"] for ctx in all_contexts)
            total_latency = sum(ctx["latency"] for ctx in all_contexts)
            total_prompt_tokens = sum(ctx["prompt_tokens"] for ctx in all_contexts)
            total_generated_tokens = sum(ctx["generated_tokens"] for ctx in all_contexts)

            baseline_results = {
                "aggregate_metrics": {
                    "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                    "f1": total_f1 / total_questions if total_questions > 0 else 0,
                    "avg_latency": total_latency / len(all_contexts),
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_generated_tokens": total_generated_tokens,
                },
                "contexts": all_contexts,
            }

            # Save cache
            if args.cache_baseline:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(baseline_results, f, indent=2)
                logging.info(f"Saved baseline to {cache_path}")

            return baseline_results
        else:
            return {}  # Non-zero ranks don't need baseline
    else:
        # Single GPU
        total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in shard_results["contexts"])
        total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in shard_results["contexts"])
        total_questions = sum(ctx["num_questions"] for ctx in shard_results["contexts"])
        total_latency = sum(ctx["latency"] for ctx in shard_results["contexts"])

        baseline_results = {
            "aggregate_metrics": {
                "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                "f1": total_f1 / total_questions if total_questions > 0 else 0,
                "avg_latency": total_latency / len(shard_results["contexts"]) if shard_results["contexts"] else 0,
                "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in shard_results["contexts"]),
                "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in shard_results["contexts"]),
            },
            "contexts": shard_results["contexts"],
        }

        if args.cache_baseline:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(baseline_results, f, indent=2)

        return baseline_results


def evaluate_checkpoint_on_shard(
    args,
    checkpoint_path: str,
    eval_contexts: List[Dict],
    tokenizer,
    model,
    epoch: int,
) -> Dict[str, Any]:
    """Evaluate checkpoint on this rank's shard."""
    mix_layers = None
    if args.mix_layers:
        mix_layers = [int(x.strip()) for x in args.mix_layers.split(',')]

    shard_results = {"contexts": []}

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = squad_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        logging.info(f"Epoch {epoch}: Evaluating {idx}/{len(eval_contexts)}: {title}")

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

        context_result = {
            "title": title,
            "metrics": result.metrics,
            "latency": result.latency,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "num_questions": len(items),
        }
        shard_results["contexts"].append(context_result)

    return shard_results


def evaluate_checkpoint(
    args,
    checkpoint_path: str,
    eval_contexts: List[Dict],
    tokenizer,
    model,
    epoch: int,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    """Evaluate checkpoint across all ranks and gather."""
    # All ranks evaluate their shard
    shard_results = evaluate_checkpoint_on_shard(
        args,
        checkpoint_path,
        eval_contexts,
        tokenizer,
        model,
        epoch,
    )

    # Gather results
    if world_size > 1 and dist.is_initialized():
        all_shards = [None for _ in range(world_size)]
        dist.all_gather_object(all_shards, shard_results)

        if rank == 0:
            # Merge
            all_contexts = []
            for shard in all_shards:
                if shard:
                    all_contexts.extend(shard["contexts"])

            # Aggregate
            total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in all_contexts)
            total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in all_contexts)
            total_questions = sum(ctx["num_questions"] for ctx in all_contexts)
            total_latency = sum(ctx["latency"] for ctx in all_contexts)

            return {
                "epoch": epoch,
                "aggregate_metrics": {
                    "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                    "f1": total_f1 / total_questions if total_questions > 0 else 0,
                    "avg_latency": total_latency / len(all_contexts),
                    "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in all_contexts),
                    "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in all_contexts),
                },
            }
        else:
            return {}  # Non-zero ranks
    else:
        # Single GPU
        total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in shard_results["contexts"])
        total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in shard_results["contexts"])
        total_questions = sum(ctx["num_questions"] for ctx in shard_results["contexts"])
        total_latency = sum(ctx["latency"] for ctx in shard_results["contexts"])

        return {
            "epoch": epoch,
            "aggregate_metrics": {
                "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                "f1": total_f1 / total_questions if total_questions > 0 else 0,
                "avg_latency": total_latency / len(shard_results["contexts"]) if shard_results["contexts"] else 0,
                "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in shard_results["contexts"]),
                "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in shard_results["contexts"]),
            },
        }


def main():
    args = parse_args()

    # Detect distributed setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # Setup logging
    log_prefix = f"[Rank {rank}] " if world_size > 1 else ""
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format=f"{log_prefix}%(message)s",
    )

    # Create output directory (only on rank 0)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize distributed backend
    if world_size > 1:
        if torch.cuda.is_available():
            device_id = rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            if rank == 0:
                logging.info(f"Using {world_size} GPUs for evaluation")
            logging.info(f"Rank {rank} using cuda:{device_id}")

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Load model and tokenizer
    if rank == 0:
        logging.info(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # For multi-GPU, each rank loads model on its own GPU
    if world_size > 1 and torch.cuda.is_available():
        device_map = {"": f"cuda:{torch.cuda.current_device()}"}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    model.eval()

    # Load evaluation data (all ranks load full dataset, then shard)
    if rank == 0:
        logging.info(f"Loading evaluation data: {args.eval_samples} samples")
    eval_contexts = load_squad_groups(
        split="validation",
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        max_contexts=args.eval_samples,
        seed=args.seed,
    )

    # Shard evaluation data across GPUs
    if world_size > 1:
        eval_contexts = eval_contexts[rank::world_size]
        logging.info(f"Evaluating {len(eval_contexts)} contexts on this shard")

    # Get or compute baseline results
    baseline_cache_path = output_dir / "baseline_results.json"
    if rank == 0:
        logging.info("=" * 60)
        logging.info("STEP 1: Baseline (Batch) Strategy")
        logging.info("=" * 60)

    baseline_results = load_or_run_baseline(
        args,
        eval_contexts,
        tokenizer,
        model,
        baseline_cache_path,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        logging.info("\nBaseline Results:")
        logging.info(f"  EM:  {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
        logging.info(f"  F1:  {baseline_results['aggregate_metrics']['f1']:.4f}")
        logging.info(f"  Latency: {baseline_results['aggregate_metrics']['avg_latency']:.2f}s")

    # Training (only on rank 0)
    safe_model_name = args.model.replace('/', '_')
    checkpoint_base = Path(args.checkpoint_dir) / args.dataset / safe_model_name
    if rank == 0:
        checkpoint_base.mkdir(parents=True, exist_ok=True)

        logging.info("\n" + "=" * 60)
        logging.info("STEP 2: Training Cross-Batch Module")
        logging.info("=" * 60)

        # Load training data
        logging.info(f"Loading training data: {args.train_samples} samples")
        train_groups = load_squad_groups(
            split="train",
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.train_samples,
            seed=args.seed,
        )

        # Create dataset
        train_dataset = SQuADGroupedDataset(
            tokenizer=tokenizer,
            groups=train_groups,
            dataset_name=args.dataset,
        )
        total_questions = sum(len(g.get("questions", [])) for g in train_groups)
        logging.info(f"Training dataset: {len(train_dataset)} contexts, {total_questions} questions")

        # Parse mix_layers
        mix_layers = None
        if args.mix_layers:
            mix_layers = [int(x.strip()) for x in args.mix_layers.split(',')]

        # Create cross-batch module
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers

        if args.module_type == "multi_layer":
            layer_indices = mix_layers if mix_layers else list(range(num_layers))
            cross_batch_module = MultiLayerCrossBatch(
                hidden_size=hidden_size,
                num_layers=num_layers,
                layer_indices=layer_indices,
                temperature=1.0,
            )
            logging.info(f"Using MultiLayerCrossBatch with {len(layer_indices)} layers: {layer_indices[:5]}...")
        elif args.module_type == "simple":
            cross_batch_module = SimpleCrossBatchGate(hidden_size=hidden_size, temperature=1.0)
        elif args.module_type == "attention":
            cross_batch_module = CrossBatchAttention(hidden_size=hidden_size, num_heads=8, temperature=1.0)
        else:  # mixer
            cross_batch_module = CrossBatchEmbeddingMixer(hidden_size=hidden_size, temperature=1.0)

        num_params = sum(p.numel() for p in cross_batch_module.parameters())
        logging.info(f"Cross-batch module parameters: {num_params:,}")

        # Create trainer
        device = str(next(model.parameters()).device)
        trainer = CrossBatchTrainer(
            model=model,
            tokenizer=tokenizer,
            cross_batch_module=cross_batch_module,
            device=device,
            learning_rate=args.lr,
            train_lm_head=False,
            local_rank=-1,
            mix_layer=-1,
        )

    # Results tracking
    all_results = {
        "config": vars(args),
        "baseline": baseline_results.get("aggregate_metrics", {}) if rank == 0 else {},
        "epochs": [],
    }

    # Train and evaluate for each epoch
    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Epoch {epoch}/{args.epochs}")
            logging.info("=" * 60)
            logging.info("Training...")

            # Train for one epoch
            history = trainer.train(
                train_dataset=train_dataset,
                num_epochs=1,
                batch_size=args.batch_size,
                save_dir=None,
                distributed=False,
                rank=0,
                world_size=1,
                grouped=True,
            )

            logging.info(f"Training complete - Loss: {history['train_loss'][-1]:.4f}, Improvement: {history['improvement'][-1]:.4f}")

            # Save checkpoint
            checkpoint_path = str(checkpoint_base / f"{args.module_type}_frozen_epoch{epoch}.pt")
            checkpoint = {
                'cross_batch_module': cross_batch_module.state_dict(),
                'config': {
                    'model': args.model,
                    'dataset': args.dataset,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'train_samples': args.train_samples,
                    'epochs': epoch,
                    'module_type': args.module_type,
                    'mix_layer': -1,
                    'mix_layers': mix_layers,
                },
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

        # Sync all ranks before evaluation
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
            # Broadcast checkpoint path
            checkpoint_path_list = [str(checkpoint_base / f"{args.module_type}_frozen_epoch{epoch}.pt")]
            dist.broadcast_object_list(checkpoint_path_list, src=0)
            checkpoint_path = checkpoint_path_list[0]
        else:
            checkpoint_path = str(checkpoint_base / f"{args.module_type}_frozen_epoch{epoch}.pt")

        # All ranks evaluate
        if rank == 0:
            logging.info(f"\nEvaluating epoch {epoch}...")

        epoch_results = evaluate_checkpoint(
            args,
            checkpoint_path,
            eval_contexts,
            tokenizer,
            model,
            epoch,
            rank=rank,
            world_size=world_size,
        )

        # Display and save results (only rank 0)
        if rank == 0:
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

    # Final summary (only rank 0)
    if rank == 0:
        logging.info("\n" + "=" * 60)
        logging.info("FINAL SUMMARY")
        logging.info("=" * 60)

        logging.info("\nBaseline (Batch):")
        logging.info(f"  EM: {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
        logging.info(f"  F1: {baseline_results['aggregate_metrics']['f1']:.4f}")

        logging.info("\nCross-Batch Results by Epoch:")
        for ep, epoch_metrics in enumerate(all_results["epochs"], start=1):
            logging.info(f"\nEpoch {ep}:")
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

    # Cleanup
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
