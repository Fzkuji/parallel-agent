"""
Complete training and evaluation pipeline for cross-batch module.

This script:
1. Runs baseline evaluation using vLLM (fast, single-sample inference)
2. Trains cross-batch module using transformers
3. Evaluates cross-batch model using transformers

Baseline uses vLLM for fast inference (same as exp2a_shared_context.py).
Cross-batch uses transformers for compatibility with the training code.

Usage:
  Multi-GPU (auto-detected):
    python scripts/train_and_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct --epochs 3 --cache-baseline

  Single GPU:
    python scripts/train_and_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct --epochs 3 --cache-baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path (before other imports)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Simple prompt format (same as exp2a_shared_context.py)
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
Give a short, direct answer. Do not explain or elaborate."""


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
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--module-type", type=str, default="multi_layer", choices=["simple", "multi_layer", "multi_layer_attention", "attention", "mixer"], help="Cross-batch module type")
    parser.add_argument("--mix-layers", type=str, default=None, help="Comma-separated layer indices for multi_layer mode (None = all layers)")

    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens for generation")

    # Paths
    parser.add_argument("--output-dir", type=str, default="outputs/train_and_eval", help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Checkpoint directory")
    parser.add_argument("--cache-baseline", action="store_true", help="Cache baseline results to avoid recomputation")
    parser.add_argument("--force", action="store_true", help="Force re-run baseline even if cached")

    # LoRA parameters
    parser.add_argument("--use-lora", action="store_true", default=False, help="Use LoRA for model training")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="LoRA target modules (comma-separated)")

    # vLLM options
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode for Qwen3 models")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    return parser.parse_args()


def context_to_items(context_payload: dict) -> List[dict]:
    """Convert context format to items format for batch strategy."""
    if "items" in context_payload:
        return context_payload["items"]

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


def baseline_worker(
    rank: int,
    world_size: int,
    gpu_id: int,
    model_name: str,
    eval_contexts: List[Dict],
    output_dir: str,
    max_new_tokens: int,
    dataset: str,
    enable_thinking: bool,
):
    """Worker process for vLLM baseline inference on a single GPU.

    IMPORTANT: This runs in a separate process with isolated CUDA context.
    """
    # Set environment variables BEFORE any CUDA imports
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_DISABLE_FRONTEND_MULTIPROCESSING"] = "1"
    os.environ["VLLM_NO_PROGRESS_BAR"] = "1"
    os.environ["TQDM_DISABLE"] = "1"

    logger.info(f"[Worker {rank}] GPU {gpu_id}: Starting baseline, {len(eval_contexts)} contexts")

    # Now import vLLM and torch
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from src.models import Question
    from src.evaluation import evaluate_predictions
    from src.inference import extract_answer

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vllm_model = LLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
    )

    logger.info(f"[Worker {rank}] Model loaded, running inference...")

    # Run inference on shard
    shard_results = {"contexts": []}

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = context_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        if idx % 10 == 0:
            logger.info(f"[Worker {rank}] Processing {idx}/{len(eval_contexts)}")

        # Build prompts
        prompts = []
        for item in items:
            prompt = f"Passage:\n{item['context']}\n\nQuestion: {item['question']}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            prompts.append(full_prompt)

        # Generate
        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        start_time = time.perf_counter()
        outputs = vllm_model.generate(prompts, sampling_params, use_tqdm=False)
        latency = time.perf_counter() - start_time

        # Process results
        question_lookup = {
            item["qid"]: Question(
                qid=item["qid"],
                text=item["question"],
                priority=1.0,
                answer_tokens=item.get("answer_tokens", 12),
                type_hint=None,
                references=item.get("references", []),
            )
            for item in items
        }

        answer_records = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, item in enumerate(items):
            output = outputs[i]
            raw_text = output.outputs[0].text
            final_answer, strict_valid = extract_answer(raw_text, dataset)
            answer_records[item["qid"]] = (final_answer, strict_valid)
            total_prompt_tokens += len(output.prompt_token_ids)
            total_completion_tokens += len(output.outputs[0].token_ids)

        metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

        shard_results["contexts"].append({
            "title": title,
            "metrics": metrics,
            "latency": latency,
            "prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_completion_tokens,
            "num_questions": len(items),
        })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f"baseline_shard_{rank}.json")
    with open(temp_file, 'w') as f:
        json.dump(shard_results, f)

    logger.info(f"[Worker {rank}] Done, saved to {temp_file}")

    # Explicitly cleanup vLLM to avoid hanging
    del vllm_model
    torch.cuda.empty_cache()

    # Force process exit to kill any lingering vLLM subprocesses
    import gc
    gc.collect()
    os._exit(0)


def run_baseline_parallel(
    args,
    all_eval_contexts: List[Dict],
    cache_path: Path,
    num_gpus: int,
) -> Dict[str, Any]:
    """Run vLLM baseline in parallel using multiprocessing."""
    import multiprocessing as mp

    # Check cache
    if cache_path.exists() and args.cache_baseline and not args.force:
        logger.info(f"Loading cached baseline from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    # Delete old cache if force
    if args.force and cache_path.exists():
        logger.info(f"--force specified, removing cached baseline")
        cache_path.unlink()
        for shard_file in cache_path.parent.glob("baseline_shard_*.json"):
            shard_file.unlink()

    logger.info(f"Running vLLM baseline on {num_gpus} GPU(s)...")

    # Shard data
    shards = [[] for _ in range(num_gpus)]
    for i, ctx in enumerate(all_eval_contexts):
        shards[i % num_gpus].append(ctx)

    # Clean up old shard files
    for shard_file in cache_path.parent.glob("baseline_shard_*.json"):
        shard_file.unlink()

    if num_gpus > 1:
        # Multi-GPU: use multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=baseline_worker,
                args=(
                    rank, num_gpus, rank, args.model,
                    shards[rank], str(cache_path.parent),
                    args.max_new_tokens, args.dataset, args.enable_thinking,
                )
            )
            p.start()
            processes.append(p)
            logger.info(f"Started baseline worker {rank} on GPU {rank} (PID: {p.pid})")

        # Wait for all workers with timeout
        timeout = 3600  # 1 hour max
        for p in processes:
            p.join(timeout=timeout)
            if p.is_alive():
                logger.warning(f"Worker {p.pid} timed out, terminating...")
                p.terminate()
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()

        logger.info("All baseline workers finished")
    else:
        # Single GPU: still run in subprocess to handle cleanup properly
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        p = mp.Process(
            target=baseline_worker,
            args=(
                0, 1, 0, args.model,
                all_eval_contexts, str(cache_path.parent),
                args.max_new_tokens, args.dataset, args.enable_thinking,
            )
        )
        p.start()
        logger.info(f"Started baseline worker on GPU 0 (PID: {p.pid})")
        p.join(timeout=3600)
        if p.is_alive():
            logger.warning("Worker timed out, terminating...")
            p.terminate()
            p.join(timeout=10)

    # Gather results
    logger.info("Gathering baseline results...")
    all_contexts = []
    for rank in range(num_gpus):
        shard_file = cache_path.parent / f"baseline_shard_{rank}.json"
        if shard_file.exists():
            with open(shard_file, 'r') as f:
                shard_data = json.load(f)
                all_contexts.extend(shard_data["contexts"])
            shard_file.unlink()

    # Aggregate
    total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in all_contexts)
    total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in all_contexts)
    total_questions = sum(ctx["num_questions"] for ctx in all_contexts)
    total_latency = sum(ctx["latency"] for ctx in all_contexts)

    baseline_results = {
        "aggregate_metrics": {
            "strict_acc": total_em / total_questions if total_questions > 0 else 0,
            "f1": total_f1 / total_questions if total_questions > 0 else 0,
            "avg_latency": total_latency / len(all_contexts) if all_contexts else 0,
            "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in all_contexts),
            "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in all_contexts),
        },
        "contexts": all_contexts,
    }

    # Save cache
    if args.cache_baseline:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        logger.info(f"Saved baseline to {cache_path}")

    return baseline_results


def run_training_and_eval(
    args,
    all_eval_contexts: List[Dict],
    baseline_results: Dict[str, Any],
    num_gpus: int,
):
    """Run training and evaluation (single process, single GPU for now).

    TODO: Add multi-GPU training support with proper DDP.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.datasets.squad import load_squad_groups
    from src.datasets.hotpot import load_hotpot_groups
    from src.strategies.cross_batch import run_cross_batch_multi_strategy
    from src.cross_batch import (
        CrossBatchTrainer,
        SQuADGroupedDataset,
        SimpleCrossBatchGate,
        MultiLayerCrossBatch,
        MultiLayerCrossBatchAttention,
        CrossBatchAttention,
        CrossBatchEmbeddingMixer,
    )

    output_dir = Path(args.output_dir)

    # Load model
    logger.info(f"\nLoading transformers model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.eval()

    # Apply LoRA if enabled
    lora_model = None
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            lora_model = get_peft_model(model, lora_config)
            lora_model.print_trainable_parameters()
            logger.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
            model = lora_model
        except ImportError:
            logger.warning("peft not installed. Cannot use LoRA.")
            args.use_lora = False

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

    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer,
        groups=train_groups,
        dataset_name=args.dataset,
    )
    total_questions = sum(len(g.get("questions", g.get("items", []))) for g in train_groups)
    logger.info(f"Training dataset: {len(train_dataset)} contexts, {total_questions} questions")

    # Create cross-batch module
    mix_layers = None
    if args.mix_layers:
        mix_layers = [int(x.strip()) for x in args.mix_layers.split(',')]

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
        logger.info(f"Using MultiLayerCrossBatch with {len(layer_indices)} layers")
    elif args.module_type == "multi_layer_attention":
        layer_indices = mix_layers if mix_layers else list(range(num_layers))
        cross_batch_module = MultiLayerCrossBatchAttention(
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_indices=layer_indices,
            num_heads=8,
            temperature=1.0,
            use_gate=True,
        )
        logger.info(f"Using MultiLayerCrossBatchAttention with {len(layer_indices)} layers")
    elif args.module_type == "simple":
        cross_batch_module = SimpleCrossBatchGate(hidden_size=hidden_size, temperature=1.0)
    elif args.module_type == "attention":
        cross_batch_module = CrossBatchAttention(hidden_size=hidden_size, num_heads=8, temperature=1.0)
    else:
        cross_batch_module = CrossBatchEmbeddingMixer(hidden_size=hidden_size, temperature=1.0)

    num_params = sum(p.numel() for p in cross_batch_module.parameters())
    logger.info(f"Cross-batch module parameters: {num_params:,}")

    # Create trainer
    device = str(next(model.parameters()).device)
    trainer = CrossBatchTrainer(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
        learning_rate=args.lr,
        train_lm_head=False,
        train_lora=args.use_lora,
        local_rank=-1,
        mix_layer=-1,
    )

    # Checkpoint directory
    safe_model_name = args.model.replace('/', '_')
    checkpoint_base = Path(args.checkpoint_dir) / args.dataset / safe_model_name
    checkpoint_base.mkdir(parents=True, exist_ok=True)

    # Results tracking
    all_results = {
        "config": vars(args),
        "baseline": baseline_results.get("aggregate_metrics", {}),
        "epochs": [],
    }

    # Train and evaluate
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info("=" * 60)

        # Train
        logger.info("Training...")
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
        logger.info(f"Training complete - Loss: {history['train_loss'][-1]:.4f}, Improvement: {history['improvement'][-1]:.4f}")

        # Save checkpoint
        checkpoint_path = str(checkpoint_base / f"{args.module_type}_frozen_epoch{epoch}.pt")
        if hasattr(trainer, 'cross_batch_module_unwrapped'):
            module_state = trainer.cross_batch_module_unwrapped.state_dict()
        else:
            module_state = cross_batch_module.state_dict()

        checkpoint = {
            'cross_batch_module': module_state,
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
                'use_lora': args.use_lora,
            },
        }
        if args.use_lora and lora_model is not None:
            lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
            checkpoint['lora'] = lora_state_dict
            logger.info(f"Saved {len(lora_state_dict)} LoRA tensors")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Evaluate
        logger.info(f"\nEvaluating epoch {epoch}...")
        eval_results = {"contexts": []}

        for idx, context_payload in enumerate(all_eval_contexts, start=1):
            items = context_to_items(context_payload)
            title = context_payload.get("title", f"context-{idx}")

            if idx % 10 == 0:
                logger.info(f"Evaluating {idx}/{len(all_eval_contexts)}")

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

            eval_results["contexts"].append({
                "title": title,
                "metrics": result.metrics,
                "latency": result.latency,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "num_questions": len(items),
            })

        # Aggregate eval results
        total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in eval_results["contexts"])
        total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in eval_results["contexts"])
        total_questions = sum(ctx["num_questions"] for ctx in eval_results["contexts"])
        total_latency = sum(ctx["latency"] for ctx in eval_results["contexts"])

        epoch_metrics = {
            "strict_acc": total_em / total_questions if total_questions > 0 else 0,
            "f1": total_f1 / total_questions if total_questions > 0 else 0,
            "avg_latency": total_latency / len(eval_results["contexts"]) if eval_results["contexts"] else 0,
            "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in eval_results["contexts"]),
            "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in eval_results["contexts"]),
        }

        baseline_em = baseline_results['aggregate_metrics']['strict_acc']
        baseline_f1 = baseline_results['aggregate_metrics']['f1']

        logger.info(f"\nEpoch {epoch} Results:")
        logger.info(f"  EM:  {epoch_metrics['strict_acc']:.4f} (Δ {epoch_metrics['strict_acc'] - baseline_em:+.4f})")
        logger.info(f"  F1:  {epoch_metrics['f1']:.4f} (Δ {epoch_metrics['f1'] - baseline_f1:+.4f})")
        logger.info(f"  Latency: {epoch_metrics['avg_latency']:.2f}s")

        all_results["epochs"].append(epoch_metrics)

        # Save results
        results_path = output_dir / f"results_epoch{epoch}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved results to {results_path}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    logger.info("\nBaseline (vLLM):")
    logger.info(f"  EM: {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
    logger.info(f"  F1: {baseline_results['aggregate_metrics']['f1']:.4f}")

    logger.info("\nCross-Batch Results by Epoch:")
    for ep, epoch_metrics in enumerate(all_results["epochs"], start=1):
        logger.info(f"\nEpoch {ep}:")
        logger.info(f"  EM: {epoch_metrics['strict_acc']:.4f} "
                   f"(Δ {epoch_metrics['strict_acc'] - baseline_results['aggregate_metrics']['strict_acc']:+.4f})")
        logger.info(f"  F1: {epoch_metrics['f1']:.4f} "
                   f"(Δ {epoch_metrics['f1'] - baseline_results['aggregate_metrics']['f1']:+.4f})")

    # Find best epoch
    if all_results["epochs"]:
        best_epoch = max(range(len(all_results["epochs"])),
                        key=lambda i: all_results["epochs"][i]["f1"]) + 1
        logger.info(f"\nBest Epoch: {best_epoch} (F1: {all_results['epochs'][best_epoch-1]['f1']:.4f})")

    # Save final results
    final_results_path = output_dir / "final_results.json"
    with open(final_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved final results to {final_results_path}")

    return all_results


def main():
    import torch
    from src.datasets.squad import load_squad_groups
    from src.datasets.hotpot import load_hotpot_groups

    args = parse_args()

    # Detect GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s)")
    else:
        num_gpus = 0
        logger.info("No GPU detected, using CPU")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation data
    logger.info(f"Loading evaluation data: {args.eval_samples} samples")

    if args.dataset == "hotpot":
        all_eval_contexts = load_hotpot_groups(
            split="validation",
            max_contexts=args.eval_samples,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            seed=args.seed + 1000,
        )
    else:
        all_eval_contexts = load_squad_groups(
            split="validation",
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.eval_samples,
            seed=args.seed + 1000,
        )

    logger.info(f"Loaded {len(all_eval_contexts)} evaluation contexts")

    # Step 1: Baseline (vLLM) - parallel across all GPUs
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Baseline (vLLM) Strategy")
    logger.info("=" * 60)

    baseline_cache_path = output_dir / "baseline_results.json"
    baseline_results = run_baseline_parallel(
        args,
        all_eval_contexts,
        baseline_cache_path,
        num_gpus=max(1, num_gpus),
    )

    logger.info("\nBaseline Results:")
    logger.info(f"  EM:  {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
    logger.info(f"  F1:  {baseline_results['aggregate_metrics']['f1']:.4f}")
    logger.info(f"  Latency: {baseline_results['aggregate_metrics']['avg_latency']:.2f}s")

    # Step 2: Training and evaluation (single GPU for now)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Training Cross-Batch Module")
    logger.info("=" * 60)

    run_training_and_eval(
        args,
        all_eval_contexts,
        baseline_results,
        num_gpus=num_gpus,
    )

    logger.info("\n" + "=" * 60)
    logger.info("DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
