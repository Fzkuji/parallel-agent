"""
Cross-batch training and evaluation pipeline with DDP support.

This module provides a high-level API for training and evaluating cross-batch modules
using Distributed Data Parallel (DDP) for multi-GPU training and parallel evaluation.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _get_available_gpus(min_free_memory_gb: float = 10.0) -> List[int]:
    """Get list of GPUs with sufficient free memory.

    Args:
        min_free_memory_gb: Minimum free memory required in GB

    Returns:
        List of GPU indices with sufficient free memory
    """
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return list(range(8))  # Fallback: assume all GPUs available

        available = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split(',')
            gpu_idx = int(parts[0].strip())
            free_mb = float(parts[1].strip())
            free_gb = free_mb / 1024
            if free_gb >= min_free_memory_gb:
                available.append(gpu_idx)
        return available if available else list(range(8))
    except Exception:
        return list(range(8))  # Fallback


@dataclass
class TrainingConfig:
    """Configuration for cross-batch training."""
    model_name: str
    dataset: str = "squad"
    module_type: str = "multi_layer"
    mix_layers: Optional[List[int]] = None
    epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 1e-5
    max_new_tokens: int = 96

    # LoRA settings
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Paths
    checkpoint_dir: str = "outputs/checkpoints"
    output_dir: str = "outputs/train_and_eval"

    # DDP settings
    num_gpus: Optional[int] = None  # Auto-detect if None


def _context_to_items(context_payload: dict) -> List[dict]:
    """Convert context format to items format."""
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


def _ddp_training_worker(
    rank: int,
    world_size: int,
    gpu_id: int,
    config_dict: Dict,
    train_groups: List[Dict],
    checkpoint_dir: str,
    epoch: int,
):
    """DDP worker process for training on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    print(f"[Worker {rank}] Starting DDP training on GPU {gpu_id}", flush=True)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from src.cross_batch import (
        CrossBatchTrainer,
        SQuADGroupedDataset,
        SimpleCrossBatchGate,
        MultiLayerCrossBatch,
        MultiLayerCrossBatchAttention,
        CrossBatchAttention,
        CrossBatchEmbeddingMixer,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    print(f"[Worker {rank}] Initialized DDP process group", flush=True)

    config = TrainingConfig(**config_dict)
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.eval()
    print(f"[Worker {rank}] Model loaded", flush=True)

    if config.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            if rank == 0:
                model.print_trainable_parameters()
        except ImportError:
            if rank == 0:
                print("peft not installed. Cannot use LoRA.", flush=True)
            config.use_lora = False

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    if config.module_type == "multi_layer":
        layer_indices = config.mix_layers if config.mix_layers else list(range(num_layers))
        cross_batch_module = MultiLayerCrossBatch(
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_indices=layer_indices,
            temperature=1.0,
        )
    elif config.module_type == "multi_layer_attention":
        layer_indices = config.mix_layers if config.mix_layers else list(range(num_layers))
        cross_batch_module = MultiLayerCrossBatchAttention(
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_indices=layer_indices,
            num_heads=8,
            temperature=1.0,
            use_gate=True,
        )
    elif config.module_type == "simple":
        cross_batch_module = SimpleCrossBatchGate(hidden_size=hidden_size, temperature=1.0)
    elif config.module_type == "attention":
        cross_batch_module = CrossBatchAttention(hidden_size=hidden_size, num_heads=8, temperature=1.0)
    else:
        cross_batch_module = CrossBatchEmbeddingMixer(hidden_size=hidden_size, temperature=1.0)

    cross_batch_module = cross_batch_module.to(device)
    if rank == 0:
        num_params = sum(p.numel() for p in cross_batch_module.parameters())
        print(f"[Worker {rank}] Cross-batch module parameters: {num_params:,}", flush=True)

    trainer = CrossBatchTrainer(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
        learning_rate=config.learning_rate,
        train_lm_head=False,
        train_lora=config.use_lora,
        local_rank=rank,
        mix_layer=-1,
    )

    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer,
        groups=train_groups,
        dataset_name=config.dataset,
    )

    if rank == 0:
        total_questions = sum(len(g.get("questions", g.get("items", []))) for g in train_groups)
        print(f"[Worker {rank}] Training dataset: {len(train_dataset)} contexts, {total_questions} questions", flush=True)

    print(f"[Worker {rank}] Starting training epoch {epoch}...", flush=True)
    history = trainer.train(
        train_dataset=train_dataset,
        num_epochs=1,
        batch_size=config.batch_size,
        save_dir=None,
        distributed=True,
        rank=rank,
        world_size=world_size,
        grouped=True,
    )
    print(f"[Worker {rank}] Training complete", flush=True)

    if rank == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"{config.module_type}_frozen_epoch{epoch}.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        module_state = trainer.cross_batch_module_unwrapped.state_dict()

        checkpoint = {
            'cross_batch_module': module_state,
            'config': {
                'model': config.model_name,
                'dataset': config.dataset,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'epochs': epoch,
                'module_type': config.module_type,
                'mix_layer': -1,
                'mix_layers': config.mix_layers,
                'use_lora': config.use_lora,
            },
            'history': history,
        }

        if config.use_lora:
            lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
            checkpoint['lora'] = lora_state_dict

        torch.save(checkpoint, checkpoint_path)
        print(f"[Worker {rank}] Saved checkpoint to {checkpoint_path}", flush=True)

    dist.destroy_process_group()
    print(f"[Worker {rank}] DDP cleanup complete", flush=True)


def _eval_worker(
    rank: int,
    world_size: int,
    gpu_id: int,
    model_name: str,
    eval_contexts: List[Dict],
    checkpoint_path: str,
    output_dir: str,
    max_new_tokens: int,
    dataset: str,
    module_type: str,
    mix_layers: Optional[List[int]],
    epoch: int,
):
    """Worker process for parallel evaluation on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[Eval Worker {rank}] GPU {gpu_id}: Starting, {len(eval_contexts)} contexts", flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.strategies.cross_batch import run_cross_batch_multi_strategy

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.eval()
    print(f"[Eval Worker {rank}] Model loaded", flush=True)

    shard_results = {"contexts": []}

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = _context_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        if idx % 10 == 0:
            print(f"[Eval Worker {rank}] Processing {idx}/{len(eval_contexts)}", flush=True)

        result = run_cross_batch_multi_strategy(
            items,
            tokenizer,
            model,
            max_new_tokens=max_new_tokens,
            strategy_name=f"collab_hidden_epoch{epoch}",
            dataset=dataset,
            mix_method=module_type,
            mix_layer=-1,
            mix_layers=mix_layers,
            checkpoint_path=checkpoint_path,
            enable_cross_batch=True,
        )

        shard_results["contexts"].append({
            "title": title,
            "metrics": result.metrics,
            "latency": result.latency,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "num_questions": len(items),
        })

    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f"eval_shard_{rank}.json")
    with open(temp_file, 'w') as f:
        json.dump(shard_results, f)

    print(f"[Eval Worker {rank}] Done, saved to {temp_file}", flush=True)


class CrossBatchPipeline:
    """High-level pipeline for cross-batch training and evaluation with parallel support."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.cross_batch_module = None
        self.trainer = None
        self.lora_model = None

    def _run_ddp_training(self, train_groups: List[Dict], checkpoint_dir: str, epoch: int, num_gpus: int, gpu_ids: List[int]):
        """Run DDP training across multiple GPUs using multiprocessing."""
        config_dict = {
            'model_name': self.config.model_name,
            'dataset': self.config.dataset,
            'module_type': self.config.module_type,
            'mix_layers': self.config.mix_layers,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'max_new_tokens': self.config.max_new_tokens,
            'use_lora': self.config.use_lora,
            'lora_r': self.config.lora_r,
            'lora_alpha': self.config.lora_alpha,
            'lora_dropout': self.config.lora_dropout,
            'lora_target_modules': self.config.lora_target_modules,
            'checkpoint_dir': self.config.checkpoint_dir,
            'output_dir': self.config.output_dir,
        }

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        logger.info(f"Starting {num_gpus} DDP workers on GPUs: {gpu_ids}")

        processes = []
        for rank in range(num_gpus):
            gpu_id = gpu_ids[rank]
            p = mp.Process(
                target=_ddp_training_worker,
                args=(rank, num_gpus, gpu_id, config_dict, train_groups, checkpoint_dir, epoch),
            )
            p.start()
            processes.append(p)
            logger.info(f"Started DDP worker {rank} on GPU {gpu_id} (PID: {p.pid})")

        for p in processes:
            p.join()

        logger.info("All DDP workers finished")

    def _run_parallel_eval(
        self,
        eval_contexts: List[Dict],
        checkpoint_path: str,
        epoch: int,
        num_gpus: int,
        gpu_ids: List[int],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Run parallel evaluation across multiple GPUs."""
        # Shard data across GPUs
        shards = [[] for _ in range(num_gpus)]
        for i, ctx in enumerate(eval_contexts):
            shards[i % num_gpus].append(ctx)

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        logger.info(f"Starting {num_gpus} eval workers on GPUs: {gpu_ids}")

        # Start all workers
        processes = []
        for rank in range(num_gpus):
            gpu_id = gpu_ids[rank]
            p = mp.Process(
                target=_eval_worker,
                args=(
                    rank, num_gpus, gpu_id, self.config.model_name,
                    shards[rank], checkpoint_path, output_dir,
                    self.config.max_new_tokens, self.config.dataset,
                    self.config.module_type, self.config.mix_layers, epoch,
                ),
            )
            p.start()
            processes.append(p)
            logger.info(f"Started eval worker {rank} on GPU {gpu_id} (PID: {p.pid})")

        for p in processes:
            p.join()

        logger.info("All eval workers finished, gathering results...")

        # Gather results from all shards
        all_contexts = []
        for rank in range(num_gpus):
            shard_file = os.path.join(output_dir, f"eval_shard_{rank}.json")
            if os.path.exists(shard_file):
                with open(shard_file, 'r') as f:
                    shard_data = json.load(f)
                    all_contexts.extend(shard_data["contexts"])
                os.unlink(shard_file)
            else:
                logger.warning(f"Missing eval shard file: {shard_file}")

        # Aggregate metrics
        total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in all_contexts)
        total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in all_contexts)
        total_questions = sum(ctx["num_questions"] for ctx in all_contexts)
        total_latency = sum(ctx["latency"] for ctx in all_contexts)

        return {
            "epoch": epoch,
            "aggregate_metrics": {
                "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                "f1": total_f1 / total_questions if total_questions > 0 else 0,
                "avg_latency": total_latency / len(all_contexts) if all_contexts else 0,
                "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in all_contexts),
                "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in all_contexts),
            },
            "contexts": all_contexts,
        }

    def run(
        self,
        train_groups: List[Dict],
        eval_contexts: List[Dict],
        baseline_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the full training and evaluation pipeline with parallel support.

        Args:
            train_groups: Training data groups
            eval_contexts: Evaluation contexts
            baseline_results: Baseline results for comparison

        Returns:
            Dictionary with all results
        """
        # Auto-detect available GPUs with sufficient free memory
        available_gpus = _get_available_gpus(min_free_memory_gb=10.0)
        logger.info(f"Available GPUs with sufficient memory: {available_gpus}")

        num_gpus = self.config.num_gpus
        if num_gpus is None:
            num_gpus = len(available_gpus) if available_gpus else 1
        else:
            num_gpus = min(num_gpus, len(available_gpus))
        num_gpus = max(1, num_gpus)

        self._gpu_ids = available_gpus[:num_gpus] if available_gpus else list(range(num_gpus))
        logger.info(f"Using {num_gpus} GPU(s) for training and evaluation: {self._gpu_ids}")

        # Checkpoint directory
        safe_model_name = self.config.model_name.replace('/', '_')
        checkpoint_base = Path(self.config.checkpoint_dir) / self.config.dataset / safe_model_name
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Results tracking
        all_results = {
            "config": {
                "model": self.config.model_name,
                "dataset": self.config.dataset,
                "module_type": self.config.module_type,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "use_lora": self.config.use_lora,
                "num_gpus": num_gpus,
            },
            "baseline": baseline_results.get("aggregate_metrics", {}),
            "epochs": [],
        }

        baseline_em = baseline_results['aggregate_metrics']['strict_acc']
        baseline_f1 = baseline_results['aggregate_metrics']['f1']

        # Train and evaluate for each epoch
        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Epoch {epoch}/{self.config.epochs}")
            logger.info("=" * 60)

            # Train with DDP
            logger.info(f"Training with DDP on {num_gpus} GPUs...")
            self._run_ddp_training(train_groups, str(checkpoint_base), epoch, num_gpus, self._gpu_ids)

            checkpoint_path = str(checkpoint_base / f"{self.config.module_type}_frozen_epoch{epoch}.pt")

            # Load training history from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            history = checkpoint.get('history', {})
            if history:
                logger.info(f"Training complete - Loss: {history.get('train_loss', [0])[-1]:.4f}, "
                           f"Improvement: {history.get('improvement', [0])[-1]:.4f}")

            # Parallel evaluation
            logger.info(f"Evaluating epoch {epoch} on {num_gpus} GPUs...")
            epoch_results = self._run_parallel_eval(
                eval_contexts, checkpoint_path, epoch, num_gpus, self._gpu_ids, str(output_dir)
            )

            # Log results
            epoch_metrics = epoch_results["aggregate_metrics"]
            logger.info(f"\nEpoch {epoch} Results:")
            logger.info(f"  EM:  {epoch_metrics['strict_acc']:.4f} (Δ {epoch_metrics['strict_acc'] - baseline_em:+.4f})")
            logger.info(f"  F1:  {epoch_metrics['f1']:.4f} (Δ {epoch_metrics['f1'] - baseline_f1:+.4f})")
            logger.info(f"  Latency: {epoch_metrics['avg_latency']:.2f}s")

            all_results["epochs"].append(epoch_metrics)

            # Save intermediate results
            results_path = output_dir / f"results_epoch{epoch}.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)

        # Final summary
        self._log_summary(all_results, baseline_results)

        # Save final results
        final_results_path = output_dir / "final_results.json"
        with open(final_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved final results to {final_results_path}")

        return all_results

    def _log_summary(self, all_results: Dict, baseline_results: Dict):
        """Log final summary."""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)

        logger.info("\nBaseline (vLLM):")
        logger.info(f"  EM: {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
        logger.info(f"  F1: {baseline_results['aggregate_metrics']['f1']:.4f}")

        logger.info("\nCross-Batch Results by Epoch:")
        baseline_em = baseline_results['aggregate_metrics']['strict_acc']
        baseline_f1 = baseline_results['aggregate_metrics']['f1']

        for ep, epoch_metrics in enumerate(all_results["epochs"], start=1):
            logger.info(f"\nEpoch {ep}:")
            logger.info(f"  EM: {epoch_metrics['strict_acc']:.4f} (Δ {epoch_metrics['strict_acc'] - baseline_em:+.4f})")
            logger.info(f"  F1: {epoch_metrics['f1']:.4f} (Δ {epoch_metrics['f1'] - baseline_f1:+.4f})")

        if all_results["epochs"]:
            best_epoch = max(range(len(all_results["epochs"])),
                            key=lambda i: all_results["epochs"][i]["f1"]) + 1
            logger.info(f"\nBest Epoch: {best_epoch} (F1: {all_results['epochs'][best_epoch-1]['f1']:.4f})")
