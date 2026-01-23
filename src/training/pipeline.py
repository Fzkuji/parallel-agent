"""
Cross-batch training and evaluation pipeline with DDP support.

This module provides a high-level API for training and evaluating cross-batch modules
using Distributed Data Parallel (DDP) for multi-GPU training.
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
    config_dict: Dict,
    train_groups: List[Dict],
    checkpoint_dir: str,
    epoch: int,
):
    """DDP worker process for training on a single GPU.

    IMPORTANT: This runs in a separate process with isolated CUDA context.
    Environment variables must be set BEFORE any CUDA imports.
    """
    # Set environment variables BEFORE importing CUDA-related modules
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    print(f"[Worker {rank}] Starting DDP training on GPU {rank}", flush=True)

    # Now import CUDA-related modules
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

    # Initialize distributed training
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    print(f"[Worker {rank}] Initialized DDP process group", flush=True)

    # Reconstruct config
    config = TrainingConfig(**config_dict)

    # Load model on this GPU (cuda:0 because CUDA_VISIBLE_DEVICES makes it the only visible GPU)
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

    # Apply LoRA if enabled
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

    # Create cross-batch module
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
    else:  # mixer
        cross_batch_module = CrossBatchEmbeddingMixer(hidden_size=hidden_size, temperature=1.0)

    cross_batch_module = cross_batch_module.to(device)
    if rank == 0:
        num_params = sum(p.numel() for p in cross_batch_module.parameters())
        print(f"[Worker {rank}] Cross-batch module parameters: {num_params:,}", flush=True)

    # Create trainer with DDP
    trainer = CrossBatchTrainer(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
        learning_rate=config.learning_rate,
        train_lm_head=False,
        train_lora=config.use_lora,
        local_rank=rank,  # Enable DDP
        mix_layer=-1,
    )

    # Create training dataset
    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer,
        groups=train_groups,
        dataset_name=config.dataset,
    )

    if rank == 0:
        total_questions = sum(len(g.get("questions", g.get("items", []))) for g in train_groups)
        print(f"[Worker {rank}] Training dataset: {len(train_dataset)} contexts, {total_questions} questions", flush=True)

    # Train for one epoch
    print(f"[Worker {rank}] Starting training epoch {epoch}...", flush=True)
    history = trainer.train(
        train_dataset=train_dataset,
        num_epochs=1,
        batch_size=config.batch_size,
        save_dir=None,  # Don't save during training, we'll save manually
        distributed=True,
        rank=rank,
        world_size=world_size,
        grouped=True,
    )
    print(f"[Worker {rank}] Training complete", flush=True)

    # Only rank 0 saves checkpoint
    if rank == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"{config.module_type}_frozen_epoch{epoch}.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get unwrapped module state
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

    # Cleanup
    dist.destroy_process_group()
    print(f"[Worker {rank}] DDP cleanup complete", flush=True)


class CrossBatchPipeline:
    """High-level pipeline for cross-batch training and evaluation with DDP support."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.cross_batch_module = None
        self.trainer = None
        self.lora_model = None

    def _run_ddp_training(self, train_groups: List[Dict], checkpoint_dir: str, epoch: int, num_gpus: int):
        """Run DDP training across multiple GPUs using multiprocessing."""
        # Convert config to dict for serialization
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

        # Set spawn method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Start all workers
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=_ddp_training_worker,
                args=(rank, num_gpus, config_dict, train_groups, checkpoint_dir, epoch),
            )
            p.start()
            processes.append(p)
            logger.info(f"Started DDP worker {rank} on GPU {rank} (PID: {p.pid})")

        # Wait for all workers
        for p in processes:
            p.join()

        logger.info("All DDP workers finished")

    def load_model_for_eval(self):
        """Load model for evaluation (single GPU)."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading model for evaluation: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        )
        self.model.eval()

        # Apply LoRA if enabled
        if self.config.use_lora:
            self._apply_lora()

        return self

    def _apply_lora(self):
        """Apply LoRA to the model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
            )
            self.lora_model = get_peft_model(self.model, lora_config)
            self.lora_model.print_trainable_parameters()
            self.model = self.lora_model
            logger.info(f"LoRA applied: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        except ImportError:
            logger.warning("peft not installed. Cannot use LoRA.")
            self.config.use_lora = False

    def evaluate(
        self,
        eval_contexts: List[Dict],
        checkpoint_path: str,
        epoch: int,
    ) -> Dict[str, Any]:
        """Evaluate on the given contexts."""
        from src.strategies.cross_batch import run_cross_batch_multi_strategy

        eval_results = {"contexts": []}

        for idx, context_payload in enumerate(eval_contexts, start=1):
            items = _context_to_items(context_payload)
            title = context_payload.get("title", f"context-{idx}")

            if idx % 10 == 0:
                logger.info(f"Evaluating {idx}/{len(eval_contexts)}")

            result = run_cross_batch_multi_strategy(
                items,
                self.tokenizer,
                self.model,
                max_new_tokens=self.config.max_new_tokens,
                strategy_name=f"collab_hidden_epoch{epoch}",
                dataset=self.config.dataset,
                mix_method=self.config.module_type,
                mix_layer=-1,
                mix_layers=self.config.mix_layers,
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

        # Aggregate metrics
        total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in eval_results["contexts"])
        total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in eval_results["contexts"])
        total_questions = sum(ctx["num_questions"] for ctx in eval_results["contexts"])
        total_latency = sum(ctx["latency"] for ctx in eval_results["contexts"])

        return {
            "epoch": epoch,
            "aggregate_metrics": {
                "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                "f1": total_f1 / total_questions if total_questions > 0 else 0,
                "avg_latency": total_latency / len(eval_results["contexts"]) if eval_results["contexts"] else 0,
                "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in eval_results["contexts"]),
                "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in eval_results["contexts"]),
            },
            "contexts": eval_results["contexts"],
        }

    def run(
        self,
        train_groups: List[Dict],
        eval_contexts: List[Dict],
        baseline_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the full training and evaluation pipeline with DDP.

        Args:
            train_groups: Training data groups
            eval_contexts: Evaluation contexts
            baseline_results: Baseline results for comparison

        Returns:
            Dictionary with all results
        """
        # Auto-detect GPUs
        num_gpus = self.config.num_gpus
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_gpus = max(1, num_gpus)
        logger.info(f"Using {num_gpus} GPU(s) for DDP training")

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
            self._run_ddp_training(train_groups, str(checkpoint_base), epoch, num_gpus)

            checkpoint_path = str(checkpoint_base / f"{self.config.module_type}_frozen_epoch{epoch}.pt")

            # Load training history from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            history = checkpoint.get('history', {})
            if history:
                logger.info(f"Training complete - Loss: {history.get('train_loss', [0])[-1]:.4f}, "
                           f"Improvement: {history.get('improvement', [0])[-1]:.4f}")

            # Load model for evaluation (after DDP training is done)
            if self.model is None:
                self.load_model_for_eval()

            # Evaluate
            logger.info(f"Evaluating epoch {epoch}...")
            epoch_results = self.evaluate(eval_contexts, checkpoint_path, epoch)

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
