"""
Cross-batch training and evaluation pipeline.

This module provides a high-level API for training and evaluating cross-batch modules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
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
from src.strategies.cross_batch import run_cross_batch_multi_strategy

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


class CrossBatchPipeline:
    """High-level pipeline for cross-batch training and evaluation."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.cross_batch_module = None
        self.trainer = None
        self.lora_model = None

    def load_model(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
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

    def create_cross_batch_module(self) -> torch.nn.Module:
        """Create the cross-batch module based on config."""
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers

        if self.config.module_type == "multi_layer":
            layer_indices = self.config.mix_layers if self.config.mix_layers else list(range(num_layers))
            self.cross_batch_module = MultiLayerCrossBatch(
                hidden_size=hidden_size,
                num_layers=num_layers,
                layer_indices=layer_indices,
                temperature=1.0,
            )
            logger.info(f"Created MultiLayerCrossBatch with {len(layer_indices)} layers")

        elif self.config.module_type == "multi_layer_attention":
            layer_indices = self.config.mix_layers if self.config.mix_layers else list(range(num_layers))
            self.cross_batch_module = MultiLayerCrossBatchAttention(
                hidden_size=hidden_size,
                num_layers=num_layers,
                layer_indices=layer_indices,
                num_heads=8,
                temperature=1.0,
                use_gate=True,
            )
            logger.info(f"Created MultiLayerCrossBatchAttention with {len(layer_indices)} layers")

        elif self.config.module_type == "simple":
            self.cross_batch_module = SimpleCrossBatchGate(hidden_size=hidden_size, temperature=1.0)
            logger.info("Created SimpleCrossBatchGate")

        elif self.config.module_type == "attention":
            self.cross_batch_module = CrossBatchAttention(hidden_size=hidden_size, num_heads=8, temperature=1.0)
            logger.info("Created CrossBatchAttention")

        else:  # mixer
            self.cross_batch_module = CrossBatchEmbeddingMixer(hidden_size=hidden_size, temperature=1.0)
            logger.info("Created CrossBatchEmbeddingMixer")

        num_params = sum(p.numel() for p in self.cross_batch_module.parameters())
        logger.info(f"Cross-batch module parameters: {num_params:,}")

        return self.cross_batch_module

    def create_trainer(self) -> CrossBatchTrainer:
        """Create the trainer."""
        device = str(next(self.model.parameters()).device)
        self.trainer = CrossBatchTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            cross_batch_module=self.cross_batch_module,
            device=device,
            learning_rate=self.config.learning_rate,
            train_lm_head=False,
            train_lora=self.config.use_lora,
            local_rank=-1,
            mix_layer=-1,
        )
        return self.trainer

    def train_epoch(self, train_dataset: SQuADGroupedDataset) -> Dict[str, Any]:
        """Train for one epoch."""
        history = self.trainer.train(
            train_dataset=train_dataset,
            num_epochs=1,
            batch_size=self.config.batch_size,
            save_dir=None,
            distributed=False,
            rank=0,
            world_size=1,
            grouped=True,
        )
        return history

    def save_checkpoint(self, epoch: int, checkpoint_dir: Path) -> str:
        """Save checkpoint for the current epoch."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = str(checkpoint_dir / f"{self.config.module_type}_frozen_epoch{epoch}.pt")

        if hasattr(self.trainer, 'cross_batch_module_unwrapped'):
            module_state = self.trainer.cross_batch_module_unwrapped.state_dict()
        else:
            module_state = self.cross_batch_module.state_dict()

        checkpoint = {
            'cross_batch_module': module_state,
            'config': {
                'model': self.config.model_name,
                'dataset': self.config.dataset,
                'hidden_size': self.model.config.hidden_size,
                'num_layers': self.model.config.num_hidden_layers,
                'epochs': epoch,
                'module_type': self.config.module_type,
                'mix_layer': -1,
                'mix_layers': self.config.mix_layers,
                'use_lora': self.config.use_lora,
            },
        }

        if self.config.use_lora and self.lora_model is not None:
            lora_state_dict = {k: v for k, v in self.model.state_dict().items() if 'lora' in k.lower()}
            checkpoint['lora'] = lora_state_dict
            logger.info(f"Saved {len(lora_state_dict)} LoRA tensors")

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        return checkpoint_path

    def evaluate(
        self,
        eval_contexts: List[Dict],
        checkpoint_path: str,
        epoch: int,
    ) -> Dict[str, Any]:
        """Evaluate on the given contexts."""
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
        """Run the full training and evaluation pipeline.

        Args:
            train_groups: Training data groups
            eval_contexts: Evaluation contexts
            baseline_results: Baseline results for comparison

        Returns:
            Dictionary with all results
        """
        # Setup
        self.load_model()
        self.create_cross_batch_module()
        self.create_trainer()

        # Create training dataset
        train_dataset = SQuADGroupedDataset(
            tokenizer=self.tokenizer,
            groups=train_groups,
            dataset_name=self.config.dataset,
        )
        total_questions = sum(len(g.get("questions", g.get("items", []))) for g in train_groups)
        logger.info(f"Training dataset: {len(train_dataset)} contexts, {total_questions} questions")

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

            # Train
            logger.info("Training...")
            history = self.train_epoch(train_dataset)
            logger.info(f"Training complete - Loss: {history['train_loss'][-1]:.4f}, Improvement: {history['improvement'][-1]:.4f}")

            # Save checkpoint
            checkpoint_path = self.save_checkpoint(epoch, checkpoint_base)

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
