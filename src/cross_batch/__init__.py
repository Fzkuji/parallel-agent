"""Cross-batch generation module for information sharing between samples during generation."""

from .attention import CrossBatchAttention, CrossBatchEmbeddingMixer
from .generator import CrossBatchGenerator
from .trainer import (
    CrossBatchTrainer,
    SQuADDataset,
    SQuADGroupedDataset,
    train_cross_batch_module,
)
from .eval import SquadEvaluator, run_comparison_eval

__all__ = [
    # Core modules
    "CrossBatchAttention",
    "CrossBatchEmbeddingMixer",
    "CrossBatchGenerator",
    # Training
    "CrossBatchTrainer",
    "SQuADDataset",
    "SQuADGroupedDataset",
    "train_cross_batch_module",
    # Evaluation
    "SquadEvaluator",
    "run_comparison_eval",
]
