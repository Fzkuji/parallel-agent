"""Cross-batch generation module for information sharing between samples during generation."""

from .attention import (
    CrossBatchAttention,
    CrossBatchEmbeddingMixer,
    SimpleCrossBatchGate,
    MultiLayerCrossBatch,
)
from .generator import CrossBatchGenerator
from .trainer import (
    CrossBatchTrainer,
    SQuADDataset,
    SQuADGroupedDataset,
    train_cross_batch_module,
)
from .eval import SquadEvaluator, run_comparison_eval
from .utils import is_instruct_model, get_eos_token

__all__ = [
    # Core modules
    "CrossBatchAttention",
    "CrossBatchEmbeddingMixer",
    "SimpleCrossBatchGate",
    "MultiLayerCrossBatch",
    "CrossBatchGenerator",
    # Training
    "CrossBatchTrainer",
    "SQuADDataset",
    "SQuADGroupedDataset",
    "train_cross_batch_module",
    # Evaluation
    "SquadEvaluator",
    "run_comparison_eval",
    # Utils
    "is_instruct_model",
    "get_eos_token",
]
