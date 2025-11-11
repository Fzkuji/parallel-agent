from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class StrategyResult:
    """Aggregated metrics for a single inference strategy."""

    name: str
    answers: Dict[str, str]
    prompt_tokens: int
    generated_tokens: int
    latency: float
    batches: int
    metrics: Dict[str, float]
    details: Dict[str, Any]
