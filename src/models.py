"""Core data structures for the dependency scheduling system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    ascii_tokens = text.strip().split()
    chinese_chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    approx = len(ascii_tokens) * 1.4 + len(chinese_chars) * 0.8
    if any(ch.isdigit() for ch in text):
        approx += 0.2 * sum(ch.isdigit() for ch in text)
    return max(1, int(approx))


@dataclass
class Question:
    """Represents a question with metadata for scheduling."""
    qid: str
    text: str
    priority: float = 1.0
    answer_tokens: int = 32
    explicit_dependencies: Iterable[str] = field(default_factory=list)
    type_hint: Optional[str] = None
    tokens: int = field(init=False)
    dependencies: Set[str] = field(default_factory=set, init=False)
    references: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tokens = estimate_tokens(self.text)
        self.dependencies = set(self.explicit_dependencies)


@dataclass
class EdgeCandidate:
    """A candidate dependency edge between two questions."""
    source: str
    target: str
    confidence: float = 1.0
    rationale: Optional[str] = None


@dataclass
class BatchAssignment:
    """Assignment of questions to a batch for parallel execution."""
    batch_id: int
    question_ids: List[str]
    depth: int
    priority_sum: float
    value_score: float
    background_tokens: int
    incremental_prefill_tokens: int
    generation_tokens: int
    total_tokens: int
    estimated_latency: float


@dataclass
class ScheduleResult:
    """Result of scheduling questions into batches."""
    batches: List[BatchAssignment]
    question_depths: Dict[str, int]
    dependency_graph: Dict[str, Set[str]]
    total_background_tokens: int
    total_incremental_prefill_tokens: int
    total_generation_tokens: int
    total_compute_tokens: int
    total_priority: float
    value_score: float
    total_estimated_latency: float
