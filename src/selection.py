"""Cost-aware dependency edge selection."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .models import EdgeCandidate, Question


def compute_dependency_cost(
    questions: Dict[str, Question],
    source: str,
    *,
    fmt_overhead: int = 6,
) -> int:
    """Compute the token cost of adding a dependency edge."""
    if source not in questions:
        return 0
    question = questions[source]
    return question.tokens + question.answer_tokens + fmt_overhead


def _creates_cycle(adjacency: Dict[str, Set[str]], source: str, target: str) -> bool:
    """Check if adding edge source->target would create a cycle."""
    stack = [target]
    visited: Set[str] = set()
    while stack:
        node = stack.pop()
        if node == source:
            return True
        for nxt in adjacency.get(node, ()):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return False


def select_dependency_edges(
    questions: Dict[str, Question],
    edge_candidates: Sequence[EdgeCandidate],
    *,
    cost_weight: float = 0.01,
    min_confidence: float = 0.35,
    max_dependencies_per_target: int = 3,
    total_cost_budget: Optional[int] = None,
    fmt_overhead: int = 6,
    prevent_cycles: bool = True,
) -> Dict[str, List[EdgeCandidate]]:
    """
    Select dependency edges based on confidence and cost constraints.

    Args:
        questions: Dictionary of question ID to Question object.
        edge_candidates: List of candidate edges to select from.
        cost_weight: Weight for cost penalty in scoring.
        min_confidence: Minimum confidence threshold for edges.
        max_dependencies_per_target: Maximum edges per target question.
        total_cost_budget: Optional global cost budget.
        fmt_overhead: Token overhead per dependency.
        prevent_cycles: Whether to prevent cyclic dependencies.

    Returns:
        Dictionary mapping target question ID to list of selected edges.
    """
    scored_edges: List[Tuple[float, EdgeCandidate, int]] = []
    for edge in edge_candidates:
        if edge.source not in questions or edge.target not in questions:
            continue
        if edge.source == edge.target:
            continue
        confidence = max(0.0, min(1.0, edge.confidence))
        if confidence < min_confidence:
            continue
        cost = compute_dependency_cost(questions, edge.source, fmt_overhead=fmt_overhead)
        score = confidence - cost_weight * cost
        scored_edges.append((score, edge, cost))

    scored_edges.sort(key=lambda item: item[0], reverse=True)
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    selected: Dict[str, List[EdgeCandidate]] = defaultdict(list)
    accumulated_cost = 0

    for score, edge, cost in scored_edges:
        if score <= 0:
            continue
        target_edges = selected[edge.target]
        if len(target_edges) >= max_dependencies_per_target:
            continue
        if prevent_cycles and _creates_cycle(adjacency, edge.source, edge.target):
            continue
        if total_cost_budget is not None and accumulated_cost + cost > total_cost_budget:
            continue
        adjacency[edge.source].add(edge.target)
        target_edges.append(edge)
        accumulated_cost += cost

    return selected


def apply_dependencies(
    questions: Dict[str, Question],
    selected_edges: Dict[str, List[EdgeCandidate]],
) -> None:
    """Apply selected edges to question dependency sets."""
    for question in questions.values():
        deps = {edge.source for edge in selected_edges.get(question.qid, [])}
        question.dependencies = deps
