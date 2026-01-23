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
    cost_weight: float = 0.01,  # Kept for API compatibility, but no longer used
    min_confidence: float = 0.0,  # Kept for API compatibility, but no longer used
    max_dependencies_per_target: int = 3,
    total_cost_budget: Optional[int] = None,
    fmt_overhead: int = 6,
    prevent_cycles: bool = True,
) -> Dict[str, List[EdgeCandidate]]:
    """
    Select dependency edges, preserving LLM-determined order.

    The LLM decides which edges are important by their order in the output.
    We simply filter out invalid edges (cycles, missing questions) and apply limits.

    Args:
        questions: Dictionary of question ID to Question object.
        edge_candidates: List of candidate edges to select from (in LLM order).
        cost_weight: Deprecated, kept for API compatibility.
        min_confidence: Deprecated, kept for API compatibility.
        max_dependencies_per_target: Maximum edges per target question.
        total_cost_budget: Optional global cost budget.
        fmt_overhead: Token overhead per dependency.
        prevent_cycles: Whether to prevent cyclic dependencies.

    Returns:
        Dictionary mapping target question ID to list of selected edges.
    """
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    selected: Dict[str, List[EdgeCandidate]] = defaultdict(list)
    accumulated_cost = 0

    # Process edges in LLM-determined order (no sorting by confidence)
    for edge in edge_candidates:
        if edge.source not in questions or edge.target not in questions:
            continue
        if edge.source == edge.target:
            continue
        target_edges = selected[edge.target]
        if len(target_edges) >= max_dependencies_per_target:
            continue
        if prevent_cycles and _creates_cycle(adjacency, edge.source, edge.target):
            continue
        cost = compute_dependency_cost(questions, edge.source, fmt_overhead=fmt_overhead)
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
