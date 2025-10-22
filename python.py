from __future__ import annotations

import heapq
from typing import Dict, Mapping, MutableMapping, Optional, Tuple, TypeVar

Node = TypeVar("Node")


def dijkstra(
    graph: Mapping[Node, Mapping[Node, float]], start: Node
) -> Tuple[Dict[Node, float], Dict[Node, Optional[Node]]]:
    """Compute the shortest paths from `start` to every reachable node.

    The graph must be represented as an adjacency map where graph[u][v] is the
    non-negative cost of the edge u -> v.
    Returns:
        distances: shortest known distance from start to each node.
        parents: previous node on the shortest path (None for start / unreachable).
    """
    distances: Dict[Node, float] = {start: 0.0}
    parents: Dict[Node, Optional[Node]] = {start: None}
    pq: list[Tuple[float, Node]] = [(0.0, start)]

    while pq:
        current_distance, node = heapq.heappop(pq)
        if current_distance > distances.get(node, float("inf")):
            continue

        for neighbor, weight in graph.get(node, {}).items():
            if weight < 0:
                raise ValueError("Dijkstra's algorithm requires non-negative edge weights.")

            next_distance = distances[node] + weight
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                parents[neighbor] = node
                heapq.heappush(pq, (next_distance, neighbor))

    return distances, parents


def build_path(
    parents: Mapping[Node, Optional[Node]], target: Node
) -> list[Node]:
    """Reconstruct a shortest path from the parents map returned by dijkstra()."""
    if target not in parents:
        return []

    path: list[Node] = []
    current: Optional[Node] = target
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


if __name__ == "__main__":
    # Example graph where the keys are nodes and values indicate outgoing edges.
    graph_example = {
        "A": {"B": 5, "C": 1},
        "B": {"C": 2, "D": 1},
        "C": {"B": 3, "D": 8, "E": 10},
        "D": {"E": 2},
        "E": {},
    }

    distances_result, parents_result = dijkstra(graph_example, "A")
    target_node = "E"
    print("最短距离:", distances_result.get(target_node))
    print("路径:", build_path(parents_result, target_node))
