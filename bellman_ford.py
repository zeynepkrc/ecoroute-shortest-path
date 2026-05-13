"""
Bellman-Ford for directed graphs with positive integer weights.

``visited_count`` counts edge relaxations attempted in the inner loop (each
examination of ``(u, v, w)`` in a pass). Early stopping ends a pass early when
no distance improves.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Edge = Tuple[int, int, int]
INF = float("inf")


def _finite_int_distance(d: float) -> Optional[int]:
    if d >= INF / 2:
        return None
    return int(d)


def bellman_ford(num_nodes: int, edges: Sequence[Edge], source: int, target: int) -> Tuple[Optional[int], int]:
    """
    Bellman-Ford with early termination when a full pass makes no change.

    For nonnegative weights this is correct; negative edges are not required
    for this project but would also be supported by the standard relaxation rule.
    """
    if not (0 <= source < num_nodes and 0 <= target < num_nodes):
        raise ValueError("source and target must be in range 0 .. num_nodes - 1.")
    if source == target:
        return 0, 0

    dist: List[float] = [INF] * num_nodes
    dist[source] = 0.0
    visited_count = 0

    for _ in range(num_nodes - 1):
        changed = False
        for u, v, w in edges:
            visited_count += 1
            if dist[u] < INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + float(w)
                changed = True
        if not changed:
            break

    return _finite_int_distance(dist[target]), visited_count
