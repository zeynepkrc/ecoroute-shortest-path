"""
Floyd-Warshall all-pairs shortest paths for directed graphs.

``visited_count`` is the number of innermost triple-loop iterations
``(k, i, j)``, i.e. ``num_nodes ** 3``, which is Theta(|V|^3) regardless of
sparsity for the dense-matrix formulation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Edge = Tuple[int, int, int]
INF = float("inf")


def floyd_warshall_preprocess(
    num_nodes: int, edges: Sequence[Edge]
) -> Tuple[List[List[float]], int, bool]:
    # Tüm düğüm çiftleri arasındaki en kısa yol matrisini oluşturur ve negatif döngü kontrolü yapar.
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")

    dist: List[List[float]] = [[INF] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        dist[i][i] = 0.0
    for u, v, w in edges:
        ww = float(w)
        if ww < dist[u][v]:
            dist[u][v] = ww

    visited_count = 0
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                visited_count += 1
                if dist[i][k] < INF and dist[k][j] < INF:
                    through = dist[i][k] + dist[k][j]
                    if through < dist[i][j]:
                        dist[i][j] = through

    has_negative_cycle = any(dist[i][i] < 0 for i in range(num_nodes))
    return dist, visited_count, has_negative_cycle


def _finite_int_distance(d: float) -> Optional[int]:
    if d >= INF / 2:
        return None
    return int(d)


def floyd_warshall(num_nodes: int, edges: Sequence[Edge], source: int, target: int) -> Tuple[Optional[int], int]:
    # Floyd-Warshall algoritmasını çalıştırarak belirtilen iki düğüm arasındaki en kısa yolu döner.
    if not (0 <= source < num_nodes and 0 <= target < num_nodes):
        raise ValueError("source and target must be in range 0 .. num_nodes - 1.")
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    if source == target:
        return 0, 0

    dist, visited_count, _ = floyd_warshall_preprocess(num_nodes, edges)
    d = dist[source][target]
    return _finite_int_distance(d), visited_count
