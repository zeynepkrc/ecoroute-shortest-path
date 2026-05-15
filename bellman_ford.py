from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

Edge = Tuple[int, int, int]
INF = float("inf")


def bellman_ford(
    num_nodes: int, edges: Sequence[Edge], source: int, target: int
) -> Tuple[Optional[float], int]:
    # Negatif ağırlıklı kenarlar barındıran yönlü graflarda en kısa yolu hesaplar ve negatif döngüleri tespit eder.
    if not (0 <= source < num_nodes and 0 <= target < num_nodes):
        raise ValueError("source and target must be in range 0 .. num_nodes - 1.")

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

    bad = [False] * num_nodes
    for u, v, w in edges:
        visited_count += 1
        if dist[u] < INF and dist[u] + w < dist[v]:
            bad[v] = True

    while True:
        progressed = False
        for u, v, w in edges:
            visited_count += 1
            if bad[u] and not bad[v]:
                bad[v] = True
                progressed = True
        if not progressed:
            break

    if bad[target]:
        return float("-inf"), visited_count
    if dist[target] >= INF:
        return None, visited_count
    return dist[target], visited_count
