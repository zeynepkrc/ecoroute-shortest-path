"""
Dijkstra's algorithm: adjacency-list + binary heap, and matrix O(V^2) variant.

Weights are positive integers; graphs are directed. ``visited_count`` counts
neighbor relaxations (edge examinations from a vertex that is *settled* /
extracted as the current minimum in the matrix version).
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Sequence, Tuple

Edge = Tuple[int, int, int]
INF = float("inf")


def _finite_int_distance(d: float) -> Optional[int]:
    if d >= INF / 2:
        return None
    return int(d)


def dijkstra_list(num_nodes: int, edges: Sequence[Edge], source: int, target: int) -> Tuple[Optional[int], int]:
    """
    Dijkstra with adjacency list and binary min-heap.

    visited_count: number of times an outgoing edge is scanned from the vertex
    popped from the heap (relaxation attempts, including non-improving checks).
    """
    if not (0 <= source < num_nodes and 0 <= target < num_nodes):
        raise ValueError("source and target must be in range 0 .. num_nodes - 1.")
    if source == target:
        return 0, 0

    graph: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for u, v, w in edges:
        graph[u].append((v, w))

    dist: List[float] = [INF] * num_nodes
    dist[source] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, source)]
    visited_count = 0
    pop_count = [0] * num_nodes

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u != dist[u]:
            continue
        pop_count[u] += 1
        if pop_count[u] > num_nodes:
            break
        if u == target:
            break
        for v, w in graph[u]:
            visited_count += 1
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return _finite_int_distance(dist[target]), visited_count


def dijkstra_matrix(num_nodes: int, edges: Sequence[Edge], source: int, target: int) -> Tuple[Optional[int], int]:
    """
    Classic Dijkstra without heap: each step scan all unsettled vertices for the
    minimum tentative distance (O(V^2) total for nonnegative weights).

    visited_count: each time we scan an entry ``(u, v)`` in the adjacency row
    of the settled vertex ``u`` (relaxation attempt).
    """
    if not (0 <= source < num_nodes and 0 <= target < num_nodes):
        raise ValueError("source and target must be in range 0 .. num_nodes - 1.")
    if source == target:
        return 0, 0

    weight: List[List[float]] = [[INF] * num_nodes for _ in range(num_nodes)]
    for u, v, w in edges:
        weight[u][v] = float(w)

    dist: List[float] = [INF] * num_nodes
    dist[source] = 0.0
    settled = [False] * num_nodes
    visited_count = 0

    for _ in range(num_nodes):
        u = -1
        best = INF
        for i in range(num_nodes):
            if not settled[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u < 0 or best >= INF:
            break
        settled[u] = True
        if u == target:
            break
        for v in range(num_nodes):
            w = weight[u][v]
            if w < INF:
                visited_count += 1
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd

    return _finite_int_distance(dist[target]), visited_count
