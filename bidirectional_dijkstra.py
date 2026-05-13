"""
Bidirectional Dijkstra: list + heap and matrix (O(V^2) per phase) variants.

``visited_count`` counts outgoing edge examinations from vertices that are
selected as the current minimum-distance frontier node (forward or backward).
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


def bidirectional_dijkstra_list(
    num_nodes: int, edges: Sequence[Edge], source: int, target: int
) -> Tuple[Optional[int], int]:
    """
    Bidirectional search on ``G`` (forward) and the transpose graph (backward).

    ``visited_count``: neighbor edge scans from popped heap nodes (forward +
    backward combined).
    """
    if not (0 <= source < num_nodes and 0 <= target < num_nodes):
        raise ValueError("source and target must be in range 0 .. num_nodes - 1.")
    if source == target:
        return 0, 0

    g_f: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    g_r: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for u, v, w in edges:
        g_f[u].append((v, w))
        g_r[v].append((u, w))

    dist_f: List[float] = [INF] * num_nodes
    dist_b: List[float] = [INF] * num_nodes
    dist_f[source] = 0.0
    dist_b[target] = 0.0

    heap_f: List[Tuple[float, int]] = [(0.0, source)]
    heap_b: List[Tuple[float, int]] = [(0.0, target)]
    visited_count = 0
    best = INF

    while heap_f and heap_b:
        min_f = heap_f[0][0]
        min_b = heap_b[0][0]
        if min_f + min_b >= best:
            break

        if min_f <= min_b:
            d_u, u = heapq.heappop(heap_f)
            if d_u != dist_f[u]:
                continue
            for v, w in g_f[u]:
                visited_count += 1
                nd = d_u + w
                if nd < dist_f[v]:
                    dist_f[v] = nd
                    heapq.heappush(heap_f, (nd, v))
                if dist_b[v] < INF:
                    cand = nd + dist_b[v]
                    if cand < best:
                        best = cand
        else:
            d_u, u = heapq.heappop(heap_b)
            if d_u != dist_b[u]:
                continue
            for v, w in g_r[u]:
                visited_count += 1
                nd = d_u + w
                if nd < dist_b[v]:
                    dist_b[v] = nd
                    heapq.heappush(heap_b, (nd, v))
                if dist_f[v] < INF:
                    cand = dist_f[v] + nd
                    if cand < best:
                        best = cand

    if best >= INF / 2:
        return None, visited_count
    return int(best), visited_count


def bidirectional_dijkstra_matrix(
    num_nodes: int, edges: Sequence[Edge], source: int, target: int
) -> Tuple[Optional[int], int]:
    """
    Matrix analogue: each side uses linear scans for the next frontier vertex
    (no binary heap). Forward uses ``W``; backward uses ``W_rev`` (transpose).
    """
    if not (0 <= source < num_nodes and 0 <= target < num_nodes):
        raise ValueError("source and target must be in range 0 .. num_nodes - 1.")
    if source == target:
        return 0, 0

    w_f: List[List[float]] = [[INF] * num_nodes for _ in range(num_nodes)]
    w_r: List[List[float]] = [[INF] * num_nodes for _ in range(num_nodes)]
    for u, v, w in edges:
        w_f[u][v] = float(w)
        w_r[v][u] = float(w)

    dist_f: List[float] = [INF] * num_nodes
    dist_b: List[float] = [INF] * num_nodes
    dist_f[source] = 0.0
    dist_b[target] = 0.0
    done_f = [False] * num_nodes
    done_b = [False] * num_nodes
    visited_count = 0
    best = INF

    def pick_min(dist: List[float], done: List[bool]) -> int:
        u = -1
        best_d = INF
        for i in range(num_nodes):
            if not done[i] and dist[i] < best_d:
                best_d = dist[i]
                u = i
        return u

    while True:
        u_f = pick_min(dist_f, done_f)
        u_b = pick_min(dist_b, done_b)
        if u_f < 0 and u_b < 0:
            break
        min_f = dist_f[u_f] if u_f >= 0 else INF
        min_b = dist_b[u_b] if u_b >= 0 else INF
        if min_f >= INF / 2 and min_b >= INF / 2:
            break
        if best < INF / 2 and min_f + min_b >= best:
            break

        if u_f >= 0 and (u_b < 0 or min_f <= min_b):
            done_f[u_f] = True
            du = dist_f[u_f]
            for v in range(num_nodes):
                w = w_f[u_f][v]
                if w >= INF:
                    continue
                visited_count += 1
                nd = du + w
                if dist_b[v] < INF:
                    cand = nd + dist_b[v]
                    if cand < best:
                        best = cand
                if nd < dist_f[v]:
                    dist_f[v] = nd
        elif u_b >= 0:
            done_b[u_b] = True
            du = dist_b[u_b]
            for v in range(num_nodes):
                w = w_r[u_b][v]
                if w >= INF:
                    continue
                visited_count += 1
                nd = du + w
                if dist_f[v] < INF:
                    cand = dist_f[v] + nd
                    if cand < best:
                        best = cand
                if nd < dist_b[v]:
                    dist_b[v] = nd
        else:
            break

    if best >= INF / 2:
        return None, visited_count
    return int(best), visited_count
