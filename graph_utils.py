"""
Random directed graph generation and fixed benchmark test cases.

All graphs use vertex labels ``0 .. num_nodes - 1``, integer edge weights, no
self-loops, and at most one directed arc per ordered pair ``(u, v)`` in the
nonnegative generator.

``create_test_cases`` returns dicts with ``name``, ``benchmark_type``,
``allow_negative``, ``num_nodes``, ``density``, ``edges``, ``queries``, and
``query_count`` for the benchmark driver.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

Edge = Tuple[int, int, int]


def generate_random_graph(
    num_nodes: int,
    density: float,
    max_weight: int = 20,
    seed: int = 42,
) -> List[Edge]:
    """
    Build a simple directed weighted graph as an edge list.

    Construction order
    --------------------
    1. Insert a guaranteed directed path ``0 -> 1 -> ... -> n-1`` (for
       ``num_nodes >= 2``) with random integer weights in ``[1, max_weight]``.
    2. Add up to ``remaining`` extra random edges so that, when possible, the
       total edge count approaches ``round(density * n * (n-1))`` without
       exceeding it. If the density budget is smaller than ``n-1``, only the
       backbone is kept (still guaranteeing a path).

    Parameters
    ----------
    num_nodes : int
        Number of vertices (must be >= 1).
    density : float in [0, 1]
        Target fraction of the maximum possible directed edges ``n * (n - 1)``.
    max_weight : int
        Upper inclusive bound for integer weights (default 20).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list of (u, v, w)
        Directed edges with ``1 <= w <= max_weight``, no self-loops, no duplicates.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be between 0 and 1 inclusive.")
    if max_weight < 1:
        raise ValueError("max_weight must be at least 1.")

    rng = random.Random(seed)
    max_edges = num_nodes * (num_nodes - 1)
    target = int(round(density * max_edges))
    target = max(0, min(target, max_edges))

    edges_set: Set[Tuple[int, int]] = set()
    edges: List[Edge] = []

    # --- 1) Guaranteed path 0 -> 1 -> ... -> n-1 --------------------------------
    if num_nodes >= 2:
        for u in range(num_nodes - 1):
            v = u + 1
            w = rng.randint(1, max_weight)
            edges.append((u, v, w))
            edges_set.add((u, v))

    backbone = len(edges)
    remaining = max(0, target - backbone)

    # --- 2) Additional random edges (no duplicates, no self-loops) --------------
    if remaining == 0:
        return edges

    # For large n, avoid materializing all n*(n-1) candidate pairs in memory.
    if max_edges > 2_000_000:
        attempts = 0
        cap = max(remaining * 80, num_nodes * num_nodes * 4)
        while len(edges) < target and attempts < cap:
            attempts += 1
            u = rng.randrange(num_nodes)
            v = rng.randrange(num_nodes)
            if u == v or (u, v) in edges_set:
                continue
            w = rng.randint(1, max_weight)
            edges.append((u, v, w))
            edges_set.add((u, v))
        return edges

    candidates: List[Tuple[int, int]] = [
        (u, v)
        for u in range(num_nodes)
        for v in range(num_nodes)
        if u != v and (u, v) not in edges_set
    ]
    rng.shuffle(candidates)
    for (u, v) in candidates[:remaining]:
        w = rng.randint(1, max_weight)
        edges.append((u, v, w))
        edges_set.add((u, v))

    return edges


def _build_query_set(num_nodes: int, seed: int, query_count: int) -> List[Tuple[int, int]]:
    """Deterministic multi-source-target pairs for benchmarking."""
    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    def add(s: int, t: int) -> None:
        if (s, t) not in seen:
            seen.add((s, t))
            pairs.append((s, t))

    if num_nodes >= 1:
        add(0, num_nodes - 1)
    if num_nodes >= 3:
        add(0, num_nodes // 2)
        add(num_nodes // 3, (2 * num_nodes) // 3)
    if num_nodes >= 2:
        add(num_nodes - 2, num_nodes - 1)
    while len(pairs) < query_count:
        s = rng.randrange(num_nodes)
        t = rng.randrange(num_nodes)
        add(s, t)
    return pairs[:query_count]


def generate_acyclic_random_graph(
    num_nodes: int,
    density: float,
    max_weight: int = 20,
    seed: int = 0,
    neg_edge_probability: float = 0.35,
    negative_ratio: Optional[float] = None,
) -> List[Edge]:
    """
    Random DAG on labels ``0 .. n-1`` using only edges ``u -> v`` with ``u < v``.

    No directed cycles exist, so negative edge weights cannot create a negative
    directed cycle. Integer weights may be negative when sampled.

    If ``negative_ratio`` is given, it overrides ``neg_edge_probability`` (same
    meaning: Bernoulli probability that a chosen edge weight is negative).
    """
    if negative_ratio is not None:
        neg_edge_probability = negative_ratio
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be between 0 and 1 inclusive.")
    if max_weight < 1:
        raise ValueError("max_weight must be at least 1.")

    rng = random.Random(seed)
    candidates: List[Tuple[int, int]] = [
        (u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)
    ]
    max_edges = len(candidates)
    target = int(round(density * max_edges))
    target = max(1, min(target, max_edges)) if max_edges else 0
    rng.shuffle(candidates)
    edges: List[Edge] = []
    for u, v in candidates[:target]:
        if rng.random() < neg_edge_probability:
            w = -rng.randint(1, min(8, max_weight))
        else:
            w = rng.randint(1, max_weight)
        edges.append((u, v, w))
    return edges


def negative_cycle_toy_edges() -> Tuple[int, List[Edge]]:
    """
    Small graph with a reachable negative directed cycle.

    Path ``0 -> 1``; cycle ``1 -> 2 -> 3 -> 1`` has total weight ``-6``.
    """
    n = 4
    edges: List[Edge] = [
        (0, 1, 2),
        (1, 2, 2),
        (2, 3, 2),
        (3, 1, -10),
    ]
    return n, edges


def create_test_cases() -> List[Dict[str, Any]]:
    """
    Benchmark suite with shared query sets and benchmark metadata.

    Each case is a dict with at least:

    - ``name``: short identifier
    - ``benchmark_type``: ``"all_algorithms"`` or ``"large_scalability"``
    - ``allow_negative``: whether the edge list may contain negative weights
    - ``num_nodes``, ``density``, ``edges``
    - ``queries``: list of ``(source, target)`` pairs
    - ``query_count``: ``len(queries)``
    """
    cases: List[Dict[str, Any]] = []
    seed_counter = 42
    query_count = 6

    # --- 1) Small/Medium: all algorithms, nonnegative weights -----------------
    sizes = [50, 100, 250, 500, 1000]
    densities = [0.02, 0.10, 0.25]
    for n in sizes:
        for d in densities:
            edges = generate_random_graph(n, d, max_weight=20, seed=seed_counter)
            seed_counter += 1
            qseed = 1000 * n + int(d * 1000)
            queries = _build_query_set(n, qseed, query_count)
            cases.append(
                {
                    "name": f"all_nonneg_n{n}_d{d:.2f}",
                    "benchmark_type": "all_algorithms",
                    "allow_negative": False,
                    "num_nodes": n,
                    "density": d,
                    "edges": edges,
                    "queries": queries,
                    "query_count": len(queries),
                }
            )

    # --- 2) Negative-edge limitation (acyclic DAG, no negative cycles) ----------
    negative_sizes = [30, 50]
    negative_densities = [0.05, 0.10]
    negative_ratio = 0.10
    for n in negative_sizes:
        for d in negative_densities:
            edges = generate_acyclic_random_graph(
                n,
                d,
                max_weight=20,
                seed=seed_counter,
                negative_ratio=negative_ratio,
            )
            seed_counter += 1
            queries = _build_query_set(n, 5000 + n + int(d * 10000), query_count)
            cases.append(
                {
                    "name": f"all_acyclic_neg_n{n}_d{d:.2f}",
                    "benchmark_type": "all_algorithms",
                    "allow_negative": True,
                    "num_nodes": n,
                    "density": d,
                    "edges": edges,
                    "queries": queries,
                    "query_count": len(queries),
                }
            )

    n_toy, edges_toy = negative_cycle_toy_edges()
    queries_toy = _build_query_set(n_toy, 7000, query_count)
    cases.append(
        {
            "name": "all_neg_cycle_toy",
            "benchmark_type": "all_algorithms",
            "allow_negative": True,
            "num_nodes": n_toy,
            "density": 0.35,
            "edges": edges_toy,
            "queries": queries_toy,
            "query_count": len(queries_toy),
        }
    )

    # --- 3) Large scalability (list-based Dijkstra only in main driver) ----------
    large_cases = []
    for n, d in large_cases:
        edges = generate_random_graph(n, d, max_weight=20, seed=seed_counter)
        seed_counter += 1
        queries = _build_query_set(n, 9000 + n, query_count)
        cases.append(
            {
                "name": f"scale_large_n{n}_d{d:.4f}",
                "benchmark_type": "large_scalability",
                "allow_negative": False,
                "num_nodes": n,
                "density": d,
                "edges": edges,
                "queries": queries,
                "query_count": len(queries),
            }
        )

    return cases
