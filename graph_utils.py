"""
Random directed graph generation and fixed benchmark test cases.

All graphs use vertex labels ``0 .. num_nodes - 1``, positive integer edge
weights, no self-loops, and at most one directed arc per ordered pair (u, v).

Every generated graph includes a directed backbone path ``0 -> 1 -> ... -> n-1``
so that ``source=0`` to ``target=n-1`` is always reachable; additional edges
respect the requested density without removing or duplicating backbone arcs.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Set, Tuple

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


def create_test_cases() -> List[Dict[str, Any]]:
    """
    Deterministic suite: Cartesian product of sizes and densities.

    Each case includes ``num_nodes``, ``density``, ``edges``, ``source=0``,
    ``target=num_nodes - 1``. Graphs always admit a ``0 -> n-1`` path via the
    backbone inserted in ``generate_random_graph``. A different RNG seed is
    used per case so graphs are not identical across densities/sizes.
    """
    sizes = [10, 50, 100, 250]
    densities = [0.02, 0.10, 0.50]
    cases: List[Dict[str, Any]] = []
    seed_counter = 42
    for n in sizes:
        for d in densities:
            edges = generate_random_graph(n, d, max_weight=20, seed=seed_counter)
            seed_counter += 1
            cases.append(
                {
                    "num_nodes": n,
                    "density": d,
                    "edges": edges,
                    "source": 0,
                    "target": n - 1,
                }
            )
    return cases
