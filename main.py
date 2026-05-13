"""
Benchmark driver: compare shortest-path algorithms on shared random graphs.

Run from the ``project`` directory::

    python main.py

Outputs ``benchmark_results.csv`` and prints a summary table. Optional
matplotlib figures are written if matplotlib is installed.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from bellman_ford import bellman_ford
from bidirectional_dijkstra import (
    bidirectional_dijkstra_list,
    bidirectional_dijkstra_matrix,
)
from dijkstra import dijkstra_list, dijkstra_matrix
from floyd_warshall import floyd_warshall
from graph_utils import create_test_cases

AlgoFn = Callable[[int, List[Tuple[int, int, int]], int, int], Tuple[Optional[int], int]]

ALGORITHMS: List[Tuple[str, AlgoFn]] = [
    ("Dijkstra List", dijkstra_list),
    ("Dijkstra Matrix", dijkstra_matrix),
    ("Bidirectional Dijkstra List", bidirectional_dijkstra_list),
    ("Bidirectional Dijkstra Matrix", bidirectional_dijkstra_matrix),
    ("Bellman-Ford", bellman_ford),
    ("Floyd-Warshall", floyd_warshall),
]


def should_skip(algorithm_name: str, num_nodes: int, density: float) -> bool:
    if algorithm_name == "Floyd-Warshall" and num_nodes > 250:
        return True
    return False


def _run_benchmark() -> List[Dict[str, Any]]:
    cases = create_test_cases()
    rows: List[Dict[str, Any]] = []

    for case in cases:
        n = case["num_nodes"]
        d = case["density"]
        edges = case["edges"]
        src = case["source"]
        tgt = case["target"]
        m = len(edges)

        for algo_name, algo_fn in ALGORITHMS:
            skip = should_skip(algo_name, n, d)
            note = ""
            if skip:
                rows.append(
                    {
                        "nodes": n,
                        "density": d,
                        "edges": m,
                        "algorithm": algo_name,
                        "distance": "",
                        "visited_count": "",
                        "runtime_ms": "",
                        "skipped": "yes",
                        "note": "Skipped by should_skip rule.",
                    }
                )
                continue

            t0 = time.perf_counter()
            distance, visited = algo_fn(n, edges, src, tgt)
            t1 = time.perf_counter()
            runtime_ms = (t1 - t0) * 1000.0

            dist_str = "" if distance is None else str(int(distance))
            rows.append(
                {
                    "nodes": n,
                    "density": d,
                    "edges": m,
                    "algorithm": algo_name,
                    "distance": dist_str,
                    "visited_count": str(int(visited)),
                    "runtime_ms": f"{runtime_ms:.6f}",
                    "skipped": "no",
                    "note": note,
                }
            )

    return rows


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "nodes",
        "density",
        "edges",
        "algorithm",
        "distance",
        "visited_count",
        "runtime_ms",
        "skipped",
        "note",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = "| Nodes | Density | Edges | Algorithm | Distance | Visited Count | Runtime ms |"
    sep = "|-------|---------|-------|-----------|----------|---------------|------------|"
    print(header)
    print(sep)
    for r in rows:
        if r["skipped"] == "yes":
            print(
                f"| {r['nodes']:5d} | {r['density']:7.2f} | {r['edges']:5d} | "
                f"{r['algorithm'][:24]:24s} | {'':9s} | {'':13s} | {'':10s} |"
            )
        else:
            print(
                f"| {r['nodes']:5d} | {r['density']:7.2f} | {r['edges']:5d} | "
                f"{r['algorithm'][:24]:24s} | {r['distance']:>9s} | {r['visited_count']:>13s} | "
                f"{float(r['runtime_ms']):10.4f} |"
            )


def _summarize(rows: List[Dict[str, Any]]) -> None:
    """Per (nodes, density, edges) group: fastest runtime and lowest visited_count."""
    groups: Dict[Tuple[int, float, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["skipped"] != "no":
            continue
        key = (int(r["nodes"]), float(r["density"]), int(r["edges"]))
        groups[key].append(r)

    print()
    print("=== Per test case: fastest algorithm (by runtime) ===")
    for key in sorted(groups.keys()):
        n, den, m = key
        best = min(groups[key], key=lambda x: float(x["runtime_ms"]))
        print(
            f"  nodes={n}, density={den:.2f}, |E|={m}: "
            f"{best['algorithm']} ({best['runtime_ms']} ms)"
        )

    print()
    print("=== Per test case: lowest visited_count ===")
    for key in sorted(groups.keys()):
        n, den, m = key
        best = min(groups[key], key=lambda x: int(x["visited_count"]))
        print(
            f"  nodes={n}, density={den:.2f}, |E|={m}: "
            f"{best['algorithm']} (visited={best['visited_count']})"
        )

    print()
    print("=== Report notes (dense vs sparse, Floyd-Warshall) ===")
    print(
        "- Floyd-Warshall costs Theta(V^3) time and Theta(V^2) memory for the "
        "distance matrix; it does not benefit from sparse edge lists, so it "
        "becomes prohibitive on large dense instances (hence the skip rule "
        "for V > 250 in this benchmark)."
    )
    print(
        "- On sparse graphs (low density), algorithms that scan only existing "
        "edges or use heaps often outperform matrix-based O(V^2) scans per step, "
        "especially when V grows."
    )
    print(
        "- Visited-count definitions differ by algorithm family; interpret "
        "comparisons as qualitative workload indicators, not a single universal "
        "notion of work."
    )


def _optional_plots(rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    active = [r for r in rows if r["skipped"] == "no"]
    if not active:
        return

    # Series: for each algorithm, x = nodes, y = runtime (one point per case)
    by_algo: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    by_algo_v: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for r in active:
        by_algo[r["algorithm"]].append((int(r["nodes"]), float(r["runtime_ms"])))
        by_algo_v[r["algorithm"]].append((int(r["nodes"]), int(r["visited_count"])))

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    for name, pts in sorted(by_algo.items()):
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax1.plot(xs, ys, marker="o", label=name)
    ax1.set_xlabel("Nodes")
    ax1.set_ylabel("Runtime (ms)")
    ax1.set_title("Runtime vs nodes (all densities shown as separate x points)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    fig1.tight_layout()
    fig1.savefig("runtime_vs_nodes.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for name, pts in sorted(by_algo_v.items()):
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax2.plot(xs, ys, marker="o", label=name)
    ax2.set_xlabel("Nodes")
    ax2.set_ylabel("Visited count")
    ax2.set_title("Visited count vs nodes")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)
    fig2.tight_layout()
    fig2.savefig("visited_vs_nodes.png", dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    out_rows = _run_benchmark()
    _write_csv(out_rows, "benchmark_results.csv")
    _print_table(out_rows)
    _summarize(out_rows)
    _optional_plots(out_rows)
    print()
    print("Wrote benchmark_results.csv (and optional PNGs if matplotlib is available).")
