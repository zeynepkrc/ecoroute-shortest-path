"""
Benchmark driver: compare shortest-path algorithms on shared random graphs.

Run from the project directory::

    python main.py

Writes ``benchmark_results.csv`` and prints a summary table. Optional
matplotlib figures are written if matplotlib is installed.
"""

from __future__ import annotations

import csv
import math
import time
import concurrent.futures
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bellman_ford import bellman_ford
from bidirectional_dijkstra import (
    bidirectional_dijkstra_list,
    bidirectional_dijkstra_matrix,
)
from dijkstra import dijkstra_list, dijkstra_matrix
from floyd_warshall import floyd_warshall_preprocess
from graph_utils import create_test_cases

Edge = Tuple[int, int, int]
INF = float("inf")

AlgoFn = Callable[[int, Sequence[Edge], int, int], Tuple[Any, int]]

ALGORITHMS: List[Tuple[str, AlgoFn]] = [
    ("Dijkstra List", dijkstra_list),
    ("Dijkstra Matrix", dijkstra_matrix),
    ("Bidirectional Dijkstra List", bidirectional_dijkstra_list),
    ("Bidirectional Dijkstra Matrix", bidirectional_dijkstra_matrix),
    ("Bellman-Ford", bellman_ford),
]

DIJKSTRA_FAMILY = {
    "Dijkstra List",
    "Dijkstra Matrix",
    "Bidirectional Dijkstra List",
    "Bidirectional Dijkstra Matrix",
}

REFERENCE_ALGOS = {"Bellman-Ford", "Floyd-Warshall"}

NEG_WEIGHT_LIMITATION = "Not guaranteed with negative weights / limitation demo"


def _normalize_distance(value: Any) -> Optional[float]:
    # Gelen uzaklık değerini (inf/-inf vb.) standart bir float veya None formatına dönüştürür.
    if value is None:
        return None
    if isinstance(value, float) and value == float("-inf"):
        return float("-inf")
    return float(value)


def _matrix_distance(dist_matrix: List[List[float]], s: int, t: int) -> Optional[float]:
    # Önceden hesaplanmış uzaklık matrisinden, belirli başlangıç ve bitiş düğümleri arasındaki uzaklığı okur.
    d = dist_matrix[s][t]
    if d >= INF / 2:
        return None
    return float(d)


def _distances_equivalent(a: Optional[float], b: Optional[float]) -> bool:
    # İki uzaklık değerinin (hesaplanan ve referans) birbirine eşit olup olmadığını kontrol eder.
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == float("-inf") or b == float("-inf"):
        return a == float("-inf") and b == float("-inf")
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) < 1e-6


def _reference_per_query(
    num_nodes: int,
    edges: Sequence[Edge],
    queries: List[Tuple[int, int]],
    dist_matrix: List[List[float]],
    has_negative_cycle: bool,
) -> List[Optional[float]]:
    # Sorgular için doğru (referans) kabul edilecek sonuçları Bellman-Ford veya matris kullanarak hesaplar.
    if has_negative_cycle:
        out: List[Optional[float]] = []
        for s, t in queries:
            d, _ = bellman_ford(num_nodes, edges, s, t)
            out.append(_normalize_distance(d))
        return out
    return [_matrix_distance(dist_matrix, s, t) for s, t in queries]


def _average_distance_summary(distances: List[Optional[float]]) -> str:
    # Sorgularda bulunan uzaklıkların (ortalama, ulaşılamayan vb.) özetini bir string olarak oluşturur.
    finites: List[float] = []
    n_none = 0
    n_neg_inf = 0
    for d in distances:
        if d is None:
            n_none += 1
        elif d == float("-inf"):
            n_neg_inf += 1
        elif math.isfinite(d):
            finites.append(d)
    parts: List[str] = []
    if finites:
        parts.append(f"mean={sum(finites) / len(finites):.4g}")
    if n_none:
        parts.append(f"unreachable={n_none}")
    if n_neg_inf:
        parts.append(f"neg_inf={n_neg_inf}")
    if not parts:
        return "n/a"
    return "; ".join(parts)


def _status_for_row(
    case: Dict[str, Any],
    algorithm_name: str,
    has_negative_cycle: bool,
    all_match_reference: bool,
) -> str:
    # Algoritmanın çalışma durumunu (Doğru, Hata, Referans vb.) test durumuna göre belirler.
    if case["allow_negative"] and algorithm_name in DIJKSTRA_FAMILY:
        return NEG_WEIGHT_LIMITATION
    if has_negative_cycle and algorithm_name in REFERENCE_ALGOS:
        return "Negative cycle detected"
    if algorithm_name in REFERENCE_ALGOS and case["allow_negative"]:
        return "Reference"
    if not case["allow_negative"]:
        return "Correct" if all_match_reference else "Mismatch"
    if algorithm_name in REFERENCE_ALGOS:
        return "Reference"
    return "Correct" if all_match_reference else "Mismatch"


def _run_floyd_warshall_benchmark_row(
    case: Dict[str, Any],
    num_nodes: int,
    queries: List[Tuple[int, int]],
    query_count: int,
    fw_shared: Tuple[List[List[float]], int, bool, float],
) -> Dict[str, Any]:
    # Floyd-Warshall algoritması için benchmark sonuçlarını hesaplar ve bir sözlük (row) olarak döner.
    dist_matrix, prep_visited, has_negative_cycle, prep_seconds = fw_shared

    t_lu0 = time.perf_counter()
    distances: List[Optional[float]] = []
    for s, t in queries:
        distances.append(_matrix_distance(dist_matrix, s, t))
    t_lu1 = time.perf_counter()
    lookup_seconds = t_lu1 - t_lu0

    total_seconds = prep_seconds + lookup_seconds
    average_seconds = total_seconds / query_count if query_count else 0.0
    average_visited = prep_visited / query_count if query_count else float(prep_visited)

    status = "Negative cycle detected" if has_negative_cycle else "Reference"

    return {
        "test_case": case["name"],
        "num_nodes": str(num_nodes),
        "density": f"{case['density']:.4f}",
        "allow_negative": str(bool(case["allow_negative"])).lower(),
        "query_count": str(query_count),
        "algorithm": "Floyd-Warshall",
        "average_distance_summary": _average_distance_summary(distances),
        "average_visited_count": f"{average_visited:.6f}",
        "total_runtime_seconds": f"{total_seconds:.9f}",
        "average_runtime_seconds": f"{average_seconds:.9f}",
        "status": status,
    }


def _run_normal_algorithm_row(
    case: Dict[str, Any],
    algorithm_name: str,
    algo_fn: AlgoFn,
    num_nodes: int,
    edges: Sequence[Edge],
    queries: List[Tuple[int, int]],
    query_count: int,
    reference: List[Optional[float]],
    has_negative_cycle: bool,
) -> Dict[str, Any]:
    # Standart algoritmaların (Dijkstra, Bellman-Ford vb.) performansını ölçer ve sonucu döndürür.
    def run_queries():
        distances_inner: List[Optional[float]] = []
        tot_vis = 0
        tot_sec = 0.0
        for s, t in queries:
            t0 = time.perf_counter()
            dist, visited = algo_fn(num_nodes, edges, s, t)
            t1 = time.perf_counter()
            tot_sec += t1 - t0
            tot_vis += visited
            distances_inner.append(_normalize_distance(dist))
        return distances_inner, tot_vis, tot_sec

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_queries)
            distances, total_visited, total_seconds = future.result(timeout=300)
    except concurrent.futures.TimeoutError:
        return {
            "test_case": case["name"],
            "num_nodes": str(num_nodes),
            "density": f"{case['density']:.4f}",
            "allow_negative": str(bool(case["allow_negative"])).lower(),
            "query_count": str(query_count),
            "algorithm": algorithm_name,
            "average_distance_summary": "SKIPPED",
            "average_visited_count": "-",
            "total_runtime_seconds": "-",
            "average_runtime_seconds": "-",
            "status": "Skipped due to 5-minute timeout limit",
        }

    average_seconds = total_seconds / query_count if query_count else 0.0
    average_visited = total_visited / query_count if query_count else 0.0

    all_match = all(
        _distances_equivalent(distances[i], reference[i]) for i in range(len(queries))
    )
    status = _status_for_row(
        case, algorithm_name, has_negative_cycle, all_match
    )

    return {
        "test_case": case["name"],
        "num_nodes": str(num_nodes),
        "density": f"{case['density']:.4f}",
        "allow_negative": str(bool(case["allow_negative"])).lower(),
        "query_count": str(query_count),
        "algorithm": algorithm_name,
        "average_distance_summary": _average_distance_summary(distances),
        "average_visited_count": f"{average_visited:.6f}",
        "total_runtime_seconds": f"{total_seconds:.9f}",
        "average_runtime_seconds": f"{average_seconds:.9f}",
        "status": status,
    }


def _skipped_row(case: Dict[str, Any], algorithm_name: str, reason: str) -> Dict[str, Any]:
    # Zaman aşımı gibi nedenlerle atlanan algoritmalar için boş/skipped durumunu içeren bir satır oluşturur.
    n = case["num_nodes"]
    return {
        "test_case": case["name"],
        "num_nodes": str(n),
        "density": f"{case['density']:.4f}",
        "allow_negative": str(bool(case["allow_negative"])).lower(),
        "query_count": str(case["query_count"]),
        "algorithm": algorithm_name,
        "average_distance_summary": "SKIPPED",
        "average_visited_count": "-",
        "total_runtime_seconds": "-",
        "average_runtime_seconds": "-",
        "status": reason,
    }


def _run_benchmark() -> List[Dict[str, Any]]:
    # Test senaryolarını sırayla tüm algoritmalar için çalıştırarak benchmark sonuçlarını tablo satırları halinde toplar.
    rows: List[Dict[str, Any]] = []
    for case in create_test_cases():
        num_nodes = case["num_nodes"]
        edges = case["edges"]
        queries: List[Tuple[int, int]] = list(case["queries"])
        query_count = int(case["query_count"])
        benchmark_type = case["benchmark_type"]

        dist_matrix: List[List[float]] = []
        prep_visited = 0
        has_negative_cycle = False
        reference: List[Optional[float]] = []
        fw_shared: Optional[Tuple[List[List[float]], int, bool, float]] = None

        def run_fw_prep():
            return floyd_warshall_preprocess(num_nodes, edges)

        t_prep0 = time.perf_counter()
        print(f"[{case['name']}] Running Floyd-Warshall preprocess...", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_fw_prep)
                dist_matrix, prep_visited, has_negative_cycle = future.result(timeout=300)
            prep_seconds = time.perf_counter() - t_prep0
            fw_shared = (dist_matrix, prep_visited, has_negative_cycle, prep_seconds)
            reference = _reference_per_query(
                num_nodes, edges, queries, dist_matrix, has_negative_cycle
            )
        except concurrent.futures.TimeoutError:
            fw_shared = None
            has_negative_cycle = False
            reference = []
            for s, t in queries:
                dist, _ = dijkstra_list(num_nodes, edges, s, t)
                reference.append(_normalize_distance(dist))

        for algo_name, algo_fn in ALGORITHMS:
            print(f"[{case['name']}] Running {algo_name}...", flush=True)
            row = _run_normal_algorithm_row(
                case,
                algo_name,
                algo_fn,
                num_nodes,
                edges,
                queries,
                query_count,
                reference,
                has_negative_cycle,
            )
            rows.append(row)

        if fw_shared is not None:
            rows.append(
                _run_floyd_warshall_benchmark_row(
                    case, num_nodes, queries, query_count, fw_shared
                )
            )
        else:
            rows.append({
                "test_case": case["name"],
                "num_nodes": str(num_nodes),
                "density": f"{case['density']:.4f}",
                "allow_negative": str(bool(case["allow_negative"])).lower(),
                "query_count": str(query_count),
                "algorithm": "Floyd-Warshall",
                "average_distance_summary": "SKIPPED",
                "average_visited_count": "-",
                "total_runtime_seconds": "-",
                "average_runtime_seconds": "-",
                "status": "Skipped due to 5-minute timeout limit",
            })

    return rows


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    # Elde edilen benchmark sonuçlarını belirtilen CSV dosyasına yazar.
    fieldnames = [
        "test_case",
        "num_nodes",
        "density",
        "allow_negative",
        "query_count",
        "algorithm",
        "average_distance_summary",
        "average_visited_count",
        "total_runtime_seconds",
        "average_runtime_seconds",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _print_table(rows: List[Dict[str, Any]]) -> None:
    # Sonuçların özetini konsol ekranına okunabilir bir tablo olarak yazdırır.
    header = (
        "| test_case | n | density | allow_neg | Q | algorithm | avg_dist | avg_vis | "
        "t_total_s | t_avg_s | status |"
    )
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for r in rows:
        tc = r["test_case"][:22] + ("…" if len(r["test_case"]) > 22 else "")
        an = r["allow_negative"]
        st = r["status"][:40] + ("…" if len(r["status"]) > 40 else "")
        print(
            f"| {tc:22s} | {r['num_nodes']:>3s} | {r['density']:>7s} | "
            f"{an:>9s} | {r['query_count']:>2s} | "
            f"{r['algorithm'][:22]:22s} | {r['average_distance_summary'][:16]:16s} | "
            f"{r['average_visited_count'][:10]:10s} | {r['total_runtime_seconds'][:11]:11s} | "
            f"{r['average_runtime_seconds'][:11]:11s} | {st:42s} |"
        )


def _summarize(rows: List[Dict[str, Any]]) -> None:
    # En hızlı algoritmaları ve çeşitli istatistiksel notları konsolda özet halinde sunar.
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["average_runtime_seconds"] == "-":
            continue
        key = (r["test_case"], int(r["num_nodes"]), r["algorithm"])
        groups[key].append(r)

    print()
    print("=== Fastest average runtime per (test_case, algorithm) ===")
    for key in sorted(groups.keys()):
        best = min(groups[key], key=lambda x: float(x["average_runtime_seconds"]))
        print(
            f"  {key[0]} / {key[1]} / {key[2]}: "
            f"{best['average_runtime_seconds']} s avg"
        )

    print()
    print("=== Notes ===")
    print(
        "- EcoRoute motivating scenario: shortest-path distance is interpreted as "
        "energy cost in kWh in the report narrative."
    )
    print(
        "- Negative edge weights represent regenerative braking or downhill "
        "energy recovery in that interpretation."
    )
    print(
        "- Code and benchmark_results.csv stay graph-theoretic and generic "
        "(distance, visited_count, runtime, status) for clarity and reproducibility."
    )
    print(
        "- Floyd-Warshall: floyd_warshall_preprocess runs once per graph; all queries "
        "read from the distance matrix. Reported total_runtime_seconds includes "
        "preprocessing plus lookup time; average_runtime_seconds divides that total "
        "by query_count. After the matrix is built, each source-target lookup is O(1)."
    )
    print(
        "- Dijkstra-based algorithms are not guaranteed correct with negative edge "
        "weights; negative-edge cases are limitation demos."
    )
    print(
        "- Bellman-Ford and Floyd-Warshall demonstrate negative-weight support and "
        "negative-cycle detection."
    )
    print(
        "- large_scalability cases run only Dijkstra List and Bidirectional Dijkstra List; "
        "other algorithms are skipped but still appear in the console table and CSV."
    )
    print(
        "- Skipped rows use average_distance_summary=SKIPPED, average_visited_count=-, "
        "runtime fields=-, and status=skip reason."
    )
    print(
        "- Visited-count semantics differ by algorithm; treat comparisons as coarse "
        "workload indicators."
    )


def _optional_plots(rows: List[Dict[str, Any]]) -> None:
    # Eğer matplotlib yüklüyse, algoritmaların çalışma süreleri ve ziyaret edilen düğüm sayıları için grafikler çizer.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    active = [r for r in rows if r["average_runtime_seconds"] != "-"]
    if not active:
        return

    by_algo: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    by_algo_v: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in active:
        try:
            n = int(r["num_nodes"])
            t_avg = float(r["average_runtime_seconds"])
            v_avg = float(r["average_visited_count"])
        except ValueError:
            continue
        by_algo[r["algorithm"]].append((n, t_avg * 1000.0))
        by_algo_v[r["algorithm"]].append((n, v_avg))

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for name, pts in sorted(by_algo.items()):
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax1.plot(xs, ys, marker="o", label=name, linestyle="-", alpha=0.85)
    ax1.set_xlabel("num_nodes")
    ax1.set_ylabel("average_runtime (ms)")
    ax1.set_title("Average runtime per query vs num_nodes (from benchmark_results schema)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    fig1.tight_layout()
    fig1.savefig("runtime_vs_nodes.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for name, pts in sorted(by_algo_v.items()):
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax2.plot(xs, ys, marker="o", label=name, linestyle="-", alpha=0.85)
    ax2.set_xlabel("num_nodes")
    ax2.set_ylabel("average_visited_count")
    ax2.set_title("Average visited count vs num_nodes")
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
