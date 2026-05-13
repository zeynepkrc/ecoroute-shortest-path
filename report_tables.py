"""
Read benchmark_results.csv and print markdown tables for reports.

Usage (from the project directory)::

    python report_tables.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

CSV_NAME = "benchmark_results.csv"
V_SUBSET = (50, 100, 250)

RUNTIME_ALGOS_AB: Sequence[str] = (
    "Dijkstra List",
    "Dijkstra Matrix",
    "Bidirectional Dijkstra List",
    "Bidirectional Dijkstra Matrix",
    "Bellman-Ford",
    "Floyd-Warshall",
)

VISITED_ALGOS_C: Sequence[str] = (
    "Dijkstra List",
    "Bidirectional Dijkstra List",
    "Bellman-Ford",
    "Floyd-Warshall",
)

MATRIX_ALGOS = ("Dijkstra Matrix", "Bidirectional Dijkstra Matrix")


def _script_csv_path() -> Path:
    return Path(__file__).resolve().parent / CSV_NAME


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _parse_density(s: str) -> float:
    return float(s)


def _is_runtime_skipped(row: Dict[str, str]) -> bool:
    if row.get("average_distance_summary", "").strip() == "SKIPPED":
        return True
    avg_rt = row.get("average_runtime_seconds", "").strip()
    return avg_rt in ("", "-")


def _runtime_ms_cell(row: Dict[str, str]) -> str:
    if _is_runtime_skipped(row):
        return "SKIPPED"
    sec = float(row["average_runtime_seconds"])
    return f"{sec * 1000.0:.4f}"


def _visited_cell(row: Dict[str, str]) -> str:
    if row.get("average_distance_summary", "").strip() == "SKIPPED":
        return "SKIPPED"
    vis = row.get("average_visited_count", "").strip()
    if vis in ("", "-"):
        return "SKIPPED"
    return f"{float(vis):.2f}"


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _index_by_key(rows: List[Dict[str, str]]) -> Dict[tuple, Dict[str, Dict[str, str]]]:
    """Map (num_nodes, density, allow_negative) -> algorithm name -> row."""
    out: Dict[tuple, Dict[str, Dict[str, str]]] = {}
    for r in rows:
        try:
            n = int(r["num_nodes"])
            d = _parse_density(r["density"])
            neg = _parse_bool(r["allow_negative"])
        except (KeyError, ValueError):
            continue
        key = (n, d, neg)
        out.setdefault(key, {})[r["algorithm"]] = r
    return out


def _print_markdown_table(headers: List[str], body_rows: List[List[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for line in body_rows:
        print("| " + " | ".join(line) + " |")


def _table_runtime_vs_v(
    by_key: Dict[tuple, Dict[str, Dict[str, str]]],
    density_target: float,
    title: str,
) -> None:
    print()
    print(f"## {title}")
    print()
    headers = ["V"] + list(RUNTIME_ALGOS_AB)
    body: List[List[str]] = []
    for v in V_SUBSET:
        key = (v, density_target, False)
        alg_map = by_key.get(key)
        if alg_map is None:
            continue
        row_cells = [str(v)]
        for algo in RUNTIME_ALGOS_AB:
            r = alg_map.get(algo)
            row_cells.append(_runtime_ms_cell(r) if r else "SKIPPED")
        body.append(row_cells)
    if not body:
        print("_No matching rows in CSV._")
        print()
        return
    _print_markdown_table(headers, body)
    print()


def _table_visited_sparse(by_key: Dict[tuple, Dict[str, Dict[str, str]]]) -> None:
    print()
    print("## Average visited count vs V \u2014 Sparse graphs")
    print()
    headers = ["V"] + list(VISITED_ALGOS_C)
    body: List[List[str]] = []
    density_target = 0.02
    for v in V_SUBSET:
        key = (v, density_target, False)
        alg_map = by_key.get(key)
        if alg_map is None:
            continue
        row_cells = [str(v)]
        for algo in VISITED_ALGOS_C:
            r = alg_map.get(algo)
            row_cells.append(_visited_cell(r) if r else "SKIPPED")
        body.append(row_cells)
    if not body:
        print("_No matching rows in CSV._")
        print()
        return
    _print_markdown_table(headers, body)
    print()


def _matrix_algorithms_cell(alg_map: Dict[str, Dict[str, str]]) -> str:
    cells = []
    for name in MATRIX_ALGOS:
        r = alg_map.get(name)
        cells.append("SKIPPED" if r is None or _is_runtime_skipped(r) else _runtime_ms_cell(r))
    if all(c == "SKIPPED" for c in cells):
        return "SKIPPED"
    return " / ".join(cells)


def _table_large_scalability(rows: List[Dict[str, str]]) -> None:
    print()
    print("## Large scalability benchmark")
    print()
    groups: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        tc = r.get("test_case", "")
        if not tc.startswith("scale_large"):
            continue
        groups.setdefault(tc, []).append(r)

    headers = [
        "V",
        "Density",
        "Dijkstra List",
        "Bidirectional Dijkstra List",
        "Matrix Algorithms",
        "Bellman-Ford",
        "Floyd-Warshall",
    ]
    body: List[List[str]] = []
    for tc in sorted(groups.keys(), key=lambda t: (int(groups[t][0]["num_nodes"]), t)):
        chunk = groups[tc]
        alg_map = {r["algorithm"]: r for r in chunk}
        sample = chunk[0]
        v = sample["num_nodes"]
        dens = sample["density"]
        row_cells = [
            v,
            dens,
            _runtime_ms_cell(alg_map["Dijkstra List"]) if "Dijkstra List" in alg_map else "SKIPPED",
            _runtime_ms_cell(alg_map["Bidirectional Dijkstra List"])
            if "Bidirectional Dijkstra List" in alg_map
            else "SKIPPED",
            _matrix_algorithms_cell(alg_map),
            _runtime_ms_cell(alg_map["Bellman-Ford"]) if "Bellman-Ford" in alg_map else "SKIPPED",
            _runtime_ms_cell(alg_map["Floyd-Warshall"]) if "Floyd-Warshall" in alg_map else "SKIPPED",
        ]
        body.append(row_cells)

    if not body:
        print("_No test cases with names starting with `scale_large`._")
        print()
        return
    _print_markdown_table(headers, body)
    print()


def main() -> int:
    path = _script_csv_path()
    if not path.is_file():
        print(f"Error: {path} not found.", file=sys.stderr)
        return 1

    rows = _load_rows(path)
    by_key = _index_by_key(rows)

    print("# Benchmark report tables")
    print()
    print(f"_Source: `{CSV_NAME}` (average runtime converted to milliseconds)._")
    print()

    _table_runtime_vs_v(
        by_key,
        0.02,
        "Average per-query runtime vs V \u2014 Sparse graphs",
    )
    _table_runtime_vs_v(
        by_key,
        0.5,
        "Average per-query runtime vs V \u2014 Dense graphs",
    )
    _table_visited_sparse(by_key)
    _table_large_scalability(rows)

    print()
    print(
        "_Note: In the EcoRoute scenario, distance values are interpreted as "
        "energy cost in kWh._"
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
