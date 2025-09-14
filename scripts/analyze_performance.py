#!/usr/bin/env python3
"""
Analyze performance results persisted by tests/test_performance.py.

Features:
- Summaries for latest commit: cross-backend compile_ms per interface.
- Deltas vs previous commit for each interface/backend.
- Simple trend stats over time (min/median/max) per interface/backend.

Usage examples:
- python scripts/analyze_performance.py
- python scripts/analyze_performance.py --results perf_results/perf_results.jsonl --latest
- python scripts/analyze_performance.py --interface demo
- python scripts/analyze_performance.py --csv perf_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Tuple


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"No results found at {path}. Run tests/test_performance.py first.")
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as inf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def latest_commit(records: List[Dict[str, Any]]) -> str:
    commits = [r.get("commit", "unknown") for r in records]
    return commits[-1] if commits else "unknown"


def group_by_commit(records: Iterable[Dict[str, Any]]):
    byc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        byc[r.get("commit", "unknown")].append(r)
    return byc


def summarize_commit(records: List[Dict[str, Any]], metric: str) -> Dict[Tuple[str, str], float]:
    """Return median metric keyed by (interface, backend)."""
    acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in records:
        if r.get("metric") != metric:
            continue
        key = (r.get("interface", ""), r.get("backend", ""))
        acc[key].append(float(r.get("value", 0.0)))
    return {k: median(vs) for k, vs in acc.items() if vs}


def cross_backend_tables(summary: Dict[Tuple[str, str], float]):
    per_iface: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (iname, be), val in summary.items():
        per_iface[iname].append((be, val))
    for iname, rows in per_iface.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda x: x[1])
        yield iname, rows


def write_csv(path: Path, latest_sum: Dict[Tuple[str, str], float], metric: str):
    if not latest_sum:
        return
    with path.open("w", newline="", encoding="utf-8") as outf:
        w = csv.writer(outf)
        w.writerow(["interface", "backend", f"{metric}_median"])
        for (iname, be), val in sorted(latest_sum.items()):
            w.writerow([iname, be, f"{val:.3f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="perf_results/perf_results.jsonl", help="Path to results JSONL")
    ap.add_argument("--interface", default=None, help="Filter to a specific interface")
    ap.add_argument("--latest", action="store_true", help="Show only latest commit summary")
    ap.add_argument("--csv", default=None, help="Optional CSV output path for latest summary")
    ap.add_argument("--metric", default="exec_ms", help="Metric to summarize (exec_ms or compile_ms)")
    args = ap.parse_args()

    path = Path(args.results)
    records = load_records(path)
    if args.interface:
        records = [r for r in records if r.get("interface") == args.interface]

    by_commit = group_by_commit(records)
    commits = list(by_commit.keys())
    if not commits:
        print("No records loaded.")
        return

    # Latest commit summary
    latest = commits[-1]
    latest_sum = summarize_commit(by_commit[latest], args.metric)
    print(f"Latest commit: {latest}")
    for iname, rows in cross_backend_tables(latest_sum):
        print(f"Interface: {iname}")
        for be, val in rows:
            print(f"  {be:>6}: {val:8.3f} ms")

    # Deltas vs previous commit, if available
    if len(commits) >= 2 and not args.latest:
        prev = commits[-2]
        prev_sum = summarize_commit(by_commit[prev], args.metric)
        print(f"\nDelta vs previous commit: {prev} -> {latest}")
        for (iname, be), latest_val in latest_sum.items():
            prev_val = prev_sum.get((iname, be))
            if prev_val is None:
                continue
            delta = latest_val - prev_val
            pct = (delta / prev_val * 100.0) if prev_val else 0.0
            print(f"{iname}:{be}  {prev_val:8.3f} -> {latest_val:8.3f}  ({delta:+.3f} ms, {pct:+.1f}%)")

    # Trend stats over all commits
    if not args.latest:
        all_sum = summarize_commit(records, args.metric)
        per_if_be: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for (iname, be), val in all_sum.items():
            per_if_be[(iname, be)].append(val)
        print("\nTrend stats (median across commits):")
        for (iname, be), vals in sorted(per_if_be.items()):
            print(f"{iname}:{be}  median={median(vals):.3f} ms  min={min(vals):.3f}  max={max(vals):.3f}")

    if args.csv:
        write_csv(Path(args.csv), latest_sum, args.metric)
        print(f"\nWrote CSV summary to {args.csv}")


if __name__ == "__main__":
    main()
