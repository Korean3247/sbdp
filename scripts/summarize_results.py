#!/usr/bin/env python3
"""Summarize results across all runs into a single table."""

import sys
import json
import csv
from pathlib import Path


def collect_summaries(output_dir: str = "outputs") -> list[dict]:
    results = []
    for summary_file in Path(output_dir).rglob("summary.json"):
        with open(summary_file, "r") as f:
            data = json.load(f)
            data["path"] = str(summary_file.parent)
            results.append(data)
    return results


def print_table(results: list[dict]):
    if not results:
        print("No results found.")
        return

    # Sort by mode, then seed
    results.sort(key=lambda r: (r.get("mode", ""), r.get("seed", 0)))

    # Header
    cols = ["mode", "seed", "retention_ratio", "best_test_acc", "final_test_acc",
            "mean_score_drift", "mean_turnover", "num_pruning_events"]
    header = " | ".join(f"{c:>20s}" for c in cols)
    print(header)
    print("-" * len(header))

    for r in results:
        row = " | ".join(f"{str(r.get(c, '')):>20s}" for c in cols)
        print(row)


def save_csv(results: list[dict], path: str = "outputs/results/summary_all.csv"):
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["mode", "seed", "retention_ratio", "best_test_acc", "final_test_acc",
            "mean_score_drift", "mean_turnover", "num_pruning_events", "run_name"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nCSV saved to {path}")


def aggregate_stats(results: list[dict]):
    """Print per-mode aggregated stats (mean ± std across seeds)."""
    import numpy as np

    modes = sorted(set(r["mode"] for r in results))
    print("\n=== Aggregated Results (mean ± std) ===\n")
    header = f"{'mode':>25s} | {'best_acc':>16s} | {'final_acc':>16s} | {'score_drift':>16s} | {'turnover':>16s}"
    print(header)
    print("-" * len(header))

    for mode in modes:
        runs = [r for r in results if r["mode"] == mode]
        if not runs:
            continue
        best_accs = np.array([r["best_test_acc"] for r in runs])
        final_accs = np.array([r["final_test_acc"] for r in runs])
        drifts = np.array([r["mean_score_drift"] for r in runs])
        turnovers = np.array([r["mean_turnover"] for r in runs])

        print(
            f"{mode:>25s} | "
            f"{best_accs.mean():.4f}±{best_accs.std():.4f} | "
            f"{final_accs.mean():.4f}±{final_accs.std():.4f} | "
            f"{drifts.mean():.4f}±{drifts.std():.4f} | "
            f"{turnovers.mean():.4f}±{turnovers.std():.4f}"
        )


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    results = collect_summaries(output_dir)
    print_table(results)
    save_csv(results)
    if len(results) > 1:
        aggregate_stats(results)


if __name__ == "__main__":
    main()
