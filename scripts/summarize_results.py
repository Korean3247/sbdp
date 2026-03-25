#!/usr/bin/env python3
"""Summarize results across all runs into a single table.

Automatically separates clean (noise_rate=0) and noisy results.
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path


def collect_summaries(output_dir: str = "outputs") -> list[dict]:
    results = []
    for summary_file in Path(output_dir).rglob("summary.json"):
        with open(summary_file, "r") as f:
            data = json.load(f)
            data["path"] = str(summary_file.parent)
            results.append(data)
    return results


def split_by_noise(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split results into clean (noise_rate=0) and noisy (noise_rate>0)."""
    clean = [r for r in results if r.get("noise_rate", 0.0) == 0.0]
    noisy = [r for r in results if r.get("noise_rate", 0.0) > 0.0]
    return clean, noisy


def filter_seeds(results: list[dict], seeds=(0, 1, 2)) -> list[dict]:
    return [r for r in results if r["seed"] in seeds]


def print_table(results: list[dict], title: str = ""):
    if not results:
        print(f"No results found{f' for {title}' if title else ''}.")
        return

    if title:
        print(f"\n{'=' * 40}")
        print(f"  {title}")
        print(f"{'=' * 40}")

    results.sort(key=lambda r: (r.get("mode", ""), r.get("retention_ratio", 1.0), r.get("seed", 0)))

    cols = ["mode", "seed", "retention_ratio", "noise_rate",
            "best_test_acc", "final_test_acc", "mean_score_drift",
            "mean_turnover", "num_pruning_events"]
    header = " | ".join(f"{c:>20s}" for c in cols)
    print(header)
    print("-" * len(header))

    for r in results:
        row = " | ".join(f"{str(r.get(c, '')):>20s}" for c in cols)
        print(row)


def save_csv(results: list[dict], path: str):
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["mode", "seed", "retention_ratio", "noise_rate",
            "best_test_acc", "final_test_acc", "mean_score_drift",
            "mean_turnover", "num_pruning_events", "run_name"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"CSV saved to {path}")


def print_paper_table(results: list[dict], title: str = ""):
    """Print paper-ready table grouped by mode and retention."""
    if not results:
        return

    if title:
        print(f"\n=== {title} ===\n")

    print(f"{'Mode':<25} {'Ret':>5} {'Best Acc':>16} {'Final Acc':>16} {'Turnover':>16} {'Score Drift':>16}")
    print("-" * 100)

    # Full baseline
    full_runs = [r for r in results if r["mode"] == "full"]
    if full_runs:
        m, s = np.mean([r["best_test_acc"] for r in full_runs]), np.std([r["best_test_acc"] for r in full_runs])
        fm, fs = np.mean([r["final_test_acc"] for r in full_runs]), np.std([r["final_test_acc"] for r in full_runs])
        print(f"{'Full Data':<25} {'1.0':>5} {m:.4f}\u00b1{s:.4f} {fm:.4f}\u00b1{fs:.4f}     {'--':>16} {'--':>16}")

    for mode_key, label in [
        ("calibrated_topk_loss", "Calibrated Top-K"),
        ("raw_topk_loss", "Raw Top-K"),
        ("random_pruning", "Random"),
    ]:
        mode_runs = [r for r in results if r["mode"] == mode_key]
        rets = sorted(set(r["retention_ratio"] for r in mode_runs))
        for ret in rets:
            runs = [r for r in mode_runs if r["retention_ratio"] == ret]
            if not runs:
                continue
            acc_m = np.mean([r["best_test_acc"] for r in runs])
            acc_s = np.std([r["best_test_acc"] for r in runs])
            facc_m = np.mean([r["final_test_acc"] for r in runs])
            facc_s = np.std([r["final_test_acc"] for r in runs])
            to_m = np.mean([r["mean_turnover"] for r in runs])
            to_s = np.std([r["mean_turnover"] for r in runs])
            sd_m = np.mean([r["mean_score_drift"] for r in runs])
            sd_s = np.std([r["mean_score_drift"] for r in runs])
            print(f"{label:<25} {ret:>5} {acc_m:.4f}\u00b1{acc_s:.4f} {facc_m:.4f}\u00b1{facc_s:.4f} {to_m:.4f}\u00b1{to_s:.4f} {sd_m:.4f}\u00b1{sd_s:.4f}")


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    all_results = collect_summaries(output_dir)
    all_results = filter_seeds(all_results)

    clean, noisy = split_by_noise(all_results)

    # --- Clean results ---
    if clean:
        print_table(clean, "Phase 1: Clean Data Results")
        save_csv(clean, "outputs/results/summary_clean.csv")
        print_paper_table(clean, "Paper Table: Clean Data (CIFAR-10)")

    # --- Noisy results ---
    if noisy:
        noise_rates = sorted(set(r.get("noise_rate", 0.0) for r in noisy))
        for nr in noise_rates:
            nr_results = [r for r in noisy if r.get("noise_rate", 0.0) == nr]
            print_table(nr_results, f"Phase 2: Label Noise {nr}")
            save_csv(nr_results, f"outputs/results/summary_noise{nr}.csv")
            print_paper_table(nr_results, f"Paper Table: Noise {nr} (CIFAR-10)")

    # --- Combined CSV ---
    save_csv(all_results, "outputs/results/summary_all.csv")

    print(f"\nTotal runs: {len(all_results)} (clean={len(clean)}, noisy={len(noisy)})")


if __name__ == "__main__":
    main()
