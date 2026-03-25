#!/usr/bin/env python3
"""Generate figures for Phase 2: Label Noise experiments."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
os.chdir(_PROJECT_ROOT)


def collect_summaries(output_dir="outputs"):
    results = []
    for summary_file in Path(output_dir).rglob("summary.json"):
        with open(summary_file) as f:
            data = json.load(f)
            data["path"] = str(summary_file.parent)
            results.append(data)
    return results


def get_stats(runs, key):
    vals = [r[key] for r in runs]
    return np.mean(vals), np.std(vals)


def fig_noise_accuracy_comparison(clean, noisy, save_dir="outputs/figures"):
    """Side-by-side: clean vs noise accuracy for each method/retention."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    modes = {
        "calibrated_topk_loss": ("Calibrated Top-K", "#2196F3"),
        "random_pruning": ("Random", "#9E9E9E"),
    }
    retentions = [0.3, 0.5]
    x = np.arange(len(retentions))
    width = 0.3

    for ax_idx, (dataset_results, title) in enumerate([
        (clean, "Clean Data"),
        (noisy, "Label Noise 20%"),
    ]):
        ax = axes[ax_idx]

        # Full baseline
        full_runs = [r for r in dataset_results if r["mode"] == "full"]
        if full_runs:
            fm, fs = get_stats(full_runs, "best_test_acc")
            ax.axhline(y=fm, color="#4CAF50", linestyle=":", linewidth=2,
                        label=f"Full ({fm:.3f})")

        for i, (mode, (label, color)) in enumerate(modes.items()):
            means, stds = [], []
            for ret in retentions:
                runs = [r for r in dataset_results
                        if r["mode"] == mode and r["retention_ratio"] == ret]
                if runs:
                    m, s = get_stats(runs, "best_test_acc")
                    means.append(m)
                    stds.append(s)
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(x + i * width, means, width, yerr=stds, label=label,
                   color=color, alpha=0.85, capsize=4)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Retention Ratio", fontsize=12)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([str(r) for r in retentions])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if ax_idx == 0:
            ax.set_ylabel("Best Test Accuracy", fontsize=12)

    fig.suptitle("Clean vs. Label Noise: Accuracy Comparison (CIFAR-10)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_noise_accuracy_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{save_dir}/fig_noise_accuracy_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig_noise_accuracy_comparison")


def fig_noise_degradation(clean, noisy, save_dir="outputs/figures"):
    """Bar chart showing accuracy drop from clean to noise for each method."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    configs = [
        ("full", 1.0, "Full Data", "#4CAF50"),
        ("calibrated_topk_loss", 0.3, "Calibrated 0.3", "#1565C0"),
        ("calibrated_topk_loss", 0.5, "Calibrated 0.5", "#42A5F5"),
        ("random_pruning", 0.3, "Random 0.3", "#757575"),
        ("random_pruning", 0.5, "Random 0.5", "#BDBDBD"),
    ]

    labels, clean_accs, noise_accs, drops = [], [], [], []
    for mode, ret, label, color in configs:
        c_runs = [r for r in clean if r["mode"] == mode and r["retention_ratio"] == ret]
        n_runs = [r for r in noisy if r["mode"] == mode and r["retention_ratio"] == ret]
        if c_runs and n_runs:
            cm, _ = get_stats(c_runs, "best_test_acc")
            nm, _ = get_stats(n_runs, "best_test_acc")
            labels.append(label)
            clean_accs.append(cm)
            noise_accs.append(nm)
            drops.append(cm - nm)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, clean_accs, width, label="Clean", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, noise_accs, width, label="Noise 20%", color="#F44336", alpha=0.85)

    # Annotate drops
    for i, drop in enumerate(drops):
        ax.annotate(f"-{drop:.3f}", xy=(x[i], min(clean_accs[i], noise_accs[i]) - 0.01),
                    ha="center", fontsize=9, color="#D32F2F", fontweight="bold")

    ax.set_ylabel("Best Test Accuracy", fontsize=12)
    ax.set_title("Accuracy Degradation Under 20% Label Noise", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.6, 1.0)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_noise_degradation.png", dpi=200)
    plt.savefig(f"{save_dir}/fig_noise_degradation.pdf")
    plt.close()
    print("Saved: fig_noise_degradation")


def fig_noise_final_gap(clean, noisy, save_dir="outputs/figures"):
    """Show best vs final accuracy gap (overfitting indicator) under noise."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    configs = [
        ("full", 1.0, "Full Data"),
        ("calibrated_topk_loss", 0.3, "Calibrated 0.3"),
        ("calibrated_topk_loss", 0.5, "Calibrated 0.5"),
        ("random_pruning", 0.3, "Random 0.3"),
        ("random_pruning", 0.5, "Random 0.5"),
    ]

    labels, best_means, final_means, gaps = [], [], [], []
    for mode, ret, label in configs:
        runs = [r for r in noisy if r["mode"] == mode and r["retention_ratio"] == ret]
        if runs:
            bm, _ = get_stats(runs, "best_test_acc")
            fm, _ = get_stats(runs, "final_test_acc")
            labels.append(label)
            best_means.append(bm)
            final_means.append(fm)
            gaps.append(bm - fm)

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, best_means, width, label="Best Acc", color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, final_means, width, label="Final Acc", color="#FF9800", alpha=0.85)

    for i, gap in enumerate(gaps):
        ax.annotate(f"gap={gap:.3f}", xy=(x[i], final_means[i] - 0.015),
                    ha="center", fontsize=9, color="#E65100", fontweight="bold")

    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Noise Overfitting: Best vs Final Accuracy (Noise 20%)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_noise_final_gap.png", dpi=200)
    plt.savefig(f"{save_dir}/fig_noise_final_gap.pdf")
    plt.close()
    print("Saved: fig_noise_final_gap")


def main():
    all_results = collect_summaries("outputs")
    all_results = [r for r in all_results if r["seed"] in [0, 1, 2]]

    clean = [r for r in all_results if r.get("noise_rate", 0.0) == 0.0]
    noisy = [r for r in all_results if r.get("noise_rate", 0.0) > 0.0]

    # Exclude historical from noise plots (not relevant)
    clean = [r for r in clean if r["mode"] != "calibrated_historical"]
    noisy = [r for r in noisy if r["mode"] != "calibrated_historical"]

    print(f"Clean runs: {len(clean)}, Noisy runs: {len(noisy)}")

    if noisy:
        fig_noise_accuracy_comparison(clean, noisy)
        fig_noise_degradation(clean, noisy)
        fig_noise_final_gap(clean, noisy)
        print("\nNoise figures saved to outputs/figures/")
    else:
        print("No noise experiment results found.")


if __name__ == "__main__":
    main()
