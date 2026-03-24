#!/usr/bin/env python3
"""Generate paper-quality figures from experiment results."""

import os
import sys
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


def filter_clean(results):
    """Remove smoke runs (seed=42) and keep only seed 0,1,2."""
    return [r for r in results if r["seed"] in [0, 1, 2]]


def group_by_mode_retention(results):
    """Group results by (mode, retention_ratio) -> list of runs."""
    groups = defaultdict(list)
    for r in results:
        key = (r["mode"], r["retention_ratio"])
        groups[key] = groups.get(key, [])
        groups[key].append(r)
    return groups


def get_stats(runs, key):
    vals = [r[key] for r in runs]
    return np.mean(vals), np.std(vals)


def fig1_accuracy_vs_retention(groups, save_dir="outputs/figures"):
    """Accuracy vs retention ratio for each mode."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    modes = {
        "calibrated_topk_loss": ("Calibrated Top-K", "#2196F3", "o", "-"),
        "raw_topk_loss": ("Raw Top-K", "#F44336", "s", "--"),
        "random_pruning": ("Random", "#9E9E9E", "^", "-."),
    }
    retentions = [0.3, 0.5, 0.7]

    # Full baseline
    full_runs = [r for r in sum(groups.values(), []) if r["mode"] == "full"]
    if full_runs:
        full_mean, full_std = get_stats(full_runs, "best_test_acc")
        ax.axhline(y=full_mean, color="#4CAF50", linestyle=":", linewidth=2, label=f"Full Data ({full_mean:.4f})")
        ax.fill_between([0.25, 0.75], full_mean - full_std, full_mean + full_std, color="#4CAF50", alpha=0.1)

    for mode, (label, color, marker, ls) in modes.items():
        means, stds, rets = [], [], []
        for ret in retentions:
            key = (mode, ret)
            if key in groups:
                m, s = get_stats(groups[key], "best_test_acc")
                means.append(m)
                stds.append(s)
                rets.append(ret)
        if means:
            means, stds = np.array(means), np.array(stds)
            ax.errorbar(rets, means, yerr=stds, label=label, color=color,
                        marker=marker, linestyle=ls, linewidth=2, markersize=8,
                        capsize=5, capthick=2)

    ax.set_xlabel("Retention Ratio", fontsize=13)
    ax.set_ylabel("Best Test Accuracy", fontsize=13)
    ax.set_title("Accuracy vs. Retention Ratio (CIFAR-10, ResNet-18)", fontsize=14)
    ax.set_xlim(0.25, 0.75)
    ax.set_xticks(retentions)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig1_accuracy_vs_retention.png", dpi=200)
    plt.savefig(f"{save_dir}/fig1_accuracy_vs_retention.pdf")
    plt.close()
    print(f"Saved: fig1_accuracy_vs_retention")


def fig2_turnover_comparison(groups, save_dir="outputs/figures"):
    """Selection turnover comparison across modes and retention ratios."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    modes = {
        "calibrated_topk_loss": ("Calibrated Top-K", "#2196F3"),
        "raw_topk_loss": ("Raw Top-K", "#F44336"),
        "random_pruning": ("Random", "#9E9E9E"),
    }
    retentions = [0.3, 0.5, 0.7]
    x = np.arange(len(retentions))
    width = 0.25

    for i, (mode, (label, color)) in enumerate(modes.items()):
        means, stds = [], []
        for ret in retentions:
            key = (mode, ret)
            if key in groups:
                m, s = get_stats(groups[key], "mean_turnover")
                means.append(m)
                stds.append(s)
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, label=label,
               color=color, alpha=0.85, capsize=4)

    ax.set_xlabel("Retention Ratio", fontsize=13)
    ax.set_ylabel("Mean Selection Turnover", fontsize=13)
    ax.set_title("Selection Stability: Turnover Comparison", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(r) for r in retentions])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig2_turnover_comparison.png", dpi=200)
    plt.savefig(f"{save_dir}/fig2_turnover_comparison.pdf")
    plt.close()
    print(f"Saved: fig2_turnover_comparison")


def fig3_score_drift_comparison(groups, save_dir="outputs/figures"):
    """Score drift comparison."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    modes = {
        "calibrated_topk_loss": ("Calibrated Top-K", "#2196F3"),
        "raw_topk_loss": ("Raw Top-K", "#F44336"),
        "random_pruning": ("Random", "#9E9E9E"),
    }
    retentions = [0.3, 0.5, 0.7]
    x = np.arange(len(retentions))
    width = 0.25

    for i, (mode, (label, color)) in enumerate(modes.items()):
        means, stds = [], []
        for ret in retentions:
            key = (mode, ret)
            if key in groups:
                m, s = get_stats(groups[key], "mean_score_drift")
                means.append(m)
                stds.append(s)
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, label=label,
               color=color, alpha=0.85, capsize=4)

    ax.set_xlabel("Retention Ratio", fontsize=13)
    ax.set_ylabel("Mean Score Drift Index", fontsize=13)
    ax.set_title("Score Drift Comparison", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(r) for r in retentions])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig3_score_drift_comparison.png", dpi=200)
    plt.savefig(f"{save_dir}/fig3_score_drift_comparison.pdf")
    plt.close()
    print(f"Saved: fig3_score_drift_comparison")


def fig4_seed_variance(groups, save_dir="outputs/figures"):
    """Per-seed accuracy scatter to show variance."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    retentions = [0.3, 0.5, 0.7]

    modes = {
        "calibrated_topk_loss": ("Calibrated", "#2196F3", "o"),
        "raw_topk_loss": ("Raw", "#F44336", "s"),
        "random_pruning": ("Random", "#9E9E9E", "^"),
    }

    for ax_idx, ret in enumerate(retentions):
        ax = axes[ax_idx]
        for mode, (label, color, marker) in modes.items():
            key = (mode, ret)
            if key in groups:
                accs = [r["best_test_acc"] for r in groups[key]]
                seeds = [r["seed"] for r in groups[key]]
                ax.scatter(seeds, accs, color=color, marker=marker, s=100,
                           label=label, zorder=3, edgecolors="white", linewidth=1)
                ax.axhline(y=np.mean(accs), color=color, linestyle="--", alpha=0.5)

        # Full baseline
        full_runs = [r for r in sum(groups.values(), []) if r["mode"] == "full"]
        if full_runs:
            full_mean = np.mean([r["best_test_acc"] for r in full_runs])
            ax.axhline(y=full_mean, color="#4CAF50", linestyle=":", linewidth=2, alpha=0.7)

        ax.set_title(f"Retention = {ret}", fontsize=13)
        ax.set_xlabel("Seed", fontsize=12)
        ax.set_xticks([0, 1, 2])
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Best Test Accuracy", fontsize=12)
            ax.legend(fontsize=10)

    fig.suptitle("Per-Seed Accuracy Variance (CIFAR-10, ResNet-18)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig4_seed_variance.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{save_dir}/fig4_seed_variance.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: fig4_seed_variance")


def fig5_raw03_collapse(output_dir="outputs", save_dir="outputs/figures"):
    """Analyze raw_topk ret=0.3 training curves to show collapse."""
    os.makedirs(save_dir, exist_ok=True)
    import csv

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = {0: "#E53935", 1: "#F57C00", 2: "#8E24AA"}

    # Raw ret=0.3
    for seed in [0, 1, 2]:
        run_dir = Path(output_dir) / f"cifar10_resnet18_raw_topk_loss_ret0.3_seed{seed}"
        metrics_file = run_dir / "metrics.csv"
        if metrics_file.exists():
            epochs, accs = [], []
            with open(metrics_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row["epoch"]))
                    accs.append(float(row["test_acc"]))
            ax.plot(epochs, accs, color=colors[seed], linewidth=1.5,
                    label=f"Raw Top-K seed={seed}", alpha=0.8)

    # Calibrated ret=0.3 for comparison
    cal_colors = {0: "#1565C0", 1: "#1E88E5", 2: "#42A5F5"}
    for seed in [0, 1, 2]:
        run_dir = Path(output_dir) / f"cifar10_resnet18_calibrated_topk_loss_ret0.3_seed{seed}"
        metrics_file = run_dir / "metrics.csv"
        if metrics_file.exists():
            epochs, accs = [], []
            with open(metrics_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row["epoch"]))
                    accs.append(float(row["test_acc"]))
            ax.plot(epochs, accs, color=cal_colors[seed], linewidth=1.5,
                    linestyle="--", label=f"Calibrated seed={seed}", alpha=0.8)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.set_title("Raw Top-K Collapse at 30% Retention vs. Calibrated Stability", fontsize=13)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig5_raw03_collapse.png", dpi=200)
    plt.savefig(f"{save_dir}/fig5_raw03_collapse.pdf")
    plt.close()
    print(f"Saved: fig5_raw03_collapse")


def print_clean_table(groups):
    """Print paper-ready LaTeX-style table."""
    print("\n=== Paper Table: Accuracy & Stability ===\n")
    print(f"{'Mode':<25} {'Ret':>5} {'Best Acc':>16} {'Turnover':>16} {'Score Drift':>16}")
    print("-" * 82)

    # Full baseline
    full_runs = [r for r in sum(groups.values(), []) if r["mode"] == "full"]
    if full_runs:
        m, s = get_stats(full_runs, "best_test_acc")
        print(f"{'Full Data':<25} {'1.0':>5} {m:.4f}±{s:.4f}       {'—':>16} {'—':>16}")

    for mode_key, label in [
        ("calibrated_topk_loss", "Calibrated Top-K"),
        ("raw_topk_loss", "Raw Top-K"),
        ("random_pruning", "Random"),
    ]:
        for ret in [0.3, 0.5, 0.7]:
            key = (mode_key, ret)
            if key in groups:
                acc_m, acc_s = get_stats(groups[key], "best_test_acc")
                to_m, to_s = get_stats(groups[key], "mean_turnover")
                sd_m, sd_s = get_stats(groups[key], "mean_score_drift")
                print(f"{label:<25} {ret:>5} {acc_m:.4f}±{acc_s:.4f} {to_m:.4f}±{to_s:.4f} {sd_m:.4f}±{sd_s:.4f}")


def main():
    results = collect_summaries("outputs")
    results = filter_clean(results)
    groups = group_by_mode_retention(results)

    print(f"Clean runs: {len(results)}")
    print_clean_table(groups)

    fig1_accuracy_vs_retention(groups)
    fig2_turnover_comparison(groups)
    fig3_score_drift_comparison(groups)
    fig4_seed_variance(groups)
    fig5_raw03_collapse()

    print(f"\nAll figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
