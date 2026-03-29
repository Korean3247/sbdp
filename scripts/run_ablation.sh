#!/bin/bash
# Phase 5: Ablation study (z-score only vs EMA only) + EL2N baseline
# On CIFAR-10 only
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SEEDS=(0 1 2)

echo "=== Phase 5: Ablation + EL2N Baseline ==="

CONFIGS=(
    "configs/cifar10_resnet18_zscore_only_ret0.3.yaml"
    "configs/cifar10_resnet18_zscore_only_ret0.5.yaml"
    "configs/cifar10_resnet18_ema_only_ret0.3.yaml"
    "configs/cifar10_resnet18_ema_only_ret0.5.yaml"
    "configs/cifar10_resnet18_el2n_ret0.3.yaml"
    "configs/cifar10_resnet18_el2n_ret0.5.yaml"
    "configs/cifar10_resnet18_el2n_ret0.7.yaml"
)

echo "Configs: ${#CONFIGS[@]}, Seeds: ${#SEEDS[@]}, Total: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))"

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ">>> Running: $config seed=$seed"
        python scripts/run_train.py --config "$config" --seed "$seed"
        echo ">>> Done"
    done
done

echo "=== Ablation + EL2N experiments complete ==="
python scripts/summarize_results.py outputs
