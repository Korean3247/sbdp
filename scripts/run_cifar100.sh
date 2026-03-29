#!/bin/bash
# Phase 4: CIFAR-100 experiments
# full + calibrated + raw + random × 3 retentions × 3 seeds = 30 runs
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SEEDS=(0 1 2)

echo "=== Phase 4: CIFAR-100 Experiment ==="

# Full baseline
CONFIGS=(
    "configs/cifar100_resnet18_full.yaml"
    "configs/cifar100_resnet18_calibrated_ret0.3.yaml"
    "configs/cifar100_resnet18_calibrated_ret0.5.yaml"
    "configs/cifar100_resnet18_calibrated_ret0.7.yaml"
    "configs/cifar100_resnet18_raw_ret0.3.yaml"
    "configs/cifar100_resnet18_raw_ret0.5.yaml"
    "configs/cifar100_resnet18_raw_ret0.7.yaml"
    "configs/cifar100_resnet18_random_ret0.3.yaml"
    "configs/cifar100_resnet18_random_ret0.5.yaml"
    "configs/cifar100_resnet18_random_ret0.7.yaml"
)

echo "Configs: ${#CONFIGS[@]}, Seeds: ${#SEEDS[@]}, Total: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))"

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ">>> Running: $config seed=$seed"
        python scripts/run_train.py --config "$config" --seed "$seed"
        echo ">>> Done"
    done
done

echo "=== CIFAR-100 experiments complete ==="
python scripts/summarize_results.py outputs
