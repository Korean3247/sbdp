#!/bin/bash
# Phase 2: Label Noise Robustness (noise_rate=0.2)
# 5 configs x 3 seeds = 15 runs
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIGS=(
    "configs/cifar10_resnet18_full_noise0.2.yaml"
    "configs/cifar10_resnet18_calibrated_noise0.2_ret0.3.yaml"
    "configs/cifar10_resnet18_calibrated_noise0.2_ret0.5.yaml"
    "configs/cifar10_resnet18_random_noise0.2_ret0.3.yaml"
    "configs/cifar10_resnet18_random_noise0.2_ret0.5.yaml"
)
SEEDS=(0 1 2)

echo "=== Phase 2: Label Noise Robustness ==="
echo "Configs: ${#CONFIGS[@]}, Seeds: ${#SEEDS[@]}, Total: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))"

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ">>> Running: $config seed=$seed"
        python scripts/run_train.py --config "$config" --seed "$seed"
        echo ">>> Done"
    done
done

echo "=== Label noise experiments complete ==="
python scripts/summarize_results.py outputs
