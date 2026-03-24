#!/bin/bash
# A-lite: Historical correction experiment
# 2 configs × 3 seeds = 6 runs
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIGS=(
    "configs/cifar10_resnet18_calibrated_historical_ret0.3.yaml"
    "configs/cifar10_resnet18_calibrated_historical_ret0.5.yaml"
)
SEEDS=(0 1 2)

echo "=== A-lite: Historical Correction Experiment ==="
echo "Configs: ${#CONFIGS[@]}, Seeds: ${#SEEDS[@]}, Total: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))"

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ">>> Running: $config seed=$seed"
        python scripts/run_train.py --config "$config" --seed "$seed"
        echo ">>> Done"
    done
done

echo "=== Historical correction experiments complete ==="
python scripts/summarize_results.py outputs
