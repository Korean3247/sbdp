#!/bin/bash
# Run all 4 modes × 3 seeds × 3 retention ratios
# full: 3 runs (retention irrelevant)
# random/raw/calibrated: 3 seeds × 3 retentions = 9 each
# Total: 3 + 27 = 30 runs
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SEEDS=(0 1 2)

echo "=== SBDP: Running all experiments ==="

# Phase 1: Full baseline (no pruning, retention irrelevant)
echo ""
echo "--- Phase 1: Full baseline ---"
for seed in "${SEEDS[@]}"; do
    echo ">>> full seed=$seed"
    python scripts/run_train.py --config configs/cifar10_resnet18_full.yaml --seed "$seed"
done

# Phase 2: Pruning modes × retention ratios
PRUNING_CONFIGS=(
    "configs/cifar10_resnet18_random.yaml"
    "configs/cifar10_resnet18_random_ret0.3.yaml"
    "configs/cifar10_resnet18_random_ret0.7.yaml"
    "configs/cifar10_resnet18_raw.yaml"
    "configs/cifar10_resnet18_raw_ret0.3.yaml"
    "configs/cifar10_resnet18_raw_ret0.7.yaml"
    "configs/cifar10_resnet18_calibrated.yaml"
    "configs/cifar10_resnet18_calibrated_ret0.3.yaml"
    "configs/cifar10_resnet18_calibrated_ret0.7.yaml"
)

echo ""
echo "--- Phase 2: Pruning experiments ---"
for config in "${PRUNING_CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ">>> Running: $config seed=$seed"
        python scripts/run_train.py --config "$config" --seed "$seed"
        echo ">>> Done: $config seed=$seed"
        echo ""
    done
done

echo "=== All experiments complete ==="
echo "Summarizing results..."
python scripts/summarize_results.py outputs
